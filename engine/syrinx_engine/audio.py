"""Audio playback via PipeWire.

CachyOS is PipeWire-native, so `sounddevice` (PortAudio) routes straight to it —
no GStreamer sinks. We stream the PCM block-by-block so we can emit a live RMS
level for the UI/pill waveform and stay cancellable between blocks.
"""

import asyncio
import logging
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("syrinx.engine.audio")

_BLOCK = 1024  # frames per write ≈ 43 ms at 24 kHz


def duration_of(pcm: bytes, sample_rate: int) -> float:
    """Seconds of float32 mono PCM (4 bytes/sample)."""
    return (len(pcm) // 4) / sample_rate if sample_rate else 0.0


def envelope(pcm: bytes, n: int = 100) -> list:
    """Downsample PCM to ``n`` peak-amplitude bars (0..1) for a static waveform."""
    samples = np.frombuffer(pcm, dtype=np.float32)
    if samples.size == 0:
        return [0.0] * n
    edges = np.linspace(0, samples.size, n + 1).astype(int)
    bars = [
        float(np.abs(samples[edges[k] : edges[k + 1]]).max()) if edges[k + 1] > edges[k] else 0.0
        for k in range(n)
    ]
    peak = max(bars)
    return [b / peak for b in bars] if peak > 0 else bars


def _play_blocking(
    data: np.ndarray,
    sample_rate: int,
    ctl,
    emit_level,
    emit_progress,
    volume,
) -> None:
    """The whole playback loop, in ONE worker thread that owns the stream.

    The stream must only ever be touched from this thread: closing a
    PortAudio stream from another thread while ``write()`` is blocked
    inside ALSA is a use-after-free (it segfaulted the engine mid-cancel,
    and once corrupted the heap into a delayed SIGABRT). Stop/pause/seek
    arrive via ``ctl`` flags checked between blocks.
    """
    import time

    import sounddevice as sd

    total = max(1, len(data))
    try:
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not open output stream (%s); skipping playback", exc)
        return

    stream.start()
    i = 0
    try:
        while i < len(data):
            if ctl.stop:
                break
            if ctl.seek is not None:
                i = min(len(data), max(0, int(ctl.seek * len(data))))
                ctl.seek = None
            if ctl.paused:
                time.sleep(0.05)
                continue
            block = data[i : i + _BLOCK]
            # Read the gain per block so a volume change applies mid-clip.
            gain = np.float32(max(0.0, min(1.0, volume()))) if volume is not None else None
            out = block if gain is None or gain == 1.0 else block * gain
            stream.write(out.reshape(-1, 1))
            if emit_level is not None and len(block):
                emit_level(float(np.sqrt(np.mean(np.square(block)))))
            if emit_progress is not None:
                emit_progress(min(1.0, (i + len(block)) / total))
            i += _BLOCK
    finally:
        # Teardown in the owning thread — always runs, never raced.
        try:
            stream.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


async def play(
    pcm: bytes,
    sample_rate: int,
    ctl=None,
    on_level: Optional[Callable[[float], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    volume: Optional[Callable[[], float]] = None,
) -> None:
    """Play float32 PCM cooperatively.

    ``ctl`` has ``stop`` / ``paused`` bool attrs and a ``seek`` (0..1 or
    None) checked between blocks. Task cancellation never touches the
    stream — it sets ``ctl.stop`` and drains the playback thread, which
    tears the stream down itself.
    """
    if not pcm:
        return

    try:
        import sounddevice  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        log.warning("audio unavailable (%s); skipping playback", exc)
        return

    data = np.frombuffer(pcm, dtype=np.float32)
    loop = asyncio.get_running_loop()
    if ctl is None:
        ctl = type("Ctl", (), {"stop": False, "paused": False, "seek": None})()

    def _emit(cb):
        # signal emission must happen on the event-loop thread
        def send(v):
            try:
                loop.call_soon_threadsafe(cb, v)
            except RuntimeError:
                pass  # loop already closed (engine shutdown)

        return send

    fut = loop.run_in_executor(
        None,
        _play_blocking,
        data,
        sample_rate,
        ctl,
        _emit(on_level) if on_level is not None else None,
        _emit(on_progress) if on_progress is not None else None,
        volume,
    )
    try:
        await fut
    except asyncio.CancelledError:
        ctl.stop = True
        # Hold the caller (and its audio lock) until the thread has closed
        # the stream — at most one block write plus teardown away.
        try:
            await asyncio.wait_for(asyncio.shield(fut), timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        raise
