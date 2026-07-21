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


async def play(
    pcm: bytes,
    sample_rate: int,
    ctl=None,
    on_level: Optional[Callable[[float], None]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> None:
    """Play float32 PCM cooperatively.

    ``ctl`` (optional) has ``stop`` / ``paused`` bool attrs and a ``seek`` (0..1
    or None) checked between blocks — enabling clean stop, pause/resume and seek
    without cancelling the task (task-cancellation mid-stream corrupts PortAudio).
    Stream teardown is synchronous so it always completes.
    """
    if not pcm:
        return

    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        log.warning("audio unavailable (%s); skipping playback", exc)
        return

    data = np.frombuffer(pcm, dtype=np.float32)
    total = max(1, len(data))
    loop = asyncio.get_running_loop()

    try:
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not open output stream (%s); skipping playback", exc)
        return

    stream.start()
    i = 0
    try:
        while i < len(data):
            if ctl is not None:
                if ctl.stop:
                    break
                if ctl.seek is not None:
                    i = min(len(data), max(0, int(ctl.seek * len(data))))
                    ctl.seek = None
                if ctl.paused:
                    await asyncio.sleep(0.05)
                    continue
            block = data[i : i + _BLOCK]
            # Blocking write off the event loop so signals stay responsive.
            await loop.run_in_executor(None, stream.write, block.reshape(-1, 1))
            if on_level is not None and len(block):
                on_level(float(np.sqrt(np.mean(np.square(block)))))
            if on_progress is not None:
                on_progress(min(1.0, (i + len(block)) / total))
            i += _BLOCK
    finally:
        # Synchronous teardown — never interrupted by cancellation.
        try:
            stream.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass
