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


async def play(
    pcm: bytes,
    sample_rate: int,
    on_level: Optional[Callable[[float], None]] = None,
) -> None:
    """Play float32 PCM, invoking ``on_level(rms)`` per block for visualization."""
    if not pcm:
        return

    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        log.warning("audio unavailable (%s); skipping playback", exc)
        return

    data = np.frombuffer(pcm, dtype=np.float32)
    loop = asyncio.get_running_loop()

    try:
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not open output stream (%s); skipping playback", exc)
        return

    stream.start()
    try:
        for i in range(0, len(data), _BLOCK):
            block = data[i : i + _BLOCK]
            # Blocking write off the event loop so signals/Cancel stay responsive.
            await loop.run_in_executor(None, stream.write, block.reshape(-1, 1))
            if on_level is not None and len(block):
                on_level(float(np.sqrt(np.mean(np.square(block)))))
    finally:
        await loop.run_in_executor(None, stream.stop)
        stream.close()
