"""Audio playback via PipeWire.

CachyOS is PipeWire-native, so we play straight to it (no GStreamer sinks).
`sounddevice` talks to PipeWire's PulseAudio/ALSA-compat layer; a future
version can use libpipewire directly for lower latency.
"""

import logging
from typing import Callable

log = logging.getLogger("syrinx.engine.audio")


async def play(pcm: bytes, on_level: Callable[[float], None] | None = None) -> None:
    """Play a PCM buffer, invoking `on_level(rms)` per frame for visualization."""
    if not pcm:
        return
    log.info("play %d bytes (stub)", len(pcm))
    # TODO(syrinx): stream via sounddevice.OutputStream; compute per-block RMS
    # and call on_level(rms) so the UI/pill can animate the waveform.
    if on_level:
        on_level(0.0)
