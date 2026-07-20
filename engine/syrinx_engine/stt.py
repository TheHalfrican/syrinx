"""Speech-to-text for dictation (whisper.cpp).

Deliberately *not* torch-based: whisper.cpp keeps dictation snappy and avoids
loading a second heavy runtime just to transcribe a few seconds of audio.
"""

import logging

log = logging.getLogger("syrinx.engine.stt")


class Transcriber:
    def __init__(self) -> None:
        self._ready = False

    async def load(self) -> None:
        log.info("loading whisper.cpp model (stub)")
        # TODO(syrinx): locate/download a ggml/gguf whisper model and init
        # whisper.cpp (via pywhispercpp, a ctypes binding, or a subprocess).
        self._ready = True

    async def transcribe(self, pcm: bytes) -> str:
        log.info("transcribe %d bytes (stub)", len(pcm))
        # TODO(syrinx): run whisper.cpp over the PCM buffer, return text.
        return ""
