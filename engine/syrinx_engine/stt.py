"""Speech-to-text for dictation.

faster-whisper (CTranslate2): CPU-fast with int8, auto-CUDA (float16) on the
4090. Kept deliberately light and torch-independent. Swappable for a whisper.cpp
binding later behind this same interface.

`transcribe()` takes a file PATH (the dictate pill records a WAV) rather than raw
PCM, so we don't marshal seconds of audio over D-Bus.
"""

import asyncio
import logging
import os

log = logging.getLogger("syrinx.engine.stt")


class Transcriber:
    def __init__(self) -> None:
        self._model = None
        self.model_size = os.environ.get("SYRINX_WHISPER_MODEL", "base.en")

    def set_model(self, identifier: str) -> None:
        """Switch the whisper model (a size name or a HF repo); reloads lazily."""
        if identifier and identifier != self.model_size:
            self.model_size = identifier
            self._model = None

    async def load(self) -> None:
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from faster_whisper import WhisperModel

        try:
            import torch

            cuda = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            cuda = False

        device = "cuda" if cuda else "cpu"
        compute = "float16" if cuda else "int8"
        log.info("loading faster-whisper %s on %s (%s)...", self.model_size, device, compute)
        self._model = WhisperModel(self.model_size, device=device, compute_type=compute)
        log.info("faster-whisper loaded")

    async def transcribe(self, audio_path: str) -> str:
        await self.load()

        def _run() -> str:
            segments, _info = self._model.transcribe(
                audio_path, language="en", vad_filter=True
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        text = await asyncio.to_thread(_run)
        log.info("transcribed: %r", text[:80])
        return text

    async def transcribe_stream(self, audio_path: str, on_partial=None) -> str:
        """Transcribe a (possibly long) file, invoking ``on_partial(text_so_far)``
        on the event loop as each segment decodes — the Transcription view shows
        text arriving live instead of a spinner."""
        await self.load()
        loop = asyncio.get_running_loop()

        def _run() -> str:
            segments, _info = self._model.transcribe(
                audio_path, language="en", vad_filter=True
            )
            parts = []
            for seg in segments:
                parts.append(seg.text.strip())
                if on_partial is not None:
                    loop.call_soon_threadsafe(on_partial, " ".join(parts))
            return " ".join(parts).strip()

        text = await asyncio.to_thread(_run)
        log.info("transcribed (stream): %r", text[:80])
        return text
