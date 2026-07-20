"""Text-to-speech facade.

Delegates to a backend selected by ``SYRINX_TTS_ENGINE`` (default: kokoro):
  - kokoro — preset voices, CPU-realtime (this box).
  - qwen   — Qwen3-TTS zero-shot voice cloning (the RTX 4090).

The public interface is unchanged, so the D-Bus service, PipeWire playback, and
Slint UI don't care which engine is active. Backends live in ``backends/``.
"""

import logging

from .backends import VoiceInfo, make_backend  # noqa: F401  (VoiceInfo re-exported)

log = logging.getLogger("syrinx.engine.tts")


class SpeechSynthesizer:
    def __init__(self) -> None:
        self._backend = make_backend()
        # Exposed as the D-Bus `Backend` property (cuda | rocm | cpu).
        self.backend = self._backend.device
        self.supports_cloning = self._backend.supports_cloning
        log.info(
            "TTS engine=%s device=%s cloning=%s",
            type(self._backend).__name__,
            self.backend,
            self.supports_cloning,
        )

    async def load(self) -> None:
        await self._backend.load()

    async def list_voices(self) -> list[VoiceInfo]:
        return await self._backend.list_voices()

    async def synthesize(self, text: str, voice_id: str) -> tuple[bytes, int]:
        return await self._backend.synthesize(text, voice_id)

    async def clone(self, name: str, sample_path: str, ref_text: str = "") -> str:
        return await self._backend.clone(name, sample_path, ref_text)
