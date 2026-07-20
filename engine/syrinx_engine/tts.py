"""Text-to-speech + voice cloning (Qwen3-TTS via torch).

Stub: the real implementation loads Qwen3-TTS and synthesizes PCM. Backend is
auto-detected from the torch build installed by the system package.
"""

import logging
from dataclasses import dataclass

log = logging.getLogger("syrinx.engine.tts")


@dataclass
class VoiceInfo:
    id: str
    name: str


def _detect_backend() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            # ROCm builds also report through the cuda API.
            return "rocm" if torch.version.hip else "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class SpeechSynthesizer:
    def __init__(self) -> None:
        self.backend = _detect_backend()
        self._model = None

    async def load(self) -> None:
        log.info("loading Qwen3-TTS on %s (stub)", self.backend)
        # TODO(syrinx): load Qwen3-TTS weights onto the detected device.
        self._model = object()

    async def list_voices(self) -> list[VoiceInfo]:
        # TODO(syrinx): built-ins + cloned profiles from the DB/config dir.
        return [VoiceInfo("default", "Default"), VoiceInfo("narrator", "Narrator")]

    async def synthesize(self, text: str, voice_id: str) -> bytes:
        log.info("synthesize (%s): %r (stub)", voice_id, text[:60])
        # TODO(syrinx): run the model -> float32/int16 PCM at the engine rate.
        return b""

    async def clone(self, name: str, sample_path: str) -> str:
        log.info("clone %r from %s (stub)", name, sample_path)
        # TODO(syrinx): extract speaker embedding, persist profile, return id.
        return name.lower().replace(" ", "-")
