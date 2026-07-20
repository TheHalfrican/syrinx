"""Text-to-speech + voice cloning.

Backend is auto-detected from the installed torch build. `synthesize()` returns
``(pcm_float32_bytes, sample_rate)``.

NOTE: synthesis is a PLACEHOLDER tone until Qwen3-TTS is wired in — it lets the
whole audio path (D-Bus -> PipeWire -> level meter) work end-to-end first.
Marked TODO(syrinx) at the real integration points.
"""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("syrinx.engine.tts")

SAMPLE_RATE = 24_000


@dataclass
class VoiceInfo:
    id: str
    name: str


def _detect_backend() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class SpeechSynthesizer:
    def __init__(self) -> None:
        self.backend = _detect_backend()
        self._model = None

    async def load(self) -> None:
        log.info("loading Qwen3-TTS on %s (placeholder)", self.backend)
        # TODO(syrinx): load Qwen3-TTS weights onto the detected device.
        self._model = object()

    async def list_voices(self) -> list[VoiceInfo]:
        # TODO(syrinx): built-ins + cloned profiles from the config dir/DB.
        return [VoiceInfo("default", "Default"), VoiceInfo("narrator", "Narrator")]

    async def synthesize(self, text: str, voice_id: str) -> tuple[bytes, int]:
        # TODO(syrinx): replace with Qwen3-TTS. For now, a short decaying chord
        # whose length scales with the text — enough to exercise playback.
        log.info("synthesize (%s): %r (placeholder)", voice_id, text[:60])
        dur = float(np.clip(0.4 + len(text) * 0.045, 0.4, 4.0))
        t = np.linspace(0.0, dur, int(dur * SAMPLE_RATE), endpoint=False)
        base = 174.6 if voice_id == "narrator" else 220.0  # F3 vs A3
        wave = (
            0.5 * np.sin(2 * np.pi * base * t)
            + 0.3 * np.sin(2 * np.pi * base * 1.5 * t)
            + 0.2 * np.sin(2 * np.pi * base * 2.0 * t)
        )
        envelope = np.exp(-t * 1.8)  # gentle decay
        pcm = (0.25 * wave * envelope).astype(np.float32)
        return pcm.tobytes(), SAMPLE_RATE

    async def clone(self, name: str, sample_path: str) -> str:
        log.info("clone %r from %s (stub)", name, sample_path)
        # TODO(syrinx): extract speaker embedding, persist profile, return id.
        return name.lower().replace(" ", "-")
