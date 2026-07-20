"""Text-to-speech.

Increment A: real synthesis via **Kokoro-82M** — tiny, CPU-realtime, 24 kHz,
40+ preset voices (Apache-2.0). This is the lightweight engine that runs great
on the laptop's CPU. Heavier voice-cloning engines (Qwen3-TTS / LuxTTS on the
4090) can be added later as additional backends behind the same interface.

`synthesize()` returns ``(pcm_float32_bytes, sample_rate)`` — unchanged, so the
D-Bus service, PipeWire playback, and Slint UI need no changes.

Reference: the Voicebox backend's kokoro_backend.py (kept at
~/Documents/FromSource/voicebox for exactly this).
"""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("syrinx.engine.tts")

KOKORO_HF_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000
DEFAULT_VOICE = "af_heart"

# Curated English voices (id, display). The Kokoro id's first letter is its
# pipeline lang_code (a/b = US/UK English), so we derive language from the id.
VOICES: list[tuple[str, str]] = [
    ("af_heart", "Heart (US ♀)"),
    ("af_bella", "Bella (US ♀)"),
    ("af_nova", "Nova (US ♀)"),
    ("af_sarah", "Sarah (US ♀)"),
    ("am_adam", "Adam (US ♂)"),
    ("am_michael", "Michael (US ♂)"),
    ("am_onyx", "Onyx (US ♂)"),
    ("am_puck", "Puck (US ♂)"),
    ("bf_emma", "Emma (UK ♀)"),
    ("bf_isabella", "Isabella (UK ♀)"),
    ("bm_george", "George (UK ♂)"),
    ("bm_fable", "Fable (UK ♂)"),
]


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
        self._pipelines: dict[str, object] = {}  # lang_code -> KPipeline

    async def load(self) -> None:
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from kokoro import KModel

        device = "cuda" if self.backend in ("cuda", "rocm") else "cpu"
        log.info("loading Kokoro-82M on %s (first run downloads ~330MB)...", device)
        self._model = KModel(repo_id=KOKORO_HF_REPO).to(device).eval()
        log.info("Kokoro-82M loaded")

    def _pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            from kokoro import KPipeline

            self._pipelines[lang_code] = KPipeline(
                lang_code=lang_code, repo_id=KOKORO_HF_REPO, model=self._model
            )
        return self._pipelines[lang_code]

    async def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(vid, name) for vid, name in VOICES]

    async def synthesize(self, text: str, voice_id: str) -> tuple[bytes, int]:
        await self.load()
        voice = voice_id if any(voice_id == v for v, _ in VOICES) else DEFAULT_VOICE
        lang_code = voice[0]  # Kokoro voice-id prefix is its pipeline lang_code

        def _run() -> tuple[bytes, int]:
            chunks: list[np.ndarray] = []
            for result in self._pipeline(lang_code)(text, voice=voice, speed=1.0):
                if result.audio is not None:
                    audio = result.audio
                    try:
                        import torch

                        if isinstance(audio, torch.Tensor):
                            audio = audio.detach().cpu().numpy()
                    except Exception:  # noqa: BLE001
                        pass
                    chunks.append(np.asarray(audio).squeeze())
            if not chunks:
                return b"", SAMPLE_RATE
            pcm = np.concatenate(chunks).astype(np.float32)
            return pcm.tobytes(), SAMPLE_RATE

        log.info("synthesize (%s): %r", voice, text[:60])
        return await asyncio.to_thread(_run)

    async def clone(self, name: str, sample_path: str) -> str:
        # Kokoro uses preset voices, not zero-shot cloning. Real cloning will
        # come with a Qwen3-TTS / LuxTTS backend (4090). TODO(syrinx).
        log.info("clone requested (%r) — not supported by Kokoro backend", name)
        return DEFAULT_VOICE
