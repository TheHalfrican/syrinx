"""Kokoro-82M backend — preset voices, CPU-realtime, 24 kHz.

Kokoro uses pre-built voice style vectors (not zero-shot cloning). Its voice-id
prefix is also its pipeline lang_code (a/b = US/UK English).
Ported from the Voicebox kokoro_backend.py reference.
"""

import asyncio
import logging

import numpy as np

from . import VoiceInfo, detect_device

log = logging.getLogger("syrinx.engine.tts.kokoro")

KOKORO_HF_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000
DEFAULT_VOICE = "af_heart"

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


class KokoroBackend:
    supports_cloning = False

    def __init__(self) -> None:
        self.device = detect_device()
        self._model = None
        self._pipelines: dict[str, object] = {}

    async def load(self) -> None:
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from kokoro import KModel

        device = "cuda" if self.device in ("cuda", "rocm") else "cpu"
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
        lang_code = voice[0]

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
            return np.concatenate(chunks).astype(np.float32).tobytes(), SAMPLE_RATE

        log.info("synthesize (%s): %r", voice, text[:60])
        return await asyncio.to_thread(_run)

    async def clone(self, name: str, sample_path: str, ref_text: str = "") -> str:
        log.info("clone requested (%r) — Kokoro is preset-only; use SYRINX_TTS_ENGINE=qwen", name)
        return DEFAULT_VOICE
