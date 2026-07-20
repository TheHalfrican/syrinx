"""Qwen3-TTS backend — zero-shot voice cloning.

The beefy engine: clones a voice from a short reference sample (+ its transcript)
and speaks arbitrary text in it. ~3.5 GB (1.7B) / ~1.2 GB (0.6B). Wants a GPU —
this is the one that lights up on the RTX 4090 (bf16 + TF32 + flash attention).
On CPU it still runs (float32), just slowly.

Cloned voices persist to ``$SYRINX_DATA_DIR/voices`` (default
``~/.local/share/syrinx/voices``): one ``<id>.pt`` per prompt + an ``index.json``
mapping id -> display name.

Grounded in the Voicebox pytorch_backend.py reference:
    Qwen3TTSModel.from_pretrained(...)
    model.create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)
    model.generate_voice_clone(text, voice_clone_prompt, language, instruct) -> (wavs, sr)

NOTE: not exercised on this iGPU box — validated on the 4090. Marked TODO(syrinx)
where a live-on-GPU check is still needed.
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

from . import VoiceInfo, detect_device

log = logging.getLogger("syrinx.engine.tts.qwen")

MODELS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "voice"


class QwenBackend:
    supports_cloning = True

    def __init__(self) -> None:
        self.device = detect_device()
        self.model_size = os.environ.get("SYRINX_MODEL", "1.7B")
        if self.model_size not in MODELS:
            self.model_size = "1.7B"
        self._model = None
        self._prompts: dict[str, object] = {}  # voice_id -> loaded prompt (cache)
        data_dir = os.environ.get(
            "SYRINX_DATA_DIR", str(Path.home() / ".local" / "share" / "syrinx")
        )
        self._voices_dir = Path(data_dir) / "voices"
        self._voices_dir.mkdir(parents=True, exist_ok=True)

    # --- model ----------------------------------------------------------

    async def load(self) -> None:
        if self._model is not None:
            return
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        model_path = MODELS[self.model_size]
        if self.device in ("cuda", "rocm"):
            # Ada/RTX 4090 fast path — see docs/HARDWARE.md.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except Exception:  # noqa: BLE001
                pass
            log.info("loading Qwen3-TTS %s on %s (bf16)...", self.model_size, self.device)
            self._model = Qwen3TTSModel.from_pretrained(
                model_path, device_map=self.device, torch_dtype=torch.bfloat16
            )
        else:
            log.info("loading Qwen3-TTS %s on cpu (float32 — slow)...", self.model_size)
            self._model = Qwen3TTSModel.from_pretrained(
                model_path, torch_dtype=torch.float32, low_cpu_mem_usage=False
            )
        log.info("Qwen3-TTS %s loaded", self.model_size)

    # --- voices / cloning ----------------------------------------------

    def _index_path(self) -> Path:
        return self._voices_dir / "index.json"

    def _read_index(self) -> dict[str, str]:
        try:
            return json.loads(self._index_path().read_text())
        except Exception:  # noqa: BLE001
            return {}

    def _write_index(self, index: dict[str, str]) -> None:
        self._index_path().write_text(json.dumps(index, indent=2))

    async def list_voices(self) -> list[VoiceInfo]:
        # Qwen has no presets — only voices the user has cloned.
        return [VoiceInfo(vid, name) for vid, name in self._read_index().items()]

    async def clone(self, name: str, sample_path: str, ref_text: str = "") -> str:
        await self.load()
        if not ref_text:
            # Qwen needs the transcript of the reference clip. Once the dictate
            # pill's whisper.cpp lands we can auto-transcribe here. TODO(syrinx).
            raise ValueError("Qwen cloning needs ref_text (transcript of the sample)")

        voice_id = _slug(name)

        def _make() -> object:
            return self._model.create_voice_clone_prompt(
                ref_audio=str(sample_path), ref_text=ref_text, x_vector_only_mode=False
            )

        prompt = await asyncio.to_thread(_make)

        import torch

        torch.save(prompt, self._voices_dir / f"{voice_id}.pt")
        index = self._read_index()
        index[voice_id] = name
        self._write_index(index)
        self._prompts[voice_id] = prompt
        log.info("cloned voice %r -> %s", name, voice_id)
        return voice_id

    def invalidate_profile(self, profile_id: str) -> None:
        """Forget a profile's cached clone prompt so it rebuilds from samples."""
        self._prompts.pop(profile_id, None)
        (self._voices_dir / f"{profile_id}.pt").unlink(missing_ok=True)
        (self._voices_dir / f"{profile_id}_combined.wav").unlink(missing_ok=True)

    def _get_prompt(self, voice_id: str):
        if voice_id not in self._prompts:
            import torch

            path = self._voices_dir / f"{voice_id}.pt"
            if not path.exists():
                raise ValueError(f"unknown cloned voice: {voice_id}")
            self._prompts[voice_id] = torch.load(
                path, map_location=self.device, weights_only=False
            )
        return self._prompts[voice_id]

    # --- synthesis ------------------------------------------------------

    async def synthesize(self, text: str, voice_id: str) -> tuple[bytes, int]:
        await self.load()
        prompt = self._get_prompt(voice_id)

        def _run() -> tuple[bytes, int]:
            wavs, sample_rate = self._model.generate_voice_clone(
                text=text, voice_clone_prompt=prompt, language="english", instruct=None
            )
            audio = wavs[0]
            try:
                import torch

                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
            except Exception:  # noqa: BLE001
                pass
            return np.asarray(audio).astype(np.float32).tobytes(), int(sample_rate)

        log.info("synthesize [qwen %s] (%s): %r", self.model_size, voice_id, text[:60])
        return await asyncio.to_thread(_run)

    # --- profile-based cloning -----------------------------------------

    def _combine_samples(self, profile) -> tuple[str, str]:
        """Concatenate a profile's reference samples into one WAV + joined text."""
        import soundfile as sf

        audios, texts, rate = [], [], 24_000
        for s in profile.samples:
            audio, rate = sf.read(s.audio_path, dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)  # to mono
            audios.append(audio)
            if s.reference_text:
                texts.append(s.reference_text)
        if not audios:
            raise ValueError(f"profile {profile.id} has no samples to clone from")
        combined = np.concatenate(audios).astype(np.float32)
        out = self._voices_dir / f"{profile.id}_combined.wav"
        sf.write(out, combined, rate)
        return str(out), " ".join(texts)

    def _profile_prompt(self, profile):
        key = profile.id
        if key in self._prompts:
            return self._prompts[key]
        import torch

        cache = self._voices_dir / f"{key}.pt"
        if cache.exists():
            self._prompts[key] = torch.load(cache, map_location=self.device, weights_only=False)
            return self._prompts[key]
        audio, text = self._combine_samples(profile)
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=audio, ref_text=text, x_vector_only_mode=False
        )
        torch.save(prompt, cache)
        self._prompts[key] = prompt
        return prompt

    async def synthesize_profile(self, profile, text: str, instruct: str = "") -> tuple[bytes, int]:
        await self.load()

        def _run() -> tuple[bytes, int]:
            prompt = self._profile_prompt(profile)
            wavs, sample_rate = self._model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt,
                language="english",
                instruct=instruct or None,
            )
            audio = wavs[0]
            try:
                import torch

                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
            except Exception:  # noqa: BLE001
                pass
            return np.asarray(audio).astype(np.float32).tobytes(), int(sample_rate)

        log.info("synthesize_profile [qwen %s] (%s): %r", self.model_size, profile.id, text[:60])
        return await asyncio.to_thread(_run)
