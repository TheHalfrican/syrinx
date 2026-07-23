"""ChatterboxVC — style-preserved voice conversion (the ⇄ tab), in-process.

Resemble's VC model is the S3 half of Chatterbox TTS on its own: the S3
tokenizer turns source speech into 25 Hz tokens that keep the words, timing
and prosody but shed timbre; S3Gen re-renders those tokens in the target
speaker's voice from a ~10 s reference embedding. It loads only
``s3gen.safetensors`` + ``conds.pt`` from the SAME ``ResembleAI/chatterbox``
snapshot the TTS backend caches — no extra downloads, ~1 GB VRAM, no T3 LLM.

Not a TTS engine: audio→audio only, never in the voice list. It joins the
synthesizer's backend dict as ``chatterbox_vc`` so the Models-tab eviction
sweep reclaims its VRAM like everything else (it reloads on the next convert).
Phase 2 adds seed_vc / vevo backends behind the same ``convert`` interface.
"""

import asyncio
import logging
import os

from .chatterbox import _ChatterboxBase, _as_f32, _patch_f32_mel

log = logging.getLogger("syrinx.engine.vc.chatterbox")


def max_source_secs() -> float:
    """Conversion source cap — S3Gen memory scales with source length.
    Chunked conversion (silence-boundary splits) is the planned lift."""
    try:
        return float(os.environ.get("SYRINX_VC_MAX_SECS", "180"))
    except ValueError:
        return 180.0


def _source_secs(path: str) -> float:
    """Best-effort duration probe; -1 for containers libsndfile can't read
    (m4a/webm) — those go straight to generate, which decodes via librosa."""
    try:
        import soundfile as sf

        info = sf.info(path)
        return info.frames / float(info.samplerate or 1)
    except Exception:  # noqa: BLE001
        try:
            import librosa

            return float(librosa.get_duration(path=path))
        except Exception:  # noqa: BLE001
            return -1.0


def check_source_cap(source_wav: str) -> None:
    """Reject over-cap sources before any model load is paid (shared by all
    VC backends)."""
    secs = _source_secs(source_wav)
    cap = max_source_secs()
    if secs > cap:
        raise ValueError(
            f"source is {secs:.0f} s — the conversion cap is {cap:.0f} s "
            "(SYRINX_VC_MAX_SECS)"
        )


class ChatterboxVCBackend(_ChatterboxBase):
    """chatterbox.vc.ChatterboxVC behind the small async VC interface:
    check_source / load / convert / unload / invalidate_profile."""

    engine_name = "chatterbox_vc"
    supports_cloning = False  # conversion only — not a cloning TTS engine

    def _load_sync(self) -> None:
        try:
            from chatterbox.vc import ChatterboxVC
        except ImportError as e:
            raise RuntimeError(
                "chatterbox-tts is not installed on this machine — "
                "voice conversion needs the GPU box"
            ) from e

        log.info("loading ChatterboxVC on %s...", self._torch_device())
        model = self._load_on(lambda dev: ChatterboxVC.from_pretrained(dev))
        # same float64 mel gotcha as the TTS backends; VC has no VoiceEncoder
        _patch_f32_mel(model.s3gen.tokenizer)
        self._model = model
        log.info("ChatterboxVC loaded")

    def check_source(self, source_wav: str) -> None:
        check_source_cap(source_wav)

    async def convert(self, source_wav: str, profile) -> tuple[bytes, int]:
        """Re-render *source_wav* in *profile*'s voice; (pcm_f32_bytes, rate).

        The target reference is the profile's combined multi-sample wav
        (``_ref_wav``, shared cache with the Chatterbox TTS backends);
        ChatterboxVC uses its first 10 s for the speaker embedding.
        """
        self.check_source(source_wav)
        await self.load()
        ref = await asyncio.to_thread(self._ref_wav, profile)

        def _run() -> tuple[bytes, int]:
            wav = self._model.generate(source_wav, target_voice_path=ref)
            rate = int(getattr(self._model, "sr", 24_000))
            return _as_f32(wav).tobytes(), rate

        log.info("convert [%s] -> profile %s", self.engine_name, profile.id)
        return await asyncio.to_thread(_run)
