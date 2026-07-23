"""Text-to-speech router.

Voices come from two places:
  - **built-in presets** from a preset engine (Kokoro) — always available.
  - **user profiles** (ProfileStore) — preset or cloned.

`synthesize(voice_id, ...)` routes each voice to the right backend:
  - "builtin:<engine>:<voice>"  -> that preset engine
  - a profile id (preset)        -> the profile's preset engine
  - a profile id (cloned)        -> the profile's cloning engine (Qwen/…), which
                                    builds a voice prompt from the profile samples

Backends live in `backends/` and are selected per-voice, lazily instantiated.
`SYRINX_TTS_ENGINE` still sets the default CLONING engine for new cloned voices.
"""

import logging
import os

from .backends import VoiceInfo, detect_device, make_backend
from .profiles import Profile, ProfileStore

log = logging.getLogger("syrinx.engine.tts")

# Preset engine whose built-in voices are always offered in the voice list.
BUILTIN_PRESET_ENGINE = "kokoro"
# Default engine used when cloning a new voice.
DEFAULT_CLONE_ENGINE = os.environ.get("SYRINX_TTS_ENGINE", "qwen")
if DEFAULT_CLONE_ENGINE == "kokoro":  # kokoro can't clone; fall back
    DEFAULT_CLONE_ENGINE = "qwen"
# Engines capable of zero-shot cloning (preset-only engines can't be the
# active clone engine, e.g. kokoro / qwen_custom_voice).
CLONING_ENGINES = {"qwen", "luxtts", "chatterbox", "chatterbox_turbo", "tada"}
# Preset engines beyond the always-on builtin whose voices are listed only
# while they're the active voice model (Models-tab "Use").
EXTRA_PRESET_ENGINES = {"qwen_custom_voice"}
# Voice-conversion engines (the ⇄ tab) — audio→audio, never in the voice
# list. Vevo2 (unified speech+singing) rides the vevo worker as a mode when
# the music pipeline lands.
VC_ENGINES = {"chatterbox_vc", "seed_vc", "vevo_timbre"}
DEFAULT_VC_ENGINE = "chatterbox_vc"


class SpeechSynthesizer:
    def __init__(self, profiles: ProfileStore) -> None:
        self._profiles = profiles
        self._backends: dict[str, object] = {}
        self.backend = detect_device()  # exposed as the D-Bus Backend property
        self.supports_cloning = True
        # Set from the Models tab ("Use" on a voice model). Profiles with an
        # explicit default_engine override it; "" falls through to the env default.
        self._clone_engine = ""
        # Extra preset engine (CustomVoice) whose voices join the list while
        # it's the active voice model.
        self._preset_engine = ""
        # Model size per engine (qwen 1.7B/0.6B, tada 1B/3B, …), recorded from
        # the Models tab so "Use" on a size variant actually takes effect.
        self._voice_sizes: dict[str, str] = {}

    def set_clone_engine(self, engine: str) -> None:
        self._clone_engine = engine if engine in CLONING_ENGINES else ""
        log.info("active clone engine -> %r", self._clone_engine or DEFAULT_CLONE_ENGINE)

    def set_voice_engine(self, engine: str, size: str = "") -> None:
        """Models-tab "Use" on a voice model — exactly one is active at a time:
        a cloning engine becomes the clone engine (and unlists any extra preset
        engine); an extra preset engine lists its voices without touching the
        clone routing. Backends for models no longer selected are evicted."""
        if size:
            self._voice_sizes[engine] = size
            be = self._backends.get(engine)
            if be is not None and getattr(be, "model_size", size) != size:
                # same engine, different size — rebuild on next use
                self._backends.pop(engine)
                unload = getattr(be, "unload", None)
                if unload:
                    unload()
                log.info("%s backend will reload at size %s", engine, size)
        if engine in EXTRA_PRESET_ENGINES:
            self._preset_engine = engine
            log.info("active preset engine -> %r", engine)
        else:
            self._preset_engine = ""
            self.set_clone_engine(engine)
        self._evict_voice_backends(keep={engine})

    def _evict_voice_backends(self, keep: set) -> None:
        """Unload voice backends that are no longer selected so their VRAM
        comes back — seven GPU engines don't fit on one card. Profiles pinned
        to an evicted engine reload it on their next generation."""
        evicted = []
        for name in list(self._backends):
            if name == BUILTIN_PRESET_ENGINE or name in keep:
                continue
            be = self._backends.pop(name)
            unload = getattr(be, "unload", None)
            try:
                if unload:
                    unload()
                evicted.append(name)
            except Exception:  # noqa: BLE001
                log.exception("unload %s failed", name)
        if evicted:
            log.info("evicted voice backends: %s", ", ".join(evicted))

    @property
    def clone_engine(self) -> str:
        return self._clone_engine or DEFAULT_CLONE_ENGINE

    def vc_backend(self, engine: str = ""):
        """Voice-conversion backend (ConvertVoice). Lives in the shared
        backend dict, so the Models-tab eviction sweep reclaims its VRAM
        too; it reloads lazily on the next convert."""
        engine = engine or DEFAULT_VC_ENGINE
        if engine not in VC_ENGINES:
            raise ValueError(
                f"unknown VC engine {engine!r} "
                f"(expected: {', '.join(sorted(VC_ENGINES))})"
            )
        if engine not in self._backends:
            if engine == "seed_vc":
                from .backends.seed_vc import SeedVCBackend

                self._backends[engine] = SeedVCBackend()
            elif engine == "vevo_timbre":
                from .backends.vevo import VevoTimbreBackend

                self._backends[engine] = VevoTimbreBackend()
            else:
                from .backends.chatterbox_vc import ChatterboxVCBackend

                self._backends[engine] = ChatterboxVCBackend()
        return self._backends[engine]

    def _be(self, engine: str):
        if engine not in self._backends:
            self._backends[engine] = make_backend(engine, self._voice_sizes.get(engine, ""))
        return self._backends[engine]

    async def load(self) -> None:
        # Warm the built-in preset engine so preset voices are instant.
        await self._be(BUILTIN_PRESET_ENGINE).load()

    async def list_voices(self) -> list[VoiceInfo]:
        voices: list[VoiceInfo] = []
        for v in await self._be(BUILTIN_PRESET_ENGINE).list_voices():
            voices.append(VoiceInfo(f"builtin:{BUILTIN_PRESET_ENGINE}:{v.id}", v.name))
        if self._preset_engine:
            for v in await self._be(self._preset_engine).list_voices():
                voices.append(VoiceInfo(f"builtin:{self._preset_engine}:{v.id}", v.name))
        for p in self._profiles.list():
            voices.append(VoiceInfo(p.id, p.name))
        return voices

    async def synthesize(self, text: str, voice_id: str, instruct: str = "") -> tuple[bytes, int]:
        if voice_id.startswith("builtin:"):
            _, engine, vid = voice_id.split(":", 2)
            return await self._be(engine).synthesize(text, vid, instruct)

        prof = self._profiles.get(voice_id)
        if prof is None:
            # Back-compat: treat an unknown id as a raw built-in preset voice.
            return await self._be(BUILTIN_PRESET_ENGINE).synthesize(text, voice_id, instruct)

        if prof.voice_type == "preset":
            engine = prof.preset_engine or BUILTIN_PRESET_ENGINE
            return await self._be(engine).synthesize(text, prof.preset_voice_id, instruct)

        # cloned
        be = self._be(prof.default_engine or self.clone_engine)
        return await be.synthesize_profile(prof, text, instruct)

    async def clone(self, name: str, sample_path: str, ref_text: str = "") -> str:
        """Legacy CloneVoice: create a cloned profile with a single sample."""
        # default_engine stays "" so the voice follows the active clone engine;
        # UpdateProfile can pin one explicitly later.
        pid = self._profiles.create(name, "cloned")
        self._profiles.add_sample(pid, sample_path, ref_text)
        return pid

    def invalidate_profile(self, profile_id: str) -> None:
        """Drop any cached clone prompt for a profile (e.g. after samples change)."""
        for be in self._backends.values():
            inv = getattr(be, "invalidate_profile", None)
            if inv:
                inv(profile_id)
