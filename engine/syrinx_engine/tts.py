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


class SpeechSynthesizer:
    def __init__(self, profiles: ProfileStore) -> None:
        self._profiles = profiles
        self._backends: dict[str, object] = {}
        self.backend = detect_device()  # exposed as the D-Bus Backend property
        self.supports_cloning = True
        # Set from the Models tab ("Use" on a voice model). Profiles with an
        # explicit default_engine override it; "" falls through to the env default.
        self._clone_engine = ""

    def set_clone_engine(self, engine: str) -> None:
        self._clone_engine = engine if engine in CLONING_ENGINES else ""
        log.info("active clone engine -> %r", self._clone_engine or DEFAULT_CLONE_ENGINE)

    @property
    def clone_engine(self) -> str:
        return self._clone_engine or DEFAULT_CLONE_ENGINE

    def _be(self, engine: str):
        if engine not in self._backends:
            self._backends[engine] = make_backend(engine)
        return self._backends[engine]

    async def load(self) -> None:
        # Warm the built-in preset engine so preset voices are instant.
        await self._be(BUILTIN_PRESET_ENGINE).load()

    async def list_voices(self) -> list[VoiceInfo]:
        voices: list[VoiceInfo] = []
        for v in await self._be(BUILTIN_PRESET_ENGINE).list_voices():
            voices.append(VoiceInfo(f"builtin:{BUILTIN_PRESET_ENGINE}:{v.id}", v.name))
        for p in self._profiles.list():
            voices.append(VoiceInfo(p.id, p.name))
        return voices

    async def synthesize(self, text: str, voice_id: str, instruct: str = "") -> tuple[bytes, int]:
        if voice_id.startswith("builtin:"):
            _, engine, vid = voice_id.split(":", 2)
            return await self._be(engine).synthesize(text, vid)

        prof = self._profiles.get(voice_id)
        if prof is None:
            # Back-compat: treat an unknown id as a raw built-in preset voice.
            return await self._be(BUILTIN_PRESET_ENGINE).synthesize(text, voice_id)

        if prof.voice_type == "preset":
            engine = prof.preset_engine or BUILTIN_PRESET_ENGINE
            return await self._be(engine).synthesize(text, prof.preset_voice_id)

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
