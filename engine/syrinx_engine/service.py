"""The ``sh.syrinx.Engine1`` D-Bus interface.

Thin layer: validate/marshal and delegate to the TTS / STT / audio modules.
Keep ML logic out of here.
"""

import asyncio
import json
import logging

from dbus_next.service import ServiceInterface, method, signal, dbus_property
from dbus_next.constants import PropertyAccess

from .tts import SpeechSynthesizer
from .stt import Transcriber
from .profiles import ProfileStore
from . import audio

log = logging.getLogger("syrinx.engine.service")


class EngineInterface(ServiceInterface):
    def __init__(self) -> None:
        super().__init__("sh.syrinx.Engine1")
        self._profiles = ProfileStore()
        self._tts = SpeechSynthesizer(self._profiles)
        self._stt = Transcriber()
        self._model_loaded = False
        self._next_gen_id = 1
        self._tasks: dict[int, asyncio.Task] = {}

    @property
    def backend_name(self) -> str:
        return self._tts.backend  # "cuda" | "rocm" | "cpu"

    async def warmup(self) -> None:
        """Load models in the background, then flip ModelLoaded."""
        await self._tts.load()
        await self._stt.load()
        self._model_loaded = True
        self.emit_properties_changed({"ModelLoaded": True})
        log.info("models loaded")

    # --- Methods --------------------------------------------------------

    @method()
    async def Speak(self, text: "s", voice_id: "s") -> "u":  # noqa: F821
        gen_id = self._next_gen_id
        self._next_gen_id += 1

        async def run() -> None:
            try:
                self.SpeakStarted(gen_id)
                self.GenerationProgress(gen_id, "synthesizing", 0.0)
                pcm, rate = await self._tts.synthesize(text, voice_id)
                self.GenerationProgress(gen_id, "playing", 1.0)
                await audio.play(
                    pcm, rate, on_level=lambda rms: self.AudioLevel(gen_id, rms)
                )
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                log.exception("Speak %d failed", gen_id)
            finally:
                self.SpeakEnded(gen_id)
                self._tasks.pop(gen_id, None)

        self._tasks[gen_id] = asyncio.create_task(run())
        return gen_id

    @method()
    async def Transcribe(self, audio_path: "s") -> "s":  # noqa: F821
        return await self._stt.transcribe(audio_path)

    @method()
    async def ListVoices(self) -> "a(ss)":  # noqa: F821
        return [[v.id, v.name] for v in await self._tts.list_voices()]

    @method()
    async def CloneVoice(self, name: "s", sample_path: "s", ref_text: "s") -> "s":  # noqa: F821
        # ref_text = transcript of the reference clip (needed by Qwen cloning).
        return await self._tts.clone(name, sample_path, ref_text)

    # --- voice profiles (JSON payloads for the structured bits) ---------

    @method()
    async def CreateProfile(self, spec_json: "s") -> "s":  # noqa: F821
        s = json.loads(spec_json)
        return self._profiles.create(
            s["name"],
            s.get("voice_type", "cloned"),
            language=s.get("language", "en"),
            description=s.get("description", ""),
            personality=s.get("personality", ""),
            default_engine=s.get("default_engine", ""),
            preset_engine=s.get("preset_engine", ""),
            preset_voice_id=s.get("preset_voice_id", ""),
        )

    @method()
    async def ListProfiles(self) -> "s":  # noqa: F821
        return json.dumps([p.summary() for p in self._profiles.list()])

    @method()
    async def GetProfile(self, profile_id: "s") -> "s":  # noqa: F821
        p = self._profiles.get(profile_id)
        return json.dumps(p.full()) if p else ""

    @method()
    async def UpdateProfile(self, profile_id: "s", patch_json: "s") -> None:  # noqa: F821
        self._profiles.update(profile_id, **json.loads(patch_json))

    @method()
    async def DeleteProfile(self, profile_id: "s") -> None:  # noqa: F821
        self._profiles.delete(profile_id)
        self._tts.invalidate_profile(profile_id)

    @method()
    async def AddSample(self, profile_id: "s", audio_path: "s", reference_text: "s") -> "s":  # noqa: F821
        # Auto-transcribe when no transcript is supplied (whisper).
        text = reference_text
        if not text.strip():
            text = await self._stt.transcribe(audio_path)
        sample = self._profiles.add_sample(profile_id, audio_path, text)
        self._tts.invalidate_profile(profile_id)  # rebuild clone prompt next synth
        return json.dumps({"sample_id": sample.id, "reference_text": sample.reference_text})

    @method()
    async def DeleteSample(self, sample_id: "s") -> None:  # noqa: F821
        self._profiles.delete_sample(sample_id)

    @method()
    def Cancel(self, gen_id: "u") -> None:  # noqa: F821
        task = self._tasks.get(gen_id)
        if task:
            task.cancel()
            log.info("cancelled %d", gen_id)

    # --- Signals (return annotation IS the D-Bus signature) -------------

    @signal()
    def GenerationProgress(self, gen_id, state, pct) -> "usd":
        return [gen_id, state, pct]

    @signal()
    def AudioLevel(self, gen_id, rms) -> "ud":
        return [gen_id, rms]

    @signal()
    def SpeakStarted(self, gen_id) -> "u":
        return gen_id

    @signal()
    def SpeakEnded(self, gen_id) -> "u":
        return gen_id

    # --- Properties (read-only) -----------------------------------------

    @dbus_property(access=PropertyAccess.READ)
    def ModelLoaded(self) -> "b":  # noqa: F821
        return self._model_loaded

    @dbus_property(access=PropertyAccess.READ)
    def Backend(self) -> "s":  # noqa: F821
        return self.backend_name
