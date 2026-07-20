"""The ``sh.syrinx.Engine1`` D-Bus interface.

Thin layer: it validates/marshals and delegates to the TTS / STT / audio
modules. Keep ML logic out of here.
"""

import asyncio
import logging

from dbus_next.service import ServiceInterface, method, signal, dbus_property
from dbus_next import Variant  # noqa: F401  (used once real opts land)

from .tts import SpeechSynthesizer
from .stt import Transcriber
from . import audio

log = logging.getLogger("syrinx.engine.service")


class EngineInterface(ServiceInterface):
    def __init__(self) -> None:
        super().__init__("sh.syrinx.Engine1")
        self._tts = SpeechSynthesizer()
        self._stt = Transcriber()
        self._model_loaded = False
        self._next_gen_id = 1

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
            self.SpeakStarted(gen_id)
            # TODO(syrinx): stream chunks; emit AudioLevel per frame.
            pcm = await self._tts.synthesize(text, voice_id)
            await audio.play(pcm, on_level=lambda rms: self.AudioLevel(gen_id, rms))
            self.SpeakEnded(gen_id)

        asyncio.create_task(run())
        return gen_id

    @method()
    async def Transcribe(self, pcm: "ay") -> "s":  # noqa: F821
        return await self._stt.transcribe(bytes(pcm))

    @method()
    async def ListVoices(self) -> "a(ss)":  # noqa: F821
        return [(v.id, v.name) for v in await self._tts.list_voices()]

    @method()
    async def CloneVoice(self, name: "s", sample_path: "s") -> "s":  # noqa: F821
        return await self._tts.clone(name, sample_path)

    @method()
    def Cancel(self, gen_id: "u") -> None:  # noqa: F821
        # TODO(syrinx): cancel the in-flight task for gen_id.
        log.info("cancel %d (stub)", gen_id)

    # --- Signals --------------------------------------------------------

    @signal()
    def GenerationProgress(self, gen_id: "u", state: "s", pct: "d") -> "(usd)":  # noqa: F821
        return [gen_id, state, pct]

    @signal()
    def AudioLevel(self, gen_id: "u", rms: "d") -> "(ud)":  # noqa: F821
        return [gen_id, rms]

    @signal()
    def SpeakStarted(self, gen_id: "u") -> "u":  # noqa: F821
        return gen_id

    @signal()
    def SpeakEnded(self, gen_id: "u") -> "u":  # noqa: F821
        return gen_id

    # --- Properties -----------------------------------------------------

    @dbus_property()
    def ModelLoaded(self) -> "b":  # noqa: F821
        return self._model_loaded

    @dbus_property()
    def Backend(self) -> "s":  # noqa: F821
        return self.backend_name
