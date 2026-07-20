"""The ``sh.syrinx.Engine1`` D-Bus interface.

Thin layer: validate/marshal and delegate to the TTS / STT / audio modules.
Keep ML logic out of here.
"""

import asyncio
import logging

from dbus_next.service import ServiceInterface, method, signal, dbus_property
from dbus_next.constants import PropertyAccess

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
    async def Transcribe(self, pcm: "ay") -> "s":  # noqa: F821
        return await self._stt.transcribe(bytes(pcm))

    @method()
    async def ListVoices(self) -> "a(ss)":  # noqa: F821
        return [[v.id, v.name] for v in await self._tts.list_voices()]

    @method()
    async def CloneVoice(self, name: "s", sample_path: "s") -> "s":  # noqa: F821
        return await self._tts.clone(name, sample_path)

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
