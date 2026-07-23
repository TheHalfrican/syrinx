"""The ``sh.syrinx.Engine1`` D-Bus interface.

Thin layer: validate/marshal and delegate to the TTS / STT / audio modules.
Keep ML logic out of here.
"""

import asyncio
import json
import logging
from pathlib import Path

from dbus_next.service import ServiceInterface, method, signal, dbus_property
from dbus_next.constants import PropertyAccess

from .tts import SpeechSynthesizer
from .stt import Transcriber
from .profiles import ProfileStore
from .history import CaptureStore, HistoryStore, SourceClipStore
from .llm import PersonalityLLM
from .models import ModelManager, spec as model_spec, detect_hardware
from . import audio, effects, settings as engine_settings

log = logging.getLogger("syrinx.engine.service")


class _PlayCtl:
    """Cooperative playback control, polled by audio.play between blocks."""

    __slots__ = ("stop", "paused", "seek")

    def __init__(self) -> None:
        self.stop = False
        self.paused = False
        self.seek = None  # float 0..1 to jump to, or None


class EngineInterface(ServiceInterface):
    def __init__(self) -> None:
        super().__init__("sh.syrinx.Engine1")
        self._profiles = ProfileStore()
        self._tts = SpeechSynthesizer(self._profiles)
        self._stt = Transcriber()
        self._history = HistoryStore()
        self._captures = CaptureStore()
        self._srcclips = SourceClipStore()
        self._fx_store = effects.EffectPresetStore()
        self._llm = PersonalityLLM()  # lazy — loads on first Compose/Rewrite
        self._models = ModelManager()
        # apply persisted active-model choices to the lazy components
        if (s := self._models.active_spec("llm")):
            self._llm.set_model(s.size)
        if (s := self._models.active_spec("stt")):
            self._stt.set_model(s.repos[0])
        if (s := self._models.active_spec("voice")):
            self._tts.set_voice_engine(s.engine, s.size)
        self._model_loaded = False
        self._next_gen_id = 1
        self._next_llm_id = 1
        self._next_tr_id = 1
        self._tasks: dict[int, asyncio.Task] = {}
        self._audio_lock = asyncio.Lock()  # only one output stream open at a time
        self._ctl: _PlayCtl | None = None  # current playback control
        self._play_epoch = 0               # latest playback request wins
        self._volume = 1.0                 # playback gain 0..1 (SetVolume)
        self._active_effect = ""           # preset id applied to generations (SetEffect)
        self._active_style = ""            # delivery instruct baked into generations (SetStyle)

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
        return self._start_speak(text, voice_id)

    def _start_speak(self, text: str, voice_id: str) -> int:
        """Synthesize, persist to history, then play. Shared by Speak/Regenerate."""
        gen_id = self._next_gen_id
        self._next_gen_id += 1

        async def run() -> None:
            try:
                self.SpeakStarted(gen_id)
                self.GenerationProgress(gen_id, "synthesizing", 0.0)
                pcm, rate = await self._tts.synthesize(text, voice_id, self._active_style)
                if self._active_effect:
                    self.GenerationProgress(gen_id, "effects", 0.9)
                    pcm = await asyncio.to_thread(
                        effects.apply_preset, pcm, rate, self._active_effect, self._fx_store
                    )
                title = await self._voice_display_name(voice_id)
                engine, lang = self._voice_meta(voice_id)
                duration = audio.duration_of(pcm, rate)
                # Persist before playback so the clip survives restarts.
                clip_id = ""
                try:
                    item = self._history.save_clip(
                        voice_id=voice_id, voice_name=title, text=text,
                        pcm=pcm, sample_rate=rate, engine=engine, language=lang,
                    )
                    clip_id = item.id
                except Exception:  # noqa: BLE001
                    log.exception("history save failed for gen %d", gen_id)
                self.GenerationProgress(gen_id, "playing", 1.0)
                bars = json.dumps(audio.envelope(pcm))
                await self._play(
                    gen_id, pcm, rate,
                    on_start=lambda: self.PlaybackInfo(gen_id, clip_id, title, duration, bars),
                )
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                log.exception("Speak %d failed", gen_id)
                # surface the failure to the app instead of a silent vanish
                self.GenerationProgress(gen_id, f"error: {str(e)[:200]}", 0.0)
            finally:
                self.SpeakEnded(gen_id)
                self._tasks.pop(gen_id, None)

        task = asyncio.create_task(run())
        self._tasks[gen_id] = task
        return gen_id

    async def _play(
        self, gen_id: int, pcm: bytes, rate: int, *, on_start=None, start_pct: float = 0.0
    ) -> None:
        """Serialized playback: one stream at a time, latest request wins."""
        self._play_epoch += 1
        epoch = self._play_epoch
        if self._ctl is not None:
            self._ctl.stop = True  # ask the current clip to end
        ctl = _PlayCtl()
        if start_pct > 0.0:
            ctl.seek = start_pct
        async with self._audio_lock:  # waits until the previous stream has closed
            if epoch != self._play_epoch:
                return  # superseded by a newer request while we waited
            self._ctl = ctl
            if on_start is not None:
                on_start()
            try:
                await audio.play(
                    pcm, rate, ctl,
                    on_level=lambda rms: self.AudioLevel(gen_id, rms),
                    on_progress=lambda p: self.PlaybackProgress(gen_id, p),
                    volume=lambda: self._volume,
                )
            finally:
                if self._ctl is ctl:
                    self._ctl = None

    async def _voice_display_name(self, voice_id: str) -> str:
        if voice_id.startswith("builtin:"):
            for v in await self._tts.list_voices():
                if v.id == voice_id:
                    return v.name
            return voice_id
        prof = self._profiles.get(voice_id)
        return prof.name if prof else voice_id

    def _voice_meta(self, voice_id: str) -> "tuple[str, str]":
        """(engine, language) for a voice id."""
        if voice_id.startswith("builtin:"):
            parts = voice_id.split(":", 2)
            return (parts[1] if len(parts) > 1 else "kokoro"), "en"
        prof = self._profiles.get(voice_id)
        if prof is None:
            return "", "en"
        if prof.voice_type == "cloned":
            # unpinned cloned voices synthesize with the active clone engine
            engine = prof.default_engine or self._tts.clone_engine
        else:
            engine = prof.preset_engine or ""
        return engine, (prof.language or "en")

    @method()
    async def Transcribe(self, audio_path: "s") -> "s":  # noqa: F821
        return await self._stt.transcribe(audio_path)

    @method()
    async def TranscribeFile(self, audio_path: "s") -> "u":  # noqa: F821
        """Async transcription for long files — Transcribe blocks the D-Bus
        reply (~25 s cap). Partial text streams via TranscribeProgress; the
        final text arrives in TranscribeResult ("" on failure)."""
        req_id = self._next_tr_id
        self._next_tr_id += 1

        async def run() -> None:
            text = ""
            try:
                text = await self._stt.transcribe_stream(
                    audio_path,
                    on_partial=lambda t: self.TranscribeProgress(req_id, t),
                )
            except Exception:  # noqa: BLE001
                log.exception("transcribe %d failed", req_id)
            self.TranscribeResult(req_id, text)

        asyncio.create_task(run())
        return req_id

    @method()
    async def ConvertVoice(self, audio_path: "s", profile_id: "s", engine: "s", label: "s", transcript: "s", mode: "s", semitones: "i") -> "u":  # noqa: F821
        """Style-preserved voice conversion (the ⇄ tab): re-render the speech
        in *audio_path* with a cloned profile's voice, keeping the source's
        delivery (words/timing/prosody — only the timbre changes). *engine*
        "" = the default (chatterbox_vc; seed_vc when *mode* is "music").
        *mode* "music" runs the song pipeline: demucs vocal split →
        f0-conditioned conversion → remix over the instrumental; *semitones*
        shifts the sung melody (octave steps keep the song's key — register
        wrangling for deep/high voices; ignored for speech mode). The history
        row stores *transcript* (the source's words) as its text and folds
        *label* into the display name ("<voice> · <label>"). Returns a
        generation id; progress and errors arrive via GenerationProgress, and
        the result auto-plays and lands in history exactly like Speak."""
        return self._start_convert(audio_path, profile_id, engine, label, transcript, mode, semitones)

    def _start_convert(
        self, audio_path: str, profile_id: str, engine: str, label: str,
        transcript: str, mode: str, semitones: int = 0,
    ) -> int:
        gen_id = self._next_gen_id
        self._next_gen_id += 1

        async def run() -> None:
            try:
                self.SpeakStarted(gen_id)
                prof = self._profiles.get(profile_id)
                if prof is None:
                    raise ValueError(f"unknown profile {profile_id!r}")
                if prof.voice_type != "cloned" or not prof.samples:
                    raise ValueError(
                        f"{prof.name} has no reference samples to convert to"
                    )
                music = mode == "music"
                be = self._tts.vc_backend(engine or ("seed_vc" if music else ""))
                if music and not hasattr(be, "convert_music"):
                    raise ValueError(f"{be.engine_name} does not support music mode")
                be.check_source(audio_path)  # cheap cap check before any load
                self.GenerationProgress(gen_id, "loading model", 0.0)
                await be.load()
                if music:
                    # stages stream back from the worker: separating /
                    # converting / remixing — forwarded verbatim
                    pcm, rate = await be.convert_music(
                        audio_path, prof,
                        on_stage=lambda s: self.GenerationProgress(gen_id, s, 0.5),
                        semitone=semitones,
                    )
                else:
                    self.GenerationProgress(gen_id, "converting", 0.3)
                    pcm, rate = await be.convert(audio_path, prof)
                duration = audio.duration_of(pcm, rate)
                # label becomes part of the display name (apply-effects style);
                # the row's text is the source transcript so the history card's
                # read-only box shows the words that were spoken
                name = f"{prof.name} ♫" if music else prof.name
                title = f"{name} · {label.strip()}" if label.strip() else name
                clip_id = ""
                # conversion recipe — Regenerate re-runs this instead of
                # re-speaking the transcript; mtime/size pin the exact source
                # take (scratch recordings get overwritten by the next ◉)
                try:
                    st = Path(audio_path).stat()
                    vc_json = json.dumps({
                        "source": str(audio_path), "engine": be.engine_name,
                        "mode": mode, "semitones": semitones,
                        "label": label.strip(),
                        "mtime": int(st.st_mtime), "size": st.st_size,
                    })
                except OSError:
                    vc_json = ""
                try:
                    item = self._history.save_clip(
                        voice_id=profile_id,
                        voice_name=title,
                        text=transcript.strip()
                        or f"[voice conversion] {Path(audio_path).name}",
                        pcm=pcm,
                        sample_rate=rate,
                        engine=be.engine_name,
                        language=prof.language or "en",
                        vc_json=vc_json,
                    )
                    clip_id = item.id
                except Exception:  # noqa: BLE001
                    log.exception("history save failed for convert %d", gen_id)
                self.GenerationProgress(gen_id, "playing", 1.0)
                bars = json.dumps(audio.envelope(pcm))
                await self._play(
                    gen_id, pcm, rate,
                    on_start=lambda: self.PlaybackInfo(
                        gen_id, clip_id, title, duration, bars
                    ),
                )
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                log.exception("ConvertVoice %d failed", gen_id)
                self.GenerationProgress(gen_id, f"error: {str(e)[:200]}", 0.0)
            finally:
                self.SpeakEnded(gen_id)
                self._tasks.pop(gen_id, None)

        task = asyncio.create_task(run())
        self._tasks[gen_id] = task
        return gen_id

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
        counts = self._profiles.sample_counts()
        out = []
        for p in self._profiles.list():
            d = p.summary()
            d["samples"] = counts.get(p.id, 0)
            out.append(d)
        return json.dumps(out)

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
    async def SetProfileAvatar(
        self, profile_id: "s", src: "s", mode: "s", sx: "i", sy: "i", sw: "i", sh: "i"  # noqa: F821
    ) -> None:
        """Attach an avatar photo + crop rect (circle|panel); empty src re-crops."""
        self._profiles.set_avatar(profile_id, src, mode, sx, sy, sw, sh)

    @method()
    async def ExportProfile(self, profile_id: "s", dest: "s") -> None:  # noqa: F821
        self._profiles.export_package(profile_id, dest)

    @method()
    async def ImportProfile(self, src: "s") -> "s":  # noqa: F821
        return self._profiles.import_package(src)

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
    async def UpdateSampleText(self, profile_id: "s", sample_id: "s", text: "s") -> None:  # noqa: F821
        """Correct a sample's reference transcript (clone prompts rebuild)."""
        self._profiles.set_sample_text(sample_id, text)
        self._tts.invalidate_profile(profile_id)

    # --- personality LLM (compose / rewrite) ---------------------------

    def _personality_of(self, voice_id: str) -> str:
        if voice_id.startswith("builtin:"):
            return ""
        prof = self._profiles.get(voice_id)
        return prof.personality if prof else ""

    @method()
    async def ComposeProfile(self, voice_id: "s", prompt: "s") -> "u":  # noqa: F821
        personality = self._personality_of(voice_id)
        if not personality:
            return 0
        return self._start_llm("compose", personality, prompt)

    @method()
    async def RewriteProfile(self, voice_id: "s", text: "s") -> "u":  # noqa: F821
        personality = self._personality_of(voice_id)
        if not personality or not text.strip():
            return 0
        return self._start_llm("rewrite", personality, text)

    @method()
    async def RefineTranscript(self, text: "s") -> "u":  # noqa: F821
        """Clean a dictation transcript via the LLM; result via LlmResult."""
        if not text.strip():
            return 0
        return self._start_llm("refine", "", text)

    def _start_llm(self, kind: str, personality: str, text: str) -> int:
        """Run compose/rewrite/refine off the D-Bus call (LLM load + inference
        is slow); deliver the result via the LlmResult signal, keyed by req_id."""
        req_id = self._next_llm_id
        self._next_llm_id += 1

        async def run() -> None:
            out = ""
            try:
                if kind == "compose":
                    out = await self._llm.compose(personality, text)
                elif kind == "refine":
                    out = await self._llm.refine(text)
                else:
                    out = await self._llm.rewrite(personality, text)
            except Exception:  # noqa: BLE001
                log.exception("llm %s failed", kind)
            self.LlmResult(req_id, out)

        asyncio.create_task(run())
        return req_id

    # --- model management ----------------------------------------------

    @method()
    async def ListModels(self) -> "s":  # noqa: F821
        return json.dumps(self._models.status())

    @method()
    async def Hardware(self) -> "s":  # noqa: F821
        return json.dumps(detect_hardware())

    @method()
    async def DownloadModel(self, model_id: "s") -> "b":  # noqa: F821
        if not model_spec(model_id):
            return False

        async def run() -> None:
            await self._models.download(
                model_id, lambda mid, pct, st: self.ModelProgress(mid, pct, st)
            )

        asyncio.create_task(run())
        return True

    @method()
    async def DeleteModel(self, model_id: "s") -> None:  # noqa: F821
        self._models.delete(model_id)

    @method()
    async def SetActiveModel(self, model_id: "s") -> "s":  # noqa: F821
        s = model_spec(model_id)
        if s and s.category == "vc":
            # conversion engines are picked per-conversion in the ⇄ tab —
            # nothing to activate (and no ACTIVE badge to claim)
            return "vc"
        category = self._models.set_active(model_id)
        if s and category == "llm":
            self._llm.set_model(s.size)
        elif s and category == "stt":
            self._stt.set_model(s.repos[0])
        elif s and category == "voice":
            # cloned profiles without a pinned default_engine follow this;
            # extra preset engines (CustomVoice) list their voices instead
            self._tts.set_voice_engine(s.engine, s.size)
        return category

    # --- generation history --------------------------------------------

    @method()
    async def ListHistory(self) -> "s":  # noqa: F821
        return json.dumps([h.to_dict() for h in self._history.list()])

    @method()
    async def PlayHistory(self, hid: "s") -> "u":  # noqa: F821
        return self._play_history(hid, 0.0)

    @method()
    async def PlayHistoryAt(self, hid: "s", pct: "d") -> "u":  # noqa: F821
        return self._play_history(hid, pct)

    @method()
    async def PlaySample(self, sample_id: "s") -> "u":  # noqa: F821
        """Audition a profile reference sample through the normal player."""
        path = self._profiles.sample_path(sample_id)
        if not path:
            return 0
        try:
            pcm, rate = effects.load_wav(path)
        except Exception:  # noqa: BLE001
            log.exception("PlaySample %s: unreadable %s", sample_id, path)
            return 0
        gen_id = self._next_gen_id
        self._next_gen_id += 1
        bars = json.dumps(audio.envelope(pcm))
        duration = len(pcm) / 4 / rate

        async def run() -> None:
            try:
                self.SpeakStarted(gen_id)
                await self._play(
                    gen_id, pcm, rate,
                    on_start=lambda: self.PlaybackInfo(gen_id, "", "Sample", duration, bars),
                )
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                log.exception("PlaySample %s failed", sample_id)
            finally:
                self.SpeakEnded(gen_id)
                self._tasks.pop(gen_id, None)

        self._tasks[gen_id] = asyncio.create_task(run())
        return gen_id

    def _play_history(self, hid: str, start_pct: float) -> int:
        item = self._history.get(hid)
        loaded = self._history.read_pcm(hid)
        if item is None or loaded is None:
            return 0
        pcm, rate = loaded
        gen_id = self._next_gen_id
        self._next_gen_id += 1
        bars = json.dumps(audio.envelope(pcm))

        async def run() -> None:
            try:
                self.SpeakStarted(gen_id)
                await self._play(
                    gen_id, pcm, rate, start_pct=start_pct,
                    on_start=lambda: self.PlaybackInfo(gen_id, hid, item.voice_name, item.duration, bars),
                )
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                log.exception("PlayHistory %s failed", hid)
            finally:
                self.SpeakEnded(gen_id)
                self._tasks.pop(gen_id, None)

        self._tasks[gen_id] = asyncio.create_task(run())
        return gen_id

    @method()
    async def PausePlayback(self) -> None:  # noqa: F821
        if self._ctl is not None:
            self._ctl.paused = True

    @method()
    async def ResumePlayback(self) -> None:  # noqa: F821
        if self._ctl is not None:
            self._ctl.paused = False

    @method()
    async def SeekPlayback(self, pct: "d") -> None:  # noqa: F821
        if self._ctl is not None:
            self._ctl.seek = pct

    @method()
    async def SetVolume(self, volume: "d") -> None:  # noqa: F821
        self._volume = max(0.0, min(1.0, volume))

    # --- effects --------------------------------------------------------

    @method()
    async def ListEffectPresets(self) -> "s":  # noqa: F821
        return json.dumps(effects.list_presets(self._fx_store))

    @method()
    async def SetEffect(self, preset_id: "s") -> None:  # noqa: F821
        known = effects.resolve_preset(preset_id, self._fx_store) is not None
        self._active_effect = preset_id if known else ""
        log.info("active effect -> %r", self._active_effect or "none")

    @method()
    async def SetStyle(self, instruct: "s") -> None:  # noqa: F821
        """Delivery direction baked into generations ("" = neutral).

        Free-text natural-language instruct ("Speak in an extremely angry
        tone…"). Honored by the qwen engines; engines without style control
        ignore it.
        """
        self._active_style = instruct
        log.info("active style -> %r", instruct[:40] if instruct else "none")

    @method()
    async def ApplyHistoryEffects(self, hid: "s", preset_id: "s") -> "s":  # noqa: F821
        """Re-process a stored clip through a preset; saves a NEW history row."""
        item = self._history.get(hid)
        if item is None or effects.resolve_preset(preset_id, self._fx_store) is None:
            return ""
        path = self._history.audio_abs_path(hid)
        pcm, rate = effects.load_wav(path)
        pcm = await asyncio.to_thread(effects.apply_preset, pcm, rate, preset_id, self._fx_store)
        new = self._history.save_clip(
            voice_id=item.voice_id,
            voice_name=f"{item.voice_name} · {effects.preset_name(preset_id, self._fx_store)}",
            text=item.text,
            pcm=pcm,
            sample_rate=rate,
            engine=item.engine,
            language=item.language,
        )
        return new.id

    # --- effect chain editor -------------------------------------------

    @method()
    async def ListEffects(self) -> "s":  # noqa: F821
        """Effect definitions (label, params with default/min/max/step)."""
        return json.dumps(effects.list_effects())

    @method()
    async def GetEffectPreset(self, preset_id: "s") -> "s":  # noqa: F821
        """Full preset incl. chain ("" if unknown)."""
        p = effects.resolve_preset(preset_id, self._fx_store)
        return json.dumps(p) if p else ""

    @method()
    async def CreateEffectPreset(self, name: "s", description: "s", chain_json: "s") -> "s":  # noqa: F821
        """New user preset; returns id ("" on invalid chain / duplicate name)."""
        try:
            chain = json.loads(chain_json)
        except json.JSONDecodeError:
            return ""
        return self._fx_store.create(name, description, chain)

    @method()
    async def UpdateEffectPreset(self, preset_id: "s", name: "s", description: "s", chain_json: "s") -> "b":  # noqa: F821
        """Rewrite a user preset in place (builtins are immutable)."""
        try:
            chain = json.loads(chain_json)
        except json.JSONDecodeError:
            return False
        return self._fx_store.update(preset_id, name, description, chain)

    @method()
    async def DeleteEffectPreset(self, preset_id: "s") -> "b":  # noqa: F821
        return self._fx_store.delete(preset_id)

    @method()
    async def PreviewEffects(self, hid: "s", chain_json: "s") -> "u":  # noqa: F821
        """Play a stored clip through an ad-hoc chain (nothing is saved)."""
        try:
            chain = json.loads(chain_json)
        except json.JSONDecodeError:
            return 0
        if effects.validate_chain(chain) is not None:
            return 0
        item = self._history.get(hid)
        loaded = self._history.read_pcm(hid)
        if item is None or loaded is None:
            return 0
        pcm, rate = loaded
        gen_id = self._next_gen_id
        self._next_gen_id += 1

        async def run() -> None:
            try:
                processed = await asyncio.to_thread(effects.apply_chain, pcm, rate, chain)
                bars = json.dumps(audio.envelope(processed))
                duration = audio.duration_of(processed, rate)
                self.SpeakStarted(gen_id)
                await self._play(
                    gen_id, processed, rate,
                    on_start=lambda: self.PlaybackInfo(
                        gen_id, "", f"{item.voice_name} · preview", duration, bars
                    ),
                )
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                log.exception("PreviewEffects %s failed", hid)
            finally:
                self.SpeakEnded(gen_id)
                self._tasks.pop(gen_id, None)

        self._tasks[gen_id] = asyncio.create_task(run())
        return gen_id

    @method()
    async def StarHistory(self, hid: "s", starred: "b") -> None:  # noqa: F821
        self._history.set_starred(hid, starred)

    @method()
    async def SetHistoryTags(self, hid: "s", tags_json: "s") -> None:  # noqa: F821
        """Replace a history row's tags (JSON array of strings)."""
        try:
            tags = json.loads(tags_json)
        except json.JSONDecodeError:
            return
        if isinstance(tags, list):
            cleaned = [str(t).strip() for t in tags if str(t).strip()]
            self._history.set_tags(hid, cleaned)

    @method()
    async def DeleteHistory(self, hid: "s") -> None:  # noqa: F821
        self._history.delete(hid)

    @method()
    async def RegenerateHistory(self, hid: "s") -> "u":  # noqa: F821
        """Re-run the generation behind a history row. TTS rows re-speak
        their text; conversion rows re-run the conversion — but only while
        the exact source take still exists (0 when it's gone or overwritten,
        so the app can say why instead of re-speaking the transcript)."""
        item = self._history.get(hid)
        if item is None:
            return 0
        vc = {}
        if item.vc_json:
            try:
                vc = json.loads(item.vc_json)
            except json.JSONDecodeError:
                vc = {}
        if vc:
            src = vc.get("source", "")
            try:
                st = Path(src).stat()
                fresh = (
                    int(st.st_mtime) == int(vc.get("mtime", -1))
                    and st.st_size == int(vc.get("size", -1))
                )
            except OSError:
                fresh = False
            if not fresh:
                log.warning("regenerate %s: conversion source gone/overwritten (%s)", hid, src)
                return 0
            return self._start_convert(
                src, item.voice_id, vc.get("engine", ""), vc.get("label", ""),
                item.text, vc.get("mode", "speech"), int(vc.get("semitones", 0)),
            )
        from .tts import VC_ENGINES

        if item.engine in VC_ENGINES:
            # conversion row from before recipes were stored — refusing beats
            # re-speaking its transcript through a TTS engine
            log.warning("regenerate %s: pre-recipe conversion row", hid)
            return 0
        return self._start_speak(item.text, item.voice_id)

    @method()
    async def ExportPackage(self, hid: "s", dest: "s") -> None:  # noqa: F821
        self._history.export_package(hid, dest)

    @method()
    async def HistoryAudioPath(self, hid: "s") -> "s":  # noqa: F821
        # Absolute WAV path so the app can copy it on "export audio".
        return self._history.audio_abs_path(hid)

    # --- transcription captures (text only) -----------------------------

    @method()
    async def SaveCapture(self, text: "s") -> "s":  # noqa: F821
        if not text.strip():
            return ""
        return self._captures.save(text).id

    @method()
    async def ListCaptures(self) -> "s":  # noqa: F821
        return json.dumps([c.to_dict() for c in self._captures.list()])

    @method()
    async def UpdateCapture(self, capture_id: "s", text: "s") -> None:  # noqa: F821
        self._captures.update(capture_id, text)

    @method()
    async def DeleteCapture(self, capture_id: "s") -> None:  # noqa: F821
        self._captures.delete(capture_id)

    # --- voice-changer source clips (named recordings/imports) ----------

    @method()
    async def SaveSourceClip(self, path: "s", name: "s", transcript: "s") -> "s":  # noqa: F821
        """Copy an audio file into the clip store; returns the new clip id
        ("" on failure). An empty name gets a time-based default; *transcript*
        is cached so re-arming the clip skips re-transcription."""
        try:
            return self._srcclips.save(path, name, transcript).id
        except Exception:  # noqa: BLE001
            log.exception("SaveSourceClip %s failed", path)
            return ""

    @method()
    async def SetSourceClipTranscript(self, clip_id: "s", transcript: "s") -> None:  # noqa: F821
        """Backfill a clip's transcript cache (saved before whisper finished)."""
        self._srcclips.set_transcript(clip_id, transcript)

    @method()
    async def ListSourceClips(self) -> "s":  # noqa: F821
        return json.dumps([c.to_dict() for c in self._srcclips.list()])

    @method()
    async def DeleteSourceClip(self, clip_id: "s") -> None:  # noqa: F821
        self._srcclips.delete(clip_id)

    def _play_file(self, path: str, title: str, start_pct: float = 0.0) -> int:
        try:
            pcm, rate = effects.load_wav(path)
        except Exception:  # noqa: BLE001
            log.exception("PlayFile: unreadable %s", path)
            return 0
        gen_id = self._next_gen_id
        self._next_gen_id += 1
        bars = json.dumps(audio.envelope(pcm))
        duration = len(pcm) / 4 / rate
        title = title or Path(path).stem

        async def run() -> None:
            try:
                self.SpeakStarted(gen_id)
                await self._play(
                    gen_id, pcm, rate, start_pct=start_pct,
                    on_start=lambda: self.PlaybackInfo(gen_id, "", title, duration, bars),
                )
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                log.exception("PlayFile %s failed", path)
            finally:
                self.SpeakEnded(gen_id)
                self._tasks.pop(gen_id, None)

        self._tasks[gen_id] = asyncio.create_task(run())
        return gen_id

    @method()
    async def PlayFile(self, path: "s", title: "s") -> "u":  # noqa: F821
        """Audition any local audio file through the normal player (0 if
        unreadable). Used by the ⇄ tab to verify sources before converting."""
        return self._play_file(path, title)

    @method()
    async def PlayFileAt(self, path: "s", title: "s", pct: "d") -> "u":  # noqa: F821
        """PlayFile from a fraction (0..1) of the way in — the trim modal's
        selection preview (the app cancels when the end handle is reached)."""
        return self._play_file(path, title, start_pct=max(0.0, min(1.0, pct)))

    @method()
    async def FileEnvelope(self, path: "s") -> "s":  # noqa: F821
        """Waveform bars + duration of any local audio file, as JSON
        {"bars": [...], "duration": secs} — the trim modal's display."""
        try:
            pcm, rate = effects.load_wav(path)
        except Exception:  # noqa: BLE001
            log.exception("FileEnvelope: unreadable %s", path)
            return "{}"
        return json.dumps({
            "bars": audio.envelope(pcm),
            "duration": audio.duration_of(pcm, rate),
        })

    @method()
    async def TrimAudio(self, path: "s", start_s: "d", end_s: "d") -> "s":  # noqa: F821
        """Cut a recording down to [start_s, end_s). WAVs are rewritten in
        place (PCM16 mono, rate kept); other formats get a sibling
        "<stem>-trimmed.wav". Returns the resulting path — "" on failure or
        a selection shorter than 0.1 s."""
        try:
            import soundfile as sf

            data, rate = sf.read(path, dtype="float32")
            if getattr(data, "ndim", 1) > 1:
                data = data.mean(axis=1)
            a = max(0, int(start_s * rate))
            b = min(len(data), int(end_s * rate))
            if b - a < int(0.1 * rate):
                return ""
            out = Path(path)
            if out.suffix.lower() != ".wav":
                out = out.with_name(out.stem + "-trimmed.wav")
            sf.write(str(out), data[a:b], int(rate), subtype="PCM_16")
            return str(out)
        except Exception:  # noqa: BLE001
            log.exception("TrimAudio %s failed", path)
            return ""

    @method()
    async def TrimHistoryClip(self, hid: "s", start_s: "d", end_s: "d") -> "b":  # noqa: F821
        """Cut a history clip to [start_s, end_s) in place — duration
        updates; text, tags and stars stay."""
        return self._history.trim(hid, start_s, end_s)

    # --- engine settings (the ⚙ tab's knobs) -----------------------------

    @method()
    async def GetSettings(self) -> "s":  # noqa: F821
        """Persisted engine settings plus the currently effective values."""
        from .backends.chatterbox_vc import max_source_secs
        from .backends.seed_vc import _steps as seedvc_steps

        return json.dumps({
            "stored": engine_settings.all_values(),
            "effective": {
                "vc_max_secs": max_source_secs(),
                "seedvc_steps": seedvc_steps(),
            },
        })

    @method()
    async def SetSetting(self, key: "s", value_json: "s") -> None:  # noqa: F821
        """Set one engine setting (JSON-encoded value; null clears it)."""
        try:
            val = json.loads(value_json)
        except json.JSONDecodeError:
            return
        engine_settings.set_value(key, val)
        log.info("setting %s -> %r", key, val)

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
    def PlaybackInfo(self, gen_id, clip_id, title, duration, bars) -> "ussds":
        # clip_id, display title, seconds, and a JSON array of waveform bars (0..1)
        return [gen_id, clip_id, title, duration, bars]

    @signal()
    def PlaybackProgress(self, gen_id, pct) -> "ud":
        return [gen_id, pct]

    @signal()
    def LlmResult(self, req_id, text) -> "us":
        return [req_id, text]

    @signal()
    def TranscribeProgress(self, req_id, partial) -> "us":
        return [req_id, partial]

    @signal()
    def TranscribeResult(self, req_id, text) -> "us":
        return [req_id, text]

    @signal()
    def ModelProgress(self, model_id, pct, status) -> "sds":
        return [model_id, pct, status]

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
