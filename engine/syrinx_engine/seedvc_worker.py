"""Standalone Seed-VC worker — runs inside the isolated .venv-seedvc.

Voice conversion via the GPL-licensed ``seed-vc`` package, which therefore
stays a process-isolated runtime dependency: this file is a thin adapter that
calls seed-vc's PUBLIC API (no logic is ported), and the package itself is
installed only in the separate venv, never vendored into the repo.

Reads one JSON request per line on stdin, converts source→target with Seed-VC
V1 (whisper-small model; the f0-conditioned 44k variant when "f0" is set, for
singing), writes raw float32 PCM to a temp file, and replies on stdout.

Request : {"id": N, "source": "<src.wav>", "target": "<ref.wav>",
           "f0": false, "steps": 25, "auto_f0": true, "semitone": 0}
          {"id": N, "cmd": "music", "source": …, "target": …, "steps": 30,
           "semitone": 0}   — demucs split → f0 convert vocals → remix
Reply   : {"id": N, "stage": "separating"|"converting"|"remixing"}  (interim)
          {"id": N, "ok": true, "raw": "<path>", "rate": 22050}
          {"id": N, "ok": false, "error": "..."}

Models load once per (f0-condition) into a reusable stream state; switching
between the speech and singing models drops the other to bound RAM. Long
sources are fed as ~20 s slices through seed-vc's own streaming crossfade.
Device: seed_vc.api picks CUDA automatically when the venv's torch sees a GPU.
"""

import json
import os
import sys
import tempfile

# Reserve the real stdout for the JSON protocol: transformers/hf downloads
# print progress straight to fd 1, which would corrupt the line protocol.
_PROTO = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402 — after the stdout/path setup above

_STATE = None       # seed_vc stream state (models + cached target features)
_STATE_F0 = None    # which model the state holds (f0 condition bool)

# One source slice per model call; seed-vc crossfades the joins itself.
CHUNK_SECS = float(os.environ.get("SYRINX_SEEDVC_CHUNK_SECS", "20"))


def _audio_data(samples_i16: np.ndarray, rate: int):
    from seed_vc.Models.audio import AudioData

    return AudioData(
        samples=samples_i16,
        mel_chunks=None,
        duration=len(samples_i16) / float(rate),
        samples_count=len(samples_i16),
        sample_rate=rate,
        metadata=None,
    )


def _load_wav_i16(path: str) -> tuple:
    import librosa

    wave, rate = librosa.load(path, sr=None, mono=True)
    i16 = np.clip(wave * 32767.0, -32768, 32767).astype(np.int16)
    return i16, int(rate)


def _ensure_state(f0: bool):
    global _STATE, _STATE_F0
    if _STATE is not None and _STATE_F0 == f0:
        return
    from seed_vc.api import create_v1_stream_state

    _STATE = None  # drop the other model before loading (15 GB box)
    print(f"seedvc-worker: loading models (f0={f0})...", file=sys.stderr, flush=True)
    _STATE = create_v1_stream_state(
        target=None,
        new_target_name=None,
        f0_condition=f0,
        fp16=False,   # autocast fp16 is a CUDA affair; harmless-off everywhere
        realtime=False,  # offline whisper-small model — quality over latency
    )
    _STATE_F0 = f0
    print("seedvc-worker: models loaded", file=sys.stderr, flush=True)


def _convert(rid, src_i16, src_rate, target_ad, target_name, f0, steps, auto_f0, semitone):
    """Chunked source → converted float32 waveform + rate (models stay warm).

    Drives the stream state's process_chunk directly instead of
    api.inference(): the wrapper round-trips every chunk through int16 with a
    plain astype, which WRAPS peaks past ±1.0 into full-scale spikes of the
    opposite sign (the vocoder routinely overshoots on hot references). The
    state's overlap buffer still does the crossfade between chunks; we keep
    the floats and stay at the model's native rate (22050, or 44100 for f0).
    """
    import torch

    _ensure_state(f0)
    parts = []
    out_rate = int(_STATE.sr)
    chunk_len = max(1, int(CHUNK_SECS * src_rate))
    slices = [src_i16[i : i + chunk_len] for i in range(0, len(src_i16), chunk_len)]
    # api.inference() is @torch.no_grad(); calling the state directly must
    # carry the same context or autograd chokes on inference-mode weights
    with torch.no_grad():
        if target_name != _STATE.target_name:  # target features cached by path
            _STATE.prepare_target(f0, target_ad, target_name)
        for i, piece in enumerate(slices):
            last = i == len(slices) - 1
            print(
                f"seedvc-worker: chunk {i + 1}/{len(slices)} (steps={steps})",
                file=sys.stderr, flush=True,
            )
            chunk = _STATE.process_chunk(
                source=_audio_data(piece, src_rate),
                length_adjust=1.0,
                diffusion_steps=steps,
                inference_cfg_rate=0.7,
                f0_condition=f0,
                auto_f0_adjust=auto_f0,
                semi_tone_shift=semitone,
                fp16_flag=False,
                end_of_stream=last,
            )
            parts.append(np.asarray(chunk, dtype=np.float32))

    audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return audio.astype(np.float32), out_rate


def _reply_raw(rid, audio: "np.ndarray", rate: int) -> dict:
    # peak safety for every reply path: playback and the saved history clip
    # hard-clip anything past ±1.0 (float32 all the way downstream)
    peak = float(np.abs(audio).max()) or 1.0
    if peak > 0.99:
        audio = audio / peak * 0.99
    out = os.path.join(tempfile.gettempdir(), f"seedvc-{rid}.raw")
    audio.astype(np.float32).tofile(out)
    return {"id": rid, "ok": True, "raw": out, "rate": rate}


def _handle(req: dict) -> dict:
    """Plain conversion: source speech → target voice."""
    rid = req.get("id")
    src_i16, src_rate = _load_wav_i16(req["source"])
    tgt_i16, tgt_rate = _load_wav_i16(req["target"])
    audio, rate = _convert(
        rid, src_i16, src_rate, _audio_data(tgt_i16, tgt_rate), req["target"],
        f0=bool(req.get("f0", False)),
        steps=int(req.get("steps", 25)),
        auto_f0=bool(req.get("auto_f0", True)),
        semitone=int(req.get("semitone", 0)),
    )
    return _reply_raw(rid, audio, rate)


def _stage(rid, name: str) -> None:
    """Interim progress line — the backend forwards it to GenerationProgress."""
    _PROTO.write(json.dumps({"id": rid, "stage": name}) + "\n")
    _PROTO.flush()


def _handle_music(req: dict) -> dict:
    """Song cover: demucs vocal split → f0-conditioned convert → remix."""
    import librosa
    import torch

    rid = req.get("id")
    _stage(rid, "separating")
    from demucs.api import Separator

    sep = Separator(model="htdemucs", device="cuda" if torch.cuda.is_available() else "cpu")
    print("seedvc-worker: demucs separating...", file=sys.stderr, flush=True)
    _origin, stems = sep.separate_audio_file(req["source"])
    demucs_sr = int(sep.samplerate)
    vocals = stems["vocals"].mean(dim=0).numpy()  # stereo -> mono
    # instrumental = every non-vocal stem, kept stereo->mono at demucs sr
    inst = sum(
        stems[name] for name in stems if name != "vocals"
    ).mean(dim=0).numpy()

    _stage(rid, "converting")
    vox_i16 = np.clip(vocals * 32767.0, -32768, 32767).astype(np.int16)
    tgt_i16, tgt_rate = _load_wav_i16(req["target"])
    converted, conv_rate = _convert(
        rid, vox_i16, demucs_sr, _audio_data(tgt_i16, tgt_rate), req["target"],
        f0=True,  # singing model — melody rides the f0 track
        steps=int(req.get("steps", 30)),
        # default OFF: auto-f0 re-registers the melody to the target voice's
        # median pitch, putting the vocal in a different key than the mix
        auto_f0=bool(req.get("auto_f0", False)),
        semitone=int(req.get("semitone", 0)),
    )

    _stage(rid, "remixing")
    if conv_rate != demucs_sr:
        converted = librosa.resample(converted, orig_sr=conv_rate, target_sr=demucs_sr)
    # align lengths (conversion can drift a few hundred samples) and sum
    n = min(len(converted), len(inst))
    mix = converted[:n] + inst[:n]  # _reply_raw peak-normalizes the sum
    return _reply_raw(rid, mix.astype(np.float32), demucs_sr)


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            resp = _handle_music(req) if req.get("cmd") == "music" else _handle(req)
        except Exception as e:  # noqa: BLE001
            print(f"seedvc-worker: error {e}", file=sys.stderr, flush=True)
            resp = {"id": rid, "ok": False, "error": str(e)}
        _PROTO.write(json.dumps(resp) + "\n")
        _PROTO.flush()


if __name__ == "__main__":
    main()
