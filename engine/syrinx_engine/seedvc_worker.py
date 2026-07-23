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
Reply   : {"id": N, "ok": true, "raw": "<path>", "rate": 22050}
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

import numpy as np

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


def _handle(req: dict) -> dict:
    from seed_vc.api import inference

    rid = req.get("id")
    f0 = bool(req.get("f0", False))
    steps = int(req.get("steps", 25))
    auto_f0 = bool(req.get("auto_f0", True))
    semitone = int(req.get("semitone", 0))
    _ensure_state(f0)

    src_i16, src_rate = _load_wav_i16(req["source"])
    tgt_i16, tgt_rate = _load_wav_i16(req["target"])
    target_ad = _audio_data(tgt_i16, tgt_rate)

    chunk_len = max(1, int(CHUNK_SECS * src_rate))
    slices = [src_i16[i : i + chunk_len] for i in range(0, len(src_i16), chunk_len)]
    parts = []
    out_rate = src_rate
    for i, piece in enumerate(slices):
        last = i == len(slices) - 1
        print(
            f"seedvc-worker: chunk {i + 1}/{len(slices)} (steps={steps})",
            file=sys.stderr, flush=True,
        )
        result = inference(
            source=_audio_data(piece, src_rate),
            target=target_ad,
            new_target_name=req["target"],  # target features cached by path
            diffusion_steps=steps,
            f0_condition=f0,
            auto_f0_adjust=auto_f0,
            semi_tone_shift=semitone,
            fp16=False,
            streaming=True,
            stream_state=_STATE,
            end_of_stream=last,
        )
        out_rate = int(result.sample_rate)
        parts.append(np.asarray(result.samples, dtype=np.float32) / 32767.0)

    audio = np.concatenate(parts) if len(parts) > 1 else parts[0]
    out = os.path.join(tempfile.gettempdir(), f"seedvc-{rid}.raw")
    audio.astype(np.float32).tofile(out)
    return {"id": rid, "ok": True, "raw": out, "rate": out_rate}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            resp = _handle(req)
        except Exception as e:  # noqa: BLE001
            print(f"seedvc-worker: error {e}", file=sys.stderr, flush=True)
            resp = {"id": rid, "ok": False, "error": str(e)}
        _PROTO.write(json.dumps(resp) + "\n")
        _PROTO.flush()


if __name__ == "__main__":
    main()
