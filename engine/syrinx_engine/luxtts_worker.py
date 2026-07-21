"""Standalone LuxTTS worker — runs inside the isolated .venv-luxtts.

Reads one JSON request per line on stdin, synthesizes with zipvoice LuxTTS,
writes raw float32 PCM to a temp file, and replies with a JSON line on stdout.
Deliberately independent of the syrinx_engine package: it only imports zipvoice
+ numpy, which is all the lux venv has.

Request : {"id": N, "sample": "<ref.wav>", "text": "<text>"}
Reply   : {"id": N, "ok": true, "raw": "<path>", "rate": 48000}
          {"id": N, "ok": false, "error": "..."}
"""

import json
import os
import sys
import tempfile

import numpy as np

_MODEL = None
_PROMPTS = {}  # sample_path -> encoded prompt


def _load():
    global _MODEL
    if _MODEL is not None:
        return
    from zipvoice.luxvoice import LuxTTS

    threads = min(os.cpu_count() or 4, 8)
    print("luxtts-worker: loading model...", file=sys.stderr, flush=True)
    _MODEL = LuxTTS(model_path="YatharthS/LuxTTS", device="cpu", threads=threads)
    print("luxtts-worker: model loaded", file=sys.stderr, flush=True)


def _handle(req: dict) -> dict:
    rid = req.get("id")
    sample = req["sample"]
    text = req["text"]
    _load()
    if sample not in _PROMPTS:
        _PROMPTS[sample] = _MODEL.encode_prompt(prompt_audio=str(sample), duration=5, rms=0.01)
    wav = _MODEL.generate_speech(
        text=text,
        encode_dict=_PROMPTS[sample],
        num_steps=4,
        guidance_scale=3.0,
        t_shift=0.5,
        speed=1.0,
        return_smooth=False,  # 48 kHz
    )
    try:
        import torch

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
    except Exception:
        pass
    audio = np.asarray(wav).squeeze().astype(np.float32)
    out = os.path.join(tempfile.gettempdir(), f"luxtts-{rid}.raw")
    audio.tofile(out)
    return {"id": rid, "ok": True, "raw": out, "rate": 48000}


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
            print(f"luxtts-worker: error {e}", file=sys.stderr, flush=True)
            resp = {"id": rid, "ok": False, "error": str(e)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
