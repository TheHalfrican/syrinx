"""Standalone Vevo worker — runs inside the isolated .venv-vevo.

Style-preserved voice conversion via Amphion's Vevo-Timbre (MIT code; the
checkpoints are CC-BY-NC — personal use, auto-downloaded, never
redistributed). Amphion has no pip package, so its clone (setup-vevo.sh)
is put on sys.path and we call its public ``VevoInferencePipeline``; cwd
moves INTO the clone because Amphion resolves config paths and its ./ckpts
download directory relative to cwd — never this repo.

Reads one JSON request per line on stdin, converts source→reference timbre,
writes raw float32 PCM to a temp file, and replies on stdout.

Request : {"id": N, "source": "<src.wav>", "target": "<ref.wav>", "steps": 32}
Reply   : {"id": N, "ok": true, "raw": "<path>", "rate": 24000}
          {"id": N, "ok": false, "error": "..."}
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Reserve the real stdout for the JSON protocol (model downloads and torch
# both chat on fd 1 otherwise).
_PROTO = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.stdout = sys.stderr

_AMPHION = os.environ.get(
    "SYRINX_VEVO_AMPHION",
    str(Path.home() / ".local" / "share" / "syrinx" / "vevo" / "Amphion"),
)
os.chdir(_AMPHION)
sys.path.insert(0, _AMPHION)

import numpy as np

_PIPELINE = None


def _load():
    global _PIPELINE
    if _PIPELINE is not None:
        return
    import torch
    from huggingface_hub import snapshot_download
    from models.vc.vevo.vevo_utils import VevoInferencePipeline

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"vevo-worker: downloading/loading models on {device}...", file=sys.stderr, flush=True)
    # Same three components as Amphion's infer_vevotimbre.py, cached in the
    # clone's ./ckpts (we are chdir'd into the clone).
    local_dir = "./ckpts/Vevo"
    tokenizer_dir = snapshot_download(
        repo_id="amphion/Vevo", cache_dir=local_dir,
        allow_patterns=["tokenizer/vq8192/*"],
    )
    fmt_dir = snapshot_download(
        repo_id="amphion/Vevo", cache_dir=local_dir,
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )
    vocoder_dir = snapshot_download(
        repo_id="amphion/Vevo", cache_dir=local_dir,
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    _PIPELINE = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=os.path.join(tokenizer_dir, "tokenizer/vq8192"),
        fmt_cfg_path="./models/vc/vevo/config/Vq8192ToMels.json",
        fmt_ckpt_path=os.path.join(fmt_dir, "acoustic_modeling/Vq8192ToMels"),
        vocoder_cfg_path="./models/vc/vevo/config/Vocoder.json",
        vocoder_ckpt_path=os.path.join(vocoder_dir, "acoustic_modeling/Vocoder"),
        device=device,
    )
    print("vevo-worker: models loaded", file=sys.stderr, flush=True)


VEVO_SR = 24_000  # vevo_utils.save_audio's sr default — the pipeline's rate


def _handle(req: dict) -> dict:
    import torch

    rid = req.get("id")
    steps = int(req.get("steps", 32))
    _load()
    print(f"vevo-worker: converting (steps={steps})", file=sys.stderr, flush=True)
    gen_audio = _PIPELINE.inference_fm(          # [1, T] float tensor
        src_wav_path=req["source"],
        timbre_ref_wav_path=req["target"],
        flow_matching_steps=steps,
    )
    # Amphion's save_audio needs torchcodec on modern torchaudio just to
    # write a wav — we only need the samples, so apply its −25 dB RMS
    # normalization ourselves and skip the file format entirely.
    wav = gen_audio.detach().float().cpu()
    rms = torch.sqrt(torch.mean(wav**2))
    gain_db = -25.0 - float(20 * torch.log10(rms + 1e-9))
    wav = (wav * (10 ** (gain_db / 20))).squeeze().numpy()
    out = os.path.join(tempfile.gettempdir(), f"vevo-{rid}.raw")
    np.ascontiguousarray(wav, dtype=np.float32).tofile(out)
    return {"id": rid, "ok": True, "raw": out, "rate": VEVO_SR}


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
            print(f"vevo-worker: error {e}", file=sys.stderr, flush=True)
            resp = {"id": rid, "ok": False, "error": str(e)}
        _PROTO.write(json.dumps(resp) + "\n")
        _PROTO.flush()


if __name__ == "__main__":
    main()
