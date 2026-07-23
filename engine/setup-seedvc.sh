#!/usr/bin/env bash
# Set up the isolated Seed-VC venv (engine/.venv-seedvc) for the ⇄ tab's
# Seed-VC conversion engine. Safe to re-run; picks CUDA torch when an NVIDIA
# GPU is present, CPU wheels otherwise.
#
# Seed-VC is GPL-3.0 — it is installed HERE, never vendored into the repo;
# the engine talks to it through seedvc_worker.py in a subprocess. Weights
# auto-download on first conversion to ~/.local/share/syrinx/seedvc (the
# worker's pinned cwd — seed-vc writes ./checkpoints relative to cwd).
set -euo pipefail
cd "$(dirname "$0")"

PY="${SYRINX_SEEDVC_PYTHON:-python3.12}"
"$PY" -m venv .venv-seedvc
.venv-seedvc/bin/pip install -U pip

if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then
    echo "== NVIDIA GPU detected — installing CUDA torch (default PyPI build)"
    .venv-seedvc/bin/pip install torch torchaudio torchvision
else
    echo "== no GPU — installing CPU torch"
    .venv-seedvc/bin/pip install torch torchaudio torchvision \
        --index-url https://download.pytorch.org/whl/cpu
fi

.venv-seedvc/bin/pip install seed-vc

# huggingface_hub 1.x removed the proxies/resume_download kwargs that
# BigVGAN's _from_pretrained (inside seed-vc) still requires, and
# transformers 5.x demands hub>=1.5 — pin the pair to the 4.x era.
.venv-seedvc/bin/pip install 'transformers==4.57.3' 'huggingface_hub<1.0'

.venv-seedvc/bin/python - <<'EOF'
import huggingface_hub, numpy, seed_vc, torch, transformers
print(
    "seed-vc venv OK",
    "· torch", torch.__version__,
    "· transformers", transformers.__version__,
    "· hub", huggingface_hub.__version__,
    "· numpy", numpy.__version__,
    "· cuda", torch.cuda.is_available(),
)
EOF
