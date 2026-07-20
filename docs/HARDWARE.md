# Syrinx — Hardware acceleration

Syrinx runs on one codebase across very different machines and auto-adapts. Two
reference targets:

| Machine | Display | Compute for TTS/STT |
|---------|---------|---------------------|
| Laptop  | Intel iGPU (UHD/Xe) | CPU (torch cpu build) |
| Desktop | 14900K iGPU **or** RTX 4090 | **RTX 4090, CUDA** (Ada / sm_89) |

## Engine backend selection

`tts.py::_detect_backend()` picks `cuda` / `rocm` / `cpu` at startup. Overrides:

- `SYRINX_BACKEND=cuda|rocm|cpu` — force a backend.
- `SYRINX_MODEL=large|small` — model tier (auto-selected by VRAM otherwise).

Surface the active backend via the `Backend` D-Bus property (already exposed).

## CUDA (RTX 4090) fast path

Apply in `SpeechSynthesizer.load()` once real Qwen3-TTS lands:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True   # free matmul speedup
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)      # flash attention
device = "cuda"
# inference under: torch.autocast("cuda", dtype=torch.bfloat16)   # Ada tensor cores
# optional: model = torch.compile(model)                          # Inductor JIT
```

- bf16 autocast: ~2× faster, half the VRAM (24 GB to spare).
- The hot `systemd --user` engine keeps weights resident on the GPU across
  requests — load cost is paid once at login, every generation is instant.

## STT

Build **whisper.cpp with CUDA** (`GGML_CUDA=1`, or an AUR `whisper.cpp-cuda`).
Desktop: `large-v3` in <1 s. Laptop: `base`/`small` on CPU stays snappy.

## Arch / CachyOS packages

- **CUDA set:** `cuda`, `cudnn`, `nvidia` (555+ for Wayland), `python-pytorch-cuda`.
- **CPU set:** `python-pytorch` (or CachyOS `-opt`).

The PKGBUILD offers these as alternative dependency sets.

## NVIDIA + Wayland (display only)

The 4090 accelerating TTS is pure CUDA compute — unaffected by Wayland. The only
Wayland consideration is if the 4090 also drives the display for the Slint UI
(want driver 555+; explicit sync is automatic). **Cleanest desktop setup:** let
the 14900K iGPU drive the display and keep the 4090 as headless compute — avoids
NVIDIA/Wayland display quirks entirely while the 4090 does all the ML.
