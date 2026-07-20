# Syrinx

A native, Wayland-first voice studio for Linux — text-to-speech, voice cloning,
and global-hotkey dictation — built for **CachyOS + Hyprland**.

Named for the *syrinx*, the vocal organ birds sing with (and, in myth, the reeds
a nymph became — the first pan-pipe).

> Deliberately **not** cross-platform. Syrinx assumes a rolling Arch system,
> Wayland/wlroots, PipeWire, and a real GPU, and uses each of them natively
> instead of abstracting over them. That's the whole point: no WebView, no
> bundled runtime, no compositor-fighting.

## Architecture at a glance

Four small pieces on the D-Bus session bus, each doing one thing well:

| Component | Language | Role |
|-----------|----------|------|
| `engine/` | Python | ML inference (Qwen3-TTS + whisper.cpp), plays audio via PipeWire, exposes `sh.syrinx.Engine1` on D-Bus. Runs as a hot `systemd --user` service. |
| `app/`    | Rust + **Slint** | The main window — native GPU-rendered UI. |
| `dictate/`| Rust | The dictation pill: a `wlr-layer-shell` overlay, PipeWire capture, `ydotool` paste. Fired by a Hyprland keybind. |
| `mcp/`    | — | MCP server exposing `syrinx.speak` to agents. |
| `shared/` | Rust | Shared D-Bus client + types for the Rust crates. |

Why this shape: every failure mode of a portable stack (WebKitGTK rendering,
2.6 GB runtime re-extraction, always-on-top hacks, synthetic-input fragility)
is designed out by using the native primitive instead — Slint's GPU renderer,
a system-installed engine, `wlr-layer-shell`, and `ydotool`.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

## Quickstart (dev)

```sh
# Engine (Python) — the hot ML service
cd engine
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m syrinx_engine            # registers sh.syrinx.Engine1 on the session bus

# App (Rust + Slint)
cargo run -p syrinx-app

# Dictate pill (Rust) — usually launched by a Hyprland keybind
cargo run -p syrinx-dictate -- toggle
```

Add to your Hyprland config (see `packaging/hyprland.conf`):

```
bind = SUPER, D, exec, syrinx-dictate toggle
```

## Status

Skeleton / design phase. Stubs mark where the real work goes (`TODO(syrinx)`).
