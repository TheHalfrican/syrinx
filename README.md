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
| `engine/` | Python | ML inference — seven TTS engines (Kokoro, Qwen TTS, Qwen CustomVoice, LuxTTS, Chatterbox, Chatterbox Turbo, TADA), three voice-conversion engines (Chatterbox VC, Seed-VC, Vevo-Timbre), faster-whisper STT, Qwen3 personality LLM, Demucs stem separation, pedalboard effects. Plays audio via PipeWire; exposes `sh.syrinx.Engine1` on D-Bus. GPL / dependency-conflicting engines run as isolated-venv worker subprocesses. |
| `app/`    | Rust + **Slint** | The main window — native GPU-rendered UI. |
| `dictate/`| Rust | The dictation pill: a `wlr-layer-shell` overlay, PipeWire capture, `ydotool` paste. Fired by a Hyprland keybind. |
| `mcp/`    | — | MCP server exposing `syrinx.speak` to agents (stub). |
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

**Feature-complete against the original design** — every navigation
destination is a real, working view. Daily-drivable on a CPU-only machine;
a CUDA GPU makes the heavy engines fast rather than possible.

- **Text-to-speech** across seven engines: Kokoro presets (language-filtered),
  and zero-shot cloning with Qwen TTS (1.7B/0.6B), Qwen CustomVoice, LuxTTS
  (CPU, faster than realtime), Chatterbox Multilingual, Chatterbox Turbo, and
  TADA — every engine chunked at sentence boundaries so long texts synthesize
  in bounded memory, with a VRAM policy that evicts deselected backends.
- **Voice conversion** (the ⇄ Voice Converter): style-preserved dubbing — the
  source's words, timing and delivery survive; only the timbre changes. Three
  models (Chatterbox VC, Seed-VC, Vevo-Timbre), mic / system-tap / file
  sources with auto-transcription, a saved-clip library with cached
  transcripts, and **music mode**: Demucs isolates the vocals, Seed-VC's
  f0-conditioned singing model converts them, and the result is remixed over
  the original instrumental.
- **Voice profiles**: create from a recording / upload / system-audio capture,
  full editing, avatars (circle or side-panel crop), portable export/import
  zips, per-profile engine pins, and a personality LLM (compose /
  speak-in-character / rewrite), plus a dedicated Voices view with sample
  audition.
- **Audio Library**: every generation, saved and searchable — full-text
  search, type / model / voice filters, starred-only, user tags, and the
  complete action set on every row.
- **Persistent history** with a shared player (loop, live volume, drag-to-seek
  waveform), star / regenerate / export, and retroactive effects.
- **Effects**: pedalboard chains — four built-in presets plus a full chain
  editor (reorder, bypass, per-parameter sliders, live preview, saved user
  presets).
- **Transcription workspace**: mic / system-audio / file import with live
  streaming partials, LLM transcript refinement, and persistent text captures.
- **Dictation pill** (`syrinx-dictate toggle`) with an LLM cleanup pass
  toggleable from Settings (or `--refine`).
- **Models tab**: download and hot-switch STT / LLM / voice engines, plus
  weight management for the conversion models.
- **Settings**: persisted theme, PipeWire capture-device pickers, dictation
  refinement, live engine knobs, default export folder.
- Multiple full-chrome UI themes (Matrix TTY, Win95, Frutiger Aero among them).

The isolated conversion engines set up with one command each:
`engine/setup-seedvc.sh` and `engine/setup-vevo.sh` (CUDA auto-detected;
Seed-VC is GPL-3.0 and Vevo's weights are CC-BY-NC, so both live outside the
main engine venv and their weights download on first use).

## Install as a desktop app (beta)

Once the engine venv exists (see Quickstart), one command turns this checkout
into something that behaves like an installed app:

```sh
scripts/install.sh
```

That builds the release binaries and installs:

| What | Where |
|------|-------|
| `syrinx-app`, `syrinx-dictate`, `syrinx-dictate-pill` | `~/.local/bin/` |
| systemd `--user` unit (`Type=dbus`) | `~/.config/systemd/user/syrinx-engine.service` |
| D-Bus activation file | `~/.local/share/dbus-1/services/sh.syrinx.Engine.service` |
| Launcher entry | `~/.local/share/applications/syrinx.desktop` |
| Icon | `~/.local/share/icons/hicolor/scalable/apps/syrinx.svg` |

Syrinx then shows up in your app launcher. **The engine is not enabled at
login and doesn't need to be** — D-Bus activation starts it the moment the app
or the dictation pill first talks to `sh.syrinx.Engine`, and systemd restarts
it on failure. The first call after a cold start waits ~15s for model warmup;
everything after that is instant.

Watch it work:

```sh
journalctl --user -u syrinx-engine -f
```

> **Beta caveat — the engine runs from this checkout.** `ExecStart` points at
> `engine/.venv/bin/syrinx-engine` inside this clone, not at `/usr/bin`. That's
> deliberate: the engine's venvs (`.venv`, `.venv-seedvc`, `.venv-vevo`) are
> large and machine-local, and its worker paths resolve relative to the engine
> source dir. So don't move or delete this directory while Syrinx is installed
> — if you do relocate it, just re-run `scripts/install.sh` from the new path.
> A self-contained system-wide package is what `packaging/PKGBUILD` is for.

Uninstall — stops the engine and removes every file listed above, leaving the
checkout untouched:

```sh
scripts/install.sh --uninstall
```
