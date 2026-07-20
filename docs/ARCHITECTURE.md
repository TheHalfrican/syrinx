# Syrinx — Architecture

## Design principle

The pain of a portable desktop app comes from abstractions that exist *only* to
be portable. Syrinx targets exactly one environment — **CachyOS + Hyprland** —
so it deletes those layers and uses the native primitive in each case:

| Portable app's pain | Native primitive Syrinx uses instead |
|---------------------|--------------------------------------|
| WebKitGTK webview (blank/white renders) | **Slint** — GPU scene graph, native Wayland |
| PyInstaller onefile re-extracting a 2.6 GB runtime every launch | **System-installed** engine (`python-pytorch-*` from the repos) as a hot `systemd --user` service |
| Always-on-top overlay hacks | **wlr-layer-shell** surface (like waybar/wofi) |
| App-level global hotkey + synthetic keycodes | **Hyprland keybind** → IPC + **ydotool** paste |
| GStreamer `autoaudiosink` plugin roulette | **PipeWire** directly |

## Processes

```
  Hyprland keybind (SUPER+D)
        │ exec
        ▼
  ┌──────────────────┐      D-Bus       ┌────────────────────────────┐
  │ syrinx-dictate   │◄────────────────►│ syrinx-engine (Python)     │
  │ (Rust)           │                  │  Qwen3-TTS (torch)         │
  │ • layer-shell UI │                  │  whisper.cpp (STT)         │
  │ • PipeWire capture│                 │  PipeWire playback         │
  │ • ydotool paste  │                  │  systemd --user, hot       │
  └──────────────────┘                  └───────────┬────────────────┘
                                                     │ D-Bus
                        ┌────────────────┐  D-Bus    │
                        │ syrinx-mcp     │◄──────────┤
                        │ (agent speak)  │           │
                        └────────────────┘  ┌────────▼─────────┐
                                             │ syrinx (Slint UI)│
                                             └──────────────────┘
```

Everything hangs off one **D-Bus session service**: `sh.syrinx.Engine1`.
Because it's D-Bus, the whole app is scriptable from a shell (`busctl call …`),
which is great for testing.

## D-Bus interface: `sh.syrinx.Engine1`

Bus name `sh.syrinx.Engine`, object `/sh/syrinx/Engine`.

**Methods**
- `Speak(s text, s voice_id, a{sv} opts) → u gen_id`
- `Transcribe(ay pcm) → s text`
- `ListVoices() → a(ss)` *(id, display_name)*
- `CloneVoice(s name, s sample_path) → s profile_id`
- `ListModels() → a{sv}` · `DownloadModel(s id) → b`
- `Cancel(u gen_id)`

**Signals**
- `GenerationProgress(u gen_id, s state, d pct)`
- `AudioLevel(u gen_id, d rms)`  ← feeds the UI waveform + the pill animation
- `SpeakStarted(u gen_id)` · `SpeakEnded(u gen_id)`
- `ModelDownloadProgress(s id, d pct)`

**Properties**
- `ModelLoaded b` · `Backend s` (`cuda`|`rocm`|`cpu`) · `GpuAvailable b`

Audio bytes never travel over D-Bus in bulk: the engine plays TTS output itself
via PipeWire and emits only lightweight `AudioLevel` samples for visualization.

## Dictation flow

1. Hyprland: `bind = SUPER, D, exec, syrinx-dictate toggle`
2. `syrinx-dictate` maps a **layer-shell** pill and starts **PipeWire** capture.
3. On stop (keybind again, or VAD), it sends PCM to `Transcribe` (whisper.cpp — no torch).
4. Result → `wl-copy <text>` then `ydotool key ctrl+v` into the focused window.
5. Pill fades out.

## Packaging

- **PKGBUILD → AUR.** Native deps: `python-pytorch-cuda` / `-rocm` / `-cpu`,
  `whisper.cpp`, `ydotool`, `wl-clipboard`, `slint` toolchain at build time.
- **`systemd --user`** unit for the engine (socket-activatable), so it's loaded
  once at login and both the app and the pill attach instantly.

## Open design questions (for later)

- STT: whisper.cpp binding vs subprocess vs a small Rust wrapper.
- Engine in-process vs always separate (separate wins for UI responsiveness).
- Model storage location + download UX (reuse `~/.cache/huggingface`?).
- Multi-GPU / backend switching at runtime (`Backend` property + restart).
