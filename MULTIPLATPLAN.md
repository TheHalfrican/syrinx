# Syrinx Multi-Platform Plan

Syrinx today is deliberately Linux-first: Wayland-native, D-Bus-wired, tuned for
CachyOS + Hyprland. This document is the roadmap for bringing the same codebase
— not a fork — to Windows and macOS once the Linux experience is fully polished.

**Status: planning only.** Nothing here blocks or changes Linux work. Every
"phase 1" item below is written so that doing it *improves* the Linux build too.

---

## Principles

1. **One codebase, three platforms.** Platform differences live behind small,
   explicit seams (transport, audio capture, text injection) — never as forks
   of app or engine logic.
2. **Linux remains the reference platform.** Ports follow Linux polish, not the
   other way around.
3. **No webviews.** The Slint UI is the cross-platform story; that's why it was
   chosen. No Tauri, no Electron.
4. **The engine contract stays thin.** All portability work happens at the
   service/transport layer; ML modules (`tts.py`, `stt.py`, `llm.py`,
   `effects.py`, backends) must not grow platform conditionals beyond device
   selection.

---

## Portability audit

| Component | Today | Portability | Action |
|---|---|---|---|
| Slint UI (app/) | winit/femtovg | ✅ Win/mac/Linux native | Font fallback, HiDPI check |
| Theme system | 5 skins | ✅ | '95 skin: Tahoma → fallback chain on mac |
| File dialogs (rfd) | ✅ | ✅ native everywhere | none |
| Avatar pipeline (image crate) | ✅ | ✅ | none |
| **IPC: D-Bus (zbus / dbus_next)** | Linux session bus | ❌ **the one architectural blocker** | Replace with JSON-RPC over localhost (see below) |
| Engine ML core (torch/transformers/faster-whisper/kokoro/pedalboard) | CPU/CUDA | ✅ pip-installable on all three | Device matrix (below) |
| Isolated-venv workers (LuxTTS pattern) | subprocess, JSON-over-stdio | ✅ pattern is portable | Verify k2 wheels per-OS |
| Audio playback (sounddevice/PortAudio) | ✅ | ✅ | none |
| Mic recording | app shells out to `parecord` | ❌ PulseAudio-only | Move recording INTO the engine via sounddevice (fixes Linux dependency too) |
| System-audio capture | `parecord --device=<sink>.monitor` | ❌ per-OS | Win: WASAPI loopback · mac: needs loopback driver (BlackHole) · phase 3 |
| Dictation (dictate/) | pw-record + wtype + wlr-layer-shell + compositor keybind | ❌ Wayland-native by design | Per-OS reimplementation, phase 3; ship v1 ports without it |
| Paths | XDG (`~/.local/share/syrinx`, XDG_RUNTIME_DIR) | ❌ hardcoded | `platformdirs` (py) + `dirs` (rs) |
| Process lifecycle | `setsid nohup` by hand | ❌ (and clunky on Linux) | App spawns/supervises the engine as a child process |
| Packaging | cargo build + venv by hand | ❌ | Per-OS installers, phase 2 |

Roughly 90% of the code needs zero changes.

---

## Phase 1 — De-Linuxing the seams (all changes also benefit Linux)

### 1.1 Transport: D-Bus → JSON-RPC over localhost

The only structural change. Design:

- **Protocol:** JSON-RPC 2.0 over a WebSocket on `127.0.0.1:<port>` (or a Unix
  socket / named pipe; WebSocket chosen because it gives framing + push in one
  well-supported package on all three platforms).
- **Methods** map 1:1 from the current `sh.syrinx.Engine1` surface — the
  contract already lives in exactly two thin places:
  - Rust: the `Engine` trait in `shared/src/lib.rs` (zbus proxy macro) →
    becomes a hand-rolled (or macro-generated) RPC client over
    `tokio-tungstenite`.
  - Python: `service.py` (dbus_next `ServiceInterface`) → same methods exposed
    through `websockets`/`aiohttp`. ML modules untouched.
- **Signals** (GenerationProgress, AudioLevel, PlaybackInfo/Progress,
  LlmResult, ModelProgress, SpeakStarted/Ended) → server-push notifications on
  the same socket. The app's `tokio::select!` loop keeps its shape; only the
  stream sources change.
- **Auth/scope:** bind to loopback only; write the ephemeral port + a session
  token to the runtime dir so only local user processes connect.
- **Debugging:** losing `busctl` costs real ergonomics — add a tiny
  `syrinx-cli` (call any method, watch the event stream) as part of this work.
- **Migration strategy:** implement the RPC server alongside D-Bus first
  (both active), port the app, then delete the D-Bus layer once stable. The
  dictate binary migrates in the same sweep (it only uses Transcribe +
  RefineTranscript + LlmResult).

### 1.2 Engine lifecycle

The app spawns `syrinx-engine` as a child process on launch (configurable to
attach to an already-running one for dev), restarts it on crash, and shuts it
down on exit. Kills the "restart each morning" ritual on Linux and is required
on Win/mac anyway. The ephemeral-port handshake from 1.1 doubles as the
readiness signal.

### 1.3 Recording moves into the engine

Replace the app-side `parecord` shell-out with engine methods
(`StartRecording(device)/StopRecording -> wav path`) implemented with
sounddevice input streams (PortAudio: WASAPI/CoreAudio/Pulse-Pipewire).
The create-voice modal keeps its exact UX. System-audio capture stays
Linux-only until phase 3 (hide the "System" tab where unsupported).

### 1.4 Paths

- Python: `platformdirs` — data dir (profiles/history/models.json), cache dir.
- Rust: `dirs` — runtime/scratch files (recording temp, dictate state).
- One data-dir override env (`SYRINX_DATA_DIR`, already exists) works everywhere.

**Phase 1 exit criteria:** full generation studio (voices, cloning, effects,
history, avatars, compose/rewrite/refine, Models tab) runs on Windows and macOS
from a source checkout; Linux runs on the same transport with the app managing
the engine.

---

## Phase 2 — ML device matrix & packaging

### 2.1 Device matrix

| Backend | Linux | Windows | macOS |
|---|---|---|---|
| Kokoro | CPU ✅ / CUDA | CPU / CUDA | CPU / MPS |
| Qwen-TTS | CUDA | CUDA | MPS (verify) / CPU — consider MLX port later |
| LuxTTS (venv) | CPU ✅ / CUDA (k2 cuda wheels) | verify k2 Windows wheels; fallback CPU torch + k2 CPU wheel | verify k2 mac wheels (CPU) |
| faster-whisper (CTranslate2) | CPU ✅ / CUDA | CPU / CUDA | CPU (no Metal in CT2 — still fast) |
| Qwen3 LLM | CPU ✅ / CUDA fp16 | CUDA fp16 | **MPS fp16** (add "mps" to llm.py device pick) |
| pedalboard | ✅ | ✅ | ✅ |

Notes:
- Device selection is already centralized (`detect_device()`, per-module
  `torch.cuda.is_available()`): extend each with an MPS branch — a few lines.
- `models.py` hardware detection: report MPS/Metal as the GPU on mac.
- The k2 wheel index (k2-fsa.github.io) is the load-bearing dependency to
  verify per-OS *before* promising LuxTTS there; Qwen-TTS is the primary
  cloning engine on GPU boxes regardless.

### 2.2 Packaging

- **Windows:** embedded CPython + pre-built venv, Rust binaries, NSIS/MSIX
  installer; CUDA torch pulled on first run (or a "full" installer variant).
- **macOS:** `.app` bundle (Slint binary), bundled Python framework,
  codesign + notarization. Universal2 or arm64-only (decide; arm64-only is
  reasonable in 2026).
- **Linux:** stays source-first; optionally AUR package and/or Flatpak later
  (Flatpak complicates D-Bus/portals less once we're on localhost RPC).
- First-run model downloads already go through the Models tab — the installers
  ship no weights.

---

## Phase 3 — Platform-native features

- **Dictation:** per-OS global hotkey + text injection:
  - Windows: RegisterHotKey + SendInput.
  - macOS: Carbon/NSEvent hotkey + CGEventPost (needs Accessibility grant).
  - The pill overlay is cosmetic; ship without it first.
- **System-audio capture:** Windows WASAPI loopback (supported by PortAudio
  builds / cpal); macOS requires a virtual loopback device (document BlackHole,
  detect its absence gracefully).
- **Auto-update:** optional; per-OS mechanisms differ, decide when packaging
  stabilizes.

---

## Risks / open questions

- **k2 wheel coverage** on Windows/mac (LuxTTS). Mitigation: LuxTTS is
  optional; Qwen-TTS covers cloning on GPU machines.
- **Qwen-TTS on MPS** — unverified; may need CPU fallback or an MLX-based
  backend for Apple Silicon.
- **Slint renderer quirks** per-OS (font metrics, HiDPI, the clip+radius
  offscreen behavior) — audit visually during phase 1 bring-up.
- **Long-path/Unicode issues on Windows** for HF cache + profile dirs — test
  with non-ASCII user names.
- **Engine cold-start UX** on first run (model downloads + venv) — needs a
  first-run screen rather than a silent wait.

## Non-goals

- No Tauri/Electron, no webview UI.
- No per-platform forks of app or engine logic.
- No cloud anything — Syrinx stays fully local on every OS.

---

## Sequencing gate

Phase 1 starts only after the Linux polish backlog is done (composer/effects
chain editor, refinement toggles UI, remaining tabs, GPU backends validated on
a CUDA desktop). Until then this document just accumulates findings — append,
don't branch.
