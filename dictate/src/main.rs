//! syrinx-dictate — the dictation pill.
//!
//! Invoked by a Hyprland keybind: `bind = SUPER, D, exec, syrinx-dictate toggle`.
//!
//! Flow:
//!   1. `toggle` → map a wlr-layer-shell overlay ("the pill") and start
//!      PipeWire mic capture.
//!   2. On the next `toggle` (or VAD silence) → stop capture, send PCM to the
//!      engine's `Transcribe` over D-Bus.
//!   3. Put the result on the clipboard (`wl-copy`) and paste it into the
//!      focused window (`ydotool key ctrl+v`).
//!   4. Fade the pill out.
//!
//! Everything here is Hyprland-native: layer-shell for the surface, ydotool
//! (uinput) for injection, PipeWire for audio. No app-level global keytap.

use std::process::Command;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let cmd = std::env::args().nth(1).unwrap_or_else(|| "toggle".into());
    match cmd.as_str() {
        "toggle" => toggle(),
        other => {
            eprintln!("syrinx-dictate: unknown command '{other}' (expected: toggle)");
            std::process::exit(2);
        }
    }
}

fn toggle() -> anyhow::Result<()> {
    tracing::info!("toggle: (stub) show/hide the pill + start/stop capture");

    // TODO(syrinx): single-instance guard (a lock/socket) so a second
    // invocation stops the first's recording instead of spawning a new pill.

    // TODO(syrinx): map the wlr-layer-shell overlay here.
    //   With gtk4-layer-shell: create a GTK window, then
    //     gtk4_layer_shell::init_for_window(&win);
    //     set_layer(Layer::Overlay);
    //     set_anchor(Edge::Bottom, true);
    //   With smithay-client-toolkit: bind zwlr_layer_shell_v1 directly.

    // TODO(syrinx): capture mic via PipeWire into a PCM buffer.

    // TODO(syrinx): let text = EngineProxy::transcribe(&pcm).await?;
    let text = "(transcription goes here)";

    paste(text)?;
    Ok(())
}

/// Inject text into the focused window the Wayland-native way:
/// set the clipboard, then synthesize Ctrl+V via ydotool (uinput).
fn paste(text: &str) -> anyhow::Result<()> {
    // Requires `wl-clipboard` and a running `ydotoold`.
    let ok = Command::new("wl-copy").arg(text).status()?.success();
    if !ok {
        anyhow::bail!("wl-copy failed (is wl-clipboard installed?)");
    }
    // 29 = Left Ctrl, 47 = V (Linux input event codes). ydotool: <code>:<press>
    Command::new("ydotool")
        .args(["key", "29:1", "47:1", "47:0", "29:0"])
        .status()?;
    Ok(())
}
