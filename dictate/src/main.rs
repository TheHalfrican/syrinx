//! syrinx-dictate — global dictation for Hyprland.
//!
//!   bind = SUPER, D, exec, syrinx-dictate toggle
//!
//! Press once to start recording, again to stop → transcribe (engine's
//! whisper over D-Bus) → paste into the focused window. State lives in a small
//! file under $XDG_RUNTIME_DIR, so the two key-presses are independent runs.
//!
//! Native-Hyprland stack: pw-record (PipeWire capture), the engine's
//! faster-whisper for STT, and wtype (virtual-keyboard protocol) for injection.
//! The wlr-layer-shell visual pill is the next increment — TODO(syrinx).

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use syrinx_shared::EngineProxy;

fn runtime_dir() -> PathBuf {
    std::env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}
fn state_path() -> PathBuf {
    runtime_dir().join("syrinx-dictate.state")
}
fn wav_path() -> PathBuf {
    runtime_dir().join("syrinx-dictate.wav")
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();
    match std::env::args().nth(1).as_deref().unwrap_or("toggle") {
        "toggle" => toggle(),
        "start" => start_recording(),
        "stop" => stop_and_transcribe(),
        other => {
            eprintln!("syrinx-dictate: unknown command '{other}' (toggle|start|stop)");
            std::process::exit(2);
        }
    }
}

fn toggle() -> Result<()> {
    if state_path().exists() {
        stop_and_transcribe()
    } else {
        start_recording()
    }
}

fn start_recording() -> Result<()> {
    let wav = wav_path();
    let _ = fs::remove_file(&wav);

    // pw-record captures until it receives SIGINT; 16 kHz mono is what whisper wants.
    let child = Command::new("pw-record")
        .args(["--rate", "16000", "--channels", "1"])
        .arg(&wav)
        .spawn()
        .context("failed to spawn pw-record (is pipewire installed?)")?;

    fs::write(state_path(), format!("{}\n{}", child.id(), wav.display()))?;
    tracing::info!("● recording — run `syrinx-dictate toggle` again to stop");
    // TODO(syrinx): map the wlr-layer-shell pill here.
    Ok(())
}

fn stop_and_transcribe() -> Result<()> {
    let state = fs::read_to_string(state_path()).context("no recording in progress")?;
    let mut lines = state.lines();
    let pid: i32 = lines
        .next()
        .unwrap_or("")
        .trim()
        .parse()
        .context("corrupt state file (pid)")?;
    let wav = lines.next().unwrap_or("").trim().to_string();
    let _ = fs::remove_file(state_path());

    // SIGINT lets pw-record flush and finalize the WAV header cleanly.
    let _ = Command::new("kill").args(["-INT", &pid.to_string()]).status();
    std::thread::sleep(std::time::Duration::from_millis(400));

    let text = transcribe(&wav)?;
    let _ = fs::remove_file(&wav);
    let text = text.trim();
    if text.is_empty() {
        tracing::info!("(no speech detected)");
        return Ok(());
    }
    tracing::info!("transcribed: {text}");
    paste(text)?;
    // TODO(syrinx): hide the pill.
    Ok(())
}

/// Ask the engine to transcribe the WAV over D-Bus.
fn transcribe(wav: &str) -> Result<String> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let conn = zbus::Connection::session().await?;
        let proxy = EngineProxy::new(&conn).await?;
        Ok::<String, anyhow::Error>(proxy.transcribe(wav).await?)
    })
}

/// Inject text into the focused window. Prefer wtype (virtual-keyboard protocol,
/// works natively on wlroots/Hyprland); fall back to the clipboard.
fn paste(text: &str) -> Result<()> {
    if Command::new("wtype")
        .arg(text)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Ok(());
    }
    let _ = Command::new("wl-copy").arg(text).status();
    tracing::warn!("wtype not found — text copied to clipboard. Install `wtype` to auto-paste.");
    Ok(())
}
