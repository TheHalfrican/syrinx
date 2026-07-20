//! Syrinx main window.
//!
//! The Slint UI runs on the main thread; a tokio worker thread owns the D-Bus
//! connection to `sh.syrinx.Engine1`. They talk via:
//!   - an mpsc channel  (UI  -> worker: Speak / Cancel)
//!   - `Weak::upgrade_in_event_loop`  (worker -> UI: voices, level, state)

slint::include_modules!();

use futures_util::StreamExt;
use slint::{ComponentHandle, ModelRc, SharedString, VecModel};
use std::rc::Rc;
use syrinx_shared::EngineProxy;
use tokio::sync::mpsc;

enum Cmd {
    Speak { text: String, voice: String },
    Cancel { gen_id: u32 },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let ui = AppWindow::new()?;
    let (tx, rx) = mpsc::unbounded_channel::<Cmd>();

    // Speak pressed -> optimistic UI + tell the worker.
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_speak(move |text, voice| {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_speaking(true);
            }
            let _ = tx.send(Cmd::Speak {
                text: text.into(),
                voice: voice.into(),
            });
        });
    }
    // Stop pressed.
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_stop(move || {
            if let Some(ui) = ui_weak.upgrade() {
                ui.set_speaking(false);
            }
            // gen_id 0 = "current"; the worker tracks the real one.
            let _ = tx.send(Cmd::Cancel { gen_id: 0 });
        });
    }

    // D-Bus worker on its own tokio runtime.
    let ui_weak = ui.as_weak();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        if let Err(e) = rt.block_on(engine_worker(ui_weak, rx)) {
            tracing::error!("engine worker exited: {e:#}");
        }
    });

    ui.run()?;
    Ok(())
}

async fn engine_worker(
    ui: slint::Weak<AppWindow>,
    mut rx: mpsc::UnboundedReceiver<Cmd>,
) -> anyhow::Result<()> {
    let conn = zbus::Connection::session().await?;
    let proxy = EngineProxy::new(&conn).await?;

    // Initial state: voices + backend.
    let voices = proxy.list_voices().await.unwrap_or_default();
    let backend = proxy.backend().await.unwrap_or_else(|_| "cpu".into());
    {
        let names: Vec<SharedString> = voices.iter().map(|(_, n)| n.into()).collect();
        ui.upgrade_in_event_loop(move |ui| {
            ui.set_voices(ModelRc::from(Rc::new(VecModel::from(names))));
            ui.set_backend(backend.into());
        })
        .ok();
    }

    let mut levels = proxy.receive_audio_level().await?;
    let mut ended = proxy.receive_speak_ended().await?;
    let mut current_gen: u32 = 0;

    loop {
        tokio::select! {
            Some(sig) = levels.next() => {
                if let Ok(a) = sig.args() {
                    let rms = a.rms as f32;
                    ui.upgrade_in_event_loop(move |ui| ui.set_level(rms)).ok();
                }
            }
            Some(_) = ended.next() => {
                ui.upgrade_in_event_loop(|ui| {
                    ui.set_speaking(false);
                    ui.set_level(0.0);
                }).ok();
            }
            cmd = rx.recv() => match cmd {
                Some(Cmd::Speak { text, voice }) => {
                    // The UI selects by display name; map it back to the voice id.
                    let voice_id = voices.iter()
                        .find(|(_, n)| *n == voice)
                        .map(|(id, _)| id.clone())
                        .unwrap_or(voice);
                    match proxy.speak(&text, &voice_id).await {
                        Ok(id) => current_gen = id,
                        Err(e) => tracing::error!("speak failed: {e}"),
                    }
                }
                Some(Cmd::Cancel { gen_id }) => {
                    let id = if gen_id == 0 { current_gen } else { gen_id };
                    if id != 0 {
                        proxy.cancel(id).await.ok();
                    }
                }
                None => break, // UI dropped the sender
            },
            else => break,
        }
    }
    Ok(())
}
