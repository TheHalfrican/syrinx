//! Syrinx main window — themed shell + TTS workspace wired to the engine.
//!
//! The Slint UI runs on the main thread; a tokio worker owns the D-Bus
//! connection to `sh.syrinx.Engine1`. Theme switching and tab nav are pure UI
//! (Slint globals); voices, generate, level, and history cross the bridge.

slint::include_modules!();

use futures_util::StreamExt;
use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::collections::HashMap;
use std::rc::Rc;
use syrinx_shared::EngineProxy;
use tokio::sync::mpsc;

enum Cmd {
    Generate { text: String, voice: String },
    Cancel { gen_id: u32 },
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let ui = AppWindow::new()?;
    let (tx, rx) = mpsc::unbounded_channel::<Cmd>();

    let history = Rc::new(VecModel::<HistItem>::default());
    ui.set_history(ModelRc::from(history.clone()));

    // Generate pressed.
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        let history = history.clone();
        ui.on_generate(move || {
            let ui = ui_weak.unwrap();
            let text: String = ui.get_text().to_string();
            let voice: String = ui.get_selected_voice().to_string();
            if text.trim().is_empty() || voice.is_empty() {
                return;
            }
            ui.set_generating(true);
            history.insert(
                0,
                HistItem {
                    voice: voice_name(&ui, &voice).into(),
                    meta: "generating…".into(),
                    text: text.clone().into(),
                },
            );
            let _ = tx.send(Cmd::Generate { text, voice });
        });
    }
    // Stop pressed.
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_stop(move || {
            ui_weak.unwrap().set_generating(false);
            let _ = tx.send(Cmd::Cancel { gen_id: 0 });
        });
    }
    // Selection is handled in Slint; nothing to do here yet.
    ui.on_select_voice(|_id| {});

    let ui_weak = ui.as_weak();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        if let Err(e) = rt.block_on(worker(ui_weak, rx)) {
            tracing::error!("engine worker exited: {e:#}");
        }
    });

    ui.run()?;
    Ok(())
}

fn voice_name(ui: &AppWindow, id: &str) -> String {
    let voices = ui.get_voices();
    for i in 0..voices.row_count() {
        if let Some(v) = voices.row_data(i) {
            if v.id == id {
                return v.name.to_string();
            }
        }
    }
    id.to_string()
}

/// Build the voice list from ListVoices (built-ins + profile ids) enriched with
/// profile details from ListProfiles.
fn build_voices(raw: Vec<(String, String)>, profiles_json: &str) -> Vec<VoiceItem> {
    let profs: Vec<serde_json::Value> = serde_json::from_str(profiles_json).unwrap_or_default();
    let mut pmap: HashMap<String, serde_json::Value> = HashMap::new();
    for p in profs {
        if let Some(id) = p.get("id").and_then(|v| v.as_str()) {
            pmap.insert(id.to_string(), p);
        }
    }
    raw.into_iter()
        .map(|(id, name)| {
            let (desc, lang, kind) = if id.starts_with("builtin:") {
                ("Kokoro preset".to_string(), "en".to_string(), "bundled".to_string())
            } else if let Some(p) = pmap.get(&id) {
                let vt = p.get("voice_type").and_then(|v| v.as_str()).unwrap_or("voice");
                let l = p.get("language").and_then(|v| v.as_str()).unwrap_or("en");
                let d = if p.get("has_personality").and_then(|v| v.as_bool()).unwrap_or(false) {
                    "Has personality"
                } else {
                    "Custom voice"
                };
                (d.to_string(), l.to_string(), vt.to_string())
            } else {
                (String::new(), "en".to_string(), "voice".to_string())
            };
            VoiceItem {
                id: id.into(),
                name: name.into(),
                desc: desc.into(),
                lang: lang.into(),
                kind: kind.into(),
            }
        })
        .collect()
}

async fn worker(
    ui: slint::Weak<AppWindow>,
    mut rx: mpsc::UnboundedReceiver<Cmd>,
) -> anyhow::Result<()> {
    let conn = zbus::Connection::session().await?;
    let proxy = EngineProxy::new(&conn).await?;

    let backend = proxy.backend().await.unwrap_or_else(|_| "cpu".into());
    let raw = proxy.list_voices().await.unwrap_or_default();
    let profiles_json = proxy.list_profiles().await.unwrap_or_else(|_| "[]".into());
    let items = build_voices(raw, &profiles_json);
    let first = items.first().map(|v| v.id.to_string()).unwrap_or_default();
    {
        ui.upgrade_in_event_loop(move |ui| {
            ui.set_backend(backend.into());
            if ui.get_selected_voice().is_empty() {
                ui.set_selected_voice(first.into());
            }
            ui.set_voices(ModelRc::from(Rc::new(VecModel::from(items))));
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
                    ui.set_generating(false);
                    ui.set_level(0.0);
                    let h = ui.get_history();
                    if h.row_count() > 0 {
                        if let Some(mut item) = h.row_data(0) {
                            item.meta = "done".into();
                            h.set_row_data(0, item);
                        }
                    }
                }).ok();
            }
            cmd = rx.recv() => match cmd {
                Some(Cmd::Generate { text, voice }) => {
                    match proxy.speak(&text, &voice).await {
                        Ok(id) => current_gen = id,
                        Err(e) => tracing::error!("speak failed: {e}"),
                    }
                }
                Some(Cmd::Cancel { gen_id }) => {
                    let id = if gen_id == 0 { current_gen } else { gen_id };
                    if id != 0 { proxy.cancel(id).await.ok(); }
                }
                None => break,
            },
            else => break,
        }
    }
    Ok(())
}
