//! Syrinx main window — themed shell + TTS workspace wired to the engine.
//!
//! The Slint UI runs on the main thread; a tokio worker owns the D-Bus
//! connection to `sh.syrinx.Engine1`. Theme switching and tab nav are pure UI
//! (Slint globals); voices, generate, level, and history cross the bridge.

slint::include_modules!();

use futures_util::StreamExt;
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::collections::HashMap;
use std::rc::Rc;
use syrinx_shared::EngineProxy;
use tokio::sync::mpsc;

enum Cmd {
    Generate { text: String, voice: String },
    Cancel { gen_id: u32 },
    Play { id: String },
    Star { id: String, on: bool },
    Delete { id: String },
    Regenerate { id: String },
    Pause,
    Resume,
    Seek { id: String, pct: f64 },
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
            ui.set_synthesizing(true);
            history.insert(
                0,
                HistItem {
                    id: "".into(),
                    voice: voice_name(&ui, &voice).into(),
                    meta: "generating…".into(),
                    text: text.clone().into(),
                    starred: false,
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

    // History actions.
    {
        let tx = tx.clone();
        ui.on_play_hist(move |id| {
            let _ = tx.send(Cmd::Play { id: id.to_string() });
        });
    }
    {
        let tx = tx.clone();
        let history = history.clone();
        ui.on_star_hist(move |id, on| {
            // optimistic UI toggle; the engine persists it
            for i in 0..history.row_count() {
                if let Some(mut it) = history.row_data(i) {
                    if it.id == id {
                        it.starred = on;
                        history.set_row_data(i, it);
                        break;
                    }
                }
            }
            let _ = tx.send(Cmd::Star { id: id.to_string(), on });
        });
    }
    {
        let tx = tx.clone();
        ui.on_delete_hist(move |id| {
            let _ = tx.send(Cmd::Delete { id: id.to_string() });
        });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_regen_hist(move |id| {
            let ui = ui_weak.unwrap();
            ui.set_generating(true);
            ui.set_synthesizing(true);
            let _ = tx.send(Cmd::Regenerate { id: id.to_string() });
        });
    }
    {
        let tx = tx.clone();
        ui.on_pause(move || { let _ = tx.send(Cmd::Pause); });
    }
    {
        let tx = tx.clone();
        ui.on_resume(move || { let _ = tx.send(Cmd::Resume); });
    }
    {
        let tx = tx.clone();
        ui.on_seek(move |id, pct| {
            let _ = tx.send(Cmd::Seek { id: id.to_string(), pct: pct as f64 });
        });
    }

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
    // Kokoro presets live in a separate model (the dropdown), not the card grid.
    let ids = ui.get_kokoro_ids();
    let names = ui.get_kokoro_names();
    for i in 0..ids.row_count() {
        if ids.row_data(i).map(|s| s.as_str() == id).unwrap_or(false) {
            if let Some(n) = names.row_data(i) {
                return n.to_string();
            }
        }
    }
    id.to_string()
}

/// The voices grid, split for the UI: bundled presets collapse into the Kokoro
/// dropdown; user-created profiles become individual cards.
struct GridData {
    grid: Vec<VoiceItem>,          // [Kokoro card, user voices…, spacer padding]
    kokoro_names: Vec<SharedString>,
    kokoro_ids: Vec<SharedString>,
    default_selected: String,      // a bundled voice, so generation works out of the box
}

/// Build the grid from ListVoices (built-ins `builtin:…` + profile ids) enriched
/// with profile details from ListProfiles.
fn build_grid(raw: Vec<(String, String)>, profiles_json: &str) -> GridData {
    let profs: Vec<serde_json::Value> = serde_json::from_str(profiles_json).unwrap_or_default();
    let mut pmap: HashMap<String, serde_json::Value> = HashMap::new();
    for p in profs {
        if let Some(id) = p.get("id").and_then(|v| v.as_str()) {
            pmap.insert(id.to_string(), p);
        }
    }

    let mut kokoro_names: Vec<SharedString> = Vec::new();
    let mut kokoro_ids: Vec<SharedString> = Vec::new();
    let mut users: Vec<VoiceItem> = Vec::new();

    for (id, name) in raw {
        if id.starts_with("builtin:") {
            kokoro_names.push(name.into());
            kokoro_ids.push(id.into());
        } else {
            let (desc, lang, kind) = if let Some(p) = pmap.get(&id) {
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
            users.push(VoiceItem {
                id: id.into(),
                name: name.into(),
                desc: desc.into(),
                lang: lang.into(),
                kind: kind.into(),
            });
        }
    }

    // grid = Kokoro Defaults card + user cards, padded to a multiple of 3 with
    // invisible spacers so the 3-column GridLayout always has full first row.
    let mut grid: Vec<VoiceItem> = Vec::with_capacity(users.len() + 3);
    grid.push(VoiceItem {
        id: "__kokoro__".into(),
        name: "Kokoro Defaults".into(),
        desc: "".into(),
        lang: "".into(),
        kind: "model-defaults".into(),
    });
    grid.extend(users);
    while grid.len() % 3 != 0 {
        grid.push(VoiceItem {
            id: "".into(),
            name: "".into(),
            desc: "".into(),
            lang: "".into(),
            kind: "empty".into(),
        });
    }

    let default_selected = kokoro_ids.first().map(|s| s.to_string()).unwrap_or_default();
    GridData { grid, kokoro_names, kokoro_ids, default_selected }
}

/// Format seconds as m:ss (Voicebox-style meta).
fn fmt_dur(d: f64) -> String {
    let s = d.round().max(0.0) as i64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// Build the history model from the engine's ListHistory JSON (newest first).
fn build_history(json: &str) -> Vec<HistItem> {
    let arr: Vec<serde_json::Value> = serde_json::from_str(json).unwrap_or_default();
    arr.iter()
        .map(|h| {
            let get = |k: &str| h.get(k).and_then(|v| v.as_str()).unwrap_or("");
            let voice = {
                let n = get("voice_name");
                if n.is_empty() { get("voice_id") } else { n }
            };
            let engine = get("engine");
            let lang = get("language");
            let dur = h.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let meta = if engine.is_empty() {
                format!("{} · {}", fmt_dur(dur), lang)
            } else {
                format!("{} · {} · {}", engine, fmt_dur(dur), lang)
            };
            HistItem {
                id: get("id").into(),
                voice: voice.into(),
                meta: meta.into(),
                text: get("text").into(),
                starred: h.get("starred").and_then(|v| v.as_bool()).unwrap_or(false),
            }
        })
        .collect()
}

/// Replace the history model's contents in place (keeps the shared VecModel).
fn set_history_model(ui: &AppWindow, items: Vec<HistItem>) {
    if let Some(vm) = ui.get_history().as_any().downcast_ref::<VecModel<HistItem>>() {
        vm.set_vec(items);
    } else {
        ui.set_history(ModelRc::from(Rc::new(VecModel::from(items))));
    }
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
    let GridData { grid, kokoro_names, kokoro_ids, default_selected } =
        build_grid(raw, &profiles_json);
    let hist_items = build_history(&proxy.list_history().await.unwrap_or_else(|_| "[]".into()));
    {
        ui.upgrade_in_event_loop(move |ui| {
            ui.set_backend(backend.into());
            if ui.get_selected_voice().is_empty() {
                ui.set_selected_voice(default_selected.into());
            }
            ui.set_kokoro_names(ModelRc::from(Rc::new(VecModel::from(kokoro_names))));
            ui.set_kokoro_ids(ModelRc::from(Rc::new(VecModel::from(kokoro_ids))));
            ui.set_voices(ModelRc::from(Rc::new(VecModel::from(grid))));
            set_history_model(&ui, hist_items);
        })
        .ok();
    }

    let mut levels = proxy.receive_audio_level().await?;
    let mut ended = proxy.receive_speak_ended().await?;
    let mut pinfo = proxy.receive_playback_info().await?;
    let mut pprog = proxy.receive_playback_progress().await?;
    let mut current_gen: u32 = 0;
    let mut player_dur: f64 = 0.0;
    let mut current_play_gen: u32 = 0;
    let mut playing = false;

    loop {
        tokio::select! {
            Some(sig) = levels.next() => {
                if let Ok(a) = sig.args() {
                    let rms = a.rms as f32;
                    ui.upgrade_in_event_loop(move |ui| ui.set_level(rms)).ok();
                }
            }
            Some(sig) = pinfo.next() => {
                if let Ok(a) = sig.args() {
                    current_play_gen = a.gen_id;
                    playing = true;
                    player_dur = a.duration;
                    let bars: Vec<f32> = serde_json::from_str(&a.bars).unwrap_or_default();
                    let title = a.title;
                    let clip_id = a.clip_id;
                    let time = format!("0:00 / {}", fmt_dur(a.duration));
                    ui.upgrade_in_event_loop(move |ui| {
                        ui.set_player_bars(ModelRc::from(Rc::new(VecModel::from(bars))));
                        ui.set_player_title(title.into());
                        ui.set_player_id(clip_id.into());
                        ui.set_player_time(time.into());
                        ui.set_play_pct(0.0);
                        ui.set_player_active(true);
                        ui.set_player_playing(true);
                        ui.set_player_paused(false);
                        ui.set_synthesizing(false);
                    }).ok();
                }
            }
            Some(sig) = pprog.next() => {
                if let Ok(a) = sig.args() {
                    if a.gen_id == current_play_gen {
                        let pct = a.pct as f32;
                        let time = format!("{} / {}", fmt_dur(a.pct * player_dur), fmt_dur(player_dur));
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_play_pct(pct);
                            ui.set_player_time(time.into());
                        }).ok();
                    }
                }
            }
            Some(sig) = ended.next() => {
                let is_current = sig.args().map(|a| a.gen_id == current_play_gen).unwrap_or(false);
                if is_current { playing = false; }
                // Refresh history only on success — never wipe the list on a failed call.
                let refreshed = proxy.list_history().await.ok().map(|j| build_history(&j));
                ui.upgrade_in_event_loop(move |ui| {
                    ui.set_generating(false);
                    ui.set_synthesizing(false);
                    ui.set_level(0.0);
                    if is_current {
                        ui.set_player_playing(false);
                        ui.set_player_paused(false);
                        ui.set_play_pct(1.0);
                    }
                    if let Some(items) = refreshed {
                        set_history_model(&ui, items);
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
                Some(Cmd::Play { id }) => {
                    match proxy.play_history(&id).await {
                        Ok(gid) if gid != 0 => current_gen = gid,
                        Ok(_) => {}
                        Err(e) => tracing::error!("play_history failed: {e}"),
                    }
                }
                Some(Cmd::Star { id, on }) => {
                    if let Err(e) = proxy.star_history(&id, on).await {
                        tracing::error!("star_history failed: {e}");
                    }
                }
                Some(Cmd::Delete { id }) => {
                    if let Err(e) = proxy.delete_history(&id).await {
                        tracing::error!("delete_history failed: {e}");
                    }
                    if let Ok(json) = proxy.list_history().await {
                        let items = build_history(&json);
                        ui.upgrade_in_event_loop(move |ui| set_history_model(&ui, items)).ok();
                    }
                }
                Some(Cmd::Regenerate { id }) => {
                    match proxy.regenerate_history(&id).await {
                        Ok(gid) if gid != 0 => current_gen = gid,
                        Ok(_) => {}
                        Err(e) => tracing::error!("regenerate_history failed: {e}"),
                    }
                }
                Some(Cmd::Pause) => { proxy.pause_playback().await.ok(); }
                Some(Cmd::Resume) => { proxy.resume_playback().await.ok(); }
                Some(Cmd::Seek { id, pct }) => {
                    if playing {
                        proxy.seek_playback(pct).await.ok();
                    } else {
                        // not playing — start from the clicked position
                        match proxy.play_history_at(&id, pct).await {
                            Ok(gid) if gid != 0 => current_gen = gid,
                            _ => {}
                        }
                    }
                }
                None => break,
            },
            else => break,
        }
    }
    Ok(())
}
