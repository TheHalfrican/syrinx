//! Syrinx main window — themed shell + TTS workspace wired to the engine.
//!
//! The Slint UI runs on the main thread; a tokio worker owns the D-Bus
//! connection to `sh.syrinx.Engine1`. Theme switching and tab nav are pure UI
//! (Slint globals); voices, generate, level, and history cross the bridge.

slint::include_modules!();

use futures_util::StreamExt;
use slint::{ComponentHandle, Model, ModelRc, SharedString, VecModel};
use std::collections::HashMap;
use std::process::Stdio;
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
    ExportAudio { id: String },
    ExportPackage { id: String },
    CvStartRecord { system: bool },
    CvStopRecord,
    CvPickFile,
    CvTranscribe,
    CvCreate { name: String, desc: String, personality: String, language: String, transcript: String, model_index: usize },
    CvCancel,
    Compose { voice_id: String, prompt: String },
    Rewrite { voice_id: String, text: String },
    DownloadModel { id: String },
    DeleteModel { id: String },
    ActivateModel { id: String },
    // Voicebox-style composer / cards / player
    GenerateInCharacter { text: String, voice: String },
    SelectVoice { id: String },
    PickLanguage { voice: String, index: usize },
    PickEngine { voice: String, index: usize },
    ToggleLoop { on: bool },
    SetVol { v: f64 },
    PickEffect { index: usize },
    ApplyFx { hid: String, index: usize },
    ExportVoice { id: String, name: String },
    EditVoice { id: String },
    DeleteVoice { id: String },
    ImportVoice,
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
            ui.set_gen_error("".into());
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
            // "Speak in character" toggle: rewrite via the personality LLM first,
            // then synthesize the rewritten line (Voicebox's persona flow).
            if ui.get_persona_on() && ui.get_selected_has_personality() {
                ui.set_llm_busy(true);
                let _ = tx.send(Cmd::GenerateInCharacter { text, voice });
            } else {
                let _ = tx.send(Cmd::Generate { text, voice });
            }
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
    // Track whether the selected voice has a personality (gates Compose/persona)
    // and feed the composer (placeholder name + per-engine language list).
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_select_voice(move |id| {
            let ui = ui_weak.unwrap();
            let voices = ui.get_voices();
            let mut hp = false;
            for i in 0..voices.row_count() {
                if let Some(v) = voices.row_data(i) {
                    if v.id == id {
                        hp = v.has_personality;
                        break;
                    }
                }
            }
            ui.set_selected_has_personality(hp);
            ui.set_selected_voice_name(voice_name(&ui, &id).into());
            let _ = tx.send(Cmd::SelectVoice { id: id.to_string() });
        });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_compose(move || {
            let ui = ui_weak.unwrap();
            ui.set_llm_busy(true);
            let _ = tx.send(Cmd::Compose {
                voice_id: ui.get_selected_voice().to_string(),
                prompt: ui.get_text().to_string(),
            });
        });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_rewrite(move || {
            let ui = ui_weak.unwrap();
            let text = ui.get_text().to_string();
            if text.trim().is_empty() {
                return;
            }
            ui.set_llm_busy(true);
            let _ = tx.send(Cmd::Rewrite {
                voice_id: ui.get_selected_voice().to_string(),
                text,
            });
        });
    }

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
        ui.on_download_model(move |id| { let _ = tx.send(Cmd::DownloadModel { id: id.to_string() }); });
    }
    {
        let tx = tx.clone();
        ui.on_delete_model(move |id| { let _ = tx.send(Cmd::DeleteModel { id: id.to_string() }); });
    }
    {
        let tx = tx.clone();
        ui.on_activate_model(move |id| { let _ = tx.send(Cmd::ActivateModel { id: id.to_string() }); });
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
    {
        let tx = tx.clone();
        ui.on_export_audio(move |id| {
            let _ = tx.send(Cmd::ExportAudio { id: id.to_string() });
        });
    }
    {
        let tx = tx.clone();
        ui.on_export_package(move |id| {
            let _ = tx.send(Cmd::ExportPackage { id: id.to_string() });
        });
    }
    // --- create-voice modal ---
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_cv_start_record(move || {
            let system = ui_weak.unwrap().get_cv_mode() == "system";
            let _ = tx.send(Cmd::CvStartRecord { system });
        });
    }
    {
        let tx = tx.clone();
        ui.on_cv_stop_record(move || { let _ = tx.send(Cmd::CvStopRecord); });
    }
    {
        let tx = tx.clone();
        ui.on_cv_pick_file(move || { let _ = tx.send(Cmd::CvPickFile); });
    }
    {
        let tx = tx.clone();
        ui.on_cv_transcribe(move || { let _ = tx.send(Cmd::CvTranscribe); });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_cv_create(move || {
            let ui = ui_weak.unwrap();
            let _ = tx.send(Cmd::CvCreate {
                name: ui.get_cv_name().to_string(),
                desc: ui.get_cv_desc().to_string(),
                personality: ui.get_cv_personality().to_string(),
                language: ui.get_cv_language().to_string(),
                transcript: ui.get_cv_transcript().to_string(),
                model_index: ui.get_cv_model_index() as usize,
            });
        });
    }
    {
        let tx = tx.clone();
        ui.on_cv_cancel(move || { let _ = tx.send(Cmd::CvCancel); });
    }

    // Composer dropdowns, card actions, player loop/volume (Voicebox parity).
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_pick_language(move |i| {
            let ui = ui_weak.unwrap();
            let _ = tx.send(Cmd::PickLanguage {
                voice: ui.get_selected_voice().to_string(),
                index: i as usize,
            });
        });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_pick_engine(move |i| {
            let ui = ui_weak.unwrap();
            let _ = tx.send(Cmd::PickEngine {
                voice: ui.get_selected_voice().to_string(),
                index: i as usize,
            });
        });
    }
    {
        let tx = tx.clone();
        ui.on_toggle_loop(move |on| { let _ = tx.send(Cmd::ToggleLoop { on }); });
    }
    {
        let tx = tx.clone();
        ui.on_pick_effect(move |i| { let _ = tx.send(Cmd::PickEffect { index: i as usize }); });
    }
    {
        let tx = tx.clone();
        ui.on_apply_fx(move |hid, i| {
            let _ = tx.send(Cmd::ApplyFx { hid: hid.to_string(), index: i as usize });
        });
    }
    {
        let tx = tx.clone();
        ui.on_set_volume(move |v| { let _ = tx.send(Cmd::SetVol { v: v as f64 }); });
    }
    {
        let tx = tx.clone();
        ui.on_export_voice(move |id, name| {
            let _ = tx.send(Cmd::ExportVoice { id: id.to_string(), name: name.to_string() });
        });
    }
    {
        let tx = tx.clone();
        ui.on_edit_voice(move |id| { let _ = tx.send(Cmd::EditVoice { id: id.to_string() }); });
    }
    {
        let tx = tx.clone();
        ui.on_delete_voice(move |id| { let _ = tx.send(Cmd::DeleteVoice { id: id.to_string() }); });
    }
    {
        let tx = tx.clone();
        ui.on_import_voice(move || { let _ = tx.send(Cmd::ImportVoice); });
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
            let hp = pmap
                .get(&id)
                .and_then(|p| p.get("has_personality"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let (desc, lang, kind) = if let Some(p) = pmap.get(&id) {
                let vt = p.get("voice_type").and_then(|v| v.as_str()).unwrap_or("voice");
                let l = p.get("language").and_then(|v| v.as_str()).unwrap_or("en");
                // the profile's own description, falling back to a kind label
                let d = match p.get("description").and_then(|v| v.as_str()) {
                    Some(d) if !d.trim().is_empty() => d.to_string(),
                    _ if hp => "Has personality".to_string(),
                    _ => "Custom voice".to_string(),
                };
                (d, l.to_string(), vt.to_string())
            } else {
                (String::new(), "en".to_string(), "voice".to_string())
            };
            users.push(VoiceItem {
                id: id.into(),
                name: name.into(),
                desc: desc.into(),
                lang: lang.into(),
                kind: kind.into(),
                has_personality: hp,
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
        has_personality: false,
    });
    grid.extend(users);
    while grid.len() % 3 != 0 {
        grid.push(VoiceItem {
            id: "".into(),
            name: "".into(),
            desc: "".into(),
            lang: "".into(),
            kind: "empty".into(),
            has_personality: false,
        });
    }

    let default_selected = kokoro_ids.first().map(|s| s.to_string()).unwrap_or_default();
    GridData { grid, kokoro_names, kokoro_ids, default_selected }
}

/// Temp WAV path for in-modal recording (runtime dir → RAM, cleaned on logout).
fn cv_wav_path() -> String {
    std::env::var("XDG_RUNTIME_DIR")
        .map(|d| format!("{d}/syrinx-cv-record.wav"))
        .unwrap_or_else(|_| "/tmp/syrinx-cv-record.wav".into())
}

/// The default sink's `.monitor` source — a passive tap for system audio (the
/// same approach as Voicebox). Works on analog/HDMI; Bluetooth A2DP monitors are
/// silent, so those need a speaker/HDMI output while capturing.
async fn default_monitor() -> Option<String> {
    let out = tokio::process::Command::new("pactl")
        .arg("get-default-sink")
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let sink = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!sink.is_empty()).then(|| format!("{sink}.monitor"))
}

/// Spawn `parecord` to `wav`, optionally from a specific `device` (a sink's
/// `.monitor` for system audio). We use `parecord` (PulseAudio) rather than
/// `pw-record`: the latter's `--target` silently no-ops for monitors here, so it
/// only ever recorded the (dead) default mic.
async fn start_pw_record(wav: &str, device: Option<&str>) -> std::io::Result<tokio::process::Child> {
    let _ = std::fs::remove_file(wav);
    let mut cmd = tokio::process::Command::new("parecord");
    cmd.args(["--file-format=wav", "--rate=24000", "--channels=1", "--format=s16le"]);
    if let Some(d) = device {
        cmd.arg(format!("--device={d}"));
    }
    cmd.arg(wav)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
}

/// RMS level of a PCM16 mono WAV (0..1), for detecting silent captures.
fn wav_rms(path: &str) -> Option<f32> {
    let bytes = std::fs::read(path).ok()?;
    let data = bytes.windows(4).position(|w| w == b"data")? + 8; // past "data"+size
    let pcm = bytes.get(data..)?;
    let mut sumsq = 0f64;
    let mut count = 0u64;
    let mut i = 0;
    while i + 1 < pcm.len() {
        let s = i16::from_le_bytes([pcm[i], pcm[i + 1]]) as f64 / 32768.0;
        sumsq += s * s;
        count += 1;
        i += 2;
    }
    (count > 0).then(|| (sumsq / count as f64).sqrt() as f32)
}

/// Label for a finished recording, warning if it came out silent.
fn recorded_label(wav: &str) -> String {
    match wav_rms(wav) {
        Some(rms) if rms < 0.006 => "⚠ silent — check input / output device".into(),
        _ => "clip recorded ✓".into(),
    }
}

/// SIGINT `pw-record` so it finalizes the WAV header, then reap it.
async fn stop_pw_record(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        let _ = tokio::process::Command::new("kill")
            .args(["-INT", &pid.to_string()])
            .status()
            .await;
    }
    let _ = child.wait().await;
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

fn size_label(mb: i64) -> String {
    if mb >= 1024 {
        format!("{:.1} GB", mb as f64 / 1024.0)
    } else {
        format!("{mb} MB")
    }
}

/// Build the three category model lists from the engine's ListModels JSON.
fn build_models(json: &str) -> (Vec<ModelItem>, Vec<ModelItem>, Vec<ModelItem>) {
    let arr: Vec<serde_json::Value> = serde_json::from_str(json).unwrap_or_default();
    let (mut voice, mut stt, mut llm) = (Vec::new(), Vec::new(), Vec::new());
    for m in arr.iter() {
        let s = |k: &str| m.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let b = |k: &str| m.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        let mb = m.get("size_mb").and_then(|v| v.as_i64()).unwrap_or(0);
        let item = ModelItem {
            id: s("id").into(),
            display: s("display").into(),
            size_label: size_label(mb).into(),
            description: s("description").into(),
            downloaded: b("downloaded"),
            downloading: b("downloading"),
            active: b("active"),
            supported: b("supported"),
            warning: s("warning").into(),
            progress: 0.0,
        };
        match m.get("category").and_then(|v| v.as_str()).unwrap_or("") {
            "voice" => voice.push(item),
            "stt" => stt.push(item),
            "llm" => llm.push(item),
            _ => {}
        }
    }
    (voice, stt, llm)
}

/// One-line hardware summary for the Models header.
fn hardware_line(json: &str) -> String {
    let h: serde_json::Value = serde_json::from_str(json).unwrap_or_default();
    let cores = h.get("cores").and_then(|v| v.as_i64()).unwrap_or(0);
    let ram = h.get("ram_gb").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let gpu = h.get("gpu").and_then(|v| v.as_bool()).unwrap_or(false);
    let name = h.get("gpu_name").and_then(|v| v.as_str()).unwrap_or("");
    let gpu_part = if gpu {
        if name.is_empty() { "GPU".to_string() } else { name.to_string() }
    } else {
        "no GPU".to_string()
    };
    format!("{cores} cores · {ram:.1} GB RAM · {gpu_part}")
}

/// Re-fetch the catalog + hardware and push into the UI. Also rebuilds the
/// composer's engine dropdown and the modal's default-model options (usable =
/// downloaded + supported voice models); returns (model id, engine) pairs in
/// dropdown order plus the active voice engine name.
async fn refresh_models(
    ui: &slint::Weak<AppWindow>,
    proxy: &EngineProxy<'_>,
) -> (Vec<(String, String)>, String) {
    let models_json = proxy.list_models().await.unwrap_or_else(|_| "[]".into());
    let hw_json = proxy.hardware().await.unwrap_or_default();
    let (voice, stt, llm) = build_models(&models_json);
    let hwline = hardware_line(&hw_json);

    let arr: Vec<serde_json::Value> = serde_json::from_str(&models_json).unwrap_or_default();
    let mut models: Vec<(String, String)> = Vec::new(); // (model id, engine)
    let mut eng_names: Vec<SharedString> = Vec::new();
    let mut active_idx: i32 = 0;
    let mut active_engine = String::from("kokoro");
    for m in &arr {
        let s = |k: &str| m.get(k).and_then(|v| v.as_str()).unwrap_or("");
        let b = |k: &str| m.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
        if s("category") == "voice" && b("downloaded") && b("supported") {
            if b("active") {
                active_idx = models.len() as i32;
                active_engine = s("engine").to_string();
            }
            models.push((s("id").to_string(), s("engine").to_string()));
            eng_names.push(s("display").into());
        }
    }
    // modal picker: "Follow active model" + the same list
    let cv_options: Vec<SharedString> = std::iter::once(SharedString::from("Follow active model"))
        .chain(eng_names.iter().cloned())
        .collect();

    ui.upgrade_in_event_loop(move |ui| {
        ui.set_voice_models(ModelRc::from(Rc::new(VecModel::from(voice))));
        ui.set_stt_models(ModelRc::from(Rc::new(VecModel::from(stt))));
        ui.set_llm_models(ModelRc::from(Rc::new(VecModel::from(llm))));
        ui.set_hardware_line(hwline.into());
        ui.set_composer_engines(ModelRc::from(Rc::new(VecModel::from(eng_names))));
        ui.set_composer_engine_index(active_idx);
        ui.set_cv_model_options(ModelRc::from(Rc::new(VecModel::from(cv_options))));
    })
    .ok();
    (models, active_engine)
}

/// Rebuild the voice-card grid from the engine (after create/edit/delete/import).
async fn refresh_grid(ui: &slint::Weak<AppWindow>, proxy: &EngineProxy<'_>) {
    let raw = proxy.list_voices().await.unwrap_or_default();
    let pj = proxy.list_profiles().await.unwrap_or_else(|_| "[]".into());
    let GridData { grid, .. } = build_grid(raw, &pj);
    ui.upgrade_in_event_loop(move |ui| {
        ui.set_voices(ModelRc::from(Rc::new(VecModel::from(grid))));
    })
    .ok();
}

/// Language of a Kokoro preset from its id convention: `builtin:kokoro:af_…`
/// — first letter = language (a/b American/British English, e Spanish, …).
fn kokoro_lang_code(voice_id: &str) -> &'static str {
    match voice_id.rsplit(':').next().and_then(|v| v.chars().next()) {
        Some('a') | Some('b') => "en",
        Some('e') => "es",
        Some('f') => "fr",
        Some('h') => "hi",
        Some('i') => "it",
        Some('j') => "ja",
        Some('p') => "pt",
        Some('z') => "zh",
        _ => "en",
    }
}

/// Kokoro id prefixes for a language code (inverse of kokoro_lang_code).
fn kokoro_prefixes(code: &str) -> &'static [char] {
    match code {
        "es" => &['e'],
        "fr" => &['f'],
        "hi" => &['h'],
        "it" => &['i'],
        "ja" => &['j'],
        "pt" => &['p'],
        "zh" => &['z'],
        _ => &['a', 'b'], // en
    }
}

/// Voicebox's per-engine language subsets (label, code), in Voicebox order.
fn langs_for_engine(engine: &str) -> Vec<(&'static str, &'static str)> {
    const ALL: &[(&str, &str)] = &[
        ("Arabic", "ar"), ("Danish", "da"), ("German", "de"), ("Greek", "el"),
        ("English", "en"), ("Spanish", "es"), ("Finnish", "fi"), ("French", "fr"),
        ("Hebrew", "he"), ("Hindi", "hi"), ("Italian", "it"), ("Japanese", "ja"),
        ("Korean", "ko"), ("Malay", "ms"), ("Dutch", "nl"), ("Norwegian", "no"),
        ("Polish", "pl"), ("Portuguese", "pt"), ("Russian", "ru"), ("Swedish", "sv"),
        ("Swahili", "sw"), ("Turkish", "tr"), ("Chinese", "zh"),
    ];
    let codes: &[&str] = match engine {
        "qwen" | "qwen_custom_voice" => &["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        "luxtts" | "chatterbox_turbo" => &["en"],
        "chatterbox" => return ALL.to_vec(),
        "tada" => &["en", "ar", "zh", "de", "es", "fr", "it", "ja", "pl", "pt"],
        _ => &["en", "es", "fr", "hi", "it", "pt", "ja", "zh"], // kokoro
    };
    codes
        .iter()
        .filter_map(|c| ALL.iter().find(|(_, code)| code == c).copied())
        .collect()
}

/// Push the language dropdown for `engine`, preselecting `current_code`;
/// returns the codes in dropdown order (for index → code lookups).
fn update_composer_langs(
    ui: &slint::Weak<AppWindow>,
    engine: &str,
    current_code: &str,
) -> Vec<&'static str> {
    let pairs = langs_for_engine(engine);
    let labels: Vec<SharedString> = pairs.iter().map(|(l, _)| SharedString::from(*l)).collect();
    let codes: Vec<&'static str> = pairs.iter().map(|(_, c)| *c).collect();
    let idx = codes.iter().position(|c| *c == current_code).unwrap_or(0) as i32;
    ui.upgrade_in_event_loop(move |ui| {
        ui.set_composer_langs(ModelRc::from(Rc::new(VecModel::from(labels))));
        ui.set_composer_lang_index(idx);
    })
    .ok();
    codes
}

/// Update a single model row's download progress in place (no refetch).
fn set_model_progress(ui: &slint::Weak<AppWindow>, id: String, pct: f32, downloading: bool) {
    ui.upgrade_in_event_loop(move |ui| {
        for model in [ui.get_voice_models(), ui.get_stt_models(), ui.get_llm_models()] {
            for i in 0..model.row_count() {
                if let Some(mut it) = model.row_data(i) {
                    if it.id.as_str() == id {
                        it.progress = pct;
                        it.downloading = downloading;
                        model.set_row_data(i, it);
                        return;
                    }
                }
            }
        }
    })
    .ok();
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
    // full preset list kept app-side so the language dropdown can filter it
    let mut kokoro_all: Vec<(String, String)> = kokoro_ids
        .iter()
        .zip(kokoro_names.iter())
        .map(|(i, n)| (i.to_string(), n.to_string()))
        .collect();
    let hist_items = build_history(&proxy.list_history().await.unwrap_or_else(|_| "[]".into()));
    {
        ui.upgrade_in_event_loop(move |ui| {
            ui.set_backend(backend.into());
            ui.set_kokoro_names(ModelRc::from(Rc::new(VecModel::from(kokoro_names))));
            ui.set_kokoro_ids(ModelRc::from(Rc::new(VecModel::from(kokoro_ids))));
            ui.set_voices(ModelRc::from(Rc::new(VecModel::from(grid))));
            if ui.get_selected_voice().is_empty() {
                ui.set_selected_voice(default_selected.clone().into());
                ui.set_selected_voice_name(voice_name(&ui, &default_selected).into());
            }
            set_history_model(&ui, hist_items);
        })
        .ok();
    }

    let mut levels = proxy.receive_audio_level().await?;
    let mut gprog = proxy.receive_generation_progress().await?;
    let mut ended = proxy.receive_speak_ended().await?;
    let mut pinfo = proxy.receive_playback_info().await?;
    let mut pprog = proxy.receive_playback_progress().await?;
    let mut llm_res = proxy.receive_llm_result().await?;
    let mut mprog = proxy.receive_model_progress().await?;
    let mut pending_llm: u32 = 0;
    let (mut voice_models, mut active_engine) = refresh_models(&ui, &proxy).await;
    let mut lang_codes = update_composer_langs(&ui, "kokoro", "en");
    // effects dropdown: "No effects" + the engine's built-in presets
    let effect_ids: Vec<String> = {
        let fx_json = proxy.list_effect_presets().await.unwrap_or_else(|_| "[]".into());
        let fx: Vec<serde_json::Value> = serde_json::from_str(&fx_json).unwrap_or_default();
        let mut labels: Vec<SharedString> = vec!["No effects".into()];
        let mut ids = vec![String::new()];
        for p in &fx {
            labels.push(p.get("name").and_then(|v| v.as_str()).unwrap_or("").into());
            ids.push(p.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string());
        }
        ui.upgrade_in_event_loop(move |ui| {
            ui.set_composer_effects(ModelRc::from(Rc::new(VecModel::from(labels))));
        }).ok();
        ids
    };
    let mut current_gen: u32 = 0;
    let mut player_dur: f64 = 0.0;
    let mut current_play_gen: u32 = 0;
    let mut playing = false;
    // Voicebox-parity player/composer state
    let mut loop_on = false;
    let mut current_clip = String::new();
    let mut last_pct: f64 = 0.0;
    let mut speak_after_llm: Option<String> = None;
    let mut cv_edit: Option<String> = None;
    let mut cv_edit_transcript = String::new();
    // create-voice modal state
    let mut cv_rec: Option<tokio::process::Child> = None;
    let cv_wav = cv_wav_path();
    let mut cv_sample: Option<String> = None;
    let mut rec_interval = tokio::time::interval(std::time::Duration::from_secs(1));
    let mut rec_elapsed: u32 = 0;
    const REC_MAX: u32 = 30;

    loop {
        tokio::select! {
            Some(sig) = levels.next() => {
                if let Ok(a) = sig.args() {
                    let rms = a.rms as f32;
                    ui.upgrade_in_event_loop(move |ui| ui.set_level(rms)).ok();
                }
            }
            Some(sig) = gprog.next() => {
                if let Ok(a) = sig.args() {
                    if let Some(msg) = a.state.strip_prefix("error:") {
                        let msg = msg.trim().to_string();
                        ui.upgrade_in_event_loop(move |ui| ui.set_gen_error(msg.into())).ok();
                    }
                }
            }
            Some(sig) = pinfo.next() => {
                if let Ok(a) = sig.args() {
                    current_play_gen = a.gen_id;
                    playing = true;
                    player_dur = a.duration;
                    last_pct = 0.0;
                    current_clip = a.clip_id.to_string();
                    let bars: Vec<f32> = serde_json::from_str(&a.bars).unwrap_or_default();
                    let title = a.title;
                    let clip_id = a.clip_id;
                    let time = format!("0:00 / {}", fmt_dur(a.duration));
                    ui.upgrade_in_event_loop(move |ui| {
                        ui.set_player_active_visible(true);
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
            Some(sig) = mprog.next() => {
                if let Ok(a) = sig.args() {
                    let id = a.model_id.to_string();
                    match a.status.as_str() {
                        "downloading" => set_model_progress(&ui, id, a.pct as f32, true),
                        _ => { // done / error
                            let r = refresh_models(&ui, &proxy).await;
                            voice_models = r.0;
                            active_engine = r.1;
                        }
                    }
                }
            }
            Some(sig) = llm_res.next() => {
                if let Ok(a) = sig.args() {
                    if a.req_id == pending_llm && pending_llm != 0 {
                        pending_llm = 0;
                        let text = a.text;
                        let ui_text = text.clone();
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_llm_busy(false);
                            if !ui_text.trim().is_empty() {
                                ui.set_text(ui_text.into());
                            }
                        }).ok();
                        // persona flow: the rewrite came back — now synthesize it
                        if let Some(voice) = speak_after_llm.take() {
                            if text.trim().is_empty() {
                                ui.upgrade_in_event_loop(|ui| {
                                    ui.set_generating(false);
                                    ui.set_synthesizing(false);
                                }).ok();
                            } else {
                                match proxy.speak(&text, &voice).await {
                                    Ok(id) => current_gen = id,
                                    Err(e) => tracing::error!("persona speak failed: {e}"),
                                }
                            }
                        }
                    }
                }
            }
            Some(sig) = pprog.next() => {
                if let Ok(a) = sig.args() {
                    if a.gen_id == current_play_gen {
                        last_pct = a.pct;
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
                // Loop: re-trigger only when the clip ran to its natural end
                // (a Stop/Cancel arrives with the progress short of 1.0).
                let looping = is_current && loop_on && last_pct > 0.97 && !current_clip.is_empty();
                // Refresh history only on success — never wipe the list on a failed call.
                let refreshed = proxy.list_history().await.ok().map(|j| build_history(&j));
                ui.upgrade_in_event_loop(move |ui| {
                    ui.set_generating(false);
                    ui.set_synthesizing(false);
                    ui.set_level(0.0);
                    if is_current && !looping {
                        ui.set_player_playing(false);
                        ui.set_player_paused(false);
                        ui.set_play_pct(1.0);
                    }
                    if let Some(items) = refreshed {
                        set_history_model(&ui, items);
                    }
                }).ok();
                if looping {
                    last_pct = 0.0;
                    if let Ok(gid) = proxy.play_history(&current_clip).await {
                        current_gen = gid;
                    }
                }
            }
            _ = rec_interval.tick(), if cv_rec.is_some() => {
                rec_elapsed += 1;
                if rec_elapsed >= REC_MAX {
                    // hit the cap — auto-stop and keep the clip
                    if let Some(mut child) = cv_rec.take() {
                        stop_pw_record(&mut child).await;
                        cv_sample = Some(cv_wav.clone());
                        let label = recorded_label(&cv_wav);
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_cv_recording(false);
                            ui.set_cv_sample_label(label.into());
                        }).ok();
                    }
                } else {
                    let e = rec_elapsed;
                    ui.upgrade_in_event_loop(move |ui| {
                        ui.set_cv_sample_label(format!("● recording… {e}s / 30s").into());
                    }).ok();
                }
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
                Some(Cmd::ExportAudio { id }) => {
                    let src = proxy.history_audio_path(&id).await.unwrap_or_default();
                    if src.is_empty() {
                        tracing::error!("export audio: no source for {id}");
                    } else if let Some(handle) = rfd::AsyncFileDialog::new()
                        .set_file_name("syrinx-clip.wav")
                        .add_filter("WAV audio", &["wav"])
                        .save_file()
                        .await
                    {
                        let dest = handle.path().to_path_buf();
                        match std::fs::copy(&src, &dest) {
                            Ok(_) => tracing::info!("exported audio -> {}", dest.display()),
                            Err(e) => tracing::error!("export audio copy failed: {e}"),
                        }
                    }
                }
                Some(Cmd::ExportPackage { id }) => {
                    if let Some(handle) = rfd::AsyncFileDialog::new()
                        .set_file_name("syrinx-clip.zip")
                        .add_filter("Zip package", &["zip"])
                        .save_file()
                        .await
                    {
                        let dest = handle.path().to_string_lossy().to_string();
                        match proxy.export_package(&id, &dest).await {
                            Ok(_) => tracing::info!("exported package -> {dest}"),
                            Err(e) => tracing::error!("export package failed: {e}"),
                        }
                    }
                }
                Some(Cmd::CvStartRecord { system }) => {
                    // System audio = passive tap of the default sink's monitor.
                    let target = if system { default_monitor().await } else { None };
                    match start_pw_record(&cv_wav, target.as_deref()).await {
                        Ok(child) => {
                            cv_rec = Some(child);
                            rec_elapsed = 0;
                            rec_interval.reset();  // first tick a full second out
                            ui.upgrade_in_event_loop(|ui| {
                                ui.set_cv_recording(true);
                                ui.set_cv_sample_label("● recording… 0s / 30s".into());
                            }).ok();
                        }
                        Err(e) => tracing::error!("pw-record failed: {e}"),
                    }
                }
                Some(Cmd::CvStopRecord) => {
                    if let Some(mut child) = cv_rec.take() {
                        stop_pw_record(&mut child).await;
                        cv_sample = Some(cv_wav.clone());
                        let label = recorded_label(&cv_wav);
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_cv_recording(false);
                            ui.set_cv_sample_label(label.into());
                        }).ok();
                    }
                }
                Some(Cmd::CvPickFile) => {
                    if let Some(handle) = rfd::AsyncFileDialog::new()
                        .add_filter("Audio", &["wav", "flac", "ogg", "mp3", "m4a", "opus"])
                        .pick_file()
                        .await
                    {
                        cv_sample = Some(handle.path().to_string_lossy().to_string());
                        let label = handle.file_name();
                        ui.upgrade_in_event_loop(move |ui| ui.set_cv_sample_label(label.into())).ok();
                    }
                }
                Some(Cmd::CvTranscribe) => {
                    if let Some(path) = cv_sample.clone() {
                        ui.upgrade_in_event_loop(|ui| ui.set_cv_transcribing(true)).ok();
                        let text = proxy.transcribe(&path).await.unwrap_or_default();
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_cv_transcribing(false);
                            ui.set_cv_transcript(text.into());
                        }).ok();
                    }
                }
                Some(Cmd::CvCreate { name, desc, personality, language, transcript, model_index }) => {
                    // "Follow active model" (index 0) stores "", else the engine
                    // of the picked voice model — the same field the composer
                    // dropdown pins.
                    let default_engine = if model_index == 0 {
                        String::new()
                    } else {
                        voice_models
                            .get(model_index - 1)
                            .map(|(_, e)| e.clone())
                            .unwrap_or_default()
                    };
                    if let Some(pid) = cv_edit.take() {
                        // edit mode: patch metadata + optionally replace audio
                        let patch = serde_json::json!({
                            "name": name, "description": desc,
                            "personality": personality, "language": language,
                            "default_engine": default_engine,
                        }).to_string();
                        match proxy.update_profile(&pid, &patch).await {
                            Ok(_) => {
                                if let Some(sample) = cv_sample.take() {
                                    // a new capture replaces the existing samples
                                    if let Ok(pj) = proxy.get_profile(&pid).await {
                                        let p: serde_json::Value =
                                            serde_json::from_str(&pj).unwrap_or_default();
                                        for s in p.get("samples").and_then(|v| v.as_array()).into_iter().flatten() {
                                            if let Some(sid) = s.get("id").and_then(|v| v.as_str()) {
                                                proxy.delete_sample(sid).await.ok();
                                            }
                                        }
                                    }
                                    if let Err(e) = proxy.add_sample(&pid, &sample, &transcript).await {
                                        tracing::error!("replace sample failed: {e}");
                                    }
                                } else if transcript.trim() != cv_edit_transcript.trim()
                                    && !transcript.trim().is_empty()
                                {
                                    // transcript-only correction on the existing sample
                                    if let Ok(pj) = proxy.get_profile(&pid).await {
                                        let p: serde_json::Value =
                                            serde_json::from_str(&pj).unwrap_or_default();
                                        if let Some(sid) = p
                                            .get("samples")
                                            .and_then(|v| v.as_array())
                                            .and_then(|a| a.first())
                                            .and_then(|s| s.get("id"))
                                            .and_then(|v| v.as_str())
                                        {
                                            proxy.update_sample_text(&pid, sid, &transcript).await.ok();
                                        }
                                    }
                                }
                                refresh_grid(&ui, &proxy).await;
                                let pid2 = pid.clone();
                                let name2 = name.clone();
                                let hp = !personality.trim().is_empty();
                                ui.upgrade_in_event_loop(move |ui| {
                                    ui.set_cv_open(false);
                                    ui.set_cv_edit_id("".into());
                                    ui.set_cv_name("".into());
                                    ui.set_cv_desc("".into());
                                    ui.set_cv_personality("".into());
                                    ui.set_cv_transcript("".into());
                                    ui.set_cv_sample_label("".into());
                                    ui.set_cv_model_index(0);
                                    if ui.get_selected_voice().as_str() == pid2 {
                                        ui.set_selected_voice_name(name2.into());
                                        ui.set_selected_has_personality(hp);
                                    }
                                }).ok();
                            }
                            Err(e) => tracing::error!("edit voice failed: {e}"),
                        }
                    } else if name.trim().is_empty() || cv_sample.is_none() {
                        tracing::warn!("create voice: needs a name and a reference sample");
                    } else {
                        ui.upgrade_in_event_loop(|ui| ui.set_cv_creating(true)).ok();
                        let sample = cv_sample.clone().unwrap();
                        let spec = serde_json::json!({
                            "name": name, "voice_type": "cloned", "language": language,
                            "description": desc, "personality": personality,
                            "default_engine": default_engine,
                        }).to_string();
                        let outcome = async {
                            let pid = proxy.create_profile(&spec).await?;
                            proxy.add_sample(&pid, &sample, &transcript).await?;
                            zbus::Result::Ok(())
                        }.await;
                        match outcome {
                            Ok(_) => {
                                let raw = proxy.list_voices().await.unwrap_or_default();
                                let pj = proxy.list_profiles().await.unwrap_or_else(|_| "[]".into());
                                let GridData { grid, kokoro_names, kokoro_ids, .. } = build_grid(raw, &pj);
                                kokoro_all = kokoro_ids
                                    .iter()
                                    .zip(kokoro_names.iter())
                                    .map(|(i, n)| (i.to_string(), n.to_string()))
                                    .collect();
                                cv_sample = None;
                                ui.upgrade_in_event_loop(move |ui| {
                                    ui.set_cv_creating(false);
                                    ui.set_cv_open(false);
                                    ui.set_cv_name("".into());
                                    ui.set_cv_desc("".into());
                                    ui.set_cv_personality("".into());
                                    ui.set_cv_transcript("".into());
                                    ui.set_cv_sample_label("".into());
                                    ui.set_kokoro_names(ModelRc::from(Rc::new(VecModel::from(kokoro_names))));
                                    ui.set_kokoro_ids(ModelRc::from(Rc::new(VecModel::from(kokoro_ids))));
                                    ui.set_voices(ModelRc::from(Rc::new(VecModel::from(grid))));
                                }).ok();
                            }
                            Err(e) => {
                                tracing::error!("create voice failed: {e}");
                                ui.upgrade_in_event_loop(|ui| ui.set_cv_creating(false)).ok();
                            }
                        }
                    }
                }
                Some(Cmd::DownloadModel { id }) => {
                    match proxy.download_model(&id).await {
                        Ok(true) => set_model_progress(&ui, id, 0.0, true),
                        _ => tracing::error!("download_model failed: {id}"),
                    }
                }
                Some(Cmd::DeleteModel { id }) => {
                    proxy.delete_model(&id).await.ok();
                    let r = refresh_models(&ui, &proxy).await;
                    voice_models = r.0;
                    active_engine = r.1;
                }
                Some(Cmd::ActivateModel { id }) => {
                    proxy.set_active_model(&id).await.ok();
                    let r = refresh_models(&ui, &proxy).await;
                    voice_models = r.0;
                    active_engine = r.1;
                }
                Some(Cmd::Compose { voice_id, prompt }) => {
                    match proxy.compose_profile(&voice_id, &prompt).await {
                        Ok(rid) if rid != 0 => pending_llm = rid,
                        _ => { ui.upgrade_in_event_loop(|ui| ui.set_llm_busy(false)).ok(); }
                    }
                }
                Some(Cmd::Rewrite { voice_id, text }) => {
                    match proxy.rewrite_profile(&voice_id, &text).await {
                        Ok(rid) if rid != 0 => pending_llm = rid,
                        _ => { ui.upgrade_in_event_loop(|ui| ui.set_llm_busy(false)).ok(); }
                    }
                }
                Some(Cmd::CvCancel) => {
                    if let Some(mut child) = cv_rec.take() {
                        stop_pw_record(&mut child).await;
                    }
                    cv_sample = None;
                    cv_edit = None;
                    cv_edit_transcript.clear();
                    ui.upgrade_in_event_loop(|ui| {
                        ui.set_cv_recording(false);
                        ui.set_cv_sample_label("".into());
                        ui.set_cv_name("".into());
                        ui.set_cv_desc("".into());
                        ui.set_cv_personality("".into());
                        ui.set_cv_transcript("".into());
                        ui.set_cv_edit_id("".into());
                        ui.set_cv_model_index(0);
                    }).ok();
                }
                Some(Cmd::GenerateInCharacter { text, voice }) => {
                    match proxy.rewrite_profile(&voice, &text).await {
                        Ok(rid) if rid != 0 => {
                            pending_llm = rid;
                            speak_after_llm = Some(voice);
                        }
                        _ => {
                            // no personality / LLM unavailable — speak the raw text
                            ui.upgrade_in_event_loop(|ui| ui.set_llm_busy(false)).ok();
                            match proxy.speak(&text, &voice).await {
                                Ok(id) => current_gen = id,
                                Err(e) => tracing::error!("speak failed: {e}"),
                            }
                        }
                    }
                }
                Some(Cmd::SelectVoice { id }) => {
                    let (engine, code) = if id.starts_with("builtin:") {
                        ("kokoro".to_string(), kokoro_lang_code(&id).to_string())
                    } else if let Ok(pj) = proxy.get_profile(&id).await {
                        let p: serde_json::Value = serde_json::from_str(&pj).unwrap_or_default();
                        let de = p.get("default_engine").and_then(|v| v.as_str()).unwrap_or("");
                        let engine = if de.is_empty() { active_engine.clone() } else { de.to_string() };
                        let code = p.get("language").and_then(|v| v.as_str()).unwrap_or("en").to_string();
                        (engine, code)
                    } else {
                        ("kokoro".to_string(), "en".to_string())
                    };
                    lang_codes = update_composer_langs(&ui, &engine, &code);
                    // the composer's model dropdown mirrors this voice's engine
                    // (profile pin, else the active model)
                    let eidx = voice_models
                        .iter()
                        .position(|(_, e)| *e == engine)
                        .unwrap_or(0) as i32;
                    ui.upgrade_in_event_loop(move |ui| ui.set_composer_engine_index(eidx)).ok();
                }
                Some(Cmd::PickLanguage { voice, index }) => {
                    if let Some(code) = lang_codes.get(index) {
                        if voice.starts_with("builtin:kokoro:") {
                            // filter the Kokoro Defaults dropdown to this language
                            let prefixes = kokoro_prefixes(code);
                            let mut filtered: Vec<(String, String)> = kokoro_all
                                .iter()
                                .filter(|(id, _)| {
                                    id.rsplit(':')
                                        .next()
                                        .and_then(|v| v.chars().next())
                                        .map(|c| prefixes.contains(&c))
                                        .unwrap_or(false)
                                })
                                .cloned()
                                .collect();
                            if filtered.is_empty() {
                                filtered = kokoro_all.clone();
                            }
                            let sel_pos = filtered.iter().position(|(id, _)| *id == voice);
                            let idx = sel_pos.unwrap_or(0) as i32;
                            let need_switch = sel_pos.is_none();
                            let (nid, nname) = filtered[idx as usize].clone();
                            let names: Vec<SharedString> =
                                filtered.iter().map(|(_, n)| n.as_str().into()).collect();
                            let ids: Vec<SharedString> =
                                filtered.iter().map(|(i, _)| i.as_str().into()).collect();
                            ui.upgrade_in_event_loop(move |ui| {
                                ui.set_kokoro_names(ModelRc::from(Rc::new(VecModel::from(names))));
                                ui.set_kokoro_ids(ModelRc::from(Rc::new(VecModel::from(ids))));
                                ui.set_kokoro_index(idx);
                                if need_switch {
                                    // old selection doesn't speak this language —
                                    // jump to the first preset that does
                                    ui.set_selected_voice(nid.as_str().into());
                                    ui.set_selected_voice_name(nname.as_str().into());
                                    ui.set_selected_has_personality(false);
                                    ui.set_kokoro_active(true);
                                }
                            })
                            .ok();
                        } else if !voice.is_empty() {
                            // persisted per cloned profile
                            let patch = serde_json::json!({"language": code}).to_string();
                            proxy.update_profile(&voice, &patch).await.ok();
                            refresh_grid(&ui, &proxy).await;
                        }
                    }
                }
                Some(Cmd::PickEngine { voice, index }) => {
                    if let Some((mid, meng)) = voice_models.get(index).cloned() {
                        if !voice.is_empty() && !voice.starts_with("builtin:") {
                            // cloned profile selected: the dropdown pins THIS
                            // voice's engine (same field as the edit modal)
                            let patch = serde_json::json!({"default_engine": meng}).to_string();
                            proxy.update_profile(&voice, &patch).await.ok();
                            if let Ok(pj) = proxy.get_profile(&voice).await {
                                let p: serde_json::Value = serde_json::from_str(&pj).unwrap_or_default();
                                let code = p.get("language").and_then(|v| v.as_str()).unwrap_or("en").to_string();
                                lang_codes = update_composer_langs(&ui, &meng, &code);
                            }
                        } else {
                            // preset selected: switch the global active voice model
                            proxy.set_active_model(&mid).await.ok();
                            let r = refresh_models(&ui, &proxy).await;
                            voice_models = r.0;
                            active_engine = r.1;
                        }
                    }
                }
                Some(Cmd::ToggleLoop { on }) => { loop_on = on; }
                Some(Cmd::SetVol { v }) => { proxy.set_volume(v).await.ok(); }
                Some(Cmd::PickEffect { index }) => {
                    if let Some(pid) = effect_ids.get(index) {
                        proxy.set_effect(pid).await.ok();
                    }
                }
                Some(Cmd::ApplyFx { hid, index }) => {
                    if let Some(pid) = effect_ids.get(index).filter(|p| !p.is_empty()) {
                        match proxy.apply_history_effects(&hid, pid).await {
                            Ok(new_id) if !new_id.is_empty() => {
                                if let Ok(j) = proxy.list_history().await {
                                    let items = build_history(&j);
                                    ui.upgrade_in_event_loop(move |ui| set_history_model(&ui, items)).ok();
                                }
                            }
                            Ok(_) => tracing::error!("apply effects: engine returned no clip"),
                            Err(e) => tracing::error!("apply effects failed: {e}"),
                        }
                    }
                }
                Some(Cmd::ExportVoice { id, name }) => {
                    let safe: String = name
                        .to_lowercase()
                        .chars()
                        .map(|c| if c.is_alphanumeric() { c } else { '-' })
                        .collect();
                    if let Some(handle) = rfd::AsyncFileDialog::new()
                        .set_file_name(format!("{}.syrinx-voice.zip", safe.trim_matches('-')))
                        .add_filter("Syrinx voice package", &["zip"])
                        .save_file()
                        .await
                    {
                        let dest = handle.path().to_string_lossy().to_string();
                        match proxy.export_profile(&id, &dest).await {
                            Ok(_) => tracing::info!("exported voice -> {dest}"),
                            Err(e) => tracing::error!("export voice failed: {e}"),
                        }
                    }
                }
                Some(Cmd::EditVoice { id }) => {
                    if let Ok(pj) = proxy.get_profile(&id).await {
                        let p: serde_json::Value = serde_json::from_str(&pj).unwrap_or_default();
                        let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let (name, desc, pers) = (s("name"), s("description"), s("personality"));
                        let lang = {
                            let l = s("language");
                            if l.is_empty() { "en".to_string() } else { l }
                        };
                        let lang_idx = ["en", "ja", "zh", "de", "es", "fr", "it", "pt"]
                            .iter()
                            .position(|c| *c == lang)
                            .unwrap_or(0) as i32;
                        // current sample transcript (edit shows/corrects it)
                        let transcript = p
                            .get("samples")
                            .and_then(|v| v.as_array())
                            .and_then(|a| a.first())
                            .and_then(|smp| smp.get("reference_text"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        cv_edit_transcript = transcript.clone();
                        // pinned engine → its dropdown slot; "" → Follow active
                        let de = s("default_engine");
                        let model_idx = if de.is_empty() {
                            0
                        } else {
                            voice_models
                                .iter()
                                .position(|(_, e)| *e == de)
                                .map(|i| i as i32 + 1)
                                .unwrap_or(0)
                        };
                        cv_edit = Some(id.clone());
                        cv_sample = None;
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_cv_name(name.into());
                            ui.set_cv_desc(desc.into());
                            ui.set_cv_personality(pers.into());
                            ui.set_cv_language(lang.into());
                            ui.set_cv_lang_index(lang_idx);
                            ui.set_cv_transcript(transcript.into());
                            ui.set_cv_model_index(model_idx);
                            ui.set_cv_sample_label("".into());
                            ui.set_cv_edit_id(id.into());
                            ui.set_cv_open(true);
                        }).ok();
                    }
                }
                Some(Cmd::DeleteVoice { id }) => {
                    match proxy.delete_profile(&id).await {
                        Ok(_) => {
                            refresh_grid(&ui, &proxy).await;
                            ui.upgrade_in_event_loop(move |ui| {
                                if ui.get_selected_voice().as_str() == id {
                                    // fall back to the first bundled preset
                                    if let Some(first) = ui.get_kokoro_ids().row_data(0) {
                                        ui.set_selected_voice(first.clone());
                                        ui.set_kokoro_active(true);
                                        ui.set_selected_has_personality(false);
                                        ui.set_selected_voice_name(voice_name(&ui, first.as_str()).into());
                                    }
                                }
                            }).ok();
                        }
                        Err(e) => tracing::error!("delete voice failed: {e}"),
                    }
                }
                Some(Cmd::ImportVoice) => {
                    if let Some(handle) = rfd::AsyncFileDialog::new()
                        .add_filter("Syrinx voice package", &["zip"])
                        .pick_file()
                        .await
                    {
                        let src = handle.path().to_string_lossy().to_string();
                        match proxy.import_profile(&src).await {
                            Ok(pid) => {
                                tracing::info!("imported voice {pid}");
                                refresh_grid(&ui, &proxy).await;
                            }
                            Err(e) => tracing::error!("import voice failed: {e}"),
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
