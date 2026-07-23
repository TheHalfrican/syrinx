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
    PickStyle { index: usize },
    ApplyFx { hid: String, index: usize },
    ExportVoice { id: String, name: String },
    EditVoice { id: String },
    DeleteVoice { id: String },
    ImportVoice,
    CvPickAvatar,
    CvStageAvatar { path: String, mode: String, sx: i32, sy: i32, sw: i32, sh: i32 },
    TrToggleRecord { system: bool },
    TrPickFile,
    TrRefine { text: String },
    TrSaveCapture { id: String, text: String },
    TrDeleteCapture { id: String },
    // voice changer (⇄ tab)
    VcLoad,
    VcToggleRecord { system: bool },
    VcPickFile,
    VcConvert { index: usize, label: String, transcript: String },
    VcSaveClip { name: String },
    VcDeleteClip { id: String },
    VcArmClip { id: String },
    VcAudition { id: String },
    // voices tab (profile table + inspector)
    VoicesLoad,
    VoicesSearch { q: String },
    VoicesInspect { id: String },
    PlaySample { id: String },
    // effects chain editor
    FxeShow,
    FxeLoad { index: usize },
    FxeNew,
    FxeAdd { index: usize },
    FxeRemove { index: usize },
    FxeToggle { index: usize },
    FxeMove { index: usize, dir: i32 },
    FxeExpand { index: usize },
    FxeParam { index: usize, norm: f32 },
    FxeSave { name: String, desc: String },
    FxeDelete,
    FxePreview { hid: String },
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
    {
        let tx = tx.clone();
        ui.on_voices_open(move || { let _ = tx.send(Cmd::VoicesLoad); });
    }
    {
        let tx = tx.clone();
        ui.on_voices_search(move |q| { let _ = tx.send(Cmd::VoicesSearch { q: q.to_string() }); });
    }
    {
        let tx = tx.clone();
        ui.on_vp_select(move |id| { let _ = tx.send(Cmd::VoicesInspect { id: id.to_string() }); });
    }
    {
        let tx = tx.clone();
        ui.on_vs_play(move |id| { let _ = tx.send(Cmd::PlaySample { id: id.to_string() }); });
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
        ui.on_pick_style(move |i| { let _ = tx.send(Cmd::PickStyle { index: i as usize }); });
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
        ui.on_cv_pick_avatar(move || { let _ = tx.send(Cmd::CvPickAvatar); });
    }
    // Transcription view.
    {
        let tx = tx.clone();
        ui.on_tr_toggle_record(move |mode| {
            let _ = tx.send(Cmd::TrToggleRecord { system: mode.as_str() == "system" });
        });
    }
    {
        let tx = tx.clone();
        ui.on_tr_pick_file(move || { let _ = tx.send(Cmd::TrPickFile); });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_tr_refine(move || {
            let ui = ui_weak.unwrap();
            let text = ui.get_tr_text().to_string();
            if !text.trim().is_empty() {
                ui.set_tr_busy(true);
                ui.set_tr_status("refining…".into());
                let _ = tx.send(Cmd::TrRefine { text });
            }
        });
    }
    // Voice changer (⇄ tab).
    {
        let tx = tx.clone();
        ui.on_vc_open(move || { let _ = tx.send(Cmd::VcLoad); });
    }
    {
        let tx = tx.clone();
        ui.on_vc_toggle_record(move |mode| {
            let _ = tx.send(Cmd::VcToggleRecord { system: mode.as_str() == "system" });
        });
    }
    {
        let tx = tx.clone();
        ui.on_vc_pick_file(move || { let _ = tx.send(Cmd::VcPickFile); });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_vc_convert(move |i| {
            let ui = ui_weak.unwrap();
            let label = ui.get_vc_result_name().to_string();
            let transcript = ui.get_vc_transcript().to_string();
            let _ = tx.send(Cmd::VcConvert { index: i.max(0) as usize, label, transcript });
        });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_vc_save_clip(move || {
            let ui = ui_weak.unwrap();
            let name = ui.get_vc_clip_name().to_string();
            let _ = tx.send(Cmd::VcSaveClip { name });
        });
    }
    {
        let tx = tx.clone();
        ui.on_vc_delete_clip(move |id| { let _ = tx.send(Cmd::VcDeleteClip { id: id.to_string() }); });
    }
    {
        let tx = tx.clone();
        ui.on_vc_arm_clip(move |id| { let _ = tx.send(Cmd::VcArmClip { id: id.to_string() }); });
    }
    {
        let tx = tx.clone();
        ui.on_vc_audition(move |id| { let _ = tx.send(Cmd::VcAudition { id: id.to_string() }); });
    }
    // Captures (persisted transcripts): save-or-update decided by tr-capture-id.
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_tr_save_capture(move || {
            let ui = ui_weak.unwrap();
            let text = ui.get_tr_text().to_string();
            if !text.trim().is_empty() {
                let id = ui.get_tr_capture_id().to_string();
                let _ = tx.send(Cmd::TrSaveCapture { id, text });
            }
        });
    }
    {
        let tx = tx.clone();
        ui.on_tr_delete_capture(move |id| {
            let _ = tx.send(Cmd::TrDeleteCapture { id: id.to_string() });
        });
    }
    // Effects chain editor.
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        let history = history.clone();
        ui.on_fxe_show(move || {
            let ui = ui_weak.unwrap();
            let can = !ui.get_player_id().is_empty()
                || history.iter().any(|h| !h.id.is_empty());
            ui.set_fxe_can_preview(can);
            let _ = tx.send(Cmd::FxeShow);
        });
    }
    {
        let tx = tx.clone();
        ui.on_fxe_load(move |i| { let _ = tx.send(Cmd::FxeLoad { index: i as usize }); });
    }
    {
        let tx = tx.clone();
        ui.on_fxe_new(move || { let _ = tx.send(Cmd::FxeNew); });
    }
    {
        let tx = tx.clone();
        ui.on_fxe_add_effect(move |i| { let _ = tx.send(Cmd::FxeAdd { index: i as usize }); });
    }
    {
        let tx = tx.clone();
        ui.on_fxe_remove(move |i| { let _ = tx.send(Cmd::FxeRemove { index: i as usize }); });
    }
    {
        let tx = tx.clone();
        ui.on_fxe_toggle(move |i| { let _ = tx.send(Cmd::FxeToggle { index: i as usize }); });
    }
    {
        let tx = tx.clone();
        ui.on_fxe_move(move |i, d| { let _ = tx.send(Cmd::FxeMove { index: i as usize, dir: d }); });
    }
    {
        let tx = tx.clone();
        ui.on_fxe_expand(move |i| { let _ = tx.send(Cmd::FxeExpand { index: i as usize }); });
    }
    {
        let tx = tx.clone();
        ui.on_fxe_param(move |i, v| { let _ = tx.send(Cmd::FxeParam { index: i as usize, norm: v }); });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_fxe_save(move || {
            let ui = ui_weak.unwrap();
            let _ = tx.send(Cmd::FxeSave {
                name: ui.get_fxe_name().to_string(),
                desc: ui.get_fxe_desc().to_string(),
            });
        });
    }
    {
        let tx = tx.clone();
        ui.on_fxe_delete(move || { let _ = tx.send(Cmd::FxeDelete); });
    }
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        let history = history.clone();
        ui.on_fxe_preview(move || {
            let ui = ui_weak.unwrap();
            // prefer the clip in the player; fall back to the newest history row
            let hid = if !ui.get_player_id().is_empty() {
                ui.get_player_id().to_string()
            } else {
                history.iter().find(|h| !h.id.is_empty()).map(|h| h.id.to_string()).unwrap_or_default()
            };
            if !hid.is_empty() {
                let _ = tx.send(Cmd::FxePreview { hid });
            }
        });
    }
    // Crop accepted: turn the dialog's zoom/pan into a square source-pixel rect.
    {
        let tx = tx.clone();
        let ui_weak = ui.as_weak();
        ui.on_crop_done(move |accepted| {
            if !accepted {
                return;
            }
            let ui = ui_weak.unwrap();
            let path = ui.get_crop_path().to_string();
            let sz = ui.get_crop_src().size();
            let (w, h) = (sz.width as f32, sz.height as f32);
            if path.is_empty() || w < 1.0 || h < 1.0 {
                return;
            }
            // mirror the dialog viewport's aspect: circle 220x220, panel 132x220.
            // The math runs in preview pixels (crop-src is downscaled), then the
            // rect is scaled back to ORIGINAL photo pixels for storage.
            let mode = ui.get_crop_mode().to_string();
            let (vw, vh): (f32, f32) = if mode == "panel" { (132.0, 220.0) } else { (220.0, 220.0) };
            let cw = (w.min(h * vw / vh) / ui.get_crop_zoom().max(1.0)).round();
            let ch = (cw * vh / vw).round();
            let sx = (ui.get_crop_cx() * w - cw / 2.0).clamp(0.0, (w - cw).max(0.0));
            let sy = (ui.get_crop_cy() * h - ch / 2.0).clamp(0.0, (h - ch).max(0.0));
            let (fw, fh) = (ui.get_crop_full_w() as f32, ui.get_crop_full_h() as f32);
            let scale = if fw > 0.0 && w > 0.0 { fw / w } else { 1.0 };
            let fsw = (cw * scale).round().min(fw).max(1.0);
            let fsh = (ch * scale).round().min(fh.max(1.0)).max(1.0);
            let fsx = (sx * scale).round().clamp(0.0, (fw - fsw).max(0.0));
            let fsy = (sy * scale).round().clamp(0.0, (fh - fsh).max(0.0));
            let _ = tx.send(Cmd::CvStageAvatar {
                path,
                mode,
                sx: fsx as i32,
                sy: fsy as i32,
                sw: fsw as i32,
                sh: fsh as i32,
            });
        });
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
/// Send-able grid entry — slint's `image` type isn't Send, so the worker builds
/// these and the UI thread converts them (loading avatar files) in
/// `to_voice_items`.
#[derive(Clone, Default)]
struct VoiceData {
    id: String,
    name: String,
    desc: String,
    lang: String,
    kind: String,
    has_personality: bool,
    avatar_path: String,
    avatar_mode: String,
    avatar_sx: i32,
    avatar_sy: i32,
    avatar_side: i32,
    avatar_sh: i32,
}

/// Baked avatar pixels: RGBA bytes + dimensions (Send-able, unlike slint::Image).
type RgbaBuf = (Vec<u8>, u32, u32);

/// Decode a photo, apply the crop rect, and downscale with a proper filter.
/// The GPU's plain bilinear sampling turns a 4K photo minified into a small
/// circle into visible pixelation — so we hand it a ≤400px thumbnail instead.
/// Cached by path + mtime + rect since grids rebake on every refresh.
fn bake_avatar_rgba(
    cache: &mut HashMap<String, RgbaBuf>,
    path: &str,
    sx: i32,
    sy: i32,
    sw: i32,
    sh: i32,
) -> Option<RgbaBuf> {
    if path.is_empty() || sw <= 0 {
        return None;
    }
    let sh = if sh > 0 { sh } else { sw };
    let mtime = std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let key = format!("{path}|{mtime}|{sx},{sy},{sw},{sh}");
    if let Some(b) = cache.get(&key) {
        return Some(b.clone());
    }
    let img = image::open(path).ok()?;
    let (w, h) = (img.width(), img.height());
    let cx = (sx.max(0) as u32).min(w.saturating_sub(1));
    let cy = (sy.max(0) as u32).min(h.saturating_sub(1));
    let cw = (sw as u32).min(w - cx).max(1);
    let ch = (sh as u32).min(h - cy).max(1);
    let thumb = img.crop_imm(cx, cy, cw, ch).thumbnail(400, 400);
    let rgba = thumb.to_rgba8();
    let buf = (rgba.as_raw().clone(), rgba.width(), rgba.height());
    cache.insert(key, buf.clone());
    Some(buf)
}

/// UI-thread half of avatar handling: turn baked RGBA into a slint Image.
fn rgba_to_image(buf: &RgbaBuf) -> slint::Image {
    let pb = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(&buf.0, buf.1, buf.2);
    slint::Image::from_rgba8(pb)
}

/// UI-thread conversion of pre-baked grid data into model rows.
fn to_voice_items(data: Vec<(VoiceData, Option<RgbaBuf>)>) -> Vec<VoiceItem> {
    data.into_iter()
        .map(|(d, baked)| {
            let (avatar, has) = match &baked {
                Some(b) => (rgba_to_image(b), true),
                None => (Default::default(), false),
            };
            VoiceItem {
                id: d.id.into(),
                name: d.name.into(),
                desc: d.desc.into(),
                lang: d.lang.into(),
                kind: d.kind.into(),
                has_personality: d.has_personality,
                avatar,
                avatar_mode: if d.avatar_mode.is_empty() { "circle".into() } else { d.avatar_mode.into() },
                has_avatar: has,
            }
        })
        .collect()
}

/// Worker-side pairing of grid entries with their baked avatar thumbnails.
fn bake_grid(
    cache: &mut HashMap<String, RgbaBuf>,
    grid: Vec<VoiceData>,
) -> Vec<(VoiceData, Option<RgbaBuf>)> {
    grid.into_iter()
        .map(|d| {
            let baked = bake_avatar_rgba(
                cache, &d.avatar_path, d.avatar_sx, d.avatar_sy, d.avatar_side, d.avatar_sh,
            );
            (d, baked)
        })
        .collect()
}

struct GridData {
    grid: Vec<VoiceData>,          // [Kokoro card, user voices…, spacer padding]
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
    let mut users: Vec<VoiceData> = Vec::new();

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
            let (desc, lang, kind, avatar_path, avatar_mode, asx, asy, aside, ash) = if let Some(p) = pmap.get(&id) {
                let vt = p.get("voice_type").and_then(|v| v.as_str()).unwrap_or("voice");
                let l = p.get("language").and_then(|v| v.as_str()).unwrap_or("en");
                // the profile's own description, falling back to a kind label
                let d = match p.get("description").and_then(|v| v.as_str()) {
                    Some(d) if !d.trim().is_empty() => d.to_string(),
                    _ if hp => "Has personality".to_string(),
                    _ => "Custom voice".to_string(),
                };
                let i = |k: &str| p.get(k).and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                (
                    d,
                    l.to_string(),
                    vt.to_string(),
                    p.get("avatar_path").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    p.get("avatar_mode").and_then(|v| v.as_str()).unwrap_or("circle").to_string(),
                    i("avatar_sx"),
                    i("avatar_sy"),
                    i("avatar_side"),
                    i("avatar_sh"),
                )
            } else {
                (String::new(), "en".to_string(), "voice".to_string(),
                 String::new(), "circle".to_string(), 0, 0, 0, 0)
            };
            users.push(VoiceData {
                id,
                name,
                desc,
                lang,
                kind,
                has_personality: hp,
                avatar_path,
                avatar_mode,
                avatar_sx: asx,
                avatar_sy: asy,
                avatar_side: aside,
                avatar_sh: ash,
            });
        }
    }

    // grid = Kokoro Defaults card + user cards, padded to a multiple of 3 with
    // invisible spacers so the 3-column GridLayout always has full first row.
    let mut grid: Vec<VoiceData> = Vec::with_capacity(users.len() + 3);
    grid.push(VoiceData {
        id: "__kokoro__".into(),
        name: "Kokoro Defaults".into(),
        kind: "model-defaults".into(),
        ..Default::default()
    });
    grid.extend(users);
    while grid.len() % 3 != 0 {
        grid.push(VoiceData {
            kind: "empty".into(),
            ..Default::default()
        });
    }

    let default_selected = kokoro_ids.first().map(|s| s.to_string()).unwrap_or_default();
    GridData { grid, kokoro_names, kokoro_ids, default_selected }
}

/// Delivery directions for the composer style dropdown. Labels must match
/// the dropdown model in main.slint; the instruct text is sent verbatim to
/// the engine (SetStyle) and honored by the qwen engines. Phrasing is
/// deliberately intense — subtle directions barely move the performance.
const STYLES: &[(&str, &str)] = &[
    ("No direction", ""),
    ("Angry", "Speak in an extremely angry, furious tone — seething, sharp, and aggressive, as if barely containing rage."),
    ("Sad", "Speak in a deeply sad, sorrowful tone — heavy, slow, and grief-stricken, as if on the verge of tears."),
    ("Happy", "Speak in an intensely happy, joyful tone — bright, warm, and beaming with delight."),
    ("Excited", "Speak with overwhelming excitement and energy — fast, breathless, and absolutely thrilled."),
    ("Fearful", "Speak in a terrified, trembling tone — shaky, urgent, and full of dread."),
    ("Whisper", "Speak in a hushed, intense whisper — quiet, breathy, close, and conspiratorial."),
    ("Serious", "Speak in a grave, deadly serious tone — measured, cold, and commanding."),
];

/// Short human message for a failed profile D-Bus call.
fn profile_err_msg(e: &zbus::Error) -> String {
    let s = e.to_string();
    if s.contains("UNIQUE constraint failed: profiles.name") {
        "A voice with that name already exists.".into()
    } else {
        s
    }
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
///
/// The wait is bounded: a recorder that shrugs off the SIGINT (freshly spawned
/// children inherit an ignored SIGINT until parecord installs its handler)
/// would otherwise park the whole worker loop in `wait()` — frozen timers,
/// dead buttons. After 2 s we SIGKILL, which cannot be ignored.
async fn stop_pw_record(child: &mut tokio::process::Child) {
    if let Some(pid) = child.id() {
        let _ = tokio::process::Command::new("kill")
            .args(["-INT", &pid.to_string()])
            .status()
            .await;
    }
    let grace = std::time::Duration::from_secs(2);
    if tokio::time::timeout(grace, child.wait()).await.is_err() {
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
}

/// Format seconds as m:ss (Voicebox-style meta).
fn fmt_dur(d: f64) -> String {
    let s = d.round().max(0.0) as i64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// Build the history model from the engine's ListHistory JSON (newest first).
/// Engines whose history rows are conversions, not TTS generations.
fn is_vc_engine(engine: &str) -> bool {
    matches!(engine, "chatterbox_vc" | "seed_vc" | "vevo_timbre" | "vevo2")
}

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
            // "⇄ VC" labels conversions in the shared list; set_history_model
            // also keys the vc-tab rail off this prefix
            let meta = if is_vc_engine(engine) {
                format!("⇄ VC · {} · {}", fmt_dur(dur), lang)
            } else if engine.is_empty() {
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
async fn refresh_grid(
    ui: &slint::Weak<AppWindow>,
    proxy: &EngineProxy<'_>,
    cache: &mut HashMap<String, RgbaBuf>,
) {
    let raw = proxy.list_voices().await.unwrap_or_default();
    let pj = proxy.list_profiles().await.unwrap_or_else(|_| "[]".into());
    let GridData { grid, .. } = build_grid(raw, &pj);
    let grid = bake_grid(cache, grid);
    ui.upgrade_in_event_loop(move |ui| {
        ui.set_voices(ModelRc::from(Rc::new(VecModel::from(to_voice_items(grid)))));
    })
    .ok();
}

/// One Voices-tab table row, thread-safe half (images bake on the UI thread).
#[derive(Clone)]
struct VpRowData {
    id: String,
    name: String,
    desc: String,
    lang: String,
    engine: String, // "follows" when unpinned
    samples: String,
    gens: String,
    baked: Option<RgbaBuf>,
}

/// Filter + convert to slint rows. UI-thread only (creates Images).
fn vp_to_rows(data: &[VpRowData], filter: &str) -> Vec<ProfileRow> {
    let q = filter.to_lowercase();
    data.iter()
        .filter(|d| {
            q.is_empty()
                || d.name.to_lowercase().contains(&q)
                || d.desc.to_lowercase().contains(&q)
                || d.lang.to_lowercase().contains(&q)
                || d.engine.to_lowercase().contains(&q)
        })
        .map(|d| ProfileRow {
            id: d.id.clone().into(),
            name: d.name.clone().into(),
            desc: d.desc.clone().into(),
            lang: d.lang.clone().into(),
            engine: d.engine.clone().into(),
            samples: d.samples.clone().into(),
            gens: d.gens.clone().into(),
            avatar: d.baked.as_ref().map(rgba_to_image).unwrap_or_default(),
            has_avatar: d.baked.is_some(),
        })
        .collect()
}

/// Fill the Voices-tab inspector from GetProfile (+ cached table row data).
async fn inspect_profile(
    ui: &slint::Weak<AppWindow>,
    proxy: &EngineProxy<'_>,
    voices_all: &[VpRowData],
    id: &str,
) {
    let Ok(pj) = proxy.get_profile(id).await else { return };
    if pj.is_empty() {
        return; // deleted since selection
    }
    let p: serde_json::Value = serde_json::from_str(&pj).unwrap_or_default();
    let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let samples: Vec<(String, String)> = p
        .get("samples")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|smp| (
                    smp.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    smp.get("reference_text").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                ))
                .collect()
        })
        .unwrap_or_default();
    let row = voices_all.iter().find(|d| d.id == id);
    let gens = row.map(|d| d.gens.clone()).unwrap_or_else(|| "0".into());
    let baked = row.and_then(|d| d.baked.clone());
    let engine = {
        let e = s("default_engine");
        if e.is_empty() { "follows".to_string() } else { e }
    };
    let (name, desc, pers, lang) = (s("name"), s("description"), s("personality"), s("language"));
    let id2 = id.to_string();
    ui.upgrade_in_event_loop(move |ui| {
        ui.set_vp_selected(id2.into());
        ui.set_vi_name(name.into());
        ui.set_vi_desc(desc.into());
        ui.set_vi_personality(pers.into());
        ui.set_vi_lang(lang.into());
        ui.set_vi_engine(engine.into());
        ui.set_vi_gens(gens.into());
        match baked {
            Some(b) => { ui.set_vi_avatar(rgba_to_image(&b)); ui.set_vi_has_avatar(true); }
            None => { ui.set_vi_avatar(Default::default()); ui.set_vi_has_avatar(false); }
        }
        ui.set_vi_samples(ModelRc::from(Rc::new(VecModel::from(
            samples
                .into_iter()
                .map(|(sid, t)| SampleRow { id: sid.into(), text: t.into() })
                .collect::<Vec<_>>(),
        ))));
    })
    .ok();
}

/// Rebuild the Voices-tab table (profiles + per-voice generation counts).
async fn refresh_voices_table(
    ui: &slint::Weak<AppWindow>,
    proxy: &EngineProxy<'_>,
    cache: &mut HashMap<String, RgbaBuf>,
    out: &mut Vec<VpRowData>,
) {
    let pj = proxy.list_profiles().await.unwrap_or_else(|_| "[]".into());
    let hj = proxy.list_history().await.unwrap_or_else(|_| "[]".into());
    let profs: Vec<serde_json::Value> = serde_json::from_str(&pj).unwrap_or_default();
    let hist: Vec<serde_json::Value> = serde_json::from_str(&hj).unwrap_or_default();
    let mut gens: HashMap<String, usize> = HashMap::new();
    for h in &hist {
        if let Some(v) = h.get("voice_id").and_then(|v| v.as_str()) {
            *gens.entry(v.to_string()).or_default() += 1;
        }
    }
    *out = profs
        .iter()
        .filter_map(|p| {
            let id = p.get("id")?.as_str()?.to_string();
            let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let iv = |k: &str| p.get(k).and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let baked = bake_avatar_rgba(
                cache, &s("avatar_path"), iv("avatar_sx"), iv("avatar_sy"),
                iv("avatar_side"), iv("avatar_sh"),
            );
            let engine = {
                let e = s("default_engine");
                if e.is_empty() { "follows".to_string() } else { e }
            };
            Some(VpRowData {
                id: id.clone(),
                name: s("name"),
                desc: s("description"),
                lang: s("language"),
                engine,
                samples: p.get("samples").and_then(|v| v.as_i64()).unwrap_or(0).to_string(),
                gens: gens.get(&id).copied().unwrap_or(0).to_string(),
                baked,
            })
        })
        .collect();
    let rows_src = out.clone();
    ui.upgrade_in_event_loop(move |ui| {
        let filter = ui.get_vp_search().to_string();
        ui.set_vp_rows(ModelRc::from(Rc::new(VecModel::from(vp_to_rows(&rows_src, &filter)))));
        // drop a selection whose profile no longer exists
        let sel = ui.get_vp_selected().to_string();
        if !sel.is_empty() && !rows_src.iter().any(|d| d.id == sel) {
            ui.set_vp_selected("".into());
        }
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
    // only the qwen engines honor delivery instructs — hide the style
    // dropdown for the rest instead of offering a knob that does nothing
    let styled = matches!(engine, "qwen" | "qwen_custom_voice");
    ui.upgrade_in_event_loop(move |ui| {
        ui.set_composer_langs(ModelRc::from(Rc::new(VecModel::from(labels))));
        ui.set_composer_lang_index(idx);
        ui.set_style_supported(styled);
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
    // the ⇄ tab's CONVERSIONS rail is the same data filtered to VC rows,
    // derived here so every history refresh keeps both models in sync
    let vc: Vec<HistItem> = items
        .iter()
        .filter(|h| h.meta.starts_with("⇄ VC"))
        .map(|h| {
            let mut h = h.clone();
            h.meta = h.meta.strip_prefix("⇄ VC · ").unwrap_or(&h.meta).into();
            h
        })
        .collect();
    if let Some(vm) = ui.get_vc_history().as_any().downcast_ref::<VecModel<HistItem>>() {
        vm.set_vec(vc);
    } else {
        ui.set_vc_history(ModelRc::from(Rc::new(VecModel::from(vc))));
    }
    if let Some(vm) = ui.get_history().as_any().downcast_ref::<VecModel<HistItem>>() {
        vm.set_vec(items);
    } else {
        ui.set_history(ModelRc::from(Rc::new(VecModel::from(items))));
    }
}

/// Build the captures model from the engine's ListCaptures JSON (newest first).
fn build_captures(json: &str) -> Vec<CaptureItem> {
    let arr: Vec<serde_json::Value> = serde_json::from_str(json).unwrap_or_default();
    arr.iter()
        .map(|c| {
            let get = |k: &str| c.get(k).and_then(|v| v.as_str()).unwrap_or("");
            CaptureItem {
                id: get("id").into(),
                text: get("text").into(),
                date: get("date").into(),
            }
        })
        .collect()
}

/// Refresh the ⇄ tab's saved-clip rail; returns (id, name, path) rows the
/// worker keeps for arming/audition/deletion by id.
async fn refresh_vc_clips(
    ui: &slint::Weak<AppWindow>,
    proxy: &EngineProxy<'_>,
) -> Vec<(String, String, String)> {
    let j = proxy.list_source_clips().await.unwrap_or_else(|_| "[]".into());
    let arr: Vec<serde_json::Value> = serde_json::from_str(&j).unwrap_or_default();
    let mut data = Vec::new();
    let mut items: Vec<SourceClipItem> = Vec::new();
    for c in &arr {
        let g = |k: &str| c.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let (id, name, path, meta) = (g("id"), g("name"), g("path"), g("meta"));
        items.push(SourceClipItem {
            id: id.clone().into(),
            name: name.clone().into(),
            meta: meta.into(),
        });
        data.push((id, name, path));
    }
    ui.upgrade_in_event_loop(move |ui| {
        ui.set_vc_clips(ModelRc::from(Rc::new(VecModel::from(items))));
    }).ok();
    data
}

fn set_captures_model(ui: &AppWindow, items: Vec<CaptureItem>) {
    if let Some(vm) = ui.get_captures().as_any().downcast_ref::<VecModel<CaptureItem>>() {
        vm.set_vec(items);
    } else {
        ui.set_captures(ModelRc::from(Rc::new(VecModel::from(items))));
    }
}

/// Refresh the composer effects dropdown ("No effects" + presets) and the
/// editor's preset list. Returns (dropdown ids, editor (id, builtin) pairs).
async fn refresh_effect_presets(
    ui: &slint::Weak<AppWindow>,
    proxy: &EngineProxy<'_>,
) -> (Vec<String>, Vec<(String, bool)>) {
    let fx_json = proxy.list_effect_presets().await.unwrap_or_else(|_| "[]".into());
    let fx: Vec<serde_json::Value> = serde_json::from_str(&fx_json).unwrap_or_default();
    let mut labels: Vec<SharedString> = vec!["No effects".into()];
    let mut ids = vec![String::new()];
    let mut pairs = Vec::new();
    let mut items = Vec::new();
    for p in &fx {
        let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let id = p.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let builtin = p.get("builtin").and_then(|v| v.as_bool()).unwrap_or(true);
        labels.push(name.into());
        ids.push(id.to_string());
        pairs.push((id.to_string(), builtin));
        items.push(FxPresetItem { id: id.into(), name: name.into(), builtin });
    }
    ui.upgrade_in_event_loop(move |ui| {
        ui.set_composer_effects(ModelRc::from(Rc::new(VecModel::from(labels))));
        ui.set_fxe_presets(ModelRc::from(Rc::new(VecModel::from(items))));
    })
    .ok();
    (ids, pairs)
}

/// Format an effect param value with decimals matched to its step size.
fn fx_fmt(v: f64, step: f64) -> String {
    if step >= 1.0 {
        format!("{v:.0}")
    } else if step >= 0.1 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// Push the editor's chain (and the expanded row's params) into the UI models.
fn fxe_sync(
    ui: &slint::Weak<AppWindow>,
    defs: &[serde_json::Value],
    chain: &[serde_json::Value],
    expanded: i32,
) {
    let def_of = |t: &str| defs.iter().find(|d| d.get("id").and_then(|v| v.as_str()) == Some(t));
    let rows: Vec<FxRowItem> = chain
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let t = e.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let label = def_of(t)
                .and_then(|d| d.get("label"))
                .and_then(|v| v.as_str())
                .unwrap_or(t);
            FxRowItem {
                label: label.into(),
                enabled: e.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
                expanded: i as i32 == expanded,
            }
        })
        .collect();
    let params: Vec<FxParamItem> = chain
        .get(usize::try_from(expanded).unwrap_or(usize::MAX))
        .map(|e| {
            let t = e.get("type").and_then(|v| v.as_str()).unwrap_or("");
            def_of(t)
                .and_then(|d| d.get("params"))
                .and_then(|p| p.as_array())
                .map(|list| {
                    list.iter()
                        .map(|pd| {
                            let name = pd.get("name").and_then(|v| v.as_str()).unwrap_or("");
                            let min = pd.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let max = pd.get("max").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            let step = pd.get("step").and_then(|v| v.as_f64()).unwrap_or(0.01);
                            let dflt = pd.get("default").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let val = e
                                .get("params")
                                .and_then(|p| p.get(name))
                                .and_then(|v| v.as_f64())
                                .unwrap_or(dflt);
                            FxParamItem {
                                label: pd
                                    .get("description")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(name)
                                    .into(),
                                value_text: fx_fmt(val, step).into(),
                                norm: ((val - min) / (max - min)).clamp(0.0, 1.0) as f32,
                            }
                        })
                        .collect()
                })
                .unwrap_or_default()
        })
        .unwrap_or_default();
    ui.upgrade_in_event_loop(move |ui| {
        ui.set_fxe_chain(ModelRc::from(Rc::new(VecModel::from(rows))));
        ui.set_fxe_params(ModelRc::from(Rc::new(VecModel::from(params))));
    })
    .ok();
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
    let capture_items = build_captures(&proxy.list_captures().await.unwrap_or_else(|_| "[]".into()));
    let mut avatar_cache: HashMap<String, RgbaBuf> = HashMap::new();
    let grid = bake_grid(&mut avatar_cache, grid);
    {
        ui.upgrade_in_event_loop(move |ui| {
            ui.set_backend(backend.into());
            ui.set_kokoro_names(ModelRc::from(Rc::new(VecModel::from(kokoro_names))));
            ui.set_kokoro_ids(ModelRc::from(Rc::new(VecModel::from(kokoro_ids))));
            ui.set_voices(ModelRc::from(Rc::new(VecModel::from(to_voice_items(grid)))));
            if ui.get_selected_voice().is_empty() {
                ui.set_selected_voice(default_selected.clone().into());
                ui.set_selected_voice_name(voice_name(&ui, &default_selected).into());
            }
            set_history_model(&ui, hist_items);
            set_captures_model(&ui, capture_items);
        })
        .ok();
    }

    let mut levels = proxy.receive_audio_level().await?;
    let mut gprog = proxy.receive_generation_progress().await?;
    let mut tprog = proxy.receive_transcribe_progress().await?;
    let mut tres = proxy.receive_transcribe_result().await?;
    let mut ended = proxy.receive_speak_ended().await?;
    let mut pinfo = proxy.receive_playback_info().await?;
    let mut pprog = proxy.receive_playback_progress().await?;
    let mut llm_res = proxy.receive_llm_result().await?;
    let mut mprog = proxy.receive_model_progress().await?;
    let mut pending_llm: u32 = 0;
    let (mut voice_models, mut active_engine) = refresh_models(&ui, &proxy).await;
    let mut lang_codes = update_composer_langs(&ui, "kokoro", "en");
    // effects dropdown: "No effects" + engine presets (builtin + user)
    let (mut effect_ids, mut fxe_presets) = refresh_effect_presets(&ui, &proxy).await;
    // effects chain editor state — the worker owns the chain JSON
    let mut fxe_defs: Vec<serde_json::Value> = Vec::new();
    let mut fxe_chain: Vec<serde_json::Value> = Vec::new();
    let mut fxe_pid = String::new(); // loaded user preset id ("" = new / builtin copy)
    let mut fxe_expanded: i32 = -1;
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
    let mut voices_all: Vec<VpRowData> = Vec::new();  // voices-tab table backing data
    let mut vp_inspected = String::new();             // profile id shown in the inspector
    let mut sample_gen: u32 = 0;                      // audition playback gen (0 = none)
    let mut sample_playing = String::new();           // sample id being auditioned
    let mut cv_edit_transcript = String::new();
    let mut cv_avatar: Option<(String, String, i32, i32, i32, i32)> = None; // staged (path, mode, sx, sy, sw, sh)
    // transcription view state
    let mut tr_rec: Option<tokio::process::Child> = None;
    let mut tr_elapsed: u32 = 0;
    let tr_wav = std::env::var("XDG_RUNTIME_DIR")
        .map(|d| format!("{d}/syrinx-transcribe.wav"))
        .unwrap_or_else(|_| "/tmp/syrinx-transcribe.wav".into());
    const TR_REC_MAX: u32 = 600; // 10 min safety cap
    let mut pending_tr: u32 = 0;
    let mut pending_tr_refine: u32 = 0;
    // voice-changer (⇄) view state
    let mut vc_rec: Option<tokio::process::Child> = None;
    let mut vc_elapsed: u32 = 0;
    let vc_wav = std::env::var("XDG_RUNTIME_DIR")
        .map(|d| format!("{d}/syrinx-convert.wav"))
        .unwrap_or_else(|_| "/tmp/syrinx-convert.wav".into());
    const VC_REC_MAX: u32 = 180; // matches the engine's SYRINX_VC_MAX_SECS default
    let mut vc_source: Option<String> = None;       // armed source path
    let mut vc_voice_ids: Vec<String> = Vec::new(); // parallel to the dropdown names
    let mut pending_vc: u32 = 0;                    // in-flight conversion gen id
    let mut vc_clips_data: Vec<(String, String, String)> = Vec::new(); // (id, name, path)
    let mut vc_audition_gen: u32 = 0;               // audition playback gen (0 = none)
    let mut vc_audition_id = String::new();         // clip id or "scratch" being played
    let mut pending_vc_tr: u32 = 0;                 // source auto-transcription req id
    // create-voice modal state
    let mut cv_rec: Option<tokio::process::Child> = None;
    let cv_wav = cv_wav_path();
    let mut cv_sample: Option<String> = None;
    let mut rec_interval = tokio::time::interval(std::time::Duration::from_secs(1));
    // The interval is only polled while a recording is live; with the default
    // Burst behavior every idle minute becomes a backlog of instant ticks the
    // moment recording starts — the elapsed counter then blows through the cap
    // in milliseconds and insta-stops the capture. Delay = never tick faster
    // than the period, regardless of backlog.
    rec_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
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
                    // conversions report to the ⇄ tab, not the composer
                    let is_vc = pending_vc != 0 && a.gen_id == pending_vc;
                    if let Some(msg) = a.state.strip_prefix("error:") {
                        let msg = msg.trim().to_string();
                        ui.upgrade_in_event_loop(move |ui| {
                            if is_vc {
                                ui.set_vc_busy(false);
                                ui.set_vc_status("".into());
                                ui.set_vc_error(msg.into());
                            } else {
                                ui.set_gen_error(msg.into());
                            }
                        }).ok();
                    } else if is_vc {
                        let stage: SharedString = match a.state.as_str() {
                            "loading model" => "loading model…".into(),
                            "converting" => "converting…".into(),
                            "playing" => "done — playing · saved to History".into(),
                            s => s.into(),
                        };
                        ui.upgrade_in_event_loop(move |ui| {
                            if stage.starts_with("done") { ui.set_vc_busy(false); }
                            ui.set_vc_status(stage);
                        }).ok();
                    }
                }
            }
            Some(sig) = tprog.next() => {
                if let Ok(a) = sig.args() {
                    if a.req_id == pending_tr && pending_tr != 0 {
                        let partial = a.partial.to_string();
                        ui.upgrade_in_event_loop(move |ui| ui.set_tr_text(partial.into())).ok();
                    } else if a.req_id == pending_vc_tr && pending_vc_tr != 0 {
                        let partial = a.partial.to_string();
                        ui.upgrade_in_event_loop(move |ui| ui.set_vc_transcript(partial.into())).ok();
                    }
                }
            }
            Some(sig) = tres.next() => {
                if let Ok(a) = sig.args() {
                    if a.req_id == pending_vc_tr && pending_vc_tr != 0 && a.req_id != pending_tr {
                        pending_vc_tr = 0;
                        let text = a.text.to_string();
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_vc_transcribing(false);
                            ui.set_vc_transcript(text.into());
                        }).ok();
                    } else if a.req_id == pending_tr && pending_tr != 0 {
                        pending_tr = 0;
                        let text = a.text.to_string();
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_tr_busy(false);
                            if text.trim().is_empty() {
                                ui.set_tr_status("no speech detected".into());
                            } else {
                                ui.set_tr_status("".into());
                                ui.set_tr_text(text.into());
                            }
                        }).ok();
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
                    // transcription-view refine result routes to tr-text
                    if a.req_id == pending_tr_refine && pending_tr_refine != 0 {
                        pending_tr_refine = 0;
                        let text = a.text.to_string();
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_tr_busy(false);
                            ui.set_tr_status("".into());
                            if !text.trim().is_empty() {
                                ui.set_tr_text(text.into());
                            }
                        }).ok();
                    } else if a.req_id == pending_llm && pending_llm != 0 {
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
                // conversion ran its course (played out or errored) — settle the ⇄ tab
                if pending_vc != 0 && sig.args().map(|a| a.gen_id == pending_vc).unwrap_or(false) {
                    pending_vc = 0;
                    ui.upgrade_in_event_loop(|ui| {
                        ui.set_vc_busy(false);
                        if ui.get_vc_status().starts_with("done") {
                            ui.set_vc_status("done · saved to History".into());
                        }
                    }).ok();
                }
                // source-clip audition finished -> flip ■ back to ▶
                if vc_audition_gen != 0
                    && sig.args().map(|a| a.gen_id == vc_audition_gen).unwrap_or(false)
                {
                    vc_audition_gen = 0;
                    vc_audition_id.clear();
                    ui.upgrade_in_event_loop(|ui| ui.set_vc_audition_id("".into())).ok();
                }
                // sample audition ran to its end (or was replaced) -> flip ■ back to ▶
                let sample_done = sample_gen != 0
                    && sig.args().map(|a| a.gen_id == sample_gen).unwrap_or(false);
                if sample_done {
                    sample_gen = 0;
                    sample_playing.clear();
                    ui.upgrade_in_event_loop(|ui| ui.set_vs_playing("".into())).ok();
                }
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
            _ = rec_interval.tick(), if cv_rec.is_some() || tr_rec.is_some() || vc_rec.is_some() => {
                // recorder died on its own (e.g. a suspended monitor source
                // erroring at first open) — surface it instead of a phantom
                // "recording" that never advances
                if let Some(child) = vc_rec.as_mut() {
                    if matches!(child.try_wait(), Ok(Some(_))) {
                        vc_rec = None;
                        ui.upgrade_in_event_loop(|ui| {
                            ui.set_vc_recording(false);
                            ui.set_vc_status("⚠ recorder exited — try again or check the source".into());
                        }).ok();
                    }
                }
                if vc_rec.is_some() {
                    vc_elapsed += 1;
                    if vc_elapsed >= VC_REC_MAX {
                        // engine caps conversion sources — stop and keep the clip
                        if let Some(mut child) = vc_rec.take() {
                            stop_pw_record(&mut child).await;
                            vc_source = Some(vc_wav.clone());
                            let label = format!("{} · stopped at the 3:00 cap", recorded_label(&vc_wav));
                            ui.upgrade_in_event_loop(move |ui| {
                                ui.set_vc_recording(false);
                                ui.set_vc_has_source(true);
                                ui.set_vc_source_label(label.into());
                                ui.set_vc_armed_id("".into());
                                ui.set_vc_armed_saved(false);
                                ui.set_vc_status("".into());
                            }).ok();
                            match proxy.transcribe_file(&vc_wav).await {
                                Ok(rid) => {
                                    pending_vc_tr = rid;
                                    ui.upgrade_in_event_loop(|ui| {
                                        ui.set_vc_transcribing(true);
                                        ui.set_vc_transcript("".into());
                                    }).ok();
                                }
                                Err(e) => tracing::error!("vc transcribe failed: {e}"),
                            }
                        }
                    } else {
                        let e = vc_elapsed;
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_vc_status(format!("● recording {}:{:02} / 3:00", e / 60, e % 60).into());
                        }).ok();
                    }
                }
                if tr_rec.is_some() {
                    tr_elapsed += 1;
                    if tr_elapsed >= TR_REC_MAX {
                        // safety cap — stop and transcribe what we have
                        if let Some(mut child) = tr_rec.take() {
                            stop_pw_record(&mut child).await;
                            match proxy.transcribe_file(&tr_wav).await {
                                Ok(id) => pending_tr = id,
                                Err(e) => tracing::error!("transcribe failed: {e}"),
                            }
                            ui.upgrade_in_event_loop(|ui| {
                                ui.set_tr_recording(false);
                                ui.set_tr_busy(true);
                                ui.set_tr_status("transcribing…".into());
                            }).ok();
                        }
                    } else {
                        let e = tr_elapsed;
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_tr_status(format!("● recording {}:{:02}", e / 60, e % 60).into());
                        }).ok();
                    }
                }
                if cv_rec.is_none() { continue; }
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
                        ui.upgrade_in_event_loop(|ui| {
                            ui.set_cv_error("".into());
                            ui.set_cv_transcribing(true);
                        }).ok();
                        let result = proxy.transcribe(&path).await;
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_cv_transcribing(false);
                            match result {
                                Ok(text) => ui.set_cv_transcript(text.into()),
                                Err(e) => ui.set_cv_error(format!("Transcribe failed: {e}").into()),
                            }
                        }).ok();
                    } else {
                        ui.upgrade_in_event_loop(|ui| {
                            ui.set_cv_error("Record or choose a reference clip first.".into());
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
                    ui.upgrade_in_event_loop(|ui| ui.set_cv_error("".into())).ok();
                    if let Some(pid) = cv_edit.clone() {
                        // edit mode: patch metadata + optionally replace audio
                        let patch = serde_json::json!({
                            "name": name, "description": desc,
                            "personality": personality, "language": language,
                            "default_engine": default_engine,
                        }).to_string();
                        match proxy.update_profile(&pid, &patch).await {
                            Ok(_) => {
                                let mut sample_err = String::new();
                                if let Some((path, amode, asx, asy, asw, ash)) = cv_avatar.take() {
                                    proxy.set_profile_avatar(&pid, &path, &amode, asx, asy, asw, ash).await.ok();
                                }
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
                                        sample_err = format!(
                                            "Replacing the sample failed: {} — record a new clip and save again.",
                                            profile_err_msg(&e)
                                        );
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
                                refresh_grid(&ui, &proxy, &mut avatar_cache).await;
                                refresh_voices_table(&ui, &proxy, &mut avatar_cache, &mut voices_all).await;
                                if vp_inspected == pid {
                                    // saved edits land in the inspector immediately
                                    inspect_profile(&ui, &proxy, &voices_all, &pid).await;
                                }
                                if sample_err.is_empty() {
                                    cv_edit = None;
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
                                        ui.set_cv_has_avatar(false);
                                        if ui.get_selected_voice().as_str() == pid2 {
                                            ui.set_selected_voice_name(name2.into());
                                            ui.set_selected_has_personality(hp);
                                        }
                                    }).ok();
                                } else {
                                    // keep the modal open in edit mode so a re-record can retry
                                    ui.upgrade_in_event_loop(move |ui| {
                                        ui.set_cv_sample_label("".into());
                                        ui.set_cv_error(sample_err.into());
                                    }).ok();
                                }
                            }
                            Err(e) => {
                                tracing::error!("edit voice failed: {e}");
                                let msg = profile_err_msg(&e);
                                ui.upgrade_in_event_loop(move |ui| ui.set_cv_error(msg.into())).ok();
                            }
                        }
                    } else if name.trim().is_empty() || cv_sample.is_none() {
                        ui.upgrade_in_event_loop(|ui| {
                            ui.set_cv_error("A name and a reference sample are both required.".into());
                        }).ok();
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
                            if let Err(e) = proxy.add_sample(&pid, &sample, &transcript).await {
                                // roll back so a failed create leaves no sample-less ghost
                                proxy.delete_profile(&pid).await.ok();
                                return Err(e);
                            }
                            zbus::Result::Ok(pid)
                        }.await;
                        match outcome {
                            Ok(pid) => {
                                if let Some((path, amode, asx, asy, asw, ash)) = cv_avatar.take() {
                                    proxy.set_profile_avatar(&pid, &path, &amode, asx, asy, asw, ash).await.ok();
                                }
                                let raw = proxy.list_voices().await.unwrap_or_default();
                                let pj = proxy.list_profiles().await.unwrap_or_else(|_| "[]".into());
                                let GridData { grid, kokoro_names, kokoro_ids, .. } = build_grid(raw, &pj);
                                kokoro_all = kokoro_ids
                                    .iter()
                                    .zip(kokoro_names.iter())
                                    .map(|(i, n)| (i.to_string(), n.to_string()))
                                    .collect();
                                let grid = bake_grid(&mut avatar_cache, grid);
                                cv_sample = None;
                                ui.upgrade_in_event_loop(move |ui| {
                                    ui.set_cv_creating(false);
                                    ui.set_cv_open(false);
                                    ui.set_cv_name("".into());
                                    ui.set_cv_desc("".into());
                                    ui.set_cv_personality("".into());
                                    ui.set_cv_transcript("".into());
                                    ui.set_cv_sample_label("".into());
                                    ui.set_cv_model_index(0);
                                    ui.set_cv_has_avatar(false);
                                    ui.set_kokoro_names(ModelRc::from(Rc::new(VecModel::from(kokoro_names))));
                                    ui.set_kokoro_ids(ModelRc::from(Rc::new(VecModel::from(kokoro_ids))));
                                    ui.set_voices(ModelRc::from(Rc::new(VecModel::from(to_voice_items(grid)))));
                                }).ok();
                                refresh_voices_table(&ui, &proxy, &mut avatar_cache, &mut voices_all).await;
                            }
                            Err(e) => {
                                tracing::error!("create voice failed: {e}");
                                let msg = profile_err_msg(&e);
                                ui.upgrade_in_event_loop(move |ui| {
                                    ui.set_cv_creating(false);
                                    ui.set_cv_error(msg.into());
                                }).ok();
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
                    cv_avatar = None;
                    ui.upgrade_in_event_loop(|ui| {
                        ui.set_cv_recording(false);
                        ui.set_cv_sample_label("".into());
                        ui.set_cv_name("".into());
                        ui.set_cv_desc("".into());
                        ui.set_cv_personality("".into());
                        ui.set_cv_transcript("".into());
                        ui.set_cv_edit_id("".into());
                        ui.set_cv_model_index(0);
                        ui.set_cv_has_avatar(false);
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
                        // builtin:<engine>:<voice> — kokoro or an extra preset
                        // engine like qwen_custom_voice
                        let engine = id.split(':').nth(1).unwrap_or("kokoro").to_string();
                        let code = if engine == "kokoro" {
                            kokoro_lang_code(&id).to_string()
                        } else {
                            "en".to_string()
                        };
                        (engine, code)
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
                            refresh_grid(&ui, &proxy, &mut avatar_cache).await;
                            refresh_voices_table(&ui, &proxy, &mut avatar_cache, &mut voices_all).await;
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
                Some(Cmd::PickStyle { index }) => {
                    if let Some((_, instruct)) = STYLES.get(index) {
                        proxy.set_style(instruct).await.ok();
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
                        // existing avatar for the modal preview (baked thumbnail)
                        let av_mode = {
                            let m = s("avatar_mode");
                            if m.is_empty() { "circle".to_string() } else { m }
                        };
                        let iv = |k: &str| p.get(k).and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                        let av_baked = bake_avatar_rgba(
                            &mut avatar_cache,
                            &s("avatar_path"),
                            iv("avatar_sx"),
                            iv("avatar_sy"),
                            iv("avatar_side"),
                            iv("avatar_sh"),
                        );
                        cv_edit = Some(id.clone());
                        cv_sample = None;
                        cv_avatar = None;
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_cv_error("".into());
                            ui.set_cv_name(name.into());
                            ui.set_cv_desc(desc.into());
                            ui.set_cv_personality(pers.into());
                            ui.set_cv_language(lang.into());
                            ui.set_cv_lang_index(lang_idx);
                            ui.set_cv_transcript(transcript.into());
                            ui.set_cv_model_index(model_idx);
                            ui.set_cv_sample_label("".into());
                            match &av_baked {
                                Some(b) => {
                                    ui.set_cv_avatar(rgba_to_image(b));
                                    ui.set_cv_avatar_mode(av_mode.into());
                                    ui.set_cv_has_avatar(true);
                                }
                                None => ui.set_cv_has_avatar(false),
                            }
                            ui.set_cv_edit_id(id.into());
                            ui.set_cv_open(true);
                        }).ok();
                    }
                }
                Some(Cmd::DeleteVoice { id }) => {
                    match proxy.delete_profile(&id).await {
                        Ok(_) => {
                            if vp_inspected == id {
                                vp_inspected.clear();
                            }
                            refresh_grid(&ui, &proxy, &mut avatar_cache).await;
                            refresh_voices_table(&ui, &proxy, &mut avatar_cache, &mut voices_all).await;
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
                Some(Cmd::TrToggleRecord { system }) => {
                    if let Some(mut child) = tr_rec.take() {
                        // stop → transcribe (unless the capture came out silent)
                        stop_pw_record(&mut child).await;
                        if wav_rms(&tr_wav).map(|r| r < 0.006).unwrap_or(true) {
                            ui.upgrade_in_event_loop(|ui| {
                                ui.set_tr_recording(false);
                                ui.set_tr_status("⚠ capture was silent — check the input device".into());
                            }).ok();
                        } else {
                            match proxy.transcribe_file(&tr_wav).await {
                                Ok(id) => {
                                    pending_tr = id;
                                    ui.upgrade_in_event_loop(|ui| {
                                        ui.set_tr_recording(false);
                                        ui.set_tr_busy(true);
                                        ui.set_tr_status("transcribing…".into());
                                    }).ok();
                                }
                                Err(e) => {
                                    tracing::error!("transcribe failed: {e}");
                                    ui.upgrade_in_event_loop(|ui| {
                                        ui.set_tr_recording(false);
                                        ui.set_tr_status("engine unavailable".into());
                                    }).ok();
                                }
                            }
                        }
                    } else {
                        let device = if system { default_monitor().await } else { None };
                        if system && device.is_none() {
                            ui.upgrade_in_event_loop(|ui| {
                                ui.set_tr_status("no default sink monitor found".into());
                            }).ok();
                        } else {
                            match start_pw_record(&tr_wav, device.as_deref()).await {
                                Ok(child) => {
                                    tr_rec = Some(child);
                                    tr_elapsed = 0;
                                    rec_interval.reset();  // first tick a full second out
                                    let mode = if system { "system" } else { "mic" };
                                    ui.upgrade_in_event_loop(move |ui| {
                                        ui.set_tr_text("".into());
                                        ui.set_tr_capture_id("".into()); // fresh source = new capture
                                        ui.set_tr_rec_mode(mode.into());
                                        ui.set_tr_recording(true);
                                        ui.set_tr_status("● recording 0:00".into());
                                    }).ok();
                                }
                                Err(e) => tracing::error!("record failed: {e}"),
                            }
                        }
                    }
                }
                Some(Cmd::TrPickFile) => {
                    if let Some(handle) = rfd::AsyncFileDialog::new()
                        .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a", "opus", "webm"])
                        .pick_file()
                        .await
                    {
                        let path = handle.path().to_string_lossy().to_string();
                        match proxy.transcribe_file(&path).await {
                            Ok(id) => {
                                pending_tr = id;
                                ui.upgrade_in_event_loop(|ui| {
                                    ui.set_tr_text("".into());
                                    ui.set_tr_capture_id("".into()); // fresh source = new capture
                                    ui.set_tr_busy(true);
                                    ui.set_tr_status("transcribing…".into());
                                }).ok();
                            }
                            Err(e) => tracing::error!("transcribe failed: {e}"),
                        }
                    }
                }
                Some(Cmd::TrRefine { text }) => {
                    match proxy.refine_transcript(&text).await {
                        Ok(rid) if rid != 0 => pending_tr_refine = rid,
                        _ => {
                            ui.upgrade_in_event_loop(|ui| {
                                ui.set_tr_busy(false);
                                ui.set_tr_status("refine unavailable".into());
                            }).ok();
                        }
                    }
                }
                Some(Cmd::TrSaveCapture { id, text }) => {
                    // "" id = new row; otherwise replace the same entry in place
                    let saved = if id.is_empty() {
                        proxy.save_capture(&text).await.ok().filter(|s| !s.is_empty())
                    } else {
                        proxy.update_capture(&id, &text).await.ok().map(|()| id.clone())
                    };
                    match saved {
                        Some(cid) => {
                            let status = if id.is_empty() { "capture saved" } else { "capture updated" };
                            let items = build_captures(
                                &proxy.list_captures().await.unwrap_or_else(|_| "[]".into()),
                            );
                            ui.upgrade_in_event_loop(move |ui| {
                                ui.set_tr_capture_id(cid.into());
                                ui.set_tr_status(status.into());
                                set_captures_model(&ui, items);
                            }).ok();
                        }
                        None => {
                            ui.upgrade_in_event_loop(|ui| {
                                ui.set_tr_status("save failed — engine unavailable".into());
                            }).ok();
                        }
                    }
                }
                Some(Cmd::TrDeleteCapture { id }) => {
                    match proxy.delete_capture(&id).await {
                        Ok(_) => {
                            let items = build_captures(
                                &proxy.list_captures().await.unwrap_or_else(|_| "[]".into()),
                            );
                            ui.upgrade_in_event_loop(move |ui| {
                                // the transcript stays in the box; it's just unsaved now
                                if ui.get_tr_capture_id().as_str() == id {
                                    ui.set_tr_capture_id("".into());
                                }
                                set_captures_model(&ui, items);
                            }).ok();
                        }
                        Err(e) => tracing::error!("delete capture failed: {e}"),
                    }
                }
                Some(Cmd::VcLoad) => {
                    // target dropdown = cloned profiles that have reference samples
                    let pj = proxy.list_profiles().await.unwrap_or_else(|_| "[]".into());
                    let profs: Vec<serde_json::Value> = serde_json::from_str(&pj).unwrap_or_default();
                    vc_voice_ids.clear();
                    let mut names: Vec<SharedString> = Vec::new();
                    for p in &profs {
                        let cloned = p.get("voice_type").and_then(|v| v.as_str()) == Some("cloned");
                        let samples = p.get("samples").and_then(|v| v.as_i64()).unwrap_or(0);
                        if cloned && samples > 0 {
                            if let (Some(id), Some(name)) = (
                                p.get("id").and_then(|v| v.as_str()),
                                p.get("name").and_then(|v| v.as_str()),
                            ) {
                                vc_voice_ids.push(id.to_string());
                                names.push(name.into());
                            }
                        }
                    }
                    let count = names.len() as i32;
                    ui.upgrade_in_event_loop(move |ui| {
                        if ui.get_vc_voice_index() >= count { ui.set_vc_voice_index(0); }
                        ui.set_vc_voice_names(ModelRc::from(Rc::new(VecModel::from(names))));
                    }).ok();
                    vc_clips_data = refresh_vc_clips(&ui, &proxy).await;
                }
                Some(Cmd::VcToggleRecord { system }) => {
                    if let Some(mut child) = vc_rec.take() {
                        // stop → arm the clip as the conversion source
                        stop_pw_record(&mut child).await;
                        if wav_rms(&vc_wav).map(|r| r < 0.006).unwrap_or(true) {
                            ui.upgrade_in_event_loop(|ui| {
                                ui.set_vc_recording(false);
                                ui.set_vc_status("⚠ capture was silent — check the input device".into());
                            }).ok();
                        } else {
                            vc_source = Some(vc_wav.clone());
                            let e = vc_elapsed;
                            let label = format!("recorded clip · {}:{:02}", e / 60, e % 60);
                            ui.upgrade_in_event_loop(move |ui| {
                                ui.set_vc_recording(false);
                                ui.set_vc_has_source(true);
                                ui.set_vc_source_label(label.into());
                                ui.set_vc_armed_id("".into());
                                ui.set_vc_armed_saved(false);
                                ui.set_vc_status("".into());
                            }).ok();
                            match proxy.transcribe_file(&vc_wav).await {
                                Ok(rid) => {
                                    pending_vc_tr = rid;
                                    ui.upgrade_in_event_loop(|ui| {
                                        ui.set_vc_transcribing(true);
                                        ui.set_vc_transcript("".into());
                                    }).ok();
                                }
                                Err(e) => tracing::error!("vc transcribe failed: {e}"),
                            }
                        }
                    } else {
                        let device = if system { default_monitor().await } else { None };
                        if system && device.is_none() {
                            ui.upgrade_in_event_loop(|ui| {
                                ui.set_vc_status("no default sink monitor found".into());
                            }).ok();
                        } else {
                            match start_pw_record(&vc_wav, device.as_deref()).await {
                                Ok(child) => {
                                    vc_rec = Some(child);
                                    vc_elapsed = 0;
                                    pending_vc_tr = 0;  // a stale transcription no longer applies
                                    rec_interval.reset();  // first tick a full second out
                                    let mode = if system { "system" } else { "mic" };
                                    ui.upgrade_in_event_loop(move |ui| {
                                        ui.set_vc_has_source(false);
                                        ui.set_vc_source_label("".into());
                                        ui.set_vc_error("".into());
                                        ui.set_vc_transcript("".into());
                                        ui.set_vc_transcribing(false);
                                        ui.set_vc_rec_mode(mode.into());
                                        ui.set_vc_recording(true);
                                        ui.set_vc_status("● recording 0:00 / 3:00".into());
                                    }).ok();
                                }
                                Err(e) => tracing::error!("record failed: {e}"),
                            }
                        }
                    }
                }
                Some(Cmd::VcPickFile) => {
                    if let Some(handle) = rfd::AsyncFileDialog::new()
                        .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a", "opus", "webm"])
                        .pick_file()
                        .await
                    {
                        let name = handle.file_name();
                        let path = handle.path().to_string_lossy().to_string();
                        vc_source = Some(path.clone());
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_vc_has_source(true);
                            ui.set_vc_source_label(name.into());
                            ui.set_vc_armed_id("".into());
                            ui.set_vc_armed_saved(false);
                            ui.set_vc_error("".into());
                            ui.set_vc_status("".into());
                        }).ok();
                        match proxy.transcribe_file(&path).await {
                            Ok(rid) => {
                                pending_vc_tr = rid;
                                ui.upgrade_in_event_loop(|ui| {
                                    ui.set_vc_transcribing(true);
                                    ui.set_vc_transcript("".into());
                                }).ok();
                            }
                            Err(e) => tracing::error!("vc transcribe failed: {e}"),
                        }
                    }
                }
                Some(Cmd::VcConvert { index, label, transcript }) => {
                    if let (Some(src), Some(pid)) =
                        (vc_source.clone(), vc_voice_ids.get(index).cloned())
                    {
                        match proxy.convert_voice(&src, &pid, "", &label, &transcript).await {
                            Ok(gid) if gid != 0 => {
                                pending_vc = gid;
                                ui.upgrade_in_event_loop(|ui| {
                                    ui.set_vc_busy(true);
                                    ui.set_vc_error("".into());
                                    ui.set_vc_status("starting…".into());
                                }).ok();
                            }
                            Ok(_) => {}
                            Err(e) => {
                                tracing::error!("convert failed: {e}");
                                ui.upgrade_in_event_loop(|ui| {
                                    ui.set_vc_status("engine unavailable".into());
                                }).ok();
                            }
                        }
                    }
                }
                Some(Cmd::VcSaveClip { name }) => {
                    if let Some(src) = vc_source.clone() {
                        match proxy.save_source_clip(&src, &name).await {
                            Ok(id) if !id.is_empty() => {
                                vc_clips_data = refresh_vc_clips(&ui, &proxy).await;
                                // arm the stored copy: the scratch wav gets
                                // overwritten by the next recording
                                if let Some((cid, cname, cpath)) =
                                    vc_clips_data.iter().find(|(cid, _, _)| *cid == id).cloned()
                                {
                                    vc_source = Some(cpath);
                                    ui.upgrade_in_event_loop(move |ui| {
                                        ui.set_vc_armed_id(cid.into());
                                        ui.set_vc_armed_saved(true);
                                        ui.set_vc_source_label(cname.into());
                                        ui.set_vc_clip_name("".into());
                                        ui.set_vc_status("clip saved".into());
                                    }).ok();
                                }
                            }
                            _ => {
                                ui.upgrade_in_event_loop(|ui| {
                                    ui.set_vc_status("⚠ save failed".into());
                                }).ok();
                            }
                        }
                    }
                }
                Some(Cmd::VcDeleteClip { id }) => {
                    if vc_audition_id == id && vc_audition_gen != 0 {
                        let _ = proxy.cancel(vc_audition_gen).await;
                        vc_audition_gen = 0;
                        vc_audition_id.clear();
                    }
                    if let Err(e) = proxy.delete_source_clip(&id).await {
                        tracing::error!("delete clip failed: {e}");
                    }
                    vc_clips_data = refresh_vc_clips(&ui, &proxy).await;
                    // deleting the armed clip disarms it — its file is gone
                    let disarm = vc_source
                        .as_deref()
                        .map(|p| !vc_clips_data.iter().any(|(_, _, cp)| cp == p)
                            && p.contains("/clips/"))
                        .unwrap_or(false);
                    if disarm {
                        vc_source = None;
                    }
                    ui.upgrade_in_event_loop(move |ui| {
                        ui.set_vc_audition_id("".into());
                        if disarm {
                            ui.set_vc_has_source(false);
                            ui.set_vc_source_label("".into());
                            ui.set_vc_armed_id("".into());
                            ui.set_vc_armed_saved(false);
                        }
                    }).ok();
                }
                Some(Cmd::VcArmClip { id }) => {
                    if let Some((cid, cname, cpath)) =
                        vc_clips_data.iter().find(|(cid, _, _)| *cid == id).cloned()
                    {
                        vc_source = Some(cpath.clone());
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_vc_has_source(true);
                            ui.set_vc_source_label(cname.into());
                            ui.set_vc_armed_id(cid.into());
                            ui.set_vc_armed_saved(true);
                            ui.set_vc_error("".into());
                            ui.set_vc_status("".into());
                        }).ok();
                        match proxy.transcribe_file(&cpath).await {
                            Ok(rid) => {
                                pending_vc_tr = rid;
                                ui.upgrade_in_event_loop(|ui| {
                                    ui.set_vc_transcribing(true);
                                    ui.set_vc_transcript("".into());
                                }).ok();
                            }
                            Err(e) => tracing::error!("vc transcribe failed: {e}"),
                        }
                    }
                }
                Some(Cmd::VcAudition { id }) => {
                    if vc_audition_id == id && vc_audition_gen != 0 {
                        // toggle off
                        let _ = proxy.cancel(vc_audition_gen).await;
                        vc_audition_gen = 0;
                        vc_audition_id.clear();
                        ui.upgrade_in_event_loop(|ui| ui.set_vc_audition_id("".into())).ok();
                    } else {
                        let resolved = if id == "scratch" {
                            vc_source.clone().map(|p| (p, "source clip".to_string()))
                        } else {
                            vc_clips_data.iter().find(|(cid, _, _)| *cid == id)
                                .map(|(_, n, p)| (p.clone(), n.clone()))
                        };
                        if let Some((path, title)) = resolved {
                            match proxy.play_file(&path, &title).await {
                                Ok(gid) if gid != 0 => {
                                    vc_audition_gen = gid;
                                    vc_audition_id = id.clone();
                                    ui.upgrade_in_event_loop(move |ui| {
                                        ui.set_vc_audition_id(id.into());
                                    }).ok();
                                }
                                _ => {
                                    ui.upgrade_in_event_loop(|ui| {
                                        ui.set_vc_status("⚠ can't play this file".into());
                                    }).ok();
                                }
                            }
                        }
                    }
                }
                Some(Cmd::VoicesLoad) => {
                    refresh_voices_table(&ui, &proxy, &mut avatar_cache, &mut voices_all).await;
                }
                Some(Cmd::VoicesSearch { q }) => {
                    let data = voices_all.clone();
                    ui.upgrade_in_event_loop(move |ui| {
                        ui.set_vp_rows(ModelRc::from(Rc::new(VecModel::from(vp_to_rows(&data, &q)))));
                    }).ok();
                }
                Some(Cmd::VoicesInspect { id }) => {
                    vp_inspected = id.clone();
                    inspect_profile(&ui, &proxy, &voices_all, &id).await;
                }
                Some(Cmd::PlaySample { id }) => {
                    if sample_playing == id && sample_gen != 0 {
                        // toggle: same sample clicked while playing -> stop
                        proxy.cancel(sample_gen).await.ok();
                        sample_gen = 0;
                        sample_playing.clear();
                        ui.upgrade_in_event_loop(|ui| ui.set_vs_playing("".into())).ok();
                    } else {
                        // engine playback is serialized (latest wins), so starting
                        // a different sample implicitly replaces the current one
                        match proxy.play_sample(&id).await {
                            Ok(g) if g != 0 => {
                                sample_gen = g;
                                sample_playing = id.clone();
                                let id2 = id.clone();
                                ui.upgrade_in_event_loop(move |ui| ui.set_vs_playing(id2.into())).ok();
                            }
                            _ => {}
                        }
                    }
                }
                Some(Cmd::FxeShow) => {
                    if fxe_defs.is_empty() {
                        let defs_json = proxy.list_effects().await.unwrap_or_else(|_| "[]".into());
                        fxe_defs = serde_json::from_str(&defs_json).unwrap_or_default();
                        let mut add: Vec<SharedString> = vec!["＋ Add effect…".into()];
                        for d in &fxe_defs {
                            add.push(d.get("label").and_then(|v| v.as_str()).unwrap_or("").into());
                        }
                        ui.upgrade_in_event_loop(move |ui| {
                            ui.set_fxe_add_model(ModelRc::from(Rc::new(VecModel::from(add))));
                        }).ok();
                    }
                    let r = refresh_effect_presets(&ui, &proxy).await;
                    effect_ids = r.0;
                    fxe_presets = r.1;
                    fxe_chain.clear();
                    fxe_pid.clear();
                    fxe_expanded = -1;
                    fxe_sync(&ui, &fxe_defs, &fxe_chain, fxe_expanded);
                    ui.upgrade_in_event_loop(|ui| {
                        ui.set_fxe_preset_index(-1);
                        ui.set_fxe_builtin(false);
                        ui.set_fxe_can_delete(false);
                        ui.set_fxe_name("".into());
                        ui.set_fxe_desc("".into());
                        ui.set_fxe_status("".into());
                        ui.set_fxe_open(true);
                    }).ok();
                }
                Some(Cmd::FxeLoad { index }) => {
                    if let Some((pid, builtin)) = fxe_presets.get(index).cloned() {
                        if let Ok(pjson) = proxy.get_effect_preset(&pid).await {
                            if let Ok(p) = serde_json::from_str::<serde_json::Value>(&pjson) {
                                fxe_chain = p.get("chain").and_then(|c| c.as_array()).cloned().unwrap_or_default();
                                fxe_expanded = -1;
                                fxe_pid = if builtin { String::new() } else { pid };
                                let name = p.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                // editing a builtin saves as the user's own copy
                                let display = if builtin { format!("{name} (custom)") } else { name.to_string() };
                                let desc = p.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                fxe_sync(&ui, &fxe_defs, &fxe_chain, fxe_expanded);
                                let idx = index as i32;
                                ui.upgrade_in_event_loop(move |ui| {
                                    ui.set_fxe_preset_index(idx);
                                    ui.set_fxe_builtin(builtin);
                                    ui.set_fxe_can_delete(!builtin);
                                    ui.set_fxe_name(display.into());
                                    ui.set_fxe_desc(desc.into());
                                    ui.set_fxe_status("".into());
                                }).ok();
                            }
                        }
                    }
                }
                Some(Cmd::FxeNew) => {
                    fxe_chain.clear();
                    fxe_pid.clear();
                    fxe_expanded = -1;
                    fxe_sync(&ui, &fxe_defs, &fxe_chain, fxe_expanded);
                    ui.upgrade_in_event_loop(|ui| {
                        ui.set_fxe_preset_index(-1);
                        ui.set_fxe_builtin(false);
                        ui.set_fxe_can_delete(false);
                        ui.set_fxe_name("".into());
                        ui.set_fxe_desc("".into());
                        ui.set_fxe_status("".into());
                    }).ok();
                }
                Some(Cmd::FxeAdd { index }) => {
                    if let Some(d) = fxe_defs.get(index) {
                        let t = d.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let mut params = serde_json::Map::new();
                        if let Some(list) = d.get("params").and_then(|p| p.as_array()) {
                            for pd in list {
                                if let (Some(n), Some(v)) =
                                    (pd.get("name").and_then(|v| v.as_str()), pd.get("default"))
                                {
                                    params.insert(n.to_string(), v.clone());
                                }
                            }
                        }
                        fxe_chain.push(serde_json::json!({"type": t, "enabled": true, "params": params}));
                        fxe_expanded = fxe_chain.len() as i32 - 1;
                        fxe_sync(&ui, &fxe_defs, &fxe_chain, fxe_expanded);
                    }
                }
                Some(Cmd::FxeRemove { index }) => {
                    if index < fxe_chain.len() {
                        fxe_chain.remove(index);
                        fxe_expanded = -1;
                        fxe_sync(&ui, &fxe_defs, &fxe_chain, fxe_expanded);
                    }
                }
                Some(Cmd::FxeToggle { index }) => {
                    if let Some(e) = fxe_chain.get_mut(index) {
                        let cur = e.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
                        e["enabled"] = serde_json::Value::Bool(!cur);
                        fxe_sync(&ui, &fxe_defs, &fxe_chain, fxe_expanded);
                    }
                }
                Some(Cmd::FxeMove { index, dir }) => {
                    let j = index as i32 + dir;
                    if index < fxe_chain.len() && j >= 0 && (j as usize) < fxe_chain.len() {
                        fxe_chain.swap(index, j as usize);
                        if fxe_expanded == index as i32 {
                            fxe_expanded = j;
                        } else if fxe_expanded == j {
                            fxe_expanded = index as i32;
                        }
                        fxe_sync(&ui, &fxe_defs, &fxe_chain, fxe_expanded);
                    }
                }
                Some(Cmd::FxeExpand { index }) => {
                    fxe_expanded = if fxe_expanded == index as i32 { -1 } else { index as i32 };
                    fxe_sync(&ui, &fxe_defs, &fxe_chain, fxe_expanded);
                }
                Some(Cmd::FxeParam { index, norm }) => {
                    if let Some(e) = fxe_chain.get_mut(usize::try_from(fxe_expanded).unwrap_or(usize::MAX)) {
                        let t = e.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let pd = fxe_defs
                            .iter()
                            .find(|d| d.get("id").and_then(|v| v.as_str()) == Some(t.as_str()))
                            .and_then(|d| d.get("params"))
                            .and_then(|p| p.as_array())
                            .and_then(|l| l.get(index))
                            .cloned();
                        if let Some(pd) = pd {
                            let name = pd.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                            let min = pd.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let max = pd.get("max").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            let step = pd.get("step").and_then(|v| v.as_f64()).unwrap_or(0.01);
                            let raw = min + norm as f64 * (max - min);
                            let snapped = ((raw / step).round() * step).clamp(min, max);
                            if !e.get("params").map(|p| p.is_object()).unwrap_or(false) {
                                e["params"] = serde_json::json!({});
                            }
                            e["params"][name.as_str()] = serde_json::json!(snapped);
                            // update the one param row in place — replacing the model
                            // mid-drag would tear down the slider under the pointer
                            let vt: SharedString = fx_fmt(snapped, step).into();
                            let nnorm = ((snapped - min) / (max - min)).clamp(0.0, 1.0) as f32;
                            ui.upgrade_in_event_loop(move |ui| {
                                let m = ui.get_fxe_params();
                                if let Some(vm) = m.as_any().downcast_ref::<VecModel<FxParamItem>>() {
                                    if let Some(mut row) = vm.row_data(index) {
                                        row.value_text = vt;
                                        row.norm = nnorm;
                                        vm.set_row_data(index, row);
                                    }
                                }
                            }).ok();
                        }
                    }
                }
                Some(Cmd::FxeSave { name, desc }) => {
                    if name.trim().is_empty() {
                        ui.upgrade_in_event_loop(|ui| ui.set_fxe_status("a name is required".into())).ok();
                    } else {
                        let chain_json = serde_json::to_string(&fxe_chain).unwrap_or_else(|_| "[]".into());
                        let saved = if fxe_pid.is_empty() {
                            proxy.create_effect_preset(name.trim(), &desc, &chain_json).await
                                .ok().filter(|s| !s.is_empty())
                        } else {
                            match proxy.update_effect_preset(&fxe_pid, name.trim(), &desc, &chain_json).await {
                                Ok(true) => Some(fxe_pid.clone()),
                                _ => None,
                            }
                        };
                        match saved {
                            Some(pid) => {
                                fxe_pid = pid.clone();
                                let r = refresh_effect_presets(&ui, &proxy).await;
                                effect_ids = r.0;
                                fxe_presets = r.1;
                                let idx = fxe_presets.iter().position(|(id, _)| *id == pid)
                                    .map(|i| i as i32).unwrap_or(-1);
                                let display = name.trim().to_string();
                                ui.upgrade_in_event_loop(move |ui| {
                                    ui.set_fxe_preset_index(idx);
                                    ui.set_fxe_builtin(false);
                                    ui.set_fxe_can_delete(true);
                                    ui.set_fxe_name(display.into());
                                    ui.set_fxe_status("saved ✓".into());
                                }).ok();
                            }
                            None => {
                                ui.upgrade_in_event_loop(|ui| {
                                    ui.set_fxe_status("couldn't save — duplicate name?".into());
                                }).ok();
                            }
                        }
                    }
                }
                Some(Cmd::FxeDelete) => {
                    if !fxe_pid.is_empty() {
                        let _ = proxy.delete_effect_preset(&fxe_pid).await;
                        fxe_pid.clear();
                        fxe_chain.clear();
                        fxe_expanded = -1;
                        let r = refresh_effect_presets(&ui, &proxy).await;
                        effect_ids = r.0;
                        fxe_presets = r.1;
                        fxe_sync(&ui, &fxe_defs, &fxe_chain, fxe_expanded);
                        ui.upgrade_in_event_loop(|ui| {
                            ui.set_fxe_preset_index(-1);
                            ui.set_fxe_can_delete(false);
                            ui.set_fxe_name("".into());
                            ui.set_fxe_desc("".into());
                            ui.set_fxe_status("preset deleted".into());
                        }).ok();
                    }
                }
                Some(Cmd::FxePreview { hid }) => {
                    let chain_json = serde_json::to_string(&fxe_chain).unwrap_or_else(|_| "[]".into());
                    let status = match proxy.preview_effects(&hid, &chain_json).await {
                        Ok(id) if id != 0 => "previewing…",
                        _ => "preview failed",
                    };
                    ui.upgrade_in_event_loop(move |ui| ui.set_fxe_status(status.into())).ok();
                }
                Some(Cmd::CvPickAvatar) => {
                    if let Some(handle) = rfd::AsyncFileDialog::new()
                        .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp"])
                        .pick_file()
                        .await
                    {
                        let path = handle.path().to_string_lossy().to_string();
                        // decode once, remember the real size, and hand the
                        // dialog a filtered ≤1200px preview (max zoom 4x on a
                        // 220px viewport needs 880px — stays sharp)
                        match image::open(&path) {
                            Ok(img) => {
                                let (fw, fh) = (img.width(), img.height());
                                let preview = if fw.max(fh) > 1200 {
                                    img.thumbnail(1200, 1200)
                                } else {
                                    img
                                };
                                let rgba = preview.to_rgba8();
                                let buf: RgbaBuf = (rgba.as_raw().clone(), rgba.width(), rgba.height());
                                ui.upgrade_in_event_loop(move |ui| {
                                    ui.set_crop_src(rgba_to_image(&buf));
                                    ui.set_crop_full_w(fw as i32);
                                    ui.set_crop_full_h(fh as i32);
                                    ui.set_crop_path(path.into());
                                    ui.set_crop_zoom(1.0);
                                    ui.set_crop_cx(0.5);
                                    ui.set_crop_cy(0.5);
                                    ui.set_crop_stage("mode".into());
                                    ui.set_crop_open(true);
                                }).ok();
                            }
                            Err(e) => tracing::error!("could not load image: {e}"),
                        }
                    }
                }
                Some(Cmd::CvStageAvatar { path, mode, sx, sy, sw, sh }) => {
                    cv_avatar = Some((path.clone(), mode.clone(), sx, sy, sw, sh));
                    let baked = bake_avatar_rgba(&mut avatar_cache, &path, sx, sy, sw, sh);
                    ui.upgrade_in_event_loop(move |ui| {
                        if let Some(b) = baked {
                            ui.set_cv_avatar(rgba_to_image(&b));
                            ui.set_cv_avatar_mode(mode.into());
                            ui.set_cv_has_avatar(true);
                        }
                    }).ok();
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
                                refresh_grid(&ui, &proxy, &mut avatar_cache).await;
                                refresh_voices_table(&ui, &proxy, &mut avatar_cache, &mut voices_all).await;
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
