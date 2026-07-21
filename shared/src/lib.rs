//! Shared types and the D-Bus client proxy for the Syrinx engine.
//!
//! Both `syrinx-app` and `syrinx-dictate` talk to the Python engine through
//! this crate, so the interface lives in exactly one place.

use serde::{Deserialize, Serialize};

/// A voice the engine can speak with (built-in or a cloned profile).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    pub id: String,
    pub display_name: String,
}

/// Which compute backend the engine loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    Cuda,
    Rocm,
    Cpu,
}

/// Async proxy for `sh.syrinx.Engine1` on the session bus.
///
/// `zbus` generates the method/signal/property plumbing from this trait.
#[zbus::proxy(
    interface = "sh.syrinx.Engine1",
    default_service = "sh.syrinx.Engine",
    default_path = "/sh/syrinx/Engine"
)]
pub trait Engine {
    /// Synthesize `text` in `voice_id`; returns a generation id.
    fn speak(&self, text: &str, voice_id: &str) -> zbus::Result<u32>;

    /// Transcribe an audio file (dictation). Returns recognized text.
    fn transcribe(&self, audio_path: &str) -> zbus::Result<String>;

    /// List available voices as (id, display_name) pairs.
    fn list_voices(&self) -> zbus::Result<Vec<(String, String)>>;

    /// Clone a voice from a sample file (+ its transcript); returns the new
    /// profile id. `ref_text` is required by the Qwen cloning backend.
    fn clone_voice(&self, name: &str, sample_path: &str, ref_text: &str) -> zbus::Result<String>;

    /// Create a voice profile from a JSON spec; returns the profile id.
    fn create_profile(&self, spec_json: &str) -> zbus::Result<String>;

    /// List voice profiles as a JSON array of summaries.
    fn list_profiles(&self) -> zbus::Result<String>;

    /// Full profile as JSON (empty string if not found).
    fn get_profile(&self, profile_id: &str) -> zbus::Result<String>;

    /// Apply a JSON patch of editable fields (name/description/language/personality/default_engine).
    fn update_profile(&self, profile_id: &str, patch_json: &str) -> zbus::Result<()>;

    /// Delete a profile and its samples.
    fn delete_profile(&self, profile_id: &str) -> zbus::Result<()>;

    /// Add a reference sample (an audio file path) to a cloned profile. An empty
    /// `reference_text` auto-transcribes via whisper. Returns JSON
    /// {sample_id, reference_text}.
    fn add_sample(&self, profile_id: &str, audio_path: &str, reference_text: &str) -> zbus::Result<String>;

    /// Compose an in-character utterance for a voice, guided by `prompt` (empty =
    /// unprompted). Returns a request id (0 if no personality); result via `LlmResult`.
    fn compose_profile(&self, voice_id: &str, prompt: &str) -> zbus::Result<u32>;

    /// Rewrite `text` in the voice's personality; returns a request id (0 if no
    /// personality). The result arrives via `LlmResult`.
    fn rewrite_profile(&self, voice_id: &str, text: &str) -> zbus::Result<u32>;

    /// Delete a reference sample.
    fn delete_sample(&self, sample_id: &str) -> zbus::Result<()>;

    // --- generation history (persisted clips) ---------------------------

    /// List saved generations as a JSON array (newest first).
    fn list_history(&self) -> zbus::Result<String>;

    /// Replay a stored clip; returns a generation id (0 if not found).
    fn play_history(&self, hid: &str) -> zbus::Result<u32>;

    /// Replay a stored clip starting at `pct` (0..1); returns a generation id.
    fn play_history_at(&self, hid: &str, pct: f64) -> zbus::Result<u32>;

    /// Pause the current playback.
    fn pause_playback(&self) -> zbus::Result<()>;

    /// Resume the current playback.
    fn resume_playback(&self) -> zbus::Result<()>;

    /// Seek the current playback to `pct` (0..1).
    fn seek_playback(&self, pct: f64) -> zbus::Result<()>;

    /// Star/unstar a history entry.
    fn star_history(&self, hid: &str, starred: bool) -> zbus::Result<()>;

    /// Delete a history entry (row + audio file).
    fn delete_history(&self, hid: &str) -> zbus::Result<()>;

    /// Re-synthesize a history entry's text/voice as a new clip; returns a gen id.
    fn regenerate_history(&self, hid: &str) -> zbus::Result<u32>;

    /// Write a `.zip` package (manifest.json + audio) for a history entry to `dest`.
    fn export_package(&self, hid: &str, dest: &str) -> zbus::Result<()>;

    /// Absolute WAV path of a history entry (for the app to copy on export-audio).
    fn history_audio_path(&self, hid: &str) -> zbus::Result<String>;

    /// Cancel an in-flight generation.
    fn cancel(&self, gen_id: u32) -> zbus::Result<()>;

    #[zbus(property)]
    fn model_loaded(&self) -> zbus::Result<bool>;

    #[zbus(property)]
    fn backend(&self) -> zbus::Result<String>;

    #[zbus(signal)]
    fn generation_progress(&self, gen_id: u32, state: String, pct: f64) -> zbus::Result<()>;

    #[zbus(signal)]
    fn audio_level(&self, gen_id: u32, rms: f64) -> zbus::Result<()>;

    /// Emitted when playback of a clip starts: id, title, seconds, JSON waveform bars.
    #[zbus(signal)]
    fn playback_info(
        &self,
        gen_id: u32,
        clip_id: String,
        title: String,
        duration: f64,
        bars: String,
    ) -> zbus::Result<()>;

    /// Playback position (0..1), emitted per audio block.
    #[zbus(signal)]
    fn playback_progress(&self, gen_id: u32, pct: f64) -> zbus::Result<()>;

    /// Result of a Compose/Rewrite request (empty text = failed / no personality).
    #[zbus(signal)]
    fn llm_result(&self, req_id: u32, text: String) -> zbus::Result<()>;

    #[zbus(signal)]
    fn speak_started(&self, gen_id: u32) -> zbus::Result<()>;

    #[zbus(signal)]
    fn speak_ended(&self, gen_id: u32) -> zbus::Result<()>;
}
