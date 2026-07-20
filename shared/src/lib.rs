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

    /// Delete a reference sample.
    fn delete_sample(&self, sample_id: &str) -> zbus::Result<()>;

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

    #[zbus(signal)]
    fn speak_started(&self, gen_id: u32) -> zbus::Result<()>;

    #[zbus(signal)]
    fn speak_ended(&self, gen_id: u32) -> zbus::Result<()>;
}
