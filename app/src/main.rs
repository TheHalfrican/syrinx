//! Syrinx main window.
//!
//! Loads the Slint UI and wires its callbacks to the engine over D-Bus.
//! Engine calls are stubbed for now — see `TODO(syrinx)`.

// Pulls in the `AppWindow` type generated from `ui/main.slint`.
slint::include_modules!();

use slint::{ComponentHandle, ModelRc, VecModel, SharedString};
use std::rc::Rc;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let ui = AppWindow::new()?;

    // Seed the voice list. TODO(syrinx): populate from Engine::list_voices()
    // over D-Bus (see `syrinx_shared::EngineProxy`).
    let voices = Rc::new(VecModel::from(vec![
        SharedString::from("Default"),
        SharedString::from("Narrator"),
    ]));
    ui.set_voices(ModelRc::from(voices));
    ui.set_backend("cpu".into()); // TODO(syrinx): read Engine `Backend` property

    // Speak pressed.
    {
        let ui_weak = ui.as_weak();
        ui.on_speak(move |text, voice| {
            let ui = ui_weak.unwrap();
            tracing::info!(%voice, "speak: {}", text);
            ui.set_speaking(true);
            // TODO(syrinx): spawn async task ->
            //   EngineProxy::speak(&text, &voice), then subscribe to
            //   AudioLevel/SpeakEnded signals to drive `level` + reset `speaking`.
        });
    }

    // Stop pressed.
    {
        let ui_weak = ui.as_weak();
        ui.on_stop(move || {
            let ui = ui_weak.unwrap();
            ui.set_speaking(false);
            // TODO(syrinx): EngineProxy::cancel(gen_id)
        });
    }

    ui.run()?;
    Ok(())
}
