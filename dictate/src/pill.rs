//! syrinx-dictate-pill — the recording overlay.
//!
//! A wlr-layer-shell surface (like waybar/wofi) shown while dictation is
//! recording. Spawned by `syrinx-dictate start`, killed by `syrinx-dictate
//! stop`. It renders natively via GTK4 — no WebView — anchored bottom-center on
//! Hyprland, non-focusable, with a pulsing "● Recording…" pill.

use std::time::Duration;

use gtk4::prelude::*;
use gtk4::{glib, Application, ApplicationWindow, Box as GtkBox, CssProvider, Label, Orientation};
use gtk4_layer_shell::{Edge, KeyboardMode, Layer, LayerShell};

const CSS: &str = "
window { background: transparent; }
.pill {
    background: rgba(20, 19, 26, 0.92);
    border-radius: 22px;
    padding: 10px 20px;
    border: 1px solid rgba(124, 92, 255, 0.5);
}
.dot   { color: #ff5c5c; font-size: 16px; }
.label { color: #efeaff; font-size: 14px; font-weight: bold; }
";

fn main() {
    // NON_UNIQUE: each spawn is its own process, so `syrinx-dictate stop` can
    // always dismiss the pill by killing the PID it recorded.
    let app = Application::builder()
        .application_id("sh.syrinx.dictate.pill")
        .flags(gtk4::gio::ApplicationFlags::NON_UNIQUE)
        .build();
    app.connect_activate(build_ui);
    // Ignore process args (we're spawned with none).
    app.run_with_args::<&str>(&[]);
}

fn build_ui(app: &Application) {
    let window = ApplicationWindow::new(app);

    // --- layer-shell: the native Hyprland way to float an overlay ---
    window.init_layer_shell();
    window.set_layer(Layer::Overlay);
    window.set_anchor(Edge::Bottom, true);
    window.set_margin(Edge::Bottom, 64);
    window.set_keyboard_mode(KeyboardMode::None); // never steal focus
    window.set_namespace(Some("syrinx-dictate"));

    // --- content ---
    let row = GtkBox::new(Orientation::Horizontal, 10);
    row.add_css_class("pill");
    let dot = Label::new(Some("●"));
    dot.add_css_class("dot");
    let text = Label::new(Some("Recording…"));
    text.add_css_class("label");
    row.append(&dot);
    row.append(&text);
    window.set_child(Some(&row));

    // --- style ---
    let css = CssProvider::new();
    css.load_from_data(CSS);
    gtk4::style_context_add_provider_for_display(
        &gtk4::gdk::Display::default().expect("no display"),
        &css,
        gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
    );

    // --- pulse the dot ---
    let dot_weak = dot.downgrade();
    glib::timeout_add_local(Duration::from_millis(500), move || {
        let Some(dot) = dot_weak.upgrade() else {
            return glib::ControlFlow::Break;
        };
        dot.set_opacity(if dot.opacity() > 0.5 { 0.25 } else { 1.0 });
        glib::ControlFlow::Continue
    });

    window.present();
}
