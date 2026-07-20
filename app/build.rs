fn main() {
    // Compile the Slint UI into generated Rust (the `AppWindow` type).
    slint_build::compile("ui/main.slint").expect("failed to compile Slint UI");
}
