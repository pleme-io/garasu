//! Build-time color-emoji embed.
//!
//! When the build environment sets `GARASU_EMOJI_FONT` to a readable
//! font file, we bake that font into the binary (the Nix build points
//! it at nixpkgs `noto-fonts-color-emoji`, so a Nix-built consumer
//! renders color emoji on EVERY runtime host — including a bare Linux
//! box or CI container with no system fonts at all). When unset (a
//! plain `cargo build`), the embed is compiled out and garasu falls
//! back to runtime discovery of a system emoji font; see
//! `text::ensure_emoji_fallback`.
fn main() {
    // Declare the cfg so `--cfg garasu_bundled_emoji` doesn't trip
    // rustc's unexpected-cfg lint.
    println!("cargo::rustc-check-cfg=cfg(garasu_bundled_emoji)");
    println!("cargo::rerun-if-env-changed=GARASU_EMOJI_FONT");

    if let Ok(path) = std::env::var("GARASU_EMOJI_FONT") {
        if !path.is_empty() && std::path::Path::new(&path).is_file() {
            // `text.rs` does `include_bytes!(env!("GARASU_EMOJI_FONT_PATH"))`
            // under this cfg.
            println!("cargo::rustc-cfg=garasu_bundled_emoji");
            println!("cargo::rustc-env=GARASU_EMOJI_FONT_PATH={path}");
            println!("cargo::rerun-if-changed={path}");
        }
    }
}
