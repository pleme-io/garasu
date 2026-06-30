# Garasu (硝子) — GPU Rendering Engine

> **★★★ CSE / Knowable Construction.** This repo operates under **Constructive Substrate Engineering** — canonical specification at [`pleme-io/theory/CONSTRUCTIVE-SUBSTRATE-ENGINEERING.md`](https://github.com/pleme-io/theory/blob/main/CONSTRUCTIVE-SUBSTRATE-ENGINEERING.md). The Compounding Directive (operational rules: solve once, load-bearing fixes only, idiom-first, models stay current, direction beats velocity) is in the org-level pleme-io/CLAUDE.md ★★★ section. Read both before non-trivial changes.


## Build & Test

```bash
cargo build
cargo test --lib
```

## Architecture

Reusable GPU rendering library for all pleme-io graphical applications.

### Modules

| Module | Purpose |
|--------|---------|
| `context.rs` | `GpuContext` — wgpu instance, adapter, device, queue, surface config |
| `text.rs` | `TextRenderer` — single-pass glyphon text (prefer `layers::TextLayerStack` for multi-pass) |
| `layers.rs` | `TextLayerStack` — per-layer-isolated multi-pass text: N text surfaces (terminal + overlays) each own their own vertex buffer + `Viewport`, sharing ONE atlas/font-system; two-phase `Frame` guard, trim-once-per-frame |
| `shader.rs` | `ShaderPipeline` — WGSL post-processing chain (texture → shader → output) |
| `window.rs` | `AppWindow` — winit window creation helpers |
| `error.rs` | `GarasuError` — unified error enum |

### Shader Plugin API

Post-processing shaders receive:
- `input_texture` (binding 0) — rendered UI as texture
- `input_sampler` (binding 1) — texture sampler
- `uniforms` (binding 2) — `{ time: f32, _pad: f32, resolution: vec2<f32> }`

Custom shaders loaded from `~/.config/{app}/shaders/*.wgsl`.

### Cross-Platform

- Metal on macOS, Vulkan on Linux (wgpu auto-selects)
- winit for window management (cross-platform)
- glyphon for text (cross-platform)

### Consumers

Used by: mado, hibiki, kagi, kekkai, tobira (planned migration)

## Design Decisions

- **Library, not framework**: consumers own the event loop and render pass
- **No async runtime**: uses `pollster` for one-shot GPU init, consumers bring their own async
- **Edition 2024**: requires Rust 1.89.0+
- **`TextLayerStack` eliminates the shared text vertex buffer (eliminate-the-shared-cell)**: one `glyphon::TextRenderer` owns ONE vertex buffer, so two `prepare→render` passes in a frame (terminal grid + an overlay) before a single `submit` make the first draw read the second's glyphs — the top-left-blank class of bug. `TextLayerStack` gives each text surface its OWN renderer (own buffer) + OWN `Viewport`, sharing one `TextAtlas`/`FontSystem` (so the ~180 ms font scan + atlas memory happen once). Two-phase `Frame` (all `prepare`s before all `render`s within the frame, `Drop` trims once) keeps an atlas grow from stranding a recorded pass. `Frame::render` KEEPS `Result<(), RenderError>` — `RemovedFromAtlas` is a real shared-atlas hazard, never swallowed. `font_system` MUST route through `text::take_or_build_font_system` (the emoji-fallback + Nix-font-dir + preload-cache contract). `TextRenderer` stays for single-pass consumers (back-compat `prepare`/`render` on `TextLayerStack` cover the migration window).
