# Garasu (硝子) — GPU Rendering Engine

> **★★★ CSE / Knowable Construction.** This repo operates under **Constructive Substrate Engineering**. The Compounding Directive (operational rules: solve once, load-bearing fixes only, idiom-first, models stay current, direction beats velocity) is in the org-level pleme-io/CLAUDE.md ★★★ section. Read it before non-trivial changes.


## Build & Test

```bash
cargo build
cargo test --lib
cargo test --test canvas --test canvas_binary
cargo test --lib --features gpu_tests   # needs a GPU adapter
cargo test --features gpu_tests --test canvas_binary   # opens a real window; skips with no display
cargo run --bin garasu-canvas -- --headless --socket /tmp/c.sock
cargo run --bin garasu-canvas -- --verify --socket /tmp/c.sock   # reads each presented surface back
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
| `frame.rs` | `FramePipeline` — uploads a CPU rgb8 frame as a texture and draws it, pixel-exact, aspect kept (`fit`); `Readback` — a texture it drew into, read back and compared with the frame (`verify`) |
| `headless.rs` | `HeadlessTarget` / `HeadlessHarness` / `HeadlessSwapchain` — offscreen targets for GPU tests; `read_texture` — the one copy-and-map readback, for offscreen and surface textures |
| `canvas/` | the `garasu-canvas/1` server (Unix only): `Server`, the `Sink` trait, `HeadlessCanvases`, `window::run` (`Options { verify }` reads each presented surface back); binary `garasu-canvas`. Protocol: [`docs/canvas-protocol.md`](docs/canvas-protocol.md) |
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

Used by: mado, hibiki, kagi, kekkai, tobira (planned migration). blue's `display` capability reaches garasu through the `garasu-canvas` binary over a socket, never by linking it.

## Design Decisions

- **Library, not framework**: consumers own the event loop and render pass
- **No async runtime**: uses `pollster` for one-shot GPU init, consumers bring their own async
- **Edition 2024**: requires Rust 1.89.0+
- **`TextLayerStack` eliminates the shared text vertex buffer (eliminate-the-shared-cell)**: one `glyphon::TextRenderer` owns ONE vertex buffer, so two `prepare→render` passes in a frame (terminal grid + an overlay) before a single `submit` make the first draw read the second's glyphs — the top-left-blank class of bug. `TextLayerStack` gives each text surface its OWN renderer (own buffer) + OWN `Viewport`, sharing one `TextAtlas`/`FontSystem` (so the ~180 ms font scan + atlas memory happen once). Two-phase `Frame` (all `prepare`s before all `render`s within the frame, `Drop` trims once) keeps an atlas grow from stranding a recorded pass. `Frame::render` KEEPS `Result<(), RenderError>` — `RemovedFromAtlas` is a real shared-atlas hazard, never swallowed. `font_system` MUST route through `text::take_or_build_font_system` (the emoji-fallback + Nix-font-dir + preload-cache contract). `TextRenderer` stays for single-pass consumers (back-compat `prepare`/`render` on `TextLayerStack` cover the migration window).
