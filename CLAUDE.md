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
| `text.rs` | `TextRenderer` — glyphon text layout, atlas, rasterization |
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
