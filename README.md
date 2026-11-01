# Garasu (硝子)

GPU rendering engine for pleme-io applications. Provides a reusable wgpu + winit + glyphon stack so every GPU-accelerated app shares the same rendering foundation.

## Components

| Module | Purpose |
|--------|---------|
| `context` | wgpu device, adapter, queue, surface lifecycle |
| `text` | glyphon text layout and rasterization |
| `shader` | WGSL post-processing pipeline (ghostty/tobira pattern) |
| `window` | winit window creation with platform defaults |
| `frame` | upload a CPU rgb8 frame as a texture, draw it, and check a read-back of what it drew |
| `canvas` | the `garasu-canvas/1` socket server and the `garasu-canvas` binary ([protocol](docs/canvas-protocol.md)) |
| `error` | Unified error type |

## Usage

```toml
[dependencies]
garasu = { git = "https://github.com/pleme-io/garasu" }
```

```rust
use garasu::{GpuContext, TextRenderer, ShaderPipeline, AppWindow};

let gpu = pollster::block_on(GpuContext::new())?;
let text = TextRenderer::new(&gpu.device, &gpu.queue, format);
```

## Build

```bash
cargo build
cargo test --lib
```

## Design

- Metal on macOS, Vulkan on Linux (via wgpu backends)
- Shader plugin system: built-in + custom `~/.config/{app}/shaders/*.wgsl`
- Uniforms: `time`, `resolution`, input texture + sampler
- Lock-free text rendering via glyphon
