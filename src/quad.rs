//! Quad pipeline — instanced solid-fill colored rectangles.
//!
//! A fleet primitive: every pleme-io GPU consumer that needs to paint
//! flat colored boxes (terminal cell backgrounds, browser block-box
//! backgrounds, UI panels, selection highlights) shares ONE instanced
//! quad pipeline instead of re-rolling a per-app rect shader.
//!
//! This is the **solid-fill subset** of mado's proven `RectPipeline`
//! (mado/src/render.rs) lifted into garasu per the PRIME DIRECTIVE:
//! mado's pipeline additionally carries a fragment-selector `mode` +
//! `pattern` payload for corner-radius / box-drawing / RLE-decoration
//! work, which is terminal-specific. Garasu's [`QuadInstance`] drops
//! those fields and keeps the load-bearing primitive — pos / size /
//! color — so a consumer that only needs solid fills (a browser
//! renderer, say) pays for nothing more.
//!
//! ## Design
//!
//! - One unit quad (six vertices, two triangles) drawn instanced; each
//!   [`QuadInstance`] expands the unit quad to its `pos` + `size` in
//!   screen-space pixels and tints it flat `color`.
//! - Screen-space → clip via a resolution uniform (`update_resolution`),
//!   the same status as garasu's existing [`crate::BLUR_SHADER`] const:
//!   the WGSL is compiled-in data, not a `format!()`-composed string.
//! - The instance buffer **grows on demand** (starting at 4096) and is
//!   never a fixed cap that silently drops draws — a 1080p page can have
//!   thousands of block boxes.
//!
//! ## Example
//!
//! ```no_run
//! use garasu::{GpuContext, QuadInstance, QuadPipeline};
//!
//! let gpu = pollster::block_on(GpuContext::new()).unwrap();
//! let format = wgpu::TextureFormat::Bgra8UnormSrgb;
//! let mut quad = QuadPipeline::new(&gpu.device, format);
//! quad.update_resolution(&gpu.queue, 800, 600);
//!
//! let instances = [QuadInstance {
//!     pos: [10.0, 10.0],
//!     size: [100.0, 40.0],
//!     color: [0.19, 0.31, 1.0, 1.0],
//! }];
//! // inside a render pass:
//! // quad.draw(&mut pass, &instances);
//! ```

/// One instanced solid-fill rectangle in screen-space pixels.
///
/// `pos` is the top-left corner (px), `size` is `[width, height]` (px),
/// `color` is linear RGBA in `[0, 1]`. The pipeline blends with
/// `ALPHA_BLENDING`, so a `color[3] < 1.0` composites over what is
/// already there.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuadInstance {
    /// Top-left corner in screen-space pixels.
    pub pos: [f32; 2],
    /// `[width, height]` in screen-space pixels.
    pub size: [f32; 2],
    /// Flat fill color, linear RGBA in `[0, 1]`.
    pub color: [f32; 4],
}

/// Resolution uniform — screen size in pixels, used by the vertex
/// shader to map pixel-space corners to clip-space NDC.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadUniforms {
    resolution: [f32; 2],
    _padding: [f32; 2],
}

/// Instanced solid-fill quad shader. Compiled-in WGSL data (the same
/// status as [`crate::BLUR_SHADER`]): a unit quad expanded per-instance
/// to `pos`/`size`, tinted flat `color`, screen-space → clip via the
/// resolution uniform.
const QUAD_SHADER: &str = r"
struct ScreenUniforms {
    resolution: vec2<f32>,
    _padding: vec2<f32>,
};

struct QuadInstance {
    @location(0) pos: vec2<f32>,
    @location(1) size: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> screen: ScreenUniforms;

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    instance: QuadInstance,
) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let pixel = instance.pos + corners[vi] * instance.size;
    let ndc = vec2<f32>(
        (pixel.x / screen.resolution.x) * 2.0 - 1.0,
        1.0 - (pixel.y / screen.resolution.y) * 2.0,
    );
    var out: VertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.color = instance.color;
    return out;
}

@fragment
fn fs_main(frag: VertexOutput) -> @location(0) vec4<f32> {
    return frag.color;
}
";

/// Initial instance-buffer capacity. Grows by `next_power_of_two` when a
/// draw needs more — never a fixed cap that drops instances.
const INITIAL_CAPACITY: usize = 4096;

/// Instanced solid-fill quad pipeline. Holds the render pipeline, the
/// resolution uniform, and a growable instance buffer.
pub struct QuadPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
}

impl QuadPipeline {
    /// Build the pipeline for a given surface/texture `format`. The
    /// pipeline blends with `ALPHA_BLENDING` so translucent quads
    /// composite over existing content.
    #[must_use]
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("garasu_quad_shader"),
            source: wgpu::ShaderSource::Wgsl(QUAD_SHADER.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("garasu_quad_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("garasu_quad_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 2,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("garasu_quad_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[instance_layout],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("garasu_quad_uniforms"),
            size: std::mem::size_of::<QuadUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("garasu_quad_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("garasu_quad_instances"),
            size: (INITIAL_CAPACITY * std::mem::size_of::<QuadInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            instance_buffer,
            instance_capacity: INITIAL_CAPACITY,
        }
    }

    /// Update the resolution uniform — call on resize (and at least once
    /// before the first `draw`). `width`/`height` are the target surface
    /// dimensions in pixels.
    pub fn update_resolution(&self, queue: &wgpu::Queue, width: u32, height: u32) {
        let uniforms = QuadUniforms {
            resolution: [width as f32, height as f32],
            _padding: [0.0; 2],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Grow the instance buffer to fit at least `count` instances. Grows
    /// to `count.next_power_of_two()` so repeated small growths amortize.
    fn ensure_capacity(&mut self, device: &wgpu::Device, count: usize) {
        if count > self.instance_capacity {
            let new_cap = count.next_power_of_two();
            self.instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("garasu_quad_instances"),
                size: (new_cap * std::mem::size_of::<QuadInstance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }
    }

    /// Draw `instances` as solid-fill quads in one instanced draw call.
    ///
    /// Writes the instance data into the (grow-on-demand) instance buffer
    /// then issues a single `draw(0..6, 0..n)`. A `draw` of zero
    /// instances is a no-op.
    ///
    /// Requires `&mut wgpu::Device` access to grow the buffer when the
    /// instance count exceeds the current capacity — pass the device and
    /// queue used to build the pipeline.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        instances: &[QuadInstance],
    ) {
        if instances.is_empty() {
            return;
        }
        self.ensure_capacity(device, instances.len());
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instance_buffer.slice(..));
        pass.draw(0..6, 0..instances.len() as u32);
    }

    /// Current instance-buffer capacity (number of instances). Grows
    /// on demand; exposed for tests + diagnostics.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.instance_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Pod / field-layout tests — pure data, no GPU. Mirror garasu's
    // TextConfig test style.
    // ---------------------------------------------------------------

    #[test]
    fn quad_instance_field_layout_is_stable() {
        // 2 + 2 + 4 f32 = 8 floats = 32 bytes, tightly packed.
        assert_eq!(std::mem::size_of::<QuadInstance>(), 32);
        assert_eq!(std::mem::align_of::<QuadInstance>(), 4);
        // Field offsets must match the WGSL VertexAttribute offsets
        // (pos@0, size@8, color@16).
        let q = QuadInstance {
            pos: [1.0, 2.0],
            size: [3.0, 4.0],
            color: [5.0, 6.0, 7.0, 8.0],
        };
        let bytes: &[u8] = bytemuck::bytes_of(&q);
        assert_eq!(bytes.len(), 32);
        // pos.x at byte 0
        assert_eq!(&bytes[0..4], 1.0_f32.to_ne_bytes());
        // size.x at byte 8
        assert_eq!(&bytes[8..12], 3.0_f32.to_ne_bytes());
        // color.x at byte 16
        assert_eq!(&bytes[16..20], 5.0_f32.to_ne_bytes());
    }

    #[test]
    fn quad_instance_is_pod_castable_as_slice() {
        let instances = [
            QuadInstance {
                pos: [0.0, 0.0],
                size: [10.0, 10.0],
                color: [1.0, 0.0, 0.0, 1.0],
            },
            QuadInstance {
                pos: [10.0, 10.0],
                size: [20.0, 20.0],
                color: [0.0, 1.0, 0.0, 1.0],
            },
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn quad_instance_clone_copy_eq() {
        let a = QuadInstance {
            pos: [1.0, 2.0],
            size: [3.0, 4.0],
            color: [0.1, 0.2, 0.3, 0.4],
        };
        let b = a; // Copy
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn quad_instance_debug_contains_fields() {
        let q = QuadInstance {
            pos: [1.0, 2.0],
            size: [3.0, 4.0],
            color: [0.1, 0.2, 0.3, 0.4],
        };
        let dbg = format!("{q:?}");
        assert!(dbg.contains("pos"));
        assert!(dbg.contains("size"));
        assert!(dbg.contains("color"));
    }

    #[test]
    fn quad_uniforms_layout_matches_shader() {
        // resolution: vec2<f32> + _padding: vec2<f32> = 16 bytes,
        // matching the WGSL ScreenUniforms struct.
        assert_eq!(std::mem::size_of::<QuadUniforms>(), 16);
    }

    #[test]
    fn quad_shader_constant_is_nonempty_wgsl() {
        assert!(!QUAD_SHADER.is_empty());
        assert!(QUAD_SHADER.contains("@vertex"));
        assert!(QUAD_SHADER.contains("@fragment"));
        assert!(QUAD_SHADER.contains("@group(0)"));
        assert!(QUAD_SHADER.contains("@binding(0)"));
        assert!(QUAD_SHADER.contains("vs_main"));
        assert!(QUAD_SHADER.contains("fs_main"));
        // The three instance attributes must be declared at the
        // matching shader_location indices.
        assert!(QUAD_SHADER.contains("@location(0) pos"));
        assert!(QUAD_SHADER.contains("@location(1) size"));
        assert!(QUAD_SHADER.contains("@location(2) color"));
    }

    // ---------------------------------------------------------------
    // GPU-driven golden test — paint a red rect, assert center pixel
    // is red. Gated behind `gpu_tests` like garasu's headless tests
    // so CI runners without a real adapter don't mis-fail.
    // ---------------------------------------------------------------

    #[cfg(feature = "gpu_tests")]
    #[test]
    fn quad_pipeline_paints_red_rect_center_pixel_is_red() {
        use crate::headless::{pixel_at, HeadlessTarget};
        use crate::GpuContext;

        let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (w, h) = (64u32, 64u32);
        let target = HeadlessTarget::new(&gpu, w, h, format);

        let mut quad = QuadPipeline::new(&gpu.device, format);
        quad.update_resolution(&gpu.queue, w, h);

        // A red rect covering the center: [16,16] .. [48,48].
        let instances = [QuadInstance {
            pos: [16.0, 16.0],
            size: [32.0, 32.0],
            color: [1.0, 0.0, 0.0, 1.0],
        }];

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quad_test_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            quad.draw(&gpu.device, &gpu.queue, &mut pass, &instances);
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let pixels = target.read_pixels_rgba8(&gpu);
        // Center pixel (32, 32) is inside the red rect.
        let px = pixel_at(&pixels, w, 32, 32);
        assert!(px[0] > 200, "center R should be high, got {px:?}");
        assert!(px[1] < 60, "center G should be low, got {px:?}");
        assert!(px[2] < 60, "center B should be low, got {px:?}");
        // A corner pixel (2, 2) is outside the rect → still black.
        let corner = pixel_at(&pixels, w, 2, 2);
        assert!(corner[0] < 60, "corner R should be low, got {corner:?}");
    }
}
