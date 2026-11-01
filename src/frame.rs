use crate::error::GarasuError;

pub const FRAME_SHADER: &str = r"
@group(0) @binding(0) var frame: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let pos = positions[idx];
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(frame));
    let texel = clamp(floor(in.uv * size), vec2<f32>(0.0), size - vec2<f32>(1.0));
    return vec4<f32>(textureLoad(frame, vec2<i32>(texel), 0).rgb, 1.0);
}
";

#[must_use]
pub fn rgb8_to_rgba8(rgb: &[u8]) -> Vec<u8> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .flat_map(|&[r, g, b]| [r, g, b, u8::MAX])
        .collect()
}

#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn fit(frame: (u32, u32), target: (u32, u32)) -> [f32; 4] {
    let (fw, fh) = (f64::from(frame.0.max(1)), f64::from(frame.1.max(1)));
    let (tw, th) = (f64::from(target.0), f64::from(target.1));
    let scale = (tw / fw).min(th / fh);
    let (w, h) = (fw * scale, fh * scale);
    [
        ((tw - w) / 2.0) as f32,
        ((th - h) / 2.0) as f32,
        w as f32,
        h as f32,
    ]
}

pub struct FramePipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

fn frame_texture(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("garasu_frame_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("garasu_frame_bg"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    (texture, bind_group)
}

impl FramePipeline {
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        target: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let format = if target.is_srgb() {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("garasu_frame_shader"),
            source: wgpu::ShaderSource::Wgsl(FRAME_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("garasu_frame_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("garasu_frame_pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("garasu_frame_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let (width, height) = (width.max(1), height.max(1));
        let (texture, bind_group) = frame_texture(device, &layout, format, width, height);
        Self {
            pipeline,
            layout,
            format,
            texture,
            bind_group,
            width,
            height,
        }
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn upload_rgb8(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgb: &[u8],
    ) -> Result<(), GarasuError> {
        let want = u64::from(width) * u64::from(height) * 3;
        if width == 0 || height == 0 || rgb.len() as u64 != want {
            return Err(GarasuError::Gpu(format!(
                "an rgb8 frame of {width}x{height} is {want} bytes, got {}",
                rgb.len()
            )));
        }
        if (width, height) != (self.width, self.height) {
            let (texture, bind_group) =
                frame_texture(device, &self.layout, self.format, width, height);
            self.texture = texture;
            self.bind_group = bind_group;
            self.width = width;
            self.height = height;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgb8_to_rgba8(rgb),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_rgb_frame_widens_to_opaque_rgba_pixel_by_pixel() {
        assert_eq!(
            rgb8_to_rgba8(&[1, 2, 3, 4, 5, 6]),
            vec![1, 2, 3, 255, 4, 5, 6, 255]
        );
        assert_eq!(rgb8_to_rgba8(&[]), Vec::<u8>::new());
    }

    #[test]
    fn a_frame_fits_its_target_keeping_its_aspect_and_centred() {
        assert_eq!(fit((3, 2), (3, 2)), [0.0, 0.0, 3.0, 2.0]);
        assert_eq!(fit((3, 2), (6, 4)), [0.0, 0.0, 6.0, 4.0]);
        assert_eq!(fit((2, 2), (6, 4)), [1.0, 0.0, 4.0, 4.0]);
        assert_eq!(fit((4, 1), (4, 4)), [0.0, 1.5, 4.0, 1.0]);
    }

    #[cfg(feature = "gpu_tests")]
    #[test]
    fn a_frame_drawn_through_the_pipeline_lands_on_the_target_pixel_for_pixel() {
        use crate::GpuContext;
        use crate::headless::HeadlessTarget;

        let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let rgb = vec![
            255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 18, 52, 86, 1, 2, 3,
        ];
        let mut frame = FramePipeline::new(&gpu.device, format, 1, 1);
        frame
            .upload_rgb8(&gpu.device, &gpu.queue, 3, 2, &rgb)
            .unwrap();
        for scale in [1u32, 2] {
            let (w, h) = (3 * scale, 2 * scale);
            let target = HeadlessTarget::new(&gpu, w, h, format);
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("frame_test_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target.view(),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                let [x, y, vw, vh] = fit((3, 2), (w, h));
                pass.set_viewport(x, y, vw, vh, 0.0, 1.0);
                frame.draw(&mut pass);
            }
            gpu.queue.submit(std::iter::once(encoder.finish()));
            let pixels = target.read_pixels_rgba8(&gpu);
            let want: Vec<u8> = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x / scale, y / scale)))
                .flat_map(|(x, y)| {
                    let at = ((y * 3 + x) * 3) as usize;
                    [rgb[at], rgb[at + 1], rgb[at + 2], 255]
                })
                .collect();
            assert_eq!(pixels, want, "scale {scale}");
        }
    }
}
