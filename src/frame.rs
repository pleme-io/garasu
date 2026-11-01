use crate::context::GpuContext;
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

fn rgb8_frame(width: u32, height: u32, rgb: &[u8]) -> Result<(), GarasuError> {
    let want = u64::from(width) * u64::from(height) * 3;
    if width == 0 || height == 0 || rgb.len() as u64 != want {
        return Err(GarasuError::Gpu(format!(
            "an rgb8 frame of {width}x{height} is {want} bytes, got {}",
            rgb.len()
        )));
    }
    Ok(())
}

fn rgb8_at(rgb: &[u8], width: u32, x: u32, y: u32) -> [u8; 3] {
    let at = (y as usize * width as usize + x as usize) * 3;
    [rgb[at], rgb[at + 1], rgb[at + 2]]
}

const SLACK: f64 = 1.0 / 64.0;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sampled(p: u32, origin: f32, extent: f32, side: u32) -> Vec<Option<u32>> {
    let (start, end) = (f64::from(p), f64::from(p) + 1.0);
    let (first, last) = (f64::from(origin), f64::from(origin) + f64::from(extent));
    if end <= first || start >= last {
        return vec![None];
    }
    let texel = |c: f64| {
        let s = (c - first) / f64::from(extent) * f64::from(side);
        Some(s.clamp(0.0, f64::from(side - 1)) as u32)
    };
    let centre = start + 0.5;
    let primary = (first..last)
        .contains(&centre)
        .then(|| texel(centre))
        .flatten();
    let mut seen = vec![primary, texel(centre - SLACK), texel(centre + SLACK)];
    if start < first || end > last {
        seen.extend([texel(centre), None]);
    }
    seen
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn footprint_centre(t: u32, origin: f32, extent: f32, side: u32, pixels: u32) -> u32 {
    let at = f64::from(origin) + (f64::from(t) + 0.5) * f64::from(extent) / f64::from(side);
    (at.max(0.0) as u32).min(pixels - 1)
}

/// The first pixel of a [`Readback`] that is not what
/// [`FramePipeline`] draws for the frame: `(x, y)` in the read-back
/// texture, `want` the frame's colour there (black outside the
/// fitted frame), `got` the colour read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mismatch {
    pub x: u32,
    pub y: u32,
    pub want: [u8; 3],
    pub got: [u8; 3],
}

/// What [`Readback::verify`] found. `blake3` hashes the frame as read
/// back at its own size (one pixel from the middle of each frame
/// pixel's footprint), so it equals the hash of the frame's rgb8
/// bytes when the texture shows the frame at a scale of 1 or more.
/// `surface` hashes the whole read-back texture as rgb8.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    pub blake3: blake3::Hash,
    pub surface: blake3::Hash,
    pub mismatch: Option<Mismatch>,
}

/// A texture [`FramePipeline`] drew into, read back as rgb8, top row
/// first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Readback {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    rgb: Vec<u8>,
}

impl Readback {
    /// Read `texture` back from the GPU. It needs `COPY_SRC` and an
    /// 8-bit `Rgba8` or `Bgra8` format.
    pub fn read(gpu: &GpuContext, texture: &wgpu::Texture) -> Result<Readback, GarasuError> {
        let size = texture.size();
        let texels = crate::headless::read_texture(gpu, texture)?;
        Readback::from_texels(size.width, size.height, texture.format(), &texels)
    }

    /// Build a readback from tightly packed 4-byte texels in
    /// `format`'s channel order.
    pub fn from_texels(
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        texels: &[u8],
    ) -> Result<Readback, GarasuError> {
        use wgpu::TextureFormat::{Bgra8Unorm, Bgra8UnormSrgb, Rgba8Unorm, Rgba8UnormSrgb};
        let [r, g, b] = match format {
            Rgba8Unorm | Rgba8UnormSrgb => [0, 1, 2],
            Bgra8Unorm | Bgra8UnormSrgb => [2, 1, 0],
            other => {
                return Err(GarasuError::Gpu(format!(
                    "a readback is Rgba8 or Bgra8, got {other:?}"
                )));
            }
        };
        let want = u64::from(width) * u64::from(height) * 4;
        if width == 0 || height == 0 || texels.len() as u64 != want {
            return Err(GarasuError::Gpu(format!(
                "a {width}x{height} readback is {want} bytes, got {}",
                texels.len()
            )));
        }
        let rgb = texels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[r], p[g], p[b]])
            .collect();
        Ok(Readback {
            width,
            height,
            format,
            rgb,
        })
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    #[must_use]
    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }

    /// The read-back pixel at `(x, y)`. Panics outside the texture.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        rgb8_at(&self.rgb, self.width, x, y)
    }

    /// The frame as read back at its own size: for each frame pixel,
    /// the read-back pixel in the middle of its footprint under
    /// [`fit`].
    #[must_use]
    pub fn frame(&self, width: u32, height: u32) -> Vec<u8> {
        let [vx, vy, vw, vh] = fit((width, height), (self.width, self.height));
        (0..height)
            .flat_map(|fy| (0..width).map(move |fx| (fx, fy)))
            .flat_map(|(fx, fy)| {
                self.pixel(
                    footprint_centre(fx, vx, vw, width, self.width),
                    footprint_centre(fy, vy, vh, height, self.height),
                )
            })
            .collect()
    }

    /// Compare every read-back pixel with what [`FramePipeline`]
    /// draws for the `width` x `height` rgb8 frame at this size:
    /// the frame fitted by [`fit`], nearest pixel, black around it.
    /// Two kinds of pixel may show either of two colours, because
    /// GPUs rasterize them differently: a pixel whose centre lies
    /// within 1/64 of a pixel of a texel edge (either texel), and a
    /// pixel that a fractional edge of the fitted frame cuts (black,
    /// or the frame's edge pixel, which the shader clamps to). At a
    /// whole-number scale and offset there are none.
    pub fn verify(&self, width: u32, height: u32, rgb: &[u8]) -> Result<Verdict, GarasuError> {
        rgb8_frame(width, height, rgb)?;
        let [vx, vy, vw, vh] = fit((width, height), (self.width, self.height));
        let shown = |x: Option<u32>, y: Option<u32>| match (x, y) {
            (Some(x), Some(y)) => rgb8_at(rgb, width, x, y),
            _ => [0, 0, 0],
        };
        let columns: Vec<_> = (0..self.width).map(|x| sampled(x, vx, vw, width)).collect();
        let mismatch = (0..self.height).find_map(|y| {
            let rows = sampled(y, vy, vh, height);
            columns.iter().zip(0..).find_map(|(xs, x)| {
                let got = self.pixel(x, y);
                let seen = xs
                    .iter()
                    .any(|&tx| rows.iter().any(|&ty| shown(tx, ty) == got));
                (!seen).then(|| Mismatch {
                    x,
                    y,
                    want: shown(xs[0], rows[0]),
                    got,
                })
            })
        });
        Ok(Verdict {
            blake3: blake3::hash(&self.frame(width, height)),
            surface: blake3::hash(&self.rgb),
            mismatch,
        })
    }
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
        rgb8_frame(width, height, rgb)?;
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

    /// One pass into `view`, a `target`-sized texture: clear it to
    /// black, then draw the frame fitted by [`fit`].
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        target: (u32, u32),
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("garasu_frame_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
        let [x, y, w, h] = fit((self.width, self.height), target);
        pass.set_viewport(x, y, w, h, 0.0, 1.0);
        self.draw(&mut pass);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAVY: [u8; 3] = [18, 52, 86];
    const RED: [u8; 3] = [255, 0, 0];

    fn red_rectangle() -> (u32, u32, Vec<u8>) {
        let (width, height) = (8, 6);
        let rgb = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .flat_map(|(x, y)| {
                if (1..5).contains(&x) && (1..3).contains(&y) {
                    RED
                } else {
                    NAVY
                }
            })
            .collect();
        (width, height, rgb)
    }

    fn upside_down(rgb: &[u8], width: u32) -> Vec<u8> {
        rgb.chunks(width as usize * 3)
            .rev()
            .flatten()
            .copied()
            .collect()
    }

    fn scaled(rgb: &[u8], width: u32, height: u32, k: u32) -> Vec<u8> {
        (0..height * k)
            .flat_map(|y| (0..width * k).map(move |x| (x / k, y / k)))
            .flat_map(|(x, y)| rgb8_at(rgb, width, x, y))
            .collect()
    }

    fn readback(width: u32, height: u32, rgb: &[u8]) -> Readback {
        Readback::from_texels(
            width,
            height,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &rgb8_to_rgba8(rgb),
        )
        .unwrap()
    }

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

    #[test]
    fn a_readback_of_the_frame_at_twice_its_size_verifies_and_hashes_back_to_the_frame() {
        let (w, h, rgb) = red_rectangle();
        let surface = scaled(&rgb, w, h, 2);
        let verdict = readback(w * 2, h * 2, &surface).verify(w, h, &rgb).unwrap();
        assert_eq!(
            verdict,
            Verdict {
                blake3: blake3::hash(&rgb),
                surface: blake3::hash(&surface),
                mismatch: None
            }
        );
        for x in [0, w * 2 - 1] {
            let mut edge = surface.clone();
            let at = x as usize * 3;
            edge[at..at + 3].copy_from_slice(&[0, 0, 0]);
            let verdict = readback(w * 2, h * 2, &edge).verify(w, h, &rgb).unwrap();
            assert_eq!(
                verdict.mismatch,
                Some(Mismatch {
                    x,
                    y: 0,
                    want: NAVY,
                    got: [0, 0, 0]
                }),
                "a black pixel at ({x}, 0), inside a whole-pixel frame edge"
            );
        }
    }

    #[test]
    fn an_upside_down_readback_names_the_first_pixel_that_differs() {
        let (w, h, rgb) = red_rectangle();
        let flipped = upside_down(&rgb, w);
        let verdict = readback(w * 2, h * 2, &scaled(&flipped, w, h, 2))
            .verify(w, h, &rgb)
            .unwrap();
        assert_eq!(
            verdict.mismatch,
            Some(Mismatch {
                x: 2,
                y: 2,
                want: RED,
                got: NAVY
            })
        );
        assert_eq!(verdict.blake3, blake3::hash(&flipped));
        assert_ne!(verdict.blake3, blake3::hash(&rgb));
    }

    #[test]
    fn a_letterboxed_readback_wants_black_around_the_frame() {
        let (w, h, rgb) = red_rectangle();
        let middle = scaled(&rgb, w, h, 2);
        let mut surface: Vec<u8> = middle
            .chunks(w as usize * 2 * 3)
            .flat_map(|line| [&[0u8; 6][..], line, &[0u8; 6][..]].concat())
            .collect();
        let shown = readback(w * 2 + 4, h * 2, &surface);
        let verdict = shown.verify(w, h, &rgb).unwrap();
        assert_eq!(
            (verdict.mismatch, verdict.blake3),
            (None, blake3::hash(&rgb))
        );
        surface[0..3].copy_from_slice(&NAVY);
        let verdict = readback(w * 2 + 4, h * 2, &surface)
            .verify(w, h, &rgb)
            .unwrap();
        assert_eq!(
            verdict.mismatch,
            Some(Mismatch {
                x: 0,
                y: 0,
                want: [0, 0, 0],
                got: NAVY
            })
        );
    }

    #[test]
    fn a_pixel_centred_on_a_texel_edge_may_show_either_side_and_no_other() {
        let frame = [RED, NAVY, RED, NAVY].concat();
        for middle in [RED, NAVY] {
            let shown = [RED, middle, NAVY].concat().repeat(3);
            let verdict = readback(3, 3, &shown).verify(2, 2, &frame).unwrap();
            assert_eq!(
                (verdict.mismatch, verdict.blake3),
                (None, blake3::hash(&frame)),
                "{middle:?}"
            );
        }
        let wrong = [
            [NAVY, RED, NAVY].concat(),
            [RED, RED, NAVY].concat().repeat(2),
        ]
        .concat();
        let verdict = readback(3, 3, &wrong).verify(2, 2, &frame).unwrap();
        assert_eq!(
            verdict.mismatch,
            Some(Mismatch {
                x: 0,
                y: 0,
                want: RED,
                got: NAVY
            })
        );
    }

    #[test]
    fn a_pixel_a_fractional_frame_edge_cuts_may_show_black_or_the_edge_pixel_and_no_other() {
        const BLACK: [u8; 3] = [0, 0, 0];
        for line in [
            [BLACK, RED, RED, BLACK],
            [RED, RED, RED, RED],
            [RED, RED, RED, BLACK],
        ] {
            let verdict = readback(4, 3, &line.concat().repeat(3))
                .verify(1, 1, &RED)
                .unwrap();
            assert_eq!(verdict.mismatch, None, "{line:?}");
        }
        let verdict = readback(4, 3, &[RED, BLACK, RED, RED].concat().repeat(3))
            .verify(1, 1, &RED)
            .unwrap();
        assert_eq!(
            verdict.mismatch,
            Some(Mismatch {
                x: 1,
                y: 0,
                want: RED,
                got: BLACK
            })
        );
        let verdict = readback(4, 3, &[NAVY, RED, RED, RED].concat().repeat(3))
            .verify(1, 1, &RED)
            .unwrap();
        assert_eq!(
            verdict.mismatch,
            Some(Mismatch {
                x: 0,
                y: 0,
                want: RED,
                got: NAVY
            })
        );
    }

    #[test]
    fn a_readback_reads_its_texels_in_their_own_channel_order() {
        let texel = [1, 2, 3, 255];
        let rgba = Readback::from_texels(1, 1, wgpu::TextureFormat::Rgba8Unorm, &texel).unwrap();
        let bgra =
            Readback::from_texels(1, 1, wgpu::TextureFormat::Bgra8UnormSrgb, &texel).unwrap();
        assert_eq!((rgba.pixel(0, 0), bgra.pixel(0, 0)), ([1, 2, 3], [3, 2, 1]));
        assert_eq!(
            (bgra.width(), bgra.height(), bgra.format(), bgra.rgb()),
            (1, 1, wgpu::TextureFormat::Bgra8UnormSrgb, &[3, 2, 1][..])
        );
        let wide = Readback::from_texels(2, 3, wgpu::TextureFormat::Rgba8Unorm, &[0; 24]).unwrap();
        assert_eq!((wide.width(), wide.height(), wide.rgb().len()), (2, 3, 18));
        assert!(Readback::from_texels(1, 1, wgpu::TextureFormat::Rgba16Float, &[0; 8]).is_err());
        assert!(Readback::from_texels(2, 1, wgpu::TextureFormat::Rgba8Unorm, &texel).is_err());
        for (w, h) in [(0, 1), (1, 0), (0, 0)] {
            assert!(Readback::from_texels(w, h, wgpu::TextureFormat::Rgba8Unorm, &[]).is_err());
        }
        assert!(rgba.verify(1, 1, &[1, 2]).is_err());
        assert!(rgba.verify(0, 1, &[]).is_err());
    }

    #[cfg(feature = "gpu_tests")]
    fn rendered(gpu: &GpuContext, frame: &FramePipeline, target: &crate::headless::HeadlessTarget) {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        frame.render(
            &mut encoder,
            target.view(),
            (target.width(), target.height()),
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    #[cfg(feature = "gpu_tests")]
    #[test]
    fn a_frame_drawn_through_the_pipeline_lands_on_the_target_pixel_for_pixel() {
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
            rendered(&gpu, &frame, &target);
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

    #[cfg(feature = "gpu_tests")]
    #[test]
    fn a_frame_read_back_from_the_texture_it_was_drawn_into_verifies_and_upside_down_does_not() {
        use crate::headless::HeadlessTarget;

        let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let (fw, fh, rgb) = red_rectangle();
        let mut frame = FramePipeline::new(&gpu.device, format, 1, 1);
        let read = |frame: &FramePipeline, w: u32, h: u32| {
            let target = HeadlessTarget::new(&gpu, w, h, format);
            rendered(&gpu, frame, &target);
            Readback::read(&gpu, target.texture()).unwrap()
        };
        frame
            .upload_rgb8(&gpu.device, &gpu.queue, fw, fh, &rgb)
            .unwrap();
        assert_eq!((frame.width(), frame.height()), (fw, fh));
        for (w, h) in [(8, 6), (16, 12), (21, 10), (13, 17)] {
            let verdict = read(&frame, w, h).verify(fw, fh, &rgb).unwrap();
            assert_eq!(verdict.mismatch, None, "{w}x{h}");
            if w >= fw && h >= fh {
                assert_eq!(verdict.blake3, blake3::hash(&rgb), "{w}x{h}");
            }
        }
        frame
            .upload_rgb8(&gpu.device, &gpu.queue, fw, fh, &upside_down(&rgb, fw))
            .unwrap();
        let verdict = read(&frame, 16, 12).verify(fw, fh, &rgb).unwrap();
        assert_eq!(
            verdict.mismatch,
            Some(Mismatch {
                x: 2,
                y: 2,
                want: RED,
                got: NAVY
            })
        );
    }
}
