//! Headless wgpu render target for deterministic GPU testing.
//!
//! Avoids the entire winit + surface dance — creates a raw
//! `wgpu::Texture` as a render target, exposes its `TextureView`
//! for any `RenderCallback`-style consumer to draw into, and
//! provides synchronous pixel readback for assertions.
//!
//! Why this exists: visual bugs (purple flash on first frame,
//! cursor afterimage between frames, uninitialized-memory regions)
//! are pipeline-correctness bugs that can't be caught by CPU-side
//! "what did we tell the GPU to draw" tests alone. They need to
//! observe the actual pixels the pipeline produces. A headless
//! target makes those observations testable in CI: no display
//! server, no window manager, no GPU context switching with the
//! foreground app.
//!
//! ## Example
//!
//! ```no_run
//! use garasu::{GpuContext, headless::HeadlessTarget};
//!
//! let gpu = pollster::block_on(GpuContext::new()).unwrap();
//! let target = HeadlessTarget::new(
//!     &gpu,
//!     800,
//!     600,
//!     wgpu::TextureFormat::Rgba8UnormSrgb,
//! );
//!
//! // ... drive your RenderCallback against target.view() ...
//!
//! let pixels = target.read_pixels_rgba8(&gpu);
//! assert_eq!(pixels.len(), 800 * 600 * 4);
//! ```

use crate::{GpuContext, TextRenderer};

/// Off-screen render target with a typed texture + view + format
/// triple and synchronous pixel readback. Hand the [`view`] to a
/// `RenderCallback`'s `RenderContext::surface_view`; after the
/// callback returns, call [`read_pixels_rgba8`] to get the raw
/// pixel bytes back from the GPU.
///
/// The texture is allocated with `RENDER_ATTACHMENT | COPY_SRC` so
/// the same texture can be both rendered into and copied out of
/// without an intermediate blit.
pub struct HeadlessTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
}

impl HeadlessTarget {
    /// Allocate a fresh offscreen texture sized `width × height`
    /// in the given format. The format choice should match what
    /// the consumer's render pipeline expects — typical mado /
    /// madori work uses `Rgba8UnormSrgb` or `Bgra8UnormSrgb`.
    #[must_use]
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("garasu-headless-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            format,
            width,
            height,
        }
    }

    /// Texture view to hand into a render callback as
    /// `surface_view`.
    #[must_use]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Raw texture handle — useful for tests that want to issue
    /// their own copy-out commands.
    #[must_use]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Synchronous pixel readback: copy the texture to a staging
    /// buffer, map it, return one `Vec<u8>` of length
    /// `width * height * 4` in tightly-packed RGBA8 order
    /// (matches the texture's own row order top-to-bottom).
    ///
    /// Panics if the format is not 4-byte-per-pixel (only Rgba8 /
    /// Bgra8 variants are supported today). Callers needing other
    /// formats should add a typed read helper for theirs.
    ///
    /// Uses `device.poll(Wait)` for synchronisation — fine for
    /// tests; do not call from a hot path on a real surface.
    pub fn read_pixels_rgba8(&self, gpu: &GpuContext) -> Vec<u8> {
        let bytes_per_pixel = match self.format {
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb => 4u32,
            other => panic!(
                "HeadlessTarget::read_pixels_rgba8: unsupported format {other:?}; \
                 only 4-bpp Rgba8/Bgra8 variants are supported"
            ),
        };
        // wgpu requires the per-row byte count to be aligned to
        // COPY_BYTES_PER_ROW_ALIGNMENT (256 today). Compute a
        // padded row stride for the staging buffer; trim the
        // padding on readback.
        let unpadded_row = self.width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = unpadded_row.div_ceil(align) * align;
        let buffer_size = u64::from(padded_row) * u64::from(self.height);

        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("garasu-headless-staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("garasu-headless-copyout"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map + wait. wgpu's map_async needs to be polled; for
        // synchronous tests, poll(Wait) drives it.
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = gpu.device.poll(wgpu::PollType::Wait);
        let _ = rx.recv().expect("map_async channel closed");

        let mapped = slice.get_mapped_range();
        // Strip per-row padding into a tight RGBA8 vec.
        let mut out = Vec::with_capacity((self.width * self.height * bytes_per_pixel) as usize);
        for row in 0..self.height {
            let start = (row * padded_row) as usize;
            let end = start + unpadded_row as usize;
            out.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        staging.unmap();
        out
    }
}

/// Tiny invariant: every pixel of `rgba8` must NOT be magenta-ish
/// (R high, G low, B high). wgpu's debug-init for uninitialised
/// textures often surfaces as magenta on macOS Metal; assert this
/// after a "should-be-cleared" frame to catch the purple-flash
/// class of bug.
///
/// `rgba8` is the pixel buffer returned by
/// [`HeadlessTarget::read_pixels_rgba8`]. Returns `Err(coord)` at
/// the first offending pixel.
pub fn assert_no_magenta_pixels(
    rgba8: &[u8],
    width: u32,
    height: u32,
) -> Result<(), (u32, u32)> {
    debug_assert_eq!(
        rgba8.len() as u32,
        width * height * 4,
        "rgba8 length must match width * height * 4"
    );
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            let r = rgba8[i];
            let g = rgba8[i + 1];
            let b = rgba8[i + 2];
            // Magenta heuristic: R > 200, G < 60, B > 200. Tight
            // enough to skip Nord frost (which has high R, high
            // G, high B) and forgiving enough to catch any uninit
            // R+B-bright pixel regardless of exact channel.
            if r > 200 && g < 60 && b > 200 {
                return Err((x, y));
            }
        }
    }
    Ok(())
}

/// Deterministic content hash of a pixel buffer, suitable for
/// golden-snapshot tests. BLAKE3 of the raw RGBA8 bytes — same
/// pipeline + same inputs ⇒ same hash, byte-for-byte.
///
/// Compare two hashes for "did the rendering change?" tests. To
/// adopt as a golden test: render once, commit the hex; later
/// runs must produce the same hex. A mismatch is a visible
/// pixel-level regression (or an intentional change that needs
/// the golden updated).
#[must_use]
pub fn frame_hash(rgba8: &[u8]) -> blake3::Hash {
    blake3::hash(rgba8)
}

/// Read one RGBA8 pixel at `(x, y)`. Returns `[r, g, b, a]`. Used
/// by tests that want to assert a specific cell location's color
/// (e.g. "the cursor cell at col 5, row 2 is the cursor color").
#[must_use]
pub fn pixel_at(rgba8: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [rgba8[i], rgba8[i + 1], rgba8[i + 2], rgba8[i + 3]]
}

/// Convert a cell coordinate to its pixel center, given the cell
/// metrics + origin offset. Useful for "the pixel at the cursor's
/// rendered position should be cursor-colored" assertions.
///
/// Returns `(x_px, y_px)` rounded to the nearest pixel.
#[must_use]
pub fn cell_center_pixel(
    col: u32,
    row: u32,
    cell_width: f32,
    cell_height: f32,
    origin_x: f32,
    origin_y: f32,
) -> (u32, u32) {
    let x = origin_x + (col as f32 + 0.5) * cell_width;
    let y = origin_y + (row as f32 + 0.5) * cell_height;
    (x.round().max(0.0) as u32, y.round().max(0.0) as u32)
}

/// One-call wrapper that ties together `HeadlessTarget` + a
/// `RenderCallback`-style closure. Pattern: build a renderer,
/// hand it to `render_one_frame`, get back the raw pixel
/// buffer, assert.
///
/// This is the canonical entry point for fleet-wide headless
/// regression tests. Every `garasu`-based GPU app (mado,
/// ayatsuri, hibikine, namimado, ...) can write:
///
/// ```ignore
/// let pixels = HeadlessHarness::new(&gpu, 800, 600, fmt)
///     .render_one_frame(|ctx| my_renderer.render(ctx));
/// assert!(assert_no_magenta_pixels(&pixels, 800, 600).is_ok());
/// ```
///
/// The harness owns the `TextRenderer` because most consumers
/// need one; the closure receives a fully-populated
/// `RenderContext` matching what the live winit loop builds.
pub struct HeadlessHarness {
    target: HeadlessTarget,
    text: TextRenderer,
}

impl HeadlessHarness {
    /// Allocate target + text renderer for the given dimensions
    /// and format. The text renderer's atlas is sized for the
    /// passed format — pass the same format you'll use for the
    /// real surface so the test matches production.
    #[must_use]
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let target = HeadlessTarget::new(gpu, width, height, format);
        let text = TextRenderer::new(&gpu.device, &gpu.queue, format);
        Self { target, text }
    }

    /// Run one frame and return the resulting RGBA8 pixel buffer.
    /// `render_fn` receives a fully-populated [`crate::text::TextRenderer`]
    /// borrow and the same `TextureView` + dimensions that the live
    /// renderer would see; build a `madori::RenderContext` (or
    /// equivalent) from these and call the consumer's render entry.
    ///
    /// Polls the GPU to completion before reading pixels, so the
    /// returned buffer reflects exactly what the pipeline produced.
    pub fn render_one_frame<F>(&mut self, gpu: &GpuContext, render_fn: F) -> Vec<u8>
    where
        F: FnOnce(&mut TextRenderer, &wgpu::TextureView, u32, u32),
    {
        render_fn(
            &mut self.text,
            self.target.view(),
            self.target.width(),
            self.target.height(),
        );
        let _ = gpu.device.poll(wgpu::PollType::Wait);
        self.target.read_pixels_rgba8(gpu)
    }

    #[must_use]
    pub fn target(&self) -> &HeadlessTarget {
        &self.target
    }

    #[must_use]
    pub fn width(&self) -> u32 {
        self.target.width()
    }

    #[must_use]
    pub fn height(&self) -> u32 {
        self.target.height()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assert_no_magenta_pixels_passes_on_solid_clear_color() {
        // 2x2 buffer filled with Nord polar-night dark.
        let nord = [46u8, 52, 64, 255];
        let mut buf = Vec::new();
        for _ in 0..4 {
            buf.extend_from_slice(&nord);
        }
        assert!(assert_no_magenta_pixels(&buf, 2, 2).is_ok());
    }

    #[test]
    fn assert_no_magenta_pixels_flags_a_magenta_pixel() {
        // 2x2 buffer; bottom-right is magenta.
        let mut buf = vec![0u8; 16];
        // pixel (1, 1) = magenta
        let i = ((1 * 2 + 1) * 4) as usize;
        buf[i] = 255;
        buf[i + 1] = 0;
        buf[i + 2] = 255;
        buf[i + 3] = 255;
        let err = assert_no_magenta_pixels(&buf, 2, 2).unwrap_err();
        assert_eq!(err, (1, 1));
    }

    #[test]
    fn assert_no_magenta_pixels_does_not_flag_nord_frost() {
        // Nord frost #88c0d0 — high in all three channels, NOT
        // magenta. The heuristic must let it through.
        let frost = [0x88u8, 0xc0, 0xd0, 0xff];
        let mut buf = Vec::new();
        for _ in 0..4 {
            buf.extend_from_slice(&frost);
        }
        assert!(assert_no_magenta_pixels(&buf, 2, 2).is_ok());
    }

    #[test]
    fn frame_hash_is_deterministic_over_same_input() {
        let buf = vec![17u8; 64];
        assert_eq!(frame_hash(&buf), frame_hash(&buf));
    }

    #[test]
    fn frame_hash_differs_when_one_byte_changes() {
        let mut a = vec![17u8; 64];
        let mut b = vec![17u8; 64];
        b[7] ^= 0x01;
        assert_ne!(frame_hash(&a), frame_hash(&b));
        // Sanity: a doesn't accidentally collide with itself
        // after a no-op clone.
        a.clone_from(&a.clone());
        assert_eq!(frame_hash(&a), frame_hash(&vec![17u8; 64]));
    }

    #[test]
    fn pixel_at_returns_correct_channels() {
        // 2×1 RGBA buffer: [R G B A, R G B A].
        let buf = [1u8, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(pixel_at(&buf, 2, 0, 0), [1, 2, 3, 4]);
        assert_eq!(pixel_at(&buf, 2, 1, 0), [5, 6, 7, 8]);
    }

    #[test]
    fn cell_center_pixel_at_origin_picks_first_cell_center() {
        // 10×20 cells, origin (0, 0). Center of (0, 0) is (5, 10).
        assert_eq!(cell_center_pixel(0, 0, 10.0, 20.0, 0.0, 0.0), (5, 10));
    }

    #[test]
    fn cell_center_pixel_respects_origin_offset() {
        // Origin (100, 50) shifts everything.
        let (x, y) = cell_center_pixel(2, 1, 10.0, 20.0, 100.0, 50.0);
        // col 2 center → 100 + 2.5*10 = 125
        // row 1 center → 50 + 1.5*20 = 80
        assert_eq!((x, y), (125, 80));
    }

    #[test]
    fn cell_center_pixel_clamps_negative_origin_to_zero() {
        let (x, y) = cell_center_pixel(0, 0, 10.0, 10.0, -100.0, -100.0);
        assert_eq!((x, y), (0, 0));
    }

    // GPU-driven tests live under #[cfg(feature = "gpu_tests")] so
    // they don't run by default — wgpu's `request_adapter` needs a
    // real adapter and CI runners without one would mis-fail.

    #[cfg(feature = "gpu_tests")]
    #[test]
    fn headless_target_clear_render_produces_expected_color() {
        let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
        let target = HeadlessTarget::new(
            &gpu,
            64,
            64,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );

        // Clear to magenta intentionally — we want to verify
        // readback works AND that assert_no_magenta_pixels fires.
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let pixels = target.read_pixels_rgba8(&gpu);
        assert_eq!(pixels.len(), 64 * 64 * 4);
        // Should be flagged as magenta.
        assert!(assert_no_magenta_pixels(&pixels, 64, 64).is_err());
    }

    #[cfg(feature = "gpu_tests")]
    #[test]
    fn headless_target_clear_to_nord_passes_no_magenta_check() {
        // The canonical "first-frame should be clean" test.
        let gpu = pollster::block_on(GpuContext::new()).expect("gpu");
        let target = HeadlessTarget::new(
            &gpu,
            32,
            32,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target.view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.180,
                            g: 0.204,
                            b: 0.251,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let pixels = target.read_pixels_rgba8(&gpu);
        assert!(assert_no_magenta_pixels(&pixels, 32, 32).is_ok());
    }
}
