//! `TextLayerStack` — per-layer-isolated multi-pass text rendering.
//!
//! ## The shared-cell bug this primitive eliminates
//!
//! [`crate::text::TextRenderer`] wraps ONE `glyphon::TextRenderer`, whose
//! single internal vertex buffer + baked glyph count are rewritten by every
//! `prepare()`. A consumer that draws text in TWO passes within one frame
//! (e.g. a terminal grid PLUS a centered overlay) issues
//! `prepare(A) → record render(A); prepare(B) → record render(B); submit`.
//! Because both `prepare`s land in the one shared buffer *before* the single
//! `submit`, the recorded draw for A — its glyph count baked at record time —
//! executes against B's vertex data. A's first N glyph slots (the top-left
//! cells, since text areas are emitted row-0-first) render B's glyphs, blanking
//! the top-left and growing with B's glyph count. (mado Ctrl-S overlay report.)
//!
//! This is the **shared-mutable-cell** class. We do not guard the cell — we
//! eliminate it. A `TextLayerStack` holds an allocate-once `Vec` of layers,
//! each owning its **own** `glyphon::TextRenderer` (hence its own vertex buffer
//! + own baked count) **and its own `Viewport`** (the second shared cell:
//! `Viewport` owns the resolution uniform), while ALL layers borrow **one**
//! shared `TextAtlas` + `FontSystem` + `SwashCache` (so the ~180 ms cosmic-text
//! scan and the glyph-atlas memory happen exactly once). `prepare(B)` writes
//! only layer B's buffer — there is no shared buffer for it to clobber A's data
//! in, so the bug has no code path.
//!
//! ## Tier-honest unrepresentability (per the org bright line)
//!
//! - **CLOBBER** (vertex buffer) — *truly-unrepresentable*: each layer owns its
//!   buffer; no shared cell exists.
//! - **VIEWPORT / RESOLUTION** — *truly-unrepresentable*: resolution is lifted
//!   out of `prepare` into [`TextLayerStack::begin_frame`]; per-prepare
//!   width/height (the relocated clobber cell) is not expressible. Each layer
//!   also owns its own `Viewport`.
//! - **RECORD-WITHOUT-PREPARE / MIS-ROUTE** — *truly-unrepresentable*:
//!   [`Frame::render`] accepts only a [`PreparedLayer`] token (minted by
//!   [`Frame::prepare`], consumed by move), never a bare id. Recording an
//!   unprepared layer has no token to pass; a token carries its own id so it
//!   cannot draw a different layer.
//! - **ATLAS-GROW** — *truly-unrepresentable for mado's interleaved shape*: a
//!   grow during a later layer's `prepare` allocates a new texture + bind group;
//!   the earlier layer's already-recorded pass keeps the still-live old binding.
//!   The `Frame` `Drop` runs `atlas.trim()` exactly once, after all renders.
//! - **CROSS-LAYER EVICTION (`RemovedFromAtlas`) + INTRA-LAYER DOUBLE-PREPARE**
//!   — *only-mitigated* (the no-mid-frame-`trim` discipline; `&mut` serializing
//!   prepares). [`Frame::render`] therefore KEEPS its `Result<(), RenderError>`
//!   — swallowing it would re-introduce the silent-corruption class. Closed
//!   truly-unrepresentably only at the engawa destination (a per-Node
//!   vertex-buffer `Resource` + the shipped `MultipleWriters` validation).
//!
//! ## Destination
//!
//! Each `TextLayer` is the per-Node GPU resource engawa's v0.2 dispatcher will
//! bind; mado's frame becomes a typed `RenderGraph` where two passes writing one
//! `vb.*` resource is a validation-time error. This primitive composes INTO that
//! destination — it is not thrown away. See `mado/docs/THEORY.md` §VIII and the
//! `pending-engawa-text:` note in `mado/CLAUDE.md`.

use glyphon::{
    Attrs, Buffer, Cache, FontSystem, Metrics, PrepareError, RenderError, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextRenderer as GlyphonRenderer, Viewport,
};

/// Typed identity for one text layer within a [`TextLayerStack`]. Minted only by
/// [`TextLayerStack::add_layer`] — a consumer cannot fabricate one or reorder by
/// integer. z-order equals allocation order (earlier layers draw under later
/// ones when their passes load-blend in sequence).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextLayerId(usize);

/// One isolated layer: its OWN glyphon renderer (own vertex buffer + own baked
/// glyph count) and its OWN viewport (own resolution uniform). The atlas /
/// font-system / swash-cache it draws against are shared, held by the stack.
struct TextLayer {
    renderer: GlyphonRenderer,
    viewport: Viewport,
}

/// An ordered set of isolated text layers sharing one glyph atlas.
///
/// Construct once ([`new`](Self::new)), allocate the layers a consumer needs
/// once ([`add_layer`](Self::add_layer)), then per frame open a [`Frame`]
/// ([`begin_frame`](Self::begin_frame)) and `prepare`/`render` each layer.
///
/// See the module docs for why this exists and the per-axis unrepresentability
/// grading.
pub struct TextLayerStack {
    /// Shared — built once via `crate::text::take_or_build_font_system()` so the
    /// emoji-fallback + Nix-font-dir + preload-cache contract is preserved
    /// fleet-wide. `pub` so unmigrated consumers reach `ctx.text.font_system`
    /// exactly as they did on `TextRenderer`.
    pub font_system: FontSystem,
    /// Shared glyph rasterization cache.
    swash_cache: SwashCache,
    /// Shared glyph atlas (texture(s) + bind group + glyph cache). The ONE
    /// expensive resource every layer reads through.
    atlas: TextAtlas,
    /// Shared pipeline / bind-group-layout cache — each layer's `Viewport::new`
    /// needs it, and so does the atlas.
    cache: Cache,
    /// Allocate-once; index == `TextLayerId.0`.
    layers: Vec<TextLayer>,
    /// Lazily-minted layer for the deprecated single-pass back-compat API
    /// ([`prepare`](Self::prepare_compat) / [`render`](Self::render_compat)).
    default_layer: Option<TextLayerId>,
    multisample: wgpu::MultisampleState,
    depth_stencil: Option<wgpu::DepthStencilState>,
}

impl TextLayerStack {
    /// Build a stack with one shared atlas / font-system / viewport-cache. No
    /// layers yet — call [`add_layer`](Self::add_layer) once per text surface.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let t_start = std::time::Instant::now();
        let font_system = crate::text::take_or_build_font_system();
        let t_font = t_start.elapsed();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let atlas = TextAtlas::new(device, queue, &cache, format);
        tracing::info!(
            target: "garasu::layers",
            font_system_ms = t_font.as_millis() as u64,
            total_ms = t_start.elapsed().as_millis() as u64,
            "text layer stack built"
        );
        Self {
            font_system,
            swash_cache,
            atlas,
            cache,
            layers: Vec::new(),
            default_layer: None,
            multisample: wgpu::MultisampleState::default(),
            depth_stencil: None,
        }
    }

    /// Allocate a NEW isolated layer — its own glyphon renderer (own vertex
    /// buffer) and its own viewport. Call ONCE per text surface at renderer
    /// init (not per frame). z-order = allocation order. Returns the typed id.
    pub fn add_layer(&mut self, device: &wgpu::Device) -> TextLayerId {
        let renderer = GlyphonRenderer::new(
            &mut self.atlas,
            device,
            self.multisample,
            self.depth_stencil.clone(),
        );
        let viewport = Viewport::new(device, &self.cache);
        self.layers.push(TextLayer { renderer, viewport });
        TextLayerId(self.layers.len() - 1)
    }

    /// Number of allocated layers.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Begin a frame at one resolution. The returned [`Frame`] borrows the stack
    /// `&mut` for the frame's life; its `Drop` calls `atlas.trim()` exactly once
    /// — after all renders, never between two prepares.
    pub fn begin_frame(&mut self, width: u32, height: u32) -> Frame<'_> {
        Frame {
            stack: self,
            width,
            height,
        }
    }

    /// Create a plain text buffer with the given metrics. Shaped via the shared
    /// font system. (Moved verbatim from `TextRenderer`.)
    pub fn create_buffer(&mut self, text: &str, font_size: f32, line_height: f32) -> Buffer {
        let metrics = Metrics::new(font_size, line_height);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_text(&mut self.font_system, text, &Attrs::new(), Shaping::Advanced);
        buffer
    }

    /// Create a rich text buffer with per-span attributes (per-cell color /
    /// weight / style). (Moved verbatim from `TextRenderer`.)
    pub fn create_rich_buffer(
        &mut self,
        spans: &[(&str, Attrs<'_>)],
        font_size: f32,
        line_height: f32,
    ) -> Buffer {
        let metrics = Metrics::new(font_size, line_height);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_rich_text(
            &mut self.font_system,
            spans.iter().map(|&(text, ref attrs)| (text, attrs.clone())),
            &Attrs::new(),
            Shaping::Advanced,
            None,
        );
        buffer
    }

    // ── deprecated single-layer back-compat ─────────────────────────────────
    //
    // Lets an unmigrated `madori` consumer that draws text in ONE pass keep its
    // exact `ctx.text.prepare(dev,q,w,h,areas)` / `ctx.text.render(pass)` call
    // shape after `RenderContext.text` becomes `&mut TextLayerStack`. Operates
    // on a lazily-minted default layer. Single-pass consumers never hit the
    // clobber (only one prepare per frame), so this is correct for them — but a
    // consumer that adds a SECOND text pass MUST migrate to the layered API
    // (`begin_frame` + per-layer `prepare`/`render`) or it reintroduces the bug.

    /// Single-pass back-compat `prepare`, name- and signature-compatible with the
    /// old `TextRenderer::prepare` so an unmigrated `madori` consumer that draws
    /// text in ONE pass compiles unchanged after `RenderContext.text` becomes
    /// `&mut TextLayerStack`. Prefer [`begin_frame`](Self::begin_frame) +
    /// [`Frame::prepare`]. A SECOND text pass on the default layer reintroduces
    /// the clobber, so multi-pass consumers MUST migrate to the layered API. Not
    /// `#[deprecated]` only to spare the unmigrated fleet a `-D warnings` break
    /// during the migration window. (No collision with [`Frame::prepare`] — that
    /// is a method on the distinct [`Frame`] type.)
    pub fn prepare<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
    ) -> Result<(), PrepareError> {
        let id = match self.default_layer {
            Some(id) => id,
            None => {
                let id = self.add_layer(device);
                self.default_layer = Some(id);
                id
            }
        };
        let layer = &mut self.layers[id.0];
        layer.viewport.update(queue, Resolution { width, height });
        layer.renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &layer.viewport,
            text_areas,
            &mut self.swash_cache,
        )
    }

    /// Single-pass back-compat `render`, name- and signature-compatible with the
    /// old `TextRenderer::render`. Draws the default layer prepared by
    /// [`prepare`](Self::prepare). Prefer [`Frame::render`]. See the back-compat
    /// note above.
    pub fn render<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
    ) -> Result<(), RenderError> {
        let Some(id) = self.default_layer else {
            // Nothing prepared through the back-compat path → nothing to draw.
            return Ok(());
        };
        let layer = &self.layers[id.0];
        layer.renderer.render(&self.atlas, &layer.viewport, pass)
    }
}

/// Two-phase frame guard borrowed from a [`TextLayerStack`] for one frame at one
/// resolution. Phase 1: [`prepare`](Self::prepare) each layer (writes only that
/// layer's buffer + viewport). Phase 2: [`render`](Self::render) the returned
/// tokens. `Drop` calls `atlas.trim()` exactly once — after all renders.
///
/// **Usage shape.** The [`PreparedLayer`] token borrows the `Frame` `&mut`, so
/// it must be consumed by `render` before the next `prepare`. The natural
/// `prepare→render→prepare→render` flow — exactly mado's one-encoder-per-pass
/// shape — satisfies this and is also what keeps the atlas-grow ordering sound
/// (a later layer's prepare can grow the atlas only after the earlier layer's
/// pass already captured its binding). Batch-all-prepares-then-all-renders is
/// intentionally not expressible.
pub struct Frame<'s> {
    stack: &'s mut TextLayerStack,
    width: u32,
    height: u32,
}

/// A linear witness that a layer was prepared. Returned by [`Frame::prepare`]
/// and consumed by move by [`Frame::render`].
///
/// Non-`Copy`/non-`Clone` (linear: rendered at most once) with a private,
/// unforgeable `id`. [`Frame::render`] accepts only this token — never a bare
/// [`TextLayerId`] — so recording a layer that was never prepared has no
/// expressible call, and the id travels with the token so it cannot draw a
/// different layer than the one prepared. It deliberately holds no borrow of the
/// `Frame`: `prepare`'s `&mut` ends when it returns, leaving `render`'s `&self`
/// free (the alternative — tying the token to `prepare`'s `&mut` — makes
/// `render(&self, …)` unrepresentable). Using a token after its frame drops is
/// the *only-mitigated* stale-token residual closed only at the engawa
/// destination; it is a logical id, never a dangling reference, so it is sound.
#[must_use = "a prepared text layer must be rendered or its prepare was wasted"]
pub struct PreparedLayer {
    id: TextLayerId,
}

impl<'s> Frame<'s> {
    /// PHASE 1 — prepare ONE layer. Writes only that layer's vertex buffer and
    /// updates only that layer's viewport (to the frame resolution). Cannot
    /// touch a sibling's buffer. Returns the linear token bound to this frame.
    pub fn prepare<'a>(
        &mut self,
        id: TextLayerId,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
    ) -> Result<PreparedLayer, PrepareError> {
        let (width, height) = (self.width, self.height);
        // Split the stack into disjoint field borrows so the per-layer renderer
        // can borrow `&mut` while the shared atlas/font-system/swash borrow too.
        let TextLayerStack {
            layers,
            font_system,
            atlas,
            swash_cache,
            ..
        } = &mut *self.stack;
        let TextLayer { renderer, viewport } = &mut layers[id.0];
        viewport.update(queue, Resolution { width, height });
        renderer.prepare(
            device,
            queue,
            font_system,
            atlas,
            &*viewport,
            text_areas,
            swash_cache,
        )?;
        Ok(PreparedLayer { id })
    }

    /// PHASE 2 — record ONE prepared layer into a pass. Consumes the token (by
    /// value) so a layer renders at most once and cannot be mis-routed. Reads
    /// only that layer's own buffer + own viewport + the shared atlas. The
    /// `Result` is RETAINED — `RemovedFromAtlas | ScreenResolutionChanged` are
    /// real shared-atlas hazards, not an empty-buffer case; swallowing them
    /// would re-create the silent-corruption class this primitive eliminates.
    pub fn render<'pass>(
        &'pass self,
        token: PreparedLayer,
        pass: &mut wgpu::RenderPass<'pass>,
    ) -> Result<(), RenderError> {
        let layer = &self.stack.layers[token.id.0];
        layer.renderer.render(&self.stack.atlas, &layer.viewport, pass)
    }

    /// Build a plain text buffer while the frame holds the stack. Delegates to
    /// [`TextLayerStack::create_buffer`].
    pub fn create_buffer(&mut self, text: &str, font_size: f32, line_height: f32) -> Buffer {
        self.stack.create_buffer(text, font_size, line_height)
    }

    /// Build a rich text buffer while the frame holds the stack. Delegates to
    /// [`TextLayerStack::create_rich_buffer`].
    pub fn create_rich_buffer(
        &mut self,
        spans: &[(&str, Attrs<'_>)],
        font_size: f32,
        line_height: f32,
    ) -> Buffer {
        self.stack.create_rich_buffer(spans, font_size, line_height)
    }

    /// Mutable access to the shared font system (for `shape_until_scroll` and
    /// other cosmic-text calls a consumer makes between create_buffer and
    /// prepare).
    pub fn font_system_mut(&mut self) -> &mut FontSystem {
        &mut self.stack.font_system
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        // Exactly once per frame, AFTER every render this frame recorded. trim()
        // resets glyphs_in_use bookkeeping only; it does not touch already-
        // recorded command buffers, so trim-then-submit (the caller drops the
        // Frame before queue.submit) is the correct ordering.
        self.stack.atlas.trim();
    }
}

#[cfg(test)]
mod tests {
    //! GPU-driven forcing functions for the shared-cell elimination. Gated
    //! behind `gpu_tests` (needs a real wgpu adapter) exactly like
    //! `crate::headless`'s GPU tests. The clobber gate is the mechanical proof
    //! that the reported top-left-blanking bug has no code path on
    //! `TextLayerStack`; it fails on any regression that reintroduces a shared
    //! vertex buffer across layers.

    #[cfg(feature = "gpu_tests")]
    use super::*;

    /// Render one frame into `target`: the terminal layer clears the surface and
    /// draws a block of glyphs at the TOP-LEFT; if `with_overlay`, a second layer
    /// draws a CENTERED block. Returns the read-back RGBA8 pixels. The two passes
    /// share one frame + one submit — exactly the shape that clobbered on the old
    /// single-`TextRenderer` path.
    #[cfg(feature = "gpu_tests")]
    fn render_two_layer_frame(
        gpu: &crate::GpuContext,
        target: &crate::headless::HeadlessTarget,
        stack: &mut TextLayerStack,
        term: TextLayerId,
        ovl: TextLayerId,
        w: u32,
        h: u32,
        with_overlay: bool,
    ) -> Vec<u8> {
        use glyphon::{Color, TextArea, TextBounds};
        let bounds = TextBounds {
            left: 0,
            top: 0,
            right: w as i32,
            bottom: h as i32,
        };
        let white = Color::rgba(255, 255, 255, 255);
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut frame = stack.begin_frame(w, h);
        // Terminal layer at the top-left (left=0, top=0), several rows.
        let term_buf = frame.create_buffer("XXXXXXXX\nXXXXXXXX\nXXXXXXXX", 16.0, 20.0);
        let t = frame
            .prepare(
                term,
                &gpu.device,
                &gpu.queue,
                [TextArea {
                    buffer: &term_buf,
                    left: 0.0,
                    top: 0.0,
                    scale: 1.0,
                    bounds,
                    default_color: white,
                    custom_glyphs: &[],
                }],
            )
            .expect("term prepare");
        {
            let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test_term"),
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
            frame.render(t, &mut p).expect("term render");
        }
        if with_overlay {
            // Centered overlay layer — far from the top-left region the assertion
            // probes. On the old shared buffer, THIS prepare overwrote the
            // terminal's leading (top-left) glyph slots.
            let ovl_buf = frame.create_buffer("OVERLAY", 16.0, 20.0);
            let o = frame
                .prepare(
                    ovl,
                    &gpu.device,
                    &gpu.queue,
                    [TextArea {
                        buffer: &ovl_buf,
                        left: w as f32 / 2.0 - 24.0,
                        top: h as f32 / 2.0,
                        scale: 1.0,
                        bounds,
                        default_color: white,
                        custom_glyphs: &[],
                    }],
                )
                .expect("overlay prepare");
            {
                let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("test_overlay"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target.view(),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                frame.render(o, &mut p).expect("overlay render");
            }
        }
        drop(frame); // trims the atlas once, after all renders
        gpu.queue.submit(std::iter::once(enc.finish()));
        let _ = gpu.device.poll(wgpu::PollType::Wait);
        target.read_pixels_rgba8(gpu)
    }

    /// THE CLOBBER GATE. Two text layers in one frame: a top-left terminal block
    /// and a centered overlay. The centered overlay's glyphs cannot reach the
    /// top-left region, so the top-left must be byte-identical with and without
    /// the overlay. On the pre-fix single-shared-buffer path the overlay's
    /// `prepare` overwrote the terminal layer's leading glyph slots and the
    /// top-left blanked — this asserts that has no code path.
    #[cfg(feature = "gpu_tests")]
    #[test]
    fn top_left_terminal_survives_centered_overlay() {
        use crate::headless::{pixel_at, HeadlessTarget};
        let gpu = pollster::block_on(crate::GpuContext::new()).expect("gpu");
        let (w, h) = (256u32, 128u32);
        let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = HeadlessTarget::new(&gpu, w, h, fmt);
        let mut stack = TextLayerStack::new(&gpu.device, &gpu.queue, fmt);
        let term = stack.add_layer(&gpu.device);
        let ovl = stack.add_layer(&gpu.device);

        let term_only = render_two_layer_frame(&gpu, &target, &mut stack, term, ovl, w, h, false);
        let with_overlay =
            render_two_layer_frame(&gpu, &target, &mut stack, term, ovl, w, h, true);

        // Top-left quadrant must be untouched by the centered overlay.
        let (qx, qy) = (72u32, 56u32);
        let mut diffs = 0usize;
        for y in 0..qy {
            for x in 0..qx {
                if pixel_at(&term_only, w, x, y) != pixel_at(&with_overlay, w, x, y) {
                    diffs += 1;
                }
            }
        }
        assert_eq!(
            diffs, 0,
            "centered overlay clobbered {diffs} top-left terminal pixels — shared-buffer regression"
        );

        // Guard against a vacuous pass: the terminal layer must actually have
        // drawn glyphs in the top-left (otherwise both frames are uniform bg).
        let bg = [0u8, 0, 0, 255];
        let non_bg = (0..qy)
            .flat_map(|y| (0..qx).map(move |x| (x, y)))
            .filter(|&(x, y)| pixel_at(&with_overlay, w, x, y) != bg)
            .count();
        assert!(
            non_bg > 0,
            "terminal layer drew no glyphs in the top-left — gate is vacuous"
        );
    }

    /// Many layers across many frames: trim-once-per-frame holds and no layer's
    /// render returns `RemovedFromAtlas`. Proxy for the atlas bookkeeping staying
    /// bounded — `Frame::drop` is the sole `trim` caller.
    #[cfg(feature = "gpu_tests")]
    #[test]
    fn many_layers_render_without_atlas_eviction_over_frames() {
        use crate::headless::HeadlessTarget;
        use glyphon::{Color, TextArea, TextBounds};
        let gpu = pollster::block_on(crate::GpuContext::new()).expect("gpu");
        let (w, h) = (128u32, 128u32);
        let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = HeadlessTarget::new(&gpu, w, h, fmt);
        let mut stack = TextLayerStack::new(&gpu.device, &gpu.queue, fmt);
        let ids: Vec<TextLayerId> = (0..4).map(|_| stack.add_layer(&gpu.device)).collect();
        assert_eq!(stack.layer_count(), 4);

        for f in 0..8u32 {
            let mut enc = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let mut frame = stack.begin_frame(w, h);
            let mut first = true;
            for (li, &id) in ids.iter().enumerate() {
                let buf = frame.create_buffer(&format!("L{li} F{f}"), 14.0, 18.0);
                let tok = frame
                    .prepare(
                        id,
                        &gpu.device,
                        &gpu.queue,
                        [TextArea {
                            buffer: &buf,
                            left: 2.0,
                            top: 2.0 + li as f32 * 18.0,
                            scale: 1.0,
                            bounds: TextBounds {
                                left: 0,
                                top: 0,
                                right: w as i32,
                                bottom: h as i32,
                            },
                            default_color: Color::rgba(255, 255, 255, 255),
                            custom_glyphs: &[],
                        }],
                    )
                    .expect("prepare");
                let load = if first {
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                } else {
                    wgpu::LoadOp::Load
                };
                first = false;
                let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target.view(),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                // A render error (e.g. RemovedFromAtlas) fails the gate.
                frame.render(tok, &mut p).expect("layer render must not error");
            }
            drop(frame);
            gpu.queue.submit(std::iter::once(enc.finish()));
            let _ = gpu.device.poll(wgpu::PollType::Wait);
        }
    }

    /// Shape guard: `add_layer` mints distinct, monotonic ids and the count
    /// tracks. The TEXT_LAYERS forcing function in mado leans on this contract.
    #[cfg(feature = "gpu_tests")]
    #[test]
    fn add_layer_mints_distinct_ids() {
        let gpu = pollster::block_on(crate::GpuContext::new()).expect("gpu");
        let mut stack = TextLayerStack::new(
            &gpu.device,
            &gpu.queue,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let a = stack.add_layer(&gpu.device);
        let b = stack.add_layer(&gpu.device);
        let c = stack.add_layer(&gpu.device);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        assert_eq!(stack.layer_count(), 3);
    }
}
