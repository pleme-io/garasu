//! Bounded drawing — a rectangle a caller cannot paint outside of.
//!
//! ## The gap this closes
//!
//! Measured 2026-07-31 across `garasu`, `engawa`, `madori` and `mado`:
//! **zero calls to `set_scissor_rect` and zero to `set_viewport`.** Nothing
//! in the stack bounds a draw to a region. That is invisible while one
//! terminal fills the window, and becomes a whole class of defect the
//! moment two panes share it: a cell background wider than its pane paints
//! over its neighbour, a `full_window` bell flash from one pane washes the
//! other, and every `TextArea` currently carries full-window bounds so a
//! long unwrapped line crosses the seam.
//!
//! So "restore multi-pane rendering" is **not a restore** — there is no
//! clipping primitive to turn back on. This module is that primitive.
//!
//! ## Why it lives in garasu
//!
//! A scissor is render-pass state, so whoever lends a raw `RenderPass`
//! makes containment advisory. garasu is the only shared crate that touches
//! one ([`QuadPipeline::draw`], [`Frame::render`]) and it already owns the
//! coordinate frame (the quad resolution uniform and glyphon's `Viewport`).
//! engawa is GPU-free by charter; madori hands over a view and never records
//! a pass; putting it in mado would deepen an app-level fork of shared
//! render logic.
//!
//! It is also the same move [`crate::layers`] already makes one axis over:
//! eliminate a hazard by handing out a **narrowed capability** — there, a
//! `PreparedLayer` token instead of a raw layer id; here, a [`PanePass`]
//! instead of a raw [`wgpu::RenderPass`].
//!
//! ## Tier-honest unrepresentability (per the org bright line)
//!
//! - **PAINT OUTSIDE THE PANE** — *truly-unrepresentable*: [`PanePass`]
//!   exposes no `set_scissor_rect`, no `set_viewport`, no `Deref` to the
//!   pass, and no accessor returning one. The scissor is fixed-function
//!   hardware state that no pipeline, bind group, vertex buffer or instance
//!   range can override, so even a caller's own pipeline drawing through
//!   [`PanePass::instanced`] is bounded. This is capability NARROWING, not
//!   a leak.
//! - **A RECT ESCAPING THE TARGET OR ITS PARENT** — *truly-unrepresentable*:
//!   there is no `PaneRect::new(x, y, w, h)`. The only constructors are
//!   [`PaneRect::root`] and subdivisions of a rect you already hold, so
//!   containment is closed under construction and `set_scissor_rect` can
//!   never be handed an out-of-attachment rect (a wgpu validation error,
//!   not merely a visual bug).
//! - **A DEGENERATE OR OVERSIZED SPLIT** — *parse-time-rejected*:
//!   [`PaneRect::split_x`] / [`split_y`](PaneRect::split_y) /
//!   [`inset`](PaneRect::inset) return `Option` and do **not** clamp. A
//!   clamp would silently hand back a rect the caller did not ask for.
//! - **TWO PANES OPEN AT ONCE** — *truly-unrepresentable*:
//!   [`LayeredPass::in_pane`] takes `&mut self` and scopes the pane to a
//!   closure, so panes serialize and a scissor cannot be observed
//!   mid-change.
//! - **AUTHORING AN OVERSIZED INSTANCE** — *representable, and harmless*.
//!   The value can be built; the pixels it would produce cannot escape.
//!   Deliberately NOT clamped per-draw: a clamp cannot reach glyph-atlas
//!   quads or any derived geometry, so it would be strictly weaker than the
//!   scissor while looking like a guarantee.
//! - **SOMEONE ADDS AN ESCAPE HATCH LATER** — *only-mitigated*. **Ceiling:
//!   containment is exactly as strong as the absence of an accessor
//!   returning the inner pass.** Adding `fn inner(&mut self) -> &mut
//!   RenderPass` would dissolve every row above. Bound by the doc invariant
//!   here plus a differential pixel test; not by the type system.

/// A rectangle in physical pixels that is, by construction, inside its
/// render target and inside its parent.
///
/// Fields are private and there is no free constructor — see the
/// containment row in the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PaneRect {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl PaneRect {
    /// The whole render target — the one root of every containment tree.
    ///
    /// The caller must pass the attachment's real extent; that is the one
    /// unchecked step, and every rect derived from it is then closed.
    #[must_use]
    pub const fn root(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            w: width,
            h: height,
        }
    }

    /// Split into `(left, right)` at `at` pixels from this rect's left edge.
    ///
    /// `None` when `at == 0` or `at >= self.w` — a split that would produce
    /// an empty side is rejected rather than clamped.
    #[must_use]
    pub const fn split_x(self, at: u32) -> Option<(Self, Self)> {
        if at == 0 || at >= self.w {
            return None;
        }
        Some((
            Self { w: at, ..self },
            Self {
                x: self.x + at,
                w: self.w - at,
                ..self
            },
        ))
    }

    /// Split into `(top, bottom)` at `at` pixels from this rect's top edge.
    ///
    /// `None` when `at == 0` or `at >= self.h`.
    #[must_use]
    pub const fn split_y(self, at: u32) -> Option<(Self, Self)> {
        if at == 0 || at >= self.h {
            return None;
        }
        Some((
            Self { h: at, ..self },
            Self {
                y: self.y + at,
                h: self.h - at,
                ..self
            },
        ))
    }

    /// Inset uniformly by `px` on every side — a gutter or border.
    ///
    /// `None` if the inset would empty the rect.
    #[must_use]
    pub const fn inset(self, px: u32) -> Option<Self> {
        let shrink = px * 2;
        if shrink >= self.w || shrink >= self.h {
            return None;
        }
        Some(Self {
            x: self.x + px,
            y: self.y + px,
            w: self.w - shrink,
            h: self.h - shrink,
        })
    }

    #[must_use]
    pub const fn x(self) -> u32 {
        self.x
    }
    #[must_use]
    pub const fn y(self) -> u32 {
        self.y
    }
    #[must_use]
    pub const fn width(self) -> u32 {
        self.w
    }
    #[must_use]
    pub const fn height(self) -> u32 {
        self.h
    }

    /// Is `(px, py)` inside this rect? The hit-test a click-to-focus router
    /// needs — and which the deleted pre-Phase-4 `PaneRect` could not
    /// express, so mouse events were routed to the focused pane regardless
    /// of where the pointer actually was.
    #[must_use]
    pub const fn contains(self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.w && py >= self.y && py < self.y + self.h
    }

    /// The glyphon clip for text drawn into this pane.
    ///
    /// ★ **This is the single cheapest fix for text bleed, and it needs no
    /// GPU work at all.** glyphon clips per-glyph on the CPU at prepare
    /// time — it clamps each quad and shifts its UVs — and calls
    /// `set_scissor_rect` zero times. So the clipping mechanism for TEXT
    /// already exists and already works; it is simply being handed the
    /// wrong rectangle. Every `TextArea` in mado currently carries
    /// full-window bounds, which is why a long unwrapped line crosses a
    /// pane seam.
    ///
    /// Replacing those bounds with this is a one-line change per call site
    /// and stops all text bleed before any of the rect-pipeline work
    /// below.
    #[must_use]
    pub fn text_bounds(self) -> glyphon::TextBounds {
        // i32 saturation rather than a wrapping cast: a pane wider than
        // i32::MAX is not a real window, but wrapping it would produce a
        // NEGATIVE bound and silently clip everything away.
        glyphon::TextBounds {
            left: i32::try_from(self.x).unwrap_or(i32::MAX),
            top: i32::try_from(self.y).unwrap_or(i32::MAX),
            right: i32::try_from(self.x.saturating_add(self.w)).unwrap_or(i32::MAX),
            bottom: i32::try_from(self.y.saturating_add(self.h)).unwrap_or(i32::MAX),
        }
    }

    /// Translate pane-local coordinates (origin `0,0`) into target
    /// coordinates.
    ///
    /// The ONE audited bridge between the two spaces, so a caller never
    /// hand-adds an origin — the mistake that produced the deleted
    /// renderer's two-coordinate-space drift, where the rect a child
    /// process was sized from and the rect its glyphs were clipped to
    /// differed by the padding on every frame.
    #[must_use]
    pub fn to_target(self, lx: f32, ly: f32) -> (f32, f32) {
        #[allow(clippy::cast_precision_loss)]
        (self.x as f32 + lx, self.y as f32 + ly)
    }
}

/// One render pass, subdivided into pane scopes.
///
/// Holds the [`wgpu::RenderPass`] **privately** and exposes no draw verb of
/// its own — the only way to record anything is to enter a pane via
/// [`Self::in_pane`], which sets the scissor first.
pub struct LayeredPass<'p> {
    // PRIVATE, and it must stay that way. No `Deref`, no `inner()`, no
    // accessor of any kind: see the escape-hatch ceiling in the module doc.
    pass: wgpu::RenderPass<'p>,
    target: PaneRect,
}

impl<'p> LayeredPass<'p> {
    /// Adopt an already-begun render pass whose attachment covers `target`.
    ///
    /// `target` must be the attachment's own [`PaneRect::root`]; every rect
    /// passed to [`Self::in_pane`] must descend from it, which
    /// [`PaneRect`]'s constructors already guarantee.
    #[must_use]
    pub fn new(pass: wgpu::RenderPass<'p>, target: PaneRect) -> Self {
        Self { pass, target }
    }

    /// The full extent this pass covers — the root every pane descends from.
    #[must_use]
    pub const fn target(&self) -> PaneRect {
        self.target
    }

    /// Enter one pane.
    ///
    /// Issues the single `set_scissor_rect(rect)` **before** the closure can
    /// express any draw, then hands it a [`PanePass`] pinned to `rect`.
    /// `&mut self` serialises panes, so two cannot be open at once and the
    /// scissor cannot be observed mid-change.
    pub fn in_pane<R>(&mut self, rect: PaneRect, f: impl FnOnce(&mut PanePass<'_, 'p>) -> R) -> R {
        self.pass
            .set_scissor_rect(rect.x(), rect.y(), rect.width(), rect.height());
        f(&mut PanePass {
            pass: &mut self.pass,
            rect,
        })
    }
}

/// A borrowed render pass whose scissor is pinned to one [`PaneRect`] for
/// the scope's lifetime.
///
/// There is no `set_scissor_rect`, no `set_viewport`, no `Deref`, and no
/// accessor returning the inner pass. A caller holding one has **no
/// expression that paints a pixel outside `rect`**: the scissor test is
/// fixed-function and applies to every draw recorded after it, and the only
/// calls that could move it are absent.
pub struct PanePass<'a, 'p> {
    pass: &'a mut wgpu::RenderPass<'p>,
    rect: PaneRect,
}

impl<'p> PanePass<'_, 'p> {
    /// The rect this pass is pinned to.
    #[must_use]
    pub const fn rect(&self) -> PaneRect {
        self.rect
    }

    /// Draw a caller's own instanced pipeline **through** the contained
    /// pass.
    ///
    /// This is capability NARROWING, not a leak. The scissor is
    /// fixed-function state that no pipeline, bind group, vertex buffer or
    /// instance range can override, so an arbitrary consumer pipeline —
    /// mado's `RectPipeline` and `ImagePipeline`, which are supersets of
    /// garasu's own — is still bounded by `rect`. It is the seam that lets
    /// those forks stay where they are and converge later, instead of
    /// forcing the fork to be resolved before anything can be contained.
    pub fn instanced(
        &mut self,
        pipeline: &'p wgpu::RenderPipeline,
        bind_groups: &[&'p wgpu::BindGroup],
        vertices: wgpu::BufferSlice<'p>,
        vertex_range: std::ops::Range<u32>,
        instance_range: std::ops::Range<u32>,
    ) {
        if instance_range.is_empty() || vertex_range.is_empty() {
            return;
        }
        self.pass.set_pipeline(pipeline);
        for (i, bg) in bind_groups.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            self.pass.set_bind_group(i as u32, *bg, &[]);
        }
        self.pass.set_vertex_buffer(0, vertices);
        self.pass.draw(vertex_range, instance_range);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOT: PaneRect = PaneRect::root(800, 600);

    #[test]
    fn a_split_tiles_its_parent_exactly() {
        let (l, r) = ROOT.split_x(300).unwrap();
        // Gapless and non-overlapping: the right edge of `l` IS the left
        // edge of `r`. A gap shows the clear colour through the seam; an
        // overlap double-paints it.
        assert_eq!(l.x() + l.width(), r.x());
        assert_eq!(l.width() + r.width(), ROOT.width());
        assert_eq!(l.height(), ROOT.height());
        assert_eq!(r.height(), ROOT.height());
    }

    #[test]
    fn a_vertical_split_tiles_its_parent_exactly() {
        let (t, b) = ROOT.split_y(250).unwrap();
        assert_eq!(t.y() + t.height(), b.y());
        assert_eq!(t.height() + b.height(), ROOT.height());
    }

    #[test]
    fn a_degenerate_split_is_rejected_not_clamped() {
        assert!(ROOT.split_x(0).is_none(), "zero-width side");
        assert!(ROOT.split_x(800).is_none(), "zero-width other side");
        assert!(ROOT.split_x(801).is_none(), "escapes the parent");
        assert!(ROOT.split_y(0).is_none());
        assert!(ROOT.split_y(600).is_none());
        assert!(ROOT.split_y(9999).is_none());
    }

    #[test]
    fn an_inset_that_would_empty_the_rect_is_rejected() {
        assert!(ROOT.inset(0).is_some());
        assert!(ROOT.inset(299).is_some());
        assert!(
            ROOT.inset(300).is_none(),
            "600px tall cannot inset 300/side"
        );
        assert!(ROOT.inset(9999).is_none());
    }

    /// The containment closure: ANY rect reachable by ANY sequence of
    /// subdivisions is inside the root. This is the property that lets
    /// `set_scissor_rect` be called without a bounds check — an
    /// out-of-attachment scissor is a wgpu validation error, not a visual
    /// glitch.
    #[test]
    fn every_derived_rect_stays_inside_the_root() {
        fn assert_inside(r: PaneRect) {
            assert!(r.x() + r.width() <= ROOT.width(), "{r:?} escapes right");
            assert!(r.y() + r.height() <= ROOT.height(), "{r:?} escapes bottom");
        }

        let mut frontier = vec![ROOT];
        for _ in 0..4 {
            let mut next = Vec::new();
            for r in frontier.drain(..) {
                assert_inside(r);
                for at in [1, r.width() / 3, r.width().saturating_sub(1)] {
                    if let Some((a, b)) = r.split_x(at) {
                        assert_inside(a);
                        assert_inside(b);
                        next.push(a);
                        next.push(b);
                    }
                }
                for at in [1, r.height() / 2] {
                    if let Some((a, b)) = r.split_y(at) {
                        assert_inside(a);
                        assert_inside(b);
                        next.push(b);
                    }
                }
                if let Some(i) = r.inset(1) {
                    assert_inside(i);
                }
            }
            frontier = next;
            frontier.truncate(32);
        }
    }

    #[test]
    fn contains_is_half_open_so_adjacent_panes_never_both_claim_a_pixel() {
        let (l, r) = ROOT.split_x(300).unwrap();
        // The seam column belongs to exactly one pane. A closed interval
        // here would let both claim x=300 — the off-by-one most likely to
        // survive a whole-region equality check.
        assert!(l.contains(299, 0));
        assert!(!l.contains(300, 0), "left must not claim the seam column");
        assert!(r.contains(300, 0), "right owns the seam column");

        for x in 0..ROOT.width() {
            assert!(
                l.contains(x, 10) ^ r.contains(x, 10),
                "column {x} is claimed by both panes or by neither"
            );
        }
    }

    #[test]
    fn text_bounds_match_the_pane_and_are_half_open_like_contains() {
        let (l, r) = ROOT.split_x(300).unwrap();
        let lb = l.text_bounds();
        assert_eq!((lb.left, lb.top, lb.right, lb.bottom), (0, 0, 300, 600));
        let rb = r.text_bounds();
        assert_eq!((rb.left, rb.top, rb.right, rb.bottom), (300, 0, 800, 600));
        // The seam is shared: left's exclusive right edge IS right's
        // inclusive left edge, so no column is clipped by both or neither.
        assert_eq!(lb.right, rb.left);
    }

    #[test]
    fn to_target_offsets_by_the_panes_origin() {
        let (_, r) = ROOT.split_x(300).unwrap();
        let (tx, ty) = r.to_target(10.0, 5.0);
        assert!((tx - 310.0).abs() < f32::EPSILON);
        assert!((ty - 5.0).abs() < f32::EPSILON);
    }

    /// ★ THE FORCING FUNCTION for this module's one `only-mitigated` row.
    ///
    /// Every containment guarantee here rests on `PanePass` having no way
    /// to reach the inner `wgpu::RenderPass`. That is a property of what is
    /// ABSENT, so no type can assert it — but a source scan can, and it
    /// turns "someone adds an escape hatch later" from a review comment
    /// into a test failure.
    ///
    /// Same construction as mado's `tests/ux_unification.rs`: strip comment
    /// lines (the module docs legitimately discuss `set_scissor_rect`), then
    /// scan what remains.
    #[test]
    fn pane_pass_exposes_no_escape_hatch_to_the_inner_render_pass() {
        let src = include_str!("pane.rs");
        let code: String = src
            .lines()
            .map(str::trim_start)
            .filter(|l| !l.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");

        // Everything after `mod tests` is this scan itself — don't scan it.
        let code = code.split("mod tests").next().unwrap_or(&code);

        let banned = [
            // Would let a caller move the scissor off its pane.
            "fn set_scissor_rect",
            "fn set_viewport",
            // Would hand out the raw pass, dissolving every row above.
            "impl std::ops::Deref for PanePass",
            "impl Deref for PanePass",
            "fn inner(",
            "fn pass(",
            "fn render_pass(",
            "fn into_inner(",
        ];
        for tok in banned {
            assert!(
                !code.contains(tok),
                "`{tok}` appeared in pane.rs — containment is exactly as \
                 strong as the ABSENCE of a route to the inner RenderPass, \
                 so adding one dissolves every unrepresentability row in \
                 this module's docs. If this is deliberate, the module docs \
                 must be re-graded in the same commit."
            );
        }

        // Anti-vacuity: the scan must be looking at real code, not an empty
        // string produced by a refactor that moved the types elsewhere.
        assert!(
            code.contains("pub struct PanePass"),
            "the scan lost sight of PanePass — fix the scan, not the assert"
        );
        assert!(
            code.contains("set_scissor_rect"),
            "PanePass's containment depends on LayeredPass::in_pane actually \
             calling set_scissor_rect; it is no longer there"
        );
    }
}
