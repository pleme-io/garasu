//! `ShapeCache` — a bounded, keyed cache of *shaped* text.
//!
//! ## The problem
//!
//! cosmic-text shaping (itemisation → font fallback → HarfBuzz-class glyph
//! positioning → line breaking) is the expensive half of drawing text, and a
//! GPU app's render loop redraws continuously. A consumer that builds a fresh
//! `Buffer` per string per frame therefore re-does *all* of that work every
//! frame to produce glyph positions that have not changed — for an address
//! bar, a status line, a start screen, or a page body that only changes on
//! navigation.
//!
//! This is the fleet's second implementation of the fix, which is why it
//! lives here rather than in an app: mado has carried a private shape cache
//! (`render.rs`, `ShapeKey`/`shape_run`, LRU of 4096) whose own doc records
//! that keying shaped runs by text + attrs "avoids ~99% of cosmic-text shape
//! calls in a typical interactive session". namimado needed the same thing,
//! so the primitive is promoted instead of copied.
//!
//! ## The key IS the recipe
//!
//! The dangerous failure mode for a shape cache is a key that omits a
//! shaping input: the cache then hands back a buffer shaped for *different*
//! inputs — wrapped to the wrong width, in the wrong font, at the wrong
//! size — and the frame is silently, subtly wrong with nothing to error on.
//!
//! So there is exactly one type, [`ShapeRequest`], and it is **both** the
//! cache key **and** the complete argument list the buffer is built from.
//! [`ShapeRequest::build`] can only read what the key carries, so "an input
//! that affects shaping but is not in the key" has no way to exist.
//!
//! This is also why mado's key could omit wrap width and this one cannot: a
//! terminal shapes single-line runs, while a browser wraps paragraphs, and a
//! cache shared by both has to carry the union.
//!
//! ## Measurement rides along
//!
//! The cached value is a [`ShapedText`], which carries the measured width,
//! height and line count beside the buffer. Measuring means walking
//! `layout_runs()`, and a caller that centres or right-aligns text needs that
//! number every frame — so caching the buffer while re-measuring it each
//! frame would leave half the win on the table.

use std::cell::{Cell, RefCell};
use std::num::NonZeroUsize;
use std::sync::Arc;

use glyphon::{Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, Style, Weight};

/// A font family, in a form that can be hashed and stored.
///
/// `glyphon::Family` borrows its name, so it cannot be a cache key. This is
/// the owned mirror; [`FamilyKey::as_family`] converts back.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FamilyKey {
    Monospace,
    SansSerif,
    Serif,
    Cursive,
    Fantasy,
    /// A specific family by name, e.g. `"JetBrainsMono Nerd Font"`.
    Name(Box<str>),
}

impl FamilyKey {
    /// Borrow as a `glyphon::Family` for the duration of `self`.
    #[must_use]
    pub fn as_family(&self) -> Family<'_> {
        match self {
            Self::Monospace => Family::Monospace,
            Self::SansSerif => Family::SansSerif,
            Self::Serif => Family::Serif,
            Self::Cursive => Family::Cursive,
            Self::Fantasy => Family::Fantasy,
            Self::Name(n) => Family::Name(n),
        }
    }

    /// A named family from any string.
    #[must_use]
    pub fn named(name: &str) -> Self {
        Self::Name(name.into())
    }
}

/// One styled run inside a shaped buffer.
///
/// A multi-span request is how a caller paints two colours (or two families)
/// in ONE shaped run — which matters because two separately shaped runs can
/// drift apart, while one run's glyph positions are computed together.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeSpan {
    pub text: Box<str>,
    pub family: FamilyKey,
    /// sRGB RGBA. `None` inherits the `TextArea`'s `default_color` at draw
    /// time, which is how a caller keeps one buffer usable in several
    /// colours without re-shaping it.
    pub color: Option<[u8; 4]>,
    /// CSS-style numeric weight (400 regular, 700 bold).
    pub weight: u16,
    pub italic: bool,
}

impl ShapeSpan {
    /// A plain run: default weight, upright, colour inherited at draw time.
    #[must_use]
    pub fn plain(text: &str, family: FamilyKey) -> Self {
        Self {
            text: text.into(),
            family,
            color: None,
            weight: 400,
            italic: false,
        }
    }

    /// Give this span its own colour.
    #[must_use]
    pub fn colored(mut self, rgba: [u8; 4]) -> Self {
        self.color = Some(rgba);
        self
    }

    /// Set the numeric weight.
    #[must_use]
    pub fn weight(mut self, w: u16) -> Self {
        self.weight = w;
        self
    }

    /// Mark the span italic.
    #[must_use]
    pub fn italic(mut self, yes: bool) -> Self {
        self.italic = yes;
        self
    }

    fn attrs(&self) -> Attrs<'_> {
        let mut a = Attrs::new()
            .family(self.family.as_family())
            .weight(Weight(self.weight));
        if self.italic {
            a = a.style(Style::Italic);
        }
        if let Some([r, g, b, alpha]) = self.color {
            a = a.color(Color::rgba(r, g, b, alpha));
        }
        a
    }
}

/// Canonicalise an `f32` for use in a hash key.
///
/// Two values that are `==` must hash the same, and `f32` breaks that twice:
/// `-0.0 == 0.0` with different bits, and `NaN != NaN` at all. A NaN key
/// would never match itself, so a caller that passed one would get a cache
/// that silently never hits — the worst kind of failure, since it looks
/// exactly like a working cache. Both are folded to `+0.0`.
fn canon(v: f32) -> u32 {
    if v.is_nan() { 0.0_f32 } else { v + 0.0 }.to_bits()
}

/// A complete description of a shaped buffer: the cache key AND the recipe.
///
/// Every field affects the resulting glyph positions, and the builder reads
/// nothing else — so a shaping input outside the key cannot exist.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeRequest {
    spans: Vec<ShapeSpan>,
    font_size_bits: u32,
    line_height_bits: u32,
    /// `set_size` bounds. `None` means unwrapped (one line, measured to its
    /// natural width) — which is a genuinely different shaping result from
    /// any wrapped width, hence part of the key.
    wrap: Option<(u32, u32)>,
}

impl ShapeRequest {
    /// An unwrapped request — a single line, measured to its natural width.
    #[must_use]
    pub fn new(spans: Vec<ShapeSpan>, font_size: f32, line_height: f32) -> Self {
        Self {
            spans,
            font_size_bits: canon(font_size),
            line_height_bits: canon(line_height),
            wrap: None,
        }
    }

    /// A one-span convenience for the common case.
    #[must_use]
    pub fn line(text: &str, family: FamilyKey, font_size: f32, line_height: f32) -> Self {
        Self::new(
            vec![ShapeSpan::plain(text, family)],
            font_size,
            line_height,
        )
    }

    /// Wrap to `width` × `height`. Changes the shaping result, so it changes
    /// the key.
    #[must_use]
    pub fn wrapped(mut self, width: f32, height: f32) -> Self {
        self.wrap = Some((canon(width), canon(height)));
        self
    }

    #[must_use]
    pub fn font_size(&self) -> f32 {
        f32::from_bits(self.font_size_bits)
    }

    #[must_use]
    pub fn line_height(&self) -> f32 {
        f32::from_bits(self.line_height_bits)
    }

    #[must_use]
    pub fn spans(&self) -> &[ShapeSpan] {
        &self.spans
    }

    /// Total UTF-8 length of the request's text — a cheap proxy for how much
    /// a cached entry costs, for a consumer that wants to bound by bytes.
    #[must_use]
    pub fn text_len(&self) -> usize {
        self.spans.iter().map(|s| s.text.len()).sum()
    }

    /// Build the shaped text. The ONLY place a buffer is produced from a
    /// request, so key and result cannot diverge.
    #[must_use]
    pub fn build(&self, fs: &mut FontSystem) -> ShapedText {
        let metrics = Metrics::new(self.font_size(), self.line_height());
        let mut buffer = Buffer::new(fs, metrics);

        // Size FIRST, then text. `set_text` lays out at the buffer's current
        // width, and `set_size` re-lays out on any change — so setting the
        // text first and the size second runs the whole line-breaking pass
        // TWICE for every wrapped run. Shaping itself is retained across the
        // two (cosmic-text keeps `shape_opt`), but the layout pass is not.
        let (w, h) = match self.wrap {
            Some((w, h)) => (Some(f32::from_bits(w)), Some(f32::from_bits(h))),
            None => (None, None),
        };
        buffer.set_size(fs, w, h);

        let attrs: Vec<(&str, Attrs<'_>)> =
            self.spans.iter().map(|s| (&*s.text, s.attrs())).collect();
        buffer.set_rich_text(
            fs,
            attrs.iter().map(|(t, a)| (*t, a.clone())),
            &Attrs::new(),
            Shaping::Advanced,
            None,
        );
        buffer.shape_until_scroll(fs, false);

        // Measure while the buffer is hot — a caller that centres or
        // right-aligns needs this every frame, so caching the glyphs but
        // re-walking the runs would leave half the saving on the table.
        let mut width = 0.0_f32;
        let mut lines = 0_usize;
        for run in buffer.layout_runs() {
            lines += 1;
            if run.line_w > width {
                width = run.line_w;
            }
        }
        let lines = lines.max(1);
        #[allow(clippy::cast_precision_loss)]
        let height = lines as f32 * self.line_height();

        ShapedText {
            buffer,
            width,
            height,
            lines,
        }
    }
}

/// A shaped buffer plus its measurements.
#[derive(Debug)]
pub struct ShapedText {
    pub buffer: Buffer,
    /// Widest laid-out line, in the same pixels the buffer was shaped at.
    pub width: f32,
    /// `lines * line_height`.
    pub height: f32,
    /// Laid-out line count, at least 1.
    pub lines: usize,
}

/// Default entry cap. Sized for a UI frame's live strings plus several
/// frames of variation, not for a document.
pub const DEFAULT_CAPACITY: usize = 1024;

/// Default retained-byte budget (32 MiB).
///
/// An entry count alone is NOT a memory bound here, and that difference is
/// the whole reason this cache carries a second limit. A terminal's cached
/// runs are cells — tens to hundreds of bytes — so mado can bound its own
/// cache at 4096 entries and call it a few MB. A browser's cached runs are
/// paragraphs: cosmic-text retains both the shaped glyphs and the laid-out
/// glyphs per line (`ShapeGlyph` ~90 B, `LayoutGlyph` ~80 B), so a
/// 2000-character paragraph is on the order of 400 KB. 4096 of those is
/// multiple gigabytes.
///
/// So eviction is bounded by BOTH: entries and estimated retained bytes,
/// whichever binds first.
pub const DEFAULT_BYTE_BUDGET: usize = 32 * 1024 * 1024;

/// Estimated retained bytes per character of shaped text.
///
/// cosmic-text keeps `shape_opt` AND `layout_opt` per line, so a character
/// costs roughly one `ShapeGlyph` (~90 B) plus one `LayoutGlyph` (~80 B)
/// plus the source text and per-line overhead. Deliberately an
/// over-estimate: budgeting low and evicting early is a performance cost,
/// budgeting high and evicting late is an out-of-memory.
const BYTES_PER_CHAR: usize = 200;

/// A bounded LRU of shaped text.
///
/// Interior-mutable on purpose: shaping needs `&mut FontSystem`, and the
/// `FontSystem` usually lives in the same struct as the cache
/// ([`crate::TextRenderer`]). Taking `&self` here lets a caller pass
/// `&mut self.font_system` alongside `&self.shape_cache` as two disjoint
/// field borrows, instead of having to restructure ownership.
pub struct ShapeCache {
    inner: RefCell<lru::LruCache<ShapeRequest, Arc<ShapedText>>>,
    /// Estimated retained bytes currently held.
    bytes: Cell<usize>,
    byte_budget: usize,
    hits: Cell<u64>,
    misses: Cell<u64>,
}

impl ShapeCache {
    /// A cache holding at most `capacity` shaped entries, within the default
    /// byte budget.
    #[must_use]
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self::with_byte_budget(capacity, DEFAULT_BYTE_BUDGET)
    }

    /// A cache bounded by BOTH an entry count and an estimated retained-byte
    /// budget, whichever binds first. Reach for this when the cached text is
    /// document-sized rather than UI-sized.
    #[must_use]
    pub fn with_byte_budget(capacity: NonZeroUsize, byte_budget: usize) -> Self {
        Self {
            inner: RefCell::new(lru::LruCache::new(capacity)),
            bytes: Cell::new(0),
            byte_budget,
            hits: Cell::new(0),
            misses: Cell::new(0),
        }
    }

    /// Estimated retained bytes currently held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes.get()
    }

    /// Return the shaped text for `req`, shaping it only on a miss.
    pub fn shaped(&self, fs: &mut FontSystem, req: &ShapeRequest) -> Arc<ShapedText> {
        if let Some(hit) = self.inner.borrow_mut().get(req) {
            self.hits.set(self.hits.get() + 1);
            return Arc::clone(hit);
        }
        self.misses.set(self.misses.get() + 1);
        let shaped = Arc::new(req.build(fs));
        let cost = req.text_len() * BYTES_PER_CHAR;
        {
            let mut inner = self.inner.borrow_mut();
            // `put` on an existing key replaces it; subtract the old cost so
            // the running total tracks what is actually held.
            if let Some(old) = inner.put(req.clone(), Arc::clone(&shaped)) {
                let _ = old;
            }
            self.bytes.set(self.bytes.get().saturating_add(cost));
            // Evict least-recently-used until inside the byte budget. Always
            // keep at least the entry just inserted, or a single run larger
            // than the whole budget would evict itself and the cache would
            // miss forever on it while doing all the eviction work.
            while self.bytes.get() > self.byte_budget && inner.len() > 1 {
                match inner.pop_lru() {
                    Some((k, _)) => {
                        let freed = k.text_len() * BYTES_PER_CHAR;
                        self.bytes.set(self.bytes.get().saturating_sub(freed));
                    }
                    None => break,
                }
            }
        }
        shaped
    }

    /// Entries currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.borrow().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `(hits, misses)` since construction. A consumer can log the ratio to
    /// confirm the cache is actually working — a cache that is silently
    /// missing every lookup (a key that varies when it should not) performs
    /// worse than none, and looks identical from the outside.
    #[must_use]
    pub fn stats(&self) -> (u64, u64) {
        (self.hits.get(), self.misses.get())
    }

    /// Drop every entry. Call when the `FontSystem`'s font database changes
    /// — cached glyph positions are only valid for the fonts they were
    /// shaped against.
    pub fn clear(&self) {
        self.inner.borrow_mut().clear();
        self.bytes.set(0);
    }
}

impl Default for ShapeCache {
    fn default() -> Self {
        Self::new(NonZeroUsize::new(DEFAULT_CAPACITY).expect("non-zero"))
    }
}

impl std::fmt::Debug for ShapeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (h, m) = self.stats();
        f.debug_struct("ShapeCache")
            .field("len", &self.len())
            .field("hits", &h)
            .field("misses", &m)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bare `FontSystem` needs no GPU, so the whole cache is testable
    /// without an adapter.
    fn fs() -> FontSystem {
        FontSystem::new()
    }

    fn req(text: &str) -> ShapeRequest {
        ShapeRequest::line(text, FamilyKey::Monospace, 16.0, 20.0)
    }

    #[test]
    fn identical_requests_shape_once() {
        let mut fs = fs();
        let c = ShapeCache::default();
        let a = c.shaped(&mut fs, &req("hello"));
        let b = c.shaped(&mut fs, &req("hello"));
        assert!(Arc::ptr_eq(&a, &b), "second lookup must reuse the shape");
        assert_eq!(c.stats(), (1, 1));
        assert_eq!(c.len(), 1);
    }

    /// THE correctness property. Every field of the request must change the
    /// key — an input that shapes differently but keys the same would hand
    /// back a buffer for the wrong inputs, and the frame would be silently
    /// wrong with nothing to error on.
    #[test]
    fn every_shaping_input_changes_the_key() {
        let base = ShapeRequest::line("hello", FamilyKey::Monospace, 16.0, 20.0);
        let variants = [
            ("text", ShapeRequest::line("hellp", FamilyKey::Monospace, 16.0, 20.0)),
            ("family", ShapeRequest::line("hello", FamilyKey::SansSerif, 16.0, 20.0)),
            ("named family", ShapeRequest::line("hello", FamilyKey::named("Menlo"), 16.0, 20.0)),
            ("font size", ShapeRequest::line("hello", FamilyKey::Monospace, 17.0, 20.0)),
            ("line height", ShapeRequest::line("hello", FamilyKey::Monospace, 16.0, 21.0)),
            ("wrap", base.clone().wrapped(100.0, 40.0)),
            (
                "wrap width",
                base.clone().wrapped(100.0, 40.0),
            ),
            (
                "colour",
                ShapeRequest::new(
                    vec![ShapeSpan::plain("hello", FamilyKey::Monospace).colored([1, 2, 3, 255])],
                    16.0,
                    20.0,
                ),
            ),
            (
                "weight",
                ShapeRequest::new(
                    vec![ShapeSpan::plain("hello", FamilyKey::Monospace).weight(700)],
                    16.0,
                    20.0,
                ),
            ),
            (
                "italic",
                ShapeRequest::new(
                    vec![ShapeSpan::plain("hello", FamilyKey::Monospace).italic(true)],
                    16.0,
                    20.0,
                ),
            ),
            (
                "span split",
                ShapeRequest::new(
                    vec![
                        ShapeSpan::plain("hel", FamilyKey::Monospace),
                        ShapeSpan::plain("lo", FamilyKey::Monospace),
                    ],
                    16.0,
                    20.0,
                ),
            ),
        ];
        for (name, v) in variants {
            assert_ne!(base, v, "{name} must change the cache key");
        }
        // Two different wrap widths are also distinct.
        assert_ne!(
            base.clone().wrapped(100.0, 40.0),
            base.clone().wrapped(200.0, 40.0),
            "wrap width must change the key",
        );
        assert_ne!(
            base.clone().wrapped(100.0, 40.0),
            base.wrapped(100.0, 80.0),
            "wrap height must change the key",
        );
    }

    /// A wrapped request genuinely shapes differently from an unwrapped one —
    /// this is the input mado's terminal-shaped key could omit and a browser
    /// cannot.
    #[test]
    fn wrapping_changes_the_shaped_result() {
        let mut fs = fs();
        let long = "the quick brown fox jumps over the lazy dog again and again";
        let wide = ShapeRequest::line(long, FamilyKey::Monospace, 16.0, 20.0)
            .wrapped(2000.0, 400.0)
            .build(&mut fs);
        let narrow = ShapeRequest::line(long, FamilyKey::Monospace, 16.0, 20.0)
            .wrapped(120.0, 400.0)
            .build(&mut fs);
        assert_eq!(wide.lines, 1, "a wide column fits one line");
        assert!(
            narrow.lines > wide.lines,
            "a narrow column must wrap to more lines ({} vs {})",
            narrow.lines,
            wide.lines,
        );
        assert!(narrow.width <= 120.0 + 0.5, "wrapped width stays in column");
    }

    /// NaN and -0.0 are the two ways `f32` breaks hashing. A NaN key that
    /// never matched itself would look exactly like a working cache while
    /// never hitting, so both fold to +0.0.
    #[test]
    fn nan_and_negative_zero_canonicalise() {
        let nan_a = ShapeRequest::line("x", FamilyKey::Monospace, f32::NAN, 20.0);
        let nan_b = ShapeRequest::line("x", FamilyKey::Monospace, f32::NAN, 20.0);
        assert_eq!(nan_a, nan_b, "two NaN requests must be equal");

        let neg = ShapeRequest::line("x", FamilyKey::Monospace, -0.0, 20.0);
        let pos = ShapeRequest::line("x", FamilyKey::Monospace, 0.0, 20.0);
        assert_eq!(neg, pos, "-0.0 and 0.0 must key the same");

        // And the cache actually hits on them.
        let mut fs = fs();
        let c = ShapeCache::default();
        let _ = c.shaped(&mut fs, &nan_a);
        let _ = c.shaped(&mut fs, &nan_b);
        assert_eq!(c.stats(), (1, 1), "a NaN key must still hit");
    }

    /// The cap is real — an unbounded cache keyed on page text is a memory
    /// leak, so eviction must actually evict.
    #[test]
    fn capacity_evicts_least_recently_used() {
        let mut fs = fs();
        let c = ShapeCache::new(NonZeroUsize::new(2).unwrap());
        let _ = c.shaped(&mut fs, &req("a"));
        let _ = c.shaped(&mut fs, &req("b"));
        assert_eq!(c.len(), 2);
        // Touch "a" so "b" becomes least-recently-used.
        let _ = c.shaped(&mut fs, &req("a"));
        let _ = c.shaped(&mut fs, &req("c"));
        assert_eq!(c.len(), 2, "capacity is a hard bound");

        // "a" survived (recently used); "b" was evicted.
        let before = c.stats().1;
        let _ = c.shaped(&mut fs, &req("a"));
        assert_eq!(c.stats().1, before, "'a' should still be cached");
        let _ = c.shaped(&mut fs, &req("b"));
        assert_eq!(c.stats().1, before + 1, "'b' should have been evicted");
    }

    #[test]
    fn measurements_ride_along_with_the_buffer() {
        let mut fs = fs();
        let s = req("hello").build(&mut fs);
        assert!(s.width > 0.0, "a non-empty run has width");
        assert_eq!(s.lines, 1);
        assert!((s.height - 20.0).abs() < 0.001, "height = lines * line_height");
    }

    /// Empty text still reports one line, so a caller advancing by
    /// `height` never advances by zero and stacks rows on top of each other.
    #[test]
    fn empty_text_is_one_line_tall() {
        let mut fs = fs();
        let s = req("").build(&mut fs);
        assert_eq!(s.lines, 1);
        assert!(s.height > 0.0);
    }

    /// The byte budget is the bound that actually matters for a browser: an
    /// entry COUNT is not a memory bound when one entry can be a paragraph.
    #[test]
    fn byte_budget_evicts_before_the_entry_count_does() {
        let mut fs = fs();
        // Room for 100 entries but only ~2 KB — the bytes must bind first.
        let c = ShapeCache::with_byte_budget(NonZeroUsize::new(100).unwrap(), 2_000);
        let big = "x".repeat(50); // 50 chars * 200 B = 10 KB estimated
        for i in 0..5 {
            let mut t = big.clone();
            t.push_str(&i.to_string());
            let _ = c.shaped(&mut fs, &req(&t));
        }
        assert!(
            c.len() < 5,
            "the byte budget should have evicted; {} entries held",
            c.len(),
        );
        assert!(c.len() >= 1, "the just-inserted entry always survives");
    }

    /// A single run larger than the whole budget must still be served — if it
    /// evicted itself the cache would miss on it forever while paying all the
    /// eviction work.
    #[test]
    fn an_entry_larger_than_the_budget_still_survives() {
        let mut fs = fs();
        let c = ShapeCache::with_byte_budget(NonZeroUsize::new(100).unwrap(), 100);
        let huge = "y".repeat(500);
        let _ = c.shaped(&mut fs, &req(&huge));
        assert_eq!(c.len(), 1);
        let before = c.stats().1;
        let _ = c.shaped(&mut fs, &req(&huge));
        assert_eq!(c.stats().1, before, "the oversized entry must still hit");
    }

    #[test]
    fn clear_resets_the_byte_total() {
        let mut fs = fs();
        let c = ShapeCache::default();
        let _ = c.shaped(&mut fs, &req("hello"));
        assert!(c.bytes() > 0);
        c.clear();
        assert_eq!(c.bytes(), 0);
    }

    #[test]
    fn clear_drops_every_entry() {
        let mut fs = fs();
        let c = ShapeCache::default();
        let _ = c.shaped(&mut fs, &req("a"));
        assert_eq!(c.len(), 1);
        c.clear();
        assert!(c.is_empty());
    }
}
