use glyphon::{
    Attrs, Buffer, Cache, FontSystem, Metrics, Resolution, Shaping, SwashCache, TextArea,
    TextAtlas, TextRenderer as GlyphonRenderer, Viewport,
};

/// Pure-data configuration for text rendering.
///
/// These are the testable parameters that feed into glyphon rendering.
/// No GPU needed to construct or validate these.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextConfig {
    /// Font size in pixels.
    pub font_size: f32,
    /// Line height in pixels.
    pub line_height: f32,
    /// Text color as RGBA (each component 0.0..=1.0).
    pub color: [f32; 4],
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            font_size: 16.0,
            line_height: 24.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// A text layout request: text content paired with rendering config.
///
/// This is a pure-data type that can be constructed and tested without a GPU.
#[derive(Debug, Clone)]
pub struct TextLayout {
    /// The text content to render.
    pub text: String,
    /// Rendering configuration.
    pub config: TextConfig,
    /// Maximum width in pixels for line wrapping.
    pub max_width: f32,
}

impl TextLayout {
    /// Create a new text layout with the given text, config, and max width.
    #[must_use]
    pub fn new(text: impl Into<String>, config: TextConfig, max_width: f32) -> Self {
        Self {
            text: text.into(),
            config,
            max_width,
        }
    }
}

/// High-level text rendering via glyphon, backed by a wgpu pipeline.
///
/// Requires a GPU device. Use `TextConfig` and `TextLayout` for the
/// testable pure-data layer.
pub struct TextRenderer {
    pub font_system: FontSystem,
    pub swash_cache: SwashCache,
    pub atlas: TextAtlas,
    pub renderer: GlyphonRenderer,
    pub viewport: Viewport,
}

// ── font preload registry ──────────────────────────────────────
//
// cosmic-text's `FontSystem::new()` scans every system font on
// first call (≈150–250 ms on macOS). The scan is independent of
// wgpu init and tear discovery, so we kick it off on a background
// thread at the earliest possible moment in the consumer's main()
// and `TextRenderer::new` later just `join()`s the handle.
//
// On a fresh launch where mado calls `garasu::preload_fonts()`
// before tear setup, the 174 ms FontSystem cost overlaps with
// the ~80 ms of tear discovery + window create + wgpu init —
// total launch shrinks by ~80 ms.

static FONT_PRELOAD: std::sync::OnceLock<
    std::sync::Mutex<Option<std::thread::JoinHandle<FontSystem>>>,
> = std::sync::OnceLock::new();

/// Begin loading the system font database on a background thread.
/// Idempotent — calling more than once is a no-op. Called by
/// consumers (mado, etc.) at the very start of `main` so the
/// 150-250 ms cosmic-text font scan overlaps with everything
/// else that happens before the first text render.
///
/// Uses the persistent on-disk fontdb cache when available
/// (`crate::font_cache`) — first run ever costs the full scan
/// (~180 ms) and writes the cache; subsequent runs deserialize
/// in ~5-15 ms.
pub fn preload_fonts() {
    let lock = FONT_PRELOAD.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = lock.lock().expect("font preload mutex poisoned");
    if guard.is_none() {
        let started = std::time::Instant::now();
        *guard = Some(
            std::thread::Builder::new()
                .name("garasu-font-preload".into())
                .spawn(move || {
                    // Try the on-disk cache first. On hit, build a
                    // FontSystem from the cached fontdb in ~5-15 ms.
                    if let Some(db) = crate::font_cache::try_load_cached_db() {
                        let locale = sys_locale_string();
                        let fs = FontSystem::new_with_locale_and_db(locale, db);
                        tracing::info!(
                            target: "garasu::text",
                            ms = started.elapsed().as_millis() as u64,
                            cache_hit = true,
                            "font preload thread finished"
                        );
                        return fs;
                    }
                    // Cache miss — full scan, then persist for next
                    // run. We have to do the scan via the canonical
                    // FontSystem::new() (which calls
                    // db.load_system_fonts()) so cosmic-text picks
                    // up its monospace/sans/serif defaults; we then
                    // pull the populated db out via a workaround:
                    // build the FontSystem, snapshot its db via
                    // a fresh scan on the side, and save.
                    let fs = FontSystem::new();
                    // Independent scan for caching. The duplicate
                    // load is ~zero cost compared to the saved time
                    // on the NEXT run, and avoids touching
                    // FontSystem's internals.
                    {
                        let mut db = fontdb::Database::new();
                        db.load_system_fonts();
                        crate::font_cache::save_cache(&db);
                    }
                    tracing::info!(
                        target: "garasu::text",
                        ms = started.elapsed().as_millis() as u64,
                        cache_hit = false,
                        "font preload thread finished"
                    );
                    fs
                })
                .expect("spawn font preload thread"),
        );
    }
}

/// Locale string for cosmic-text. Mirrors what `FontSystem::new()`
/// does internally — sys_locale or "en-US" fallback.
fn sys_locale_string() -> String {
    // cosmic-text doesn't re-export sys_locale; we'd add it as
    // a dep just to mirror the default. Use the same fallback
    // string instead — locale-specific font selection is rare
    // in terminal use, and operators can FontSystem::new() the
    // long way if they need it.
    "en-US".to_string()
}

/// Take the preloaded `FontSystem` if `preload_fonts()` was
/// called, otherwise build one synchronously. Used by
/// `TextRenderer::new`.
fn take_or_build_font_system() -> FontSystem {
    if let Some(lock) = FONT_PRELOAD.get() {
        if let Some(handle) = lock.lock().expect("font preload mutex poisoned").take() {
            return handle
                .join()
                .expect("font preload thread panicked");
        }
    }
    // No preload — build synchronously.
    FontSystem::new()
}

impl TextRenderer {
    /// Create a new text renderer for the given device and texture format.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let t_start = std::time::Instant::now();
        let font_system = take_or_build_font_system();
        let t_font = t_start.elapsed();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let renderer =
            GlyphonRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);
        tracing::info!(
            target: "garasu::text",
            font_system_ms = t_font.as_millis() as u64,
            total_ms = t_start.elapsed().as_millis() as u64,
            "text renderer built"
        );

        Self {
            font_system,
            swash_cache,
            atlas,
            renderer,
            viewport,
        }
    }

    /// Create a text buffer with the given content and metrics.
    pub fn create_buffer(&mut self, text: &str, font_size: f32, line_height: f32) -> Buffer {
        let metrics = Metrics::new(font_size, line_height);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_text(
            &mut self.font_system,
            text,
            &Attrs::new(),
            Shaping::Advanced,
        );
        buffer
    }

    /// Create a text buffer with per-span colors and attributes.
    ///
    /// Each span is `(&str, Attrs)` — text with its own color, weight, style.
    /// This enables per-character coloring in terminal rendering.
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

    /// Prepare text areas for rendering in the current frame.
    pub fn prepare<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        text_areas: impl IntoIterator<Item = TextArea<'a>>,
    ) -> Result<(), glyphon::PrepareError> {
        self.viewport.update(queue, Resolution { width, height });

        self.renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            text_areas,
            &mut self.swash_cache,
        )
    }

    /// Render prepared text into the given render pass.
    pub fn render<'pass>(
        &'pass self,
        pass: &mut wgpu::RenderPass<'pass>,
    ) -> Result<(), glyphon::RenderError> {
        self.renderer.render(&self.atlas, &self.viewport, pass)
    }
}

#[cfg(test)]
mod color_emoji_regression {
    //! Regression guards for color-emoji rendering through garasu's
    //! text stack (cosmic-text resolve → swash raster → glyphon).
    //!
    //! Why these exist: a fleet operator read the prompt's Rust
    //! segment 🦀 (red-orange crab, styled "bold red" by seki) as a
    //! "broken red glyph" (2026-06-23). Investigation proved the
    //! pipeline is correct — 🦀 resolves to the host emoji font AND
    //! rasterizes as `SwashContent::Color`, so glyphon paints the
    //! real emoji, not an fg-tinted mask. These tests LOCK that:
    //!   * a cosmic-text / swash bump that drops sbix/COLR support,
    //!   * a `font_cache` format change that drops `.ttc`/emoji faces,
    //! would silently turn every color emoji into a red mask again.
    //! Each test is a forcing-function that fails the build instead.
    //!
    //! Host-gating: the assertions only run when the host actually
    //! has a color-emoji font (true on macOS / the darwin fleet). On
    //! an emoji-less host (minimal Linux CI) they skip loudly rather
    //! than fail — the *separate*, genuine gap of bundling a fallback
    //! emoji font for emoji-less hosts is tracked in the repo's font
    //! TODO, not papered over by a flaky assert here.

    use glyphon::cosmic_text::{CacheKey, SwashContent};
    use glyphon::{Attrs, Buffer, FontSystem, Metrics, Shaping, SwashCache};

    /// Color emoji that every test asserts as a matrix row. All are
    /// pure-emoji codepoints (no text-presentation variant) so a
    /// correct stack MUST resolve + color-raster them.
    const COLOR_EMOJI: &[(&str, &str)] = &[
        ("crab", "🦀"),
        ("fire", "🔥"),
        ("rocket", "🚀"),
        ("check", "✅"),
    ];

    /// True when the db carries a color-emoji-capable family. Mirrors
    /// the cosmic-text fallback the renderer relies on.
    fn has_emoji_font(fs: &FontSystem) -> bool {
        fs.db().faces().any(|f| {
            f.families
                .iter()
                .any(|(n, _)| n.to_lowercase().contains("emoji"))
        })
    }

    /// Resolve `s` to its first glyph: `Some(glyph_id)` (0 == .notdef).
    fn first_glyph(fs: &mut FontSystem, s: &str) -> Option<u16> {
        let mut buf = Buffer::new(fs, Metrics::new(16.0, 20.0));
        buf.set_text(fs, s, &Attrs::new(), Shaping::Advanced);
        buf.shape_until_scroll(fs, false);
        buf.layout_runs().flat_map(|r| r.glyphs.iter()).map(|g| g.glyph_id).next()
    }

    /// The full-pipeline invariant: on an emoji-capable host, every
    /// color emoji resolves to a real glyph AND rasterizes as
    /// `SwashContent::Color` (so glyphon paints the emoji, not an
    /// fg-tinted mask). Matrix: failures aggregate into one assert.
    #[test]
    fn color_emoji_resolve_and_raster_as_color() {
        let mut fs = FontSystem::new();
        if !has_emoji_font(&fs) {
            eprintln!(
                "SKIP color_emoji_resolve_and_raster_as_color: host has no \
                 color-emoji font (minimal Linux/CI). The emoji-less-host \
                 fallback-font gap is tracked separately."
            );
            return;
        }
        let mut swash = SwashCache::new();
        let mut failures = Vec::new();
        for &(name, glyph) in COLOR_EMOJI {
            let Some(gid) = first_glyph(&mut fs, glyph) else {
                failures.push(format!("{name} {glyph}: no glyph produced"));
                continue;
            };
            if gid == 0 {
                failures.push(format!("{name} {glyph}: resolved to .notdef"));
                continue;
            }
            // Raster the glyph and confirm a Color image.
            let mut buf = Buffer::new(&mut fs, Metrics::new(16.0, 20.0));
            buf.set_text(&mut fs, glyph, &Attrs::new(), Shaping::Advanced);
            buf.shape_until_scroll(&mut fs, false);
            let key: Option<CacheKey> = buf
                .layout_runs()
                .flat_map(|r| r.glyphs.iter())
                .map(|g| g.physical((0.0, 0.0), 1.0).cache_key)
                .next();
            match key.and_then(|k| swash.get_image(&mut fs, k).clone()) {
                Some(img) if img.content == SwashContent::Color && !img.data.is_empty() => {}
                Some(img) => failures.push(format!(
                    "{name} {glyph}: rastered as {:?} ({} bytes), expected non-empty Color",
                    img.content,
                    img.data.len()
                )),
                None => failures.push(format!("{name} {glyph}: no raster image")),
            }
        }
        assert!(
            failures.is_empty(),
            "{} color-emoji row(s) failed:\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }

    /// The `font_cache` save→reload round-trip MUST preserve emoji
    /// resolution. The cache serializes a lossy `CachedFace`
    /// projection and reconstructs the db via `push_face_info`; a
    /// future field drop that loses `.ttc` index / emoji families
    /// would regress every color emoji to .notdef. This locks it.
    #[test]
    fn font_cache_roundtrip_preserves_emoji_resolution() {
        let mut fresh = FontSystem::new();
        if !has_emoji_font(&fresh) {
            eprintln!(
                "SKIP font_cache_roundtrip_preserves_emoji_resolution: host \
                 has no color-emoji font (minimal Linux/CI)."
            );
            return;
        }
        // Baseline glyph ids from the fresh scan.
        let baseline: Vec<(&str, u16)> = COLOR_EMOJI
            .iter()
            .map(|&(name, g)| (name, first_glyph(&mut fresh, g).unwrap_or(0)))
            .collect();

        // Round-trip the db through the on-disk cache reconstruction.
        crate::font_cache::save_cache(fresh.db());
        let db = crate::font_cache::try_load_cached_db()
            .expect("cache just saved must reload (schema + fingerprint match)");
        let mut reloaded = FontSystem::new_with_locale_and_db("en-US".to_string(), db);

        let mut failures = Vec::new();
        for (name, want) in baseline {
            let got = first_glyph(&mut reloaded, COLOR_EMOJI.iter().find(|e| e.0 == name).unwrap().1)
                .unwrap_or(0);
            if got == 0 {
                failures.push(format!("{name}: .notdef after cache round-trip (was {want})"));
            }
        }
        assert!(
            failures.is_empty(),
            "cache round-trip dropped emoji resolution:\n  - {}",
            failures.join("\n  - ")
        );
    }
}
