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
                    // One path for both cache hit + miss; guarantees the
                    // color-emoji fallback and keeps the cache a faithful
                    // system-font snapshot. See `build_font_system_scanned`.
                    let fs = build_font_system_scanned();
                    tracing::info!(
                        target: "garasu::text",
                        ms = started.elapsed().as_millis() as u64,
                        "font preload thread finished"
                    );
                    fs
                })
                .expect("spawn font preload thread"),
        );
    }
}

// ── color-emoji fallback guarantee ──────────────────────────────
//
// On a host WITHOUT a system color-emoji font (minimal Linux, a CI
// container, a Nix sandbox), cosmic-text resolves emoji codepoints to
// `.notdef` and the consumer paints an fg-tinted tofu box — the
// 2026-06-23 "red crab" report (the prompt's 🦀 Rust segment). We make
// color emoji render on EVERY host by guaranteeing an emoji-capable
// face is in the db before the FontSystem is built:
//
//   1. build-time embed — when `GARASU_EMOJI_FONT` was set at build
//      (the Nix build points it at nixpkgs `noto-fonts-color-emoji`),
//      `build.rs` bakes the bytes in; the binary is self-contained and
//      needs no font files on the runtime host.
//   2. runtime discovery — otherwise probe a small set of well-known
//      emoji-font locations and load the first that exists.
//
// If the db already carries an emoji family (macOS Apple Color Emoji,
// a Nix-installed Noto on the font path), this is a no-op — the system
// font always wins.

#[cfg(garasu_bundled_emoji)]
const BUNDLED_EMOJI_FONT: &[u8] = include_bytes!(env!("GARASU_EMOJI_FONT_PATH"));

/// Well-known color-emoji font paths probed at runtime when no font was
/// embedded at build time and the host scan found no emoji family.
/// Noto first (the small 10 MB color font); Apple Color Emoji last (a
/// 192 MB `.ttc` only reached on a macOS box whose scan somehow missed
/// it — in practice never, since macOS always has it in the scan).
const EMOJI_FALLBACK_PATHS: &[&str] = &[
    "/usr/share/fonts/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
    "/run/current-system/sw/share/X11/fonts/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
];

/// True when the db carries a color-emoji-capable family — the exact
/// signal cosmic-text's per-glyph fallback needs to resolve an emoji
/// codepoint instead of returning `.notdef`.
fn db_has_emoji(db: &fontdb::Database) -> bool {
    db.faces().any(|f| {
        f.families
            .iter()
            .any(|(name, _)| name.to_lowercase().contains("emoji"))
    })
}

/// Ensure `db` carries a color-emoji face. No-op (returns `false`) when
/// one is already present. Returns `true` when a fallback was loaded.
/// Tries the build-time embed first, then runtime discovery — so a
/// Nix-built binary is self-contained and a plain `cargo`-built one
/// still finds a system Noto/Apple font when present.
fn ensure_emoji_fallback(db: &mut fontdb::Database) -> bool {
    if db_has_emoji(db) {
        return false;
    }
    #[cfg(garasu_bundled_emoji)]
    if !BUNDLED_EMOJI_FONT.is_empty() {
        db.load_font_data(BUNDLED_EMOJI_FONT.to_vec());
        tracing::info!(
            target: "garasu::text",
            source = "bundled",
            "loaded color-emoji fallback font"
        );
        return true;
    }
    for path in EMOJI_FALLBACK_PATHS {
        if let Ok(bytes) = std::fs::read(path) {
            db.load_font_data(bytes);
            tracing::info!(
                target: "garasu::text",
                source = %path,
                "loaded color-emoji fallback font"
            );
            return true;
        }
    }
    tracing::debug!(
        target: "garasu::text",
        "no color-emoji fallback available — emoji may render as tofu"
    );
    false
}

/// Explicitly load Nix-managed font directories into `db`.
///
/// `fontdb::load_system_fonts` scans the OS-standard dirs (and, on
/// Linux, fontconfig) — but a font installed by Nix (`home.packages` /
/// a module's `extraPackages` / NixOS `fonts.packages`) lands in a Nix
/// profile that the OS scan misses, most notably the **nix-darwin
/// per-user profile** (`/etc/profiles/per-user/$USER/share/fonts`).
/// Loading these dirs explicitly is what makes a Nix-declared emoji
/// font (mado's `extraPackages = [ "noto-fonts-color-emoji" ]`)
/// discoverable on every Nix platform — closing the loop between the
/// declared dependency and the font the renderer actually sees.
/// `load_fonts_dir` is recursive and a no-op for a missing dir.
fn load_nix_font_dirs(db: &mut fontdb::Database) {
    let mut dirs: Vec<std::path::PathBuf> = vec![
        // System profiles (NixOS + nix-darwin).
        "/run/current-system/sw/share/fonts".into(),
        "/run/current-system/sw/share/X11/fonts".into(),
    ];
    if let Some(user) = std::env::var_os("USER") {
        // nix-darwin per-user profile — where `extraPackages` lands.
        dirs.push(
            std::path::Path::new("/etc/profiles/per-user")
                .join(&user)
                .join("share/fonts"),
        );
    }
    if let Some(home) = dirs::home_dir() {
        // Classic single-user Nix + Linux home-manager profile.
        dirs.push(home.join(".nix-profile/share/fonts"));
        dirs.push(home.join(".local/state/nix/profiles/profile/share/fonts"));
    }
    for dir in dirs {
        if dir.is_dir() {
            db.load_fonts_dir(&dir);
            tracing::debug!(target: "garasu::text", dir = %dir.display(), "loaded nix font dir");
        }
    }
}

/// Build a `FontSystem` from the system font scan (cache-backed),
/// guaranteeing a color-emoji fallback before construction. The single
/// place both the preload thread and the synchronous path route
/// through, so the emoji guarantee + cache behaviour can't drift apart.
///
/// Cache contract: only the *system* scan is persisted (the emoji
/// fallback is re-applied on every build from the embed / system path,
/// never cached), so the on-disk cache stays a faithful snapshot of the
/// host's own fonts.
fn build_font_system_scanned() -> FontSystem {
    let locale = sys_locale_string();
    // Cache hit — reconstruct the system db, then layer Nix font dirs +
    // the emoji fallback (both re-applied every build, never cached).
    if let Some(mut db) = crate::font_cache::try_load_cached_db() {
        load_nix_font_dirs(&mut db);
        ensure_emoji_fallback(&mut db);
        return FontSystem::new_with_locale_and_db(locale, db);
    }
    // Cache miss — one system scan (the old path scanned twice: once via
    // `FontSystem::new()` and again to populate the cache). Persist the
    // system-only db, then layer Nix font dirs + the emoji fallback.
    let mut db = fontdb::Database::new();
    db.load_system_fonts();
    crate::font_cache::save_cache(&db);
    load_nix_font_dirs(&mut db);
    ensure_emoji_fallback(&mut db);
    FontSystem::new_with_locale_and_db(locale, db)
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
    // No preload — build synchronously (same emoji-guaranteed path).
    build_font_system_scanned()
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

    /// The bare-host guarantee: starting from a db with NO system
    /// fonts (a minimal Linux/CI container / Nix sandbox),
    /// `ensure_emoji_fallback` must add an emoji family AND make 🦀
    /// resolve to a real glyph — via the build-time embed
    /// (`GARASU_EMOJI_FONT`) or a discovered system Noto/Apple font.
    /// This is the test that proves the emoji-less-host gap is closed.
    /// Skips loudly only when neither an embed nor any known system
    /// emoji font exists (then there is genuinely nothing to load).
    #[test]
    fn emoji_fallback_closes_bare_host_gap() {
        let mut db = fontdb::Database::new();
        assert!(!super::db_has_emoji(&db), "fresh empty db has no emoji family");

        let loaded = super::ensure_emoji_fallback(&mut db);
        if !loaded {
            eprintln!(
                "SKIP emoji_fallback_closes_bare_host_gap: no build-time embed \
                 and no system emoji font on this host — nothing to fall back to."
            );
            return;
        }
        assert!(
            super::db_has_emoji(&db),
            "ensure_emoji_fallback must add a color-emoji family"
        );
        let mut fs = FontSystem::new_with_locale_and_db("en-US".to_string(), db);
        let gid = first_glyph(&mut fs, "🦀").unwrap_or(0);
        assert_ne!(gid, 0, "🦀 must resolve via the fallback font, not .notdef");
    }

    /// `ensure_emoji_fallback` is a no-op when the db already has an
    /// emoji family — the host's own font always wins (no double-load,
    /// no overriding Apple Color Emoji with a bundled Noto).
    #[test]
    fn emoji_fallback_is_noop_when_host_has_emoji() {
        let fs = FontSystem::new();
        if !has_emoji_font(&fs) {
            eprintln!("SKIP emoji_fallback_is_noop_when_host_has_emoji: emoji-less host");
            return;
        }
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        assert!(super::db_has_emoji(&db), "host scan carries an emoji family");
        assert!(
            !super::ensure_emoji_fallback(&mut db),
            "fallback must be a no-op when the host already has emoji"
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
