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

impl TextRenderer {
    /// Create a new text renderer for the given device and texture format.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let renderer =
            GlyphonRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);

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
