//! Garasu (硝子) — GPU rendering engine for pleme-io applications.
//!
//! Provides a reusable wgpu + winit + glyphon rendering stack:
//! - `GpuContext`: wgpu device, queue, surface lifecycle
//! - `TextRenderer`: glyphon-backed text layout and rasterization
//! - `TextConfig` / `TextLayout`: pure-data text configuration (testable without GPU)
//! - `ShaderPipeline`: WGSL shader loading and management (testable without GPU)
//! - `AppWindow` / `WindowConfig`: winit window creation with sensible defaults
//! - `GarasuError`: unified error type

pub mod context;
pub mod error;
pub mod shader;
pub mod text;
pub mod window;

pub use context::GpuContext;
pub use error::GarasuError;
pub use shader::{ShaderConfig, ShaderPipeline, ShaderSource, BLUR_SHADER};
pub use text::{TextConfig, TextLayout, TextRenderer};
pub use window::{AppWindow, WindowConfig};

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::path::PathBuf;

    // ---------------------------------------------------------------
    // ShaderPipeline tests — pure logic, no GPU
    // ---------------------------------------------------------------

    #[test]
    fn shader_pipeline_new_is_empty() {
        let pipeline = ShaderPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
    }

    #[test]
    fn shader_pipeline_default_is_empty() {
        let pipeline = ShaderPipeline::default();
        assert!(pipeline.is_empty());
    }

    #[test]
    fn shader_pipeline_add_builtin_and_list() {
        let mut pipeline = ShaderPipeline::new();
        pipeline.add_builtin("blur", BLUR_SHADER);
        assert_eq!(pipeline.len(), 1);
        assert!(!pipeline.is_empty());
        let list = pipeline.list();
        assert_eq!(list[0].name, "blur");
        assert!(list[0].enabled);
    }

    #[test]
    fn shader_pipeline_add_inline_and_get_source() {
        let mut pipeline = ShaderPipeline::new();
        let wgsl = "@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }".to_owned();
        pipeline.add_inline("test", wgsl.clone());
        let source = pipeline.get_source("test").unwrap().unwrap();
        assert_eq!(source, wgsl);
    }

    #[test]
    fn shader_pipeline_add_file_valid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wgsl");
        std::fs::write(&path, "// test shader").unwrap();
        let mut pipeline = ShaderPipeline::new();
        let result = pipeline.add_file("file_shader", path);
        assert!(result.is_ok());
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn shader_pipeline_add_file_missing_returns_error() {
        let mut pipeline = ShaderPipeline::new();
        let result = pipeline.add_file("missing", PathBuf::from("/nonexistent/shader.wgsl"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not found"), "error should mention not found: {msg}");
        assert_eq!(pipeline.len(), 0);
    }

    #[test]
    fn shader_pipeline_get_source_reads_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("custom.wgsl");
        let content = "// custom WGSL shader content\n@vertex fn vs() {}";
        std::fs::write(&path, content).unwrap();
        let mut pipeline = ShaderPipeline::new();
        pipeline.add_file("custom", path).unwrap();
        let source = pipeline.get_source("custom").unwrap().unwrap();
        assert_eq!(source, content);
    }

    #[test]
    fn shader_pipeline_get_source_builtin() {
        let mut pipeline = ShaderPipeline::new();
        pipeline.add_builtin("blur", BLUR_SHADER);
        let source = pipeline.get_source("blur").unwrap().unwrap();
        assert_eq!(source, BLUR_SHADER);
    }

    #[test]
    fn shader_pipeline_get_source_not_found() {
        let pipeline = ShaderPipeline::new();
        assert!(pipeline.get_source("nonexistent").is_none());
    }

    #[test]
    fn shader_pipeline_enable_disable() {
        let mut pipeline = ShaderPipeline::new();
        pipeline.add_builtin("blur", BLUR_SHADER);

        // starts enabled
        assert!(pipeline.is_enabled("blur"));

        // disable
        assert!(pipeline.disable("blur"));
        assert!(!pipeline.is_enabled("blur"));

        // re-enable
        assert!(pipeline.enable("blur"));
        assert!(pipeline.is_enabled("blur"));
    }

    #[test]
    fn shader_pipeline_enable_nonexistent_returns_false() {
        let mut pipeline = ShaderPipeline::new();
        assert!(!pipeline.enable("ghost"));
    }

    #[test]
    fn shader_pipeline_disable_nonexistent_returns_false() {
        let mut pipeline = ShaderPipeline::new();
        assert!(!pipeline.disable("ghost"));
    }

    #[test]
    fn shader_pipeline_is_enabled_nonexistent_returns_false() {
        let pipeline = ShaderPipeline::new();
        assert!(!pipeline.is_enabled("ghost"));
    }

    #[test]
    fn shader_pipeline_active_returns_only_enabled() {
        let mut pipeline = ShaderPipeline::new();
        pipeline.add_builtin("blur", BLUR_SHADER);
        pipeline.add_inline("glow", "// glow".to_owned());
        pipeline.add_inline("crt", "// crt".to_owned());

        // all 3 enabled by default
        assert_eq!(pipeline.active().len(), 3);

        // disable one
        pipeline.disable("glow");
        let active = pipeline.active();
        assert_eq!(active.len(), 2);
        assert!(active.iter().all(|s| s.name != "glow"));

        // disable another
        pipeline.disable("blur");
        let active = pipeline.active();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "crt");
    }

    #[test]
    fn shader_pipeline_multiple_builtins() {
        let mut pipeline = ShaderPipeline::new();
        pipeline.add_builtin("a", "// a");
        pipeline.add_builtin("b", "// b");
        pipeline.add_builtin("c", "// c");
        assert_eq!(pipeline.len(), 3);
        let names: Vec<&str> = pipeline.list().iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn blur_shader_constant_is_nonempty_wgsl() {
        assert!(!BLUR_SHADER.is_empty());
        // Should contain key WGSL constructs
        assert!(BLUR_SHADER.contains("@vertex"));
        assert!(BLUR_SHADER.contains("@fragment"));
        assert!(BLUR_SHADER.contains("@group(0)"));
        assert!(BLUR_SHADER.contains("@binding(0)"));
        assert!(BLUR_SHADER.contains("vs_main"));
        assert!(BLUR_SHADER.contains("fs_main"));
    }

    #[test]
    fn shader_source_builtin_variant() {
        let src = ShaderSource::Builtin("// test");
        assert!(matches!(src, ShaderSource::Builtin("// test")));
    }

    #[test]
    fn shader_source_file_variant() {
        let src = ShaderSource::File(PathBuf::from("/tmp/test.wgsl"));
        assert!(matches!(src, ShaderSource::File(_)));
    }

    #[test]
    fn shader_source_inline_variant() {
        let src = ShaderSource::Inline("// inline".to_owned());
        assert!(matches!(src, ShaderSource::Inline(_)));
    }

    #[test]
    fn shader_config_fields() {
        let config = ShaderConfig {
            name: "test".to_owned(),
            source: ShaderSource::Builtin("// test"),
            enabled: true,
        };
        assert_eq!(config.name, "test");
        assert!(config.enabled);
    }

    #[test]
    fn shader_pipeline_get_source_file_deleted_returns_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ephemeral.wgsl");
        std::fs::write(&path, "// ephemeral").unwrap();
        let mut pipeline = ShaderPipeline::new();
        pipeline.add_file("ephemeral", path.clone()).unwrap();

        // Delete the file after adding
        std::fs::remove_file(&path).unwrap();
        let result = pipeline.get_source("ephemeral").unwrap();
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // TextConfig / TextLayout tests — pure data, no GPU
    // ---------------------------------------------------------------

    #[test]
    fn text_config_default_has_reasonable_values() {
        let config = TextConfig::default();
        assert!(config.font_size > 0.0);
        assert!(config.line_height > 0.0);
        assert!(config.line_height >= config.font_size);
        // default color is white (opaque)
        for (actual, expected) in config.color.iter().zip(&[1.0_f32, 1.0, 1.0, 1.0]) {
            assert!((actual - expected).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn text_config_custom_values() {
        let config = TextConfig {
            font_size: 24.0,
            line_height: 32.0,
            color: [0.0, 0.5, 1.0, 0.8],
        };
        assert!((config.font_size - 24.0).abs() < f32::EPSILON);
        assert!((config.line_height - 32.0).abs() < f32::EPSILON);
        assert!((config.color[1] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn text_layout_stores_fields() {
        let config = TextConfig::default();
        let layout = TextLayout::new("hello world", config.clone(), 400.0);
        assert_eq!(layout.text, "hello world");
        assert!((layout.max_width - 400.0).abs() < f32::EPSILON);
        assert!((layout.config.font_size - config.font_size).abs() < f32::EPSILON);
    }

    #[test]
    fn text_layout_with_empty_text() {
        let layout = TextLayout::new("", TextConfig::default(), 100.0);
        assert!(layout.text.is_empty());
    }

    // ---------------------------------------------------------------
    // WindowConfig tests — pure data, no display server
    // ---------------------------------------------------------------

    #[test]
    fn window_config_default_has_reasonable_values() {
        let config = WindowConfig::default();
        assert!(config.width > 0);
        assert!(config.height > 0);
        assert!(!config.title.is_empty());
        assert!(config.decorations);
        assert!(!config.transparent);
    }

    #[test]
    fn window_config_custom_values() {
        let config = WindowConfig {
            width: 1920,
            height: 1080,
            title: "my app".to_owned(),
            transparent: true,
            decorations: false,
        };
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.title, "my app");
        assert!(config.transparent);
        assert!(!config.decorations);
    }

    // ---------------------------------------------------------------
    // GarasuError tests — error messages and variants
    // ---------------------------------------------------------------

    #[test]
    fn error_gpu_displays_correctly() {
        let err = GarasuError::Gpu("adapter not found".to_owned());
        let msg = err.to_string();
        assert!(msg.contains("GPU error"));
        assert!(msg.contains("adapter not found"));
    }

    #[test]
    fn error_shader_displays_correctly() {
        let err = GarasuError::Shader("invalid WGSL".to_owned());
        let msg = err.to_string();
        assert!(msg.contains("shader error"));
        assert!(msg.contains("invalid WGSL"));
    }

    #[test]
    fn error_window_displays_correctly() {
        let err = GarasuError::Window("no display".to_owned());
        let msg = err.to_string();
        assert!(msg.contains("window error"));
        assert!(msg.contains("no display"));
    }

    #[test]
    fn error_io_displays_correctly() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file missing");
        let err = GarasuError::Io(io_err);
        let msg = err.to_string();
        assert!(msg.contains("I/O error"));
        assert!(msg.contains("file missing"));
    }

    #[test]
    fn error_io_from_conversion() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "denied");
        let err: GarasuError = io_err.into();
        assert!(matches!(err, GarasuError::Io(_)));
    }

    // ---------------------------------------------------------------
    // TextConfig serde round-trip tests
    // ---------------------------------------------------------------

    #[test]
    fn text_config_serde_json_round_trip() {
        let config = TextConfig {
            font_size: 18.5,
            line_height: 28.0,
            color: [0.2, 0.4, 0.6, 0.9],
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: TextConfig = serde_json::from_str(&json).unwrap();
        assert!((deserialized.font_size - 18.5).abs() < f32::EPSILON);
        assert!((deserialized.line_height - 28.0).abs() < f32::EPSILON);
        assert!((deserialized.color[0] - 0.2).abs() < f32::EPSILON);
        assert!((deserialized.color[3] - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn text_config_deserialize_from_json_literal() {
        let json = r#"{"font_size":12.0,"line_height":16.0,"color":[1.0,0.0,0.0,1.0]}"#;
        let config: TextConfig = serde_json::from_str(json).unwrap();
        assert!((config.font_size - 12.0).abs() < f32::EPSILON);
        assert!((config.color[0] - 1.0).abs() < f32::EPSILON);
        assert!((config.color[1]).abs() < f32::EPSILON);
    }

    #[test]
    fn text_config_deserialize_missing_field_fails() {
        let json = r#"{"font_size":12.0}"#;
        let result = serde_json::from_str::<TextConfig>(json);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // WindowConfig serde round-trip tests
    // ---------------------------------------------------------------

    #[test]
    fn window_config_serde_json_round_trip() {
        let config = WindowConfig {
            width: 2560,
            height: 1440,
            title: "test window".to_owned(),
            transparent: true,
            decorations: false,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WindowConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.width, 2560);
        assert_eq!(deserialized.height, 1440);
        assert_eq!(deserialized.title, "test window");
        assert!(deserialized.transparent);
        assert!(!deserialized.decorations);
    }

    #[test]
    fn window_config_deserialize_from_json_literal() {
        let json = r#"{"width":640,"height":480,"title":"retro","transparent":false,"decorations":true}"#;
        let config: WindowConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.title, "retro");
    }

    // ---------------------------------------------------------------
    // Clone + Debug trait tests
    // ---------------------------------------------------------------

    #[test]
    fn text_config_clone_is_independent() {
        let original = TextConfig {
            font_size: 20.0,
            line_height: 30.0,
            color: [0.1, 0.2, 0.3, 0.4],
        };
        let mut cloned = original.clone();
        cloned.font_size = 99.0;
        // original must be unchanged
        assert!((original.font_size - 20.0).abs() < f32::EPSILON);
        assert!((cloned.font_size - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn text_config_debug_contains_fields() {
        let config = TextConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("font_size"));
        assert!(debug.contains("line_height"));
        assert!(debug.contains("color"));
    }

    #[test]
    fn window_config_clone_is_independent() {
        let original = WindowConfig {
            width: 800,
            height: 600,
            title: "original".to_owned(),
            transparent: false,
            decorations: true,
        };
        let mut cloned = original.clone();
        cloned.title = "cloned".to_owned();
        cloned.width = 1024;
        assert_eq!(original.title, "original");
        assert_eq!(original.width, 800);
        assert_eq!(cloned.title, "cloned");
        assert_eq!(cloned.width, 1024);
    }

    #[test]
    fn window_config_debug_contains_fields() {
        let config = WindowConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("width"));
        assert!(debug.contains("height"));
        assert!(debug.contains("title"));
        assert!(debug.contains("transparent"));
        assert!(debug.contains("decorations"));
    }

    // ---------------------------------------------------------------
    // TextLayout edge cases
    // ---------------------------------------------------------------

    #[test]
    fn text_layout_with_unicode_text() {
        let layout = TextLayout::new("硝子ガラス🪟", TextConfig::default(), 500.0);
        assert_eq!(layout.text, "硝子ガラス🪟");
    }

    #[test]
    fn text_layout_with_multiline_text() {
        let text = "line one\nline two\nline three";
        let layout = TextLayout::new(text, TextConfig::default(), 300.0);
        assert_eq!(layout.text, text);
        assert_eq!(layout.text.lines().count(), 3);
    }

    #[test]
    fn text_layout_zero_max_width() {
        let layout = TextLayout::new("narrow", TextConfig::default(), 0.0);
        assert!((layout.max_width).abs() < f32::EPSILON);
    }

    #[test]
    fn text_layout_clone_is_independent() {
        let original = TextLayout::new("hello", TextConfig::default(), 200.0);
        let mut cloned = original.clone();
        cloned.text = "world".to_owned();
        cloned.max_width = 999.0;
        assert_eq!(original.text, "hello");
        assert!((original.max_width - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn text_layout_debug_format() {
        let layout = TextLayout::new("test", TextConfig::default(), 100.0);
        let debug = format!("{layout:?}");
        assert!(debug.contains("TextLayout"));
        assert!(debug.contains("test"));
        assert!(debug.contains("max_width"));
    }

    // ---------------------------------------------------------------
    // GarasuError additional tests
    // ---------------------------------------------------------------

    #[test]
    fn error_debug_format_contains_variant_name() {
        let err = GarasuError::Gpu("test".to_owned());
        let debug = format!("{err:?}");
        assert!(debug.contains("Gpu"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn error_source_io_variant_has_source() {
        use std::error::Error;
        let io_err = io::Error::new(io::ErrorKind::BrokenPipe, "pipe broke");
        let err = GarasuError::Io(io_err);
        // thiserror #[from] sets source() for the Io variant
        assert!(err.source().is_some());
    }

    #[test]
    fn error_source_non_io_variants_have_no_source() {
        use std::error::Error;
        let gpu_err = GarasuError::Gpu("test".to_owned());
        let shader_err = GarasuError::Shader("test".to_owned());
        let window_err = GarasuError::Window("test".to_owned());
        assert!(gpu_err.source().is_none());
        assert!(shader_err.source().is_none());
        assert!(window_err.source().is_none());
    }

    // ---------------------------------------------------------------
    // ShaderSource / ShaderConfig edge cases
    // ---------------------------------------------------------------

    #[test]
    fn shader_source_clone_preserves_variant() {
        let builtin = ShaderSource::Builtin("// code");
        let inline = ShaderSource::Inline("// inline code".to_owned());
        let file = ShaderSource::File(PathBuf::from("/tmp/shader.wgsl"));

        let builtin_clone = builtin.clone();
        let inline_clone = inline.clone();
        let file_clone = file.clone();

        assert!(matches!(builtin_clone, ShaderSource::Builtin("// code")));
        assert!(matches!(inline_clone, ShaderSource::Inline(ref s) if s == "// inline code"));
        assert!(matches!(file_clone, ShaderSource::File(ref p) if p == &PathBuf::from("/tmp/shader.wgsl")));
    }

    #[test]
    fn shader_config_debug_format() {
        let config = ShaderConfig {
            name: "test_shader".to_owned(),
            source: ShaderSource::Builtin("// test"),
            enabled: false,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("test_shader"));
        assert!(debug.contains("Builtin"));
        assert!(debug.contains("false"));
    }

    #[test]
    fn shader_pipeline_preserves_insertion_order() {
        let mut pipeline = ShaderPipeline::new();
        pipeline.add_inline("z_last", "// z".to_owned());
        pipeline.add_builtin("a_first", "// a");
        pipeline.add_inline("m_middle", "// m".to_owned());
        let names: Vec<&str> = pipeline.list().iter().map(|s| s.name.as_str()).collect();
        // insertion order, not alphabetical
        assert_eq!(names, vec!["z_last", "a_first", "m_middle"]);
    }

    #[test]
    fn shader_pipeline_disable_all_yields_empty_active() {
        let mut pipeline = ShaderPipeline::new();
        pipeline.add_builtin("a", "// a");
        pipeline.add_builtin("b", "// b");
        pipeline.disable("a");
        pipeline.disable("b");
        assert!(pipeline.active().is_empty());
        // but list still has them
        assert_eq!(pipeline.len(), 2);
    }
}
