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
}
