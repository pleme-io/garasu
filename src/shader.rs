use crate::error::GarasuError;
use std::path::PathBuf;

/// A built-in Gaussian blur post-processing shader in WGSL.
pub const BLUR_SHADER: &str = r"
struct Uniforms {
    time: f32,
    _pad: f32,
    resolution: vec2<f32>,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

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
    out.uv = (pos + vec2<f32>(1.0)) * 0.5;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(1.0) / uniforms.resolution;
    var color = vec4<f32>(0.0);
    let offsets = array<f32, 5>(-2.0, -1.0, 0.0, 1.0, 2.0);
    let weights = array<f32, 5>(0.06136, 0.24477, 0.38774, 0.24477, 0.06136);
    for (var i = 0u; i < 5u; i = i + 1u) {
        let offset = vec2<f32>(offsets[i] * texel.x, 0.0);
        color = color + textureSample(input_texture, input_sampler, in.uv + offset) * weights[i];
    }
    return color;
}
";

/// Where a shader's WGSL source comes from.
#[derive(Debug, Clone)]
pub enum ShaderSource {
    /// Compiled into the binary as a static string.
    Builtin(&'static str),
    /// Loaded from a file on disk.
    File(PathBuf),
    /// Provided as an inline string at runtime.
    Inline(String),
}

/// Configuration for a single shader in the pipeline.
#[derive(Debug, Clone)]
pub struct ShaderConfig {
    /// Human-readable name for this shader.
    pub name: String,
    /// Where the WGSL source comes from.
    pub source: ShaderSource,
    /// Whether this shader is active in the pipeline.
    pub enabled: bool,
}

/// Manages an ordered list of shader configurations.
///
/// This is the pure-logic layer for shader management. It handles
/// loading, enabling/disabling, and retrieving WGSL source without
/// requiring a GPU device. The actual wgpu pipeline creation happens
/// separately when a `GpuContext` is available.
pub struct ShaderPipeline {
    shaders: Vec<ShaderConfig>,
}

impl ShaderPipeline {
    /// Create an empty shader pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            shaders: Vec::new(),
        }
    }

    /// Add a built-in shader (compiled into the binary).
    pub fn add_builtin(&mut self, name: &str, wgsl: &'static str) {
        self.shaders.push(ShaderConfig {
            name: name.to_owned(),
            source: ShaderSource::Builtin(wgsl),
            enabled: true,
        });
    }

    /// Add a shader loaded from a file. Returns an error if the file does not exist.
    pub fn add_file(&mut self, name: &str, path: PathBuf) -> Result<(), GarasuError> {
        if !path.exists() {
            return Err(GarasuError::Shader(format!(
                "shader file not found: {}",
                path.display()
            )));
        }
        self.shaders.push(ShaderConfig {
            name: name.to_owned(),
            source: ShaderSource::File(path),
            enabled: true,
        });
        Ok(())
    }

    /// Add an inline WGSL shader provided as a string.
    pub fn add_inline(&mut self, name: &str, wgsl: String) {
        self.shaders.push(ShaderConfig {
            name: name.to_owned(),
            source: ShaderSource::Inline(wgsl),
            enabled: true,
        });
    }

    /// Enable a shader by name. Returns `false` if the name was not found.
    pub fn enable(&mut self, name: &str) -> bool {
        if let Some(shader) = self.shaders.iter_mut().find(|s| s.name == name) {
            shader.enabled = true;
            true
        } else {
            false
        }
    }

    /// Disable a shader by name. Returns `false` if the name was not found.
    pub fn disable(&mut self, name: &str) -> bool {
        if let Some(shader) = self.shaders.iter_mut().find(|s| s.name == name) {
            shader.enabled = false;
            true
        } else {
            false
        }
    }

    /// Check whether a shader is enabled. Returns `false` if not found.
    #[must_use]
    pub fn is_enabled(&self, name: &str) -> bool {
        self.shaders
            .iter()
            .find(|s| s.name == name)
            .is_some_and(|s| s.enabled)
    }

    /// Return all shader configs in pipeline order.
    #[must_use]
    pub fn list(&self) -> &[ShaderConfig] {
        &self.shaders
    }

    /// Return only the enabled shader configs.
    #[must_use]
    pub fn active(&self) -> Vec<&ShaderConfig> {
        self.shaders.iter().filter(|s| s.enabled).collect()
    }

    /// Read the WGSL source for a shader by name.
    ///
    /// For `File` variants, this reads the file from disk each time.
    /// Returns `None` if the shader name is not found.
    #[must_use]
    pub fn get_source(&self, name: &str) -> Option<Result<String, GarasuError>> {
        let shader = self.shaders.iter().find(|s| s.name == name)?;
        Some(match &shader.source {
            ShaderSource::Builtin(s) => Ok((*s).to_owned()),
            ShaderSource::Inline(s) => Ok(s.clone()),
            ShaderSource::File(path) => {
                std::fs::read_to_string(path).map_err(GarasuError::Io)
            }
        })
    }

    /// Number of shaders in the pipeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shaders.len()
    }

    /// Whether the pipeline has no shaders.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shaders.is_empty()
    }
}

impl Default for ShaderPipeline {
    fn default() -> Self {
        Self::new()
    }
}
