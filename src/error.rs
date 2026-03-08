/// Errors produced by the garasu rendering engine.
#[derive(Debug, thiserror::Error)]
pub enum GarasuError {
    /// GPU adapter not found or GPU operation failed.
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Shader compilation or loading failed.
    #[error("shader error: {0}")]
    Shader(String),

    /// Window creation or configuration failed.
    #[error("window error: {0}")]
    Window(String),

    /// File I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
