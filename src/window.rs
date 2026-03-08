use crate::error::GarasuError;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowAttributes};

/// Pure-data window configuration.
///
/// Testable without a display server or GPU.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WindowConfig {
    /// Window width in physical pixels.
    pub width: u32,
    /// Window height in physical pixels.
    pub height: u32,
    /// Window title.
    pub title: String,
    /// Whether the window background is transparent.
    pub transparent: bool,
    /// Whether the window has OS decorations (title bar, borders).
    pub decorations: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            title: "garasu".to_owned(),
            transparent: false,
            decorations: true,
        }
    }
}

/// Thin wrapper around winit window creation.
///
/// Provides helper methods that translate `WindowConfig` into
/// winit window attributes. Requires an active event loop
/// (display server), so the actual creation is not unit-testable.
pub struct AppWindow;

impl AppWindow {
    /// Create an event loop suitable for the platform.
    pub fn event_loop() -> Result<EventLoop<()>, GarasuError> {
        EventLoop::new().map_err(|e| GarasuError::Window(e.to_string()))
    }

    /// Create a window from a `WindowConfig`.
    pub fn create_from_config(
        event_loop: &winit::event_loop::ActiveEventLoop,
        config: &WindowConfig,
    ) -> Result<Window, GarasuError> {
        let attrs = WindowAttributes::default()
            .with_title(&config.title)
            .with_inner_size(winit::dpi::PhysicalSize::new(config.width, config.height))
            .with_transparent(config.transparent)
            .with_decorations(config.decorations);

        event_loop
            .create_window(attrs)
            .map_err(|e| GarasuError::Window(e.to_string()))
    }

    /// Create a window with the given title and size.
    pub fn create(
        event_loop: &winit::event_loop::ActiveEventLoop,
        title: &str,
        width: u32,
        height: u32,
    ) -> Result<Window, GarasuError> {
        let attrs = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(winit::dpi::PhysicalSize::new(width, height));

        event_loop
            .create_window(attrs)
            .map_err(|e| GarasuError::Window(e.to_string()))
    }
}
