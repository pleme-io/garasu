use crate::error::GarasuError;

/// Core GPU context: wgpu instance, adapter, device, and queue.
///
/// Create once at startup and share across rendering subsystems.
/// Requires a GPU to be available; returns a clear error if not.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Initialize GPU context with default backends (Metal on macOS, Vulkan on Linux).
    ///
    /// This is async because wgpu adapter/device requests are async.
    /// Use `pollster::block_on` if you need synchronous initialization.
    pub async fn new() -> Result<Self, GarasuError> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| GarasuError::Gpu(format!("no suitable GPU adapter found: {e}")))?;

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| GarasuError::Gpu(format!("device request failed: {e}")))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Create a surface for a window and configure it.
    pub fn configure_surface(
        &self,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
    ) -> wgpu::TextureFormat {
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        format
    }
}
