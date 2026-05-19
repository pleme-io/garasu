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
    /// Initialize GPU context with default backends (Metal on macOS, Vulkan on Linux)
    /// and the **LowPower** power preference.
    ///
    /// Why LowPower as default: on Intel Macs `HighPerformance` triggers the
    /// discrete GPU which takes 50-200 ms to wake from sleep — visible in
    /// terminal-class apps as "the window appears blank for a beat before
    /// painting." On M-series Macs there's only one GPU so the preference
    /// is functionally equivalent but the LowPower request skips some
    /// adapter enumeration time. Terminal emulators / TUI overlays / status
    /// bars don't need the watts.
    ///
    /// Heavy-GPU consumers (3D, compute, video) should explicitly call
    /// [`Self::new_with_power`] with `PowerPreference::HighPerformance`.
    pub async fn new() -> Result<Self, GarasuError> {
        Self::new_with_power(wgpu::PowerPreference::LowPower).await
    }

    /// Initialize GPU context with an explicit power preference.
    pub async fn new_with_power(
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, GarasuError> {
        let t_start = std::time::Instant::now();
        let instance = wgpu::Instance::default();
        let t_instance = t_start.elapsed();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| GarasuError::Gpu(format!("no suitable GPU adapter found: {e}")))?;
        let t_adapter = t_start.elapsed();

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| GarasuError::Gpu(format!("device request failed: {e}")))?;
        let t_device = t_start.elapsed();

        // Per-phase tracing so consumers can read the breakdown out of
        // their stderr without external profilers. Use `RUST_LOG=garasu::ctx=info`.
        tracing::info!(
            target: "garasu::ctx",
            instance_ms = t_instance.as_millis() as u64,
            adapter_ms = (t_adapter - t_instance).as_millis() as u64,
            device_ms = (t_device - t_adapter).as_millis() as u64,
            total_ms = t_device.as_millis() as u64,
            power = ?power_preference,
            "gpu context ready"
        );

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
