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
        Self::build(wgpu::Instance::default(), None, power_preference).await
    }

    /// The wgpu instance, alone — stage 1 of the **surface-aware** path.
    ///
    /// A surface and the adapter that presents to it must come from the SAME
    /// instance, and wgpu's ordering is instance → surface → adapter. So a
    /// caller who wants [`Self::new_for_surface`] needs the instance in hand
    /// *before* the context exists. This hands it over.
    #[must_use]
    pub fn instance() -> wgpu::Instance {
        wgpu::Instance::default()
    }

    /// Initialize a context whose adapter can **actually present to `surface`**.
    ///
    /// ── WHY THIS EXISTS (measured 2026-08-19, on Linux) ───────────────────
    /// [`Self::new`] requests an adapter with `compatible_surface: None`, which
    /// is harmless on macOS — one Metal adapter, and it presents to everything.
    /// On Linux it is not harmless: several adapters are typically enumerated
    /// (a hardware Vulkan one, llvmpipe, sometimes a GL one), and wgpu is free
    /// to hand back one that **cannot present to the surface you then create**.
    ///
    /// ★ The failure does not surface as an error. `get_capabilities()` on a
    /// mismatched surface/adapter pair returns a struct whose `formats` and
    /// `alpha_modes` vectors are simply **EMPTY**, so the damage lands in the
    /// caller as `index out of bounds: the len is 0 but the index is 0` —
    /// pointing at the consumer's indexing rather than at the adapter choice
    /// two layers up. That is exactly how it was found: mado connected to a
    /// Wayland compositor on Linux, garasu reported `gpu context ready`, and
    /// madori panicked on `caps.formats[0]`.
    ///
    /// So: if you have a surface, use this. `new()` remains correct for the
    /// headless and single-adapter cases, and every existing consumer is
    /// untouched.
    ///
    /// # Errors
    /// Returns [`GarasuError::Gpu`] if no adapter can present to `surface`, or
    /// if device creation fails.
    pub async fn new_for_surface(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, GarasuError> {
        Self::build(instance, Some(surface), power_preference).await
    }

    async fn build(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, GarasuError> {
        let t_start = std::time::Instant::now();
        let surface_aware = compatible_surface.is_some();
        let t_instance = t_start.elapsed();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface,
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
            surface_aware,
            adapter = ?adapter.get_info().name,
            backend = ?adapter.get_info().backend,
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
