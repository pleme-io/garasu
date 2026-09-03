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

        // ── ★ NEVER SILENTLY LAND ON THE CPU RASTERIZER ───────────────────
        //
        // `force_fallback_adapter: false` asks wgpu not to *force* a fallback.
        // It does not stop it from *choosing* one: a software Vulkan ICD such
        // as lavapipe advertises itself like any other adapter, and under
        // `PowerPreference::LowPower` — our default, chosen for a macOS
        // reason — it is a perfectly reasonable answer to "give me the
        // low-power device". It is the lowest-power device on the machine.
        //
        // Measured on plo 2026-08-20: an RTX 3070 with a working
        // `nvidia_icd.json` and live `/dev/nvidia*` nodes, and mado rendering
        // through `llvmpipe (LLVM 21.1.7)` at 34 fps — on the CPU, with the
        // GPU idle at 39 MiB. Nothing logged an error, because nothing was
        // wrong as far as wgpu was concerned.
        //
        // So the choice is checked rather than trusted: if the adapter we were
        // handed is a CPU device, look for a hardware one and prefer it. The
        // fallback is still available when it is genuinely all there is —
        // headless CI, a VM with no passthrough — which is why this re-selects
        // instead of failing.
        let adapter = if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            // ★ AND IT MUST BE ABLE TO PRESENT TO *THIS* SURFACE.
            // `enumerate_adapters` does not filter by surface compatibility, and
            // an incompatible adapter does not error — it returns EMPTY
            // capabilities, which then fails later and far away. So when we
            // have a surface, an adapter only counts if it advertises at least
            // one format for it.
            // ★ ABSENT AND REFUSED ARE DIFFERENT ANSWERS — separated 2026-09-03
            // because collapsing them cost a real investigation.
            //
            // plo has a GeForce RTX 3070, driver 580.142 loaded with seven
            // kernel modules, live `/dev/nvidia*` nodes and `nvidia_icd.json`
            // installed — and this code logged **"no hardware GPU adapter on
            // this machine"**. That sentence is false, and it is false in the
            // most expensive direction: it sends the reader to check hardware
            // and drivers, all of which are fine, instead of to the one thing
            // that is not.
            //
            // The real cause is one layer up. omoya is a CPU compositor and so
            // advertises **linear-modifier dmabuf only** (`nuri_renderer.rs`:
            // a tiled modifier is a layout only a GPU can decode, and a CPU
            // blitter reading it paints structured noise). NVIDIA does not
            // present linear. So the adapter exists, works, and simply cannot
            // present to THIS surface — a refusal with a reason, not an
            // absence. Every GPU client on that seat therefore renders on
            // llvmpipe, which is why the terminal feels slow.
            //
            // So the two are counted separately: hardware adapters that exist
            // at all, and the subset that can present here.
            let hardware_adapters: Vec<_> = instance
                .enumerate_adapters(wgpu::Backends::all())
                .into_iter()
                .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
                .collect();
            // `enumerate_adapters` does not filter by surface compatibility,
            // and an incompatible adapter does not error — it returns EMPTY
            // capabilities, which fails later and far away. So with a surface
            // in hand, an adapter only counts if it advertises a format for it.
            let presentable = hardware_adapters.into_iter().fold(
                (None, Vec::new()),
                |(chosen, mut rejected), a| {
                    if chosen.is_some() {
                        rejected.push(a.get_info().name);
                        return (chosen, rejected);
                    }
                    let can_present = compatible_surface
                        .is_none_or(|s| !s.get_capabilities(&a).formats.is_empty());
                    if can_present {
                        (Some(a), rejected)
                    } else {
                        rejected.push(a.get_info().name);
                        (None, rejected)
                    }
                },
            );
            // ★ The verdict is computed ONCE, by the tested classifier, and
            // logged on every arm — so the branch taken and the word reported
            // cannot disagree. Two independent decisions is how the old
            // message came to say "absent" about a refusal.
            let verdict = CpuFallback::classify(presentable.0.is_some(), presentable.1.len());
            match presentable {
                (Some(hw), _) => {
                    tracing::info!(
                        target: "garasu::ctx",
                        rejected = ?adapter.get_info().name,
                        chosen = ?hw.get_info().name,
                        device_type = ?hw.get_info().device_type,
                        verdict = ?verdict,
                        "the requested power preference selected a CPU adapter; \
                         a hardware adapter exists and was preferred"
                    );
                    hw
                }
                // ★ REFUSED: the hardware is here and cannot present.
                (None, unpresentable) if !unpresentable.is_empty() => {
                    tracing::warn!(
                        target: "garasu::ctx",
                        adapter = ?adapter.get_info().name,
                        hardware_found = ?unpresentable,
                        verdict = ?verdict,
                        "a hardware GPU is present but CANNOT PRESENT to this \
                         surface, so rendering falls back to the CPU. This is a \
                         surface-compatibility refusal, NOT a missing driver — \
                         check the compositor's advertised dmabuf formats and \
                         modifiers before checking drivers. A CPU compositor \
                         advertising linear-only modifiers will refuse every \
                         GPU that cannot present linear."
                    );
                    adapter
                }
                // ★ ABSENT: there is genuinely nothing. Headless CI, a VM with
                // no passthrough. Say so once, loudly, rather than letting a
                // consumer wonder why it renders at 30 fps.
                (None, _) => {
                    tracing::warn!(
                        target: "garasu::ctx",
                        adapter = ?adapter.get_info().name,
                        verdict = ?verdict,
                        "no hardware GPU adapter on this machine — rendering on the CPU"
                    );
                    adapter
                }
            }
        } else {
            adapter
        };
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
            // Logged because "which adapter" was not enough to tell CPU from
            // GPU at a glance — `llvmpipe` reads like a driver name, not a
            // verdict, and that is how software rendering went unnoticed.
            device_type = ?adapter.get_info().device_type,
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

/// Why a CPU adapter ended up in use — `absent` and `refused` kept apart.
///
/// ── ★ WHY THIS IS A TYPE AND NOT TWO LOG STRINGS ─────────────────────────
/// The two cases demand opposite investigations. *Absent* means buy or enable
/// a GPU. *Refused* means the GPU is fine and the SURFACE will not take it —
/// on a pleme-io seat, a CPU compositor advertising linear-only dmabuf
/// modifiers, which NVIDIA cannot present. Collapsing them into "no hardware
/// GPU adapter on this machine" cost a real investigation on plo 2026-09-03:
/// the machine has an RTX 3070 with the driver loaded, and the message said
/// it had no GPU.
///
/// This is `kotae`'s distinction — an answer must say WHICH of the things
/// happened — applied to adapter selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuFallback {
    /// Not a fallback: a hardware adapter was found and preferred.
    NotAFallback,
    /// Hardware exists; none of it can present to this surface.
    Refused {
        /// How many hardware adapters were seen and could not present.
        hardware_found: usize,
    },
    /// No hardware adapter exists at all — headless CI, a VM without passthrough.
    Absent,
}

impl CpuFallback {
    /// Classify from the two counts the selection actually produces.
    ///
    /// Deliberately takes counts rather than adapters so it is testable on a
    /// machine with no GPU — which is most CI, and is exactly where a wrong
    /// verdict would otherwise go unnoticed.
    #[must_use]
    pub fn classify(chose_hardware: bool, unpresentable: usize) -> Self {
        if chose_hardware {
            Self::NotAFallback
        } else if unpresentable > 0 {
            Self::Refused {
                hardware_found: unpresentable,
            }
        } else {
            Self::Absent
        }
    }

    /// True when the operator should look at the COMPOSITOR, not the drivers.
    #[must_use]
    pub fn blames_the_surface(&self) -> bool {
        matches!(self, Self::Refused { .. })
    }
}

#[cfg(test)]
mod cpu_fallback_tests {
    use super::CpuFallback;

    /// ★ THE CASE THAT WAS MISREPORTED ON plo, pinned.
    ///
    /// One hardware adapter seen, none able to present. Before this type the
    /// code said "no hardware GPU adapter on this machine" — with an RTX 3070
    /// in the box and its driver loaded.
    #[test]
    fn hardware_that_cannot_present_is_refused_not_absent() {
        let v = CpuFallback::classify(false, 1);
        assert_eq!(v, CpuFallback::Refused { hardware_found: 1 });
        assert_ne!(
            v,
            CpuFallback::Absent,
            "reporting a refusal as an absence sends the reader to check \
             drivers that are already working"
        );
        assert!(
            v.blames_the_surface(),
            "a refusal must point at the surface — that is the whole reason \
             the two cases are distinguishable"
        );
    }

    /// ★ AND THE CONVERSE, so the type is not just always-Refused.
    #[test]
    fn nothing_at_all_is_absent_and_blames_no_surface() {
        let v = CpuFallback::classify(false, 0);
        assert_eq!(v, CpuFallback::Absent);
        assert!(
            !v.blames_the_surface(),
            "with no hardware at all there is no surface refusal to report, \
             and claiming one would send the reader to a compositor that is \
             behaving correctly"
        );
    }

    /// ★ Choosing hardware is not a fallback, however many were rejected on
    /// the way — a machine with several GPUs must not report a CPU fallback
    /// merely because one of them could not present.
    #[test]
    fn choosing_hardware_is_never_a_fallback() {
        for rejected in [0usize, 1, 7] {
            assert_eq!(
                CpuFallback::classify(true, rejected),
                CpuFallback::NotAFallback,
                "rejected={rejected} still chose hardware"
            );
        }
    }
}
