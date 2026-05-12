//! Runtime posture detection + recommendation for GPU applications.
//!
//! Every GPU app on the garasu stack faces the same question: how do I
//! pick the right fps target, vsync mode, and animation budget for the
//! machine I'm running on? This module supplies the typed answer:
//!
//! 1. [`detect_displays`] / [`detect_gpu`] / [`detect_platform`] each
//!    return a typed slice of the runtime posture from the closest
//!    available source ([`winit::event_loop::ActiveEventLoop`], a
//!    `wgpu::Adapter`, the build target).
//! 2. [`detect_all`] composes them into a [`RuntimePosture`] —
//!    serializable, snapshotable, fleet-comparable.
//! 3. [`recommend`] applies a small typed rule table to a
//!    [`RuntimePosture`] + [`RecommendationProfile`] and returns a
//!    [`RuntimeBudget`] — the values consumers should plug into their
//!    runtime configuration.
//!
//! Consumers merge the budget under user-supplied overrides:
//! *hardcoded fallback* ← *detected recommendation* ← *user config* ←
//! *profile* ← *CLI flag*.
//!
//! ## Compounding
//!
//! Today's only consumer is mado. The same typed posture also covers
//! hibikine, kagibako, namimado, future GPU apps on the substrate —
//! adding a new platform sensor or recommendation rule lands here once,
//! every consumer inherits the change.

use serde::{Deserialize, Serialize};

/// One physical display attached to the system at the moment of
/// detection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Display {
    /// Human-readable name (e.g. "Built-in Retina Display"). `None` if
    /// the platform doesn't expose one.
    pub name: Option<String>,
    /// Physical pixel dimensions (width, height).
    pub size: (u32, u32),
    /// HiDPI scale factor — logical pixels × `scale_factor` = physical
    /// pixels.
    pub scale_factor: f64,
    /// Refresh rate in whole Hz. `None` if the platform doesn't
    /// expose one (some Wayland configurations, headless / virtual
    /// displays).
    pub refresh_hz: Option<u32>,
    /// True if this is the user's primary display. Heuristic — if the
    /// platform doesn't expose a primary signal, the first enumerated
    /// monitor wins.
    pub primary: bool,
}

/// GPU posture surfaced by `wgpu::AdapterInfo`. Only the fields we
/// actually use for recommendations live here — keeps the wire format
/// stable when wgpu adds new info.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuPosture {
    /// `"Metal"`, `"Vulkan"`, `"Dx12"`, `"Gl"`, `"BrowserWebGpu"`,
    /// `"Empty"`. Matches `wgpu::Backend::Debug` rendering.
    pub backend: String,
    /// 16-bit PCI vendor id rendered as `"0x106b"` etc. Useful for
    /// detecting "is this an Apple Silicon GPU" / "Intel iGPU".
    pub vendor: String,
    /// Adapter-reported device name, e.g. `"Apple M1 Pro"`.
    pub device_name: String,
    /// `"DiscreteGpu"` / `"IntegratedGpu"` / `"VirtualGpu"` / `"Cpu"` /
    /// `"Other"`. Useful for power-vs-performance heuristics.
    pub device_type: String,
    /// Maximum 2D texture dimension — informs glyph atlas + framebuffer
    /// sizing.
    pub max_texture_dimension_2d: u32,
}

/// Build-target identification. Pure compile-time facts; no I/O.
///
/// Stored as `String` rather than `&'static str` so the type
/// deserializes cleanly from snapshot artifacts produced on other
/// machines / other builds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Platform {
    /// `"macos"` / `"linux"` / `"windows"` / `"freebsd"` / etc.
    pub os: String,
    /// `"aarch64"` / `"x86_64"` / `"arm"` / etc.
    pub arch: String,
}

/// A complete snapshot of the runtime's user-perceivable posture.
/// Serializable so consumers can persist, transmit, or attest it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimePosture {
    pub displays: Vec<Display>,
    pub gpu: Option<GpuPosture>,
    pub platform: Platform,
    /// True if any detected display reports refresh > 60 Hz. Convenience
    /// flag — saves consumers iterating `displays` for the common case.
    pub high_refresh: bool,
}

impl RuntimePosture {
    /// Best single refresh rate to render at — primary display's value
    /// if known, else the maximum across all displays, else `None`.
    #[must_use]
    pub fn effective_refresh_hz(&self) -> Option<u32> {
        self.displays
            .iter()
            .find(|d| d.primary)
            .and_then(|d| d.refresh_hz)
            .or_else(|| self.displays.iter().filter_map(|d| d.refresh_hz).max())
    }
}

/// Inputs to [`recommend`] — typically supplied from user-supplied caps
/// in app config. All `None` produces a zero-clamp recommendation.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct RecommendationProfile {
    /// Upper bound on recommended fps. The recommender never returns a
    /// value above this. `None` = no ceiling.
    pub fps_cap: Option<u32>,
    /// Upper bound on recommended fps when on battery power. `None` =
    /// same as `fps_cap`. (Battery-power detection lands in a follow-up
    /// — this field is plumbed now for forward compatibility.)
    pub battery_fps_cap: Option<u32>,
    /// Force the battery cap regardless of actual power source. Useful
    /// for "power-saver" presets driven by the operator.
    #[serde(default)]
    pub force_battery_mode: bool,
}

/// Output of [`recommend`] — the values the consumer should layer
/// between its hardcoded fallback and any user override.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeBudget {
    /// Suggested frames-per-second target. `None` = no recommendation
    /// (consumer should use its own hardcoded baseline).
    pub fps_target: Option<u32>,
    /// Suggested vsync state. `true` matches the safe-baseline default
    /// across every platform we ship on.
    pub vsync: bool,
}

/// Apply the rule table to a posture + profile.
///
/// Current rules:
///
/// - `fps_target` = primary display's refresh rate, clamped by the
///   relevant cap (`battery_fps_cap` if `force_battery_mode`, else
///   `fps_cap`).
/// - `vsync` always `true`. (Reserved for future per-platform tuning.)
#[must_use]
pub fn recommend(posture: &RuntimePosture, profile: &RecommendationProfile) -> RuntimeBudget {
    let refresh = posture.effective_refresh_hz();
    let cap = if profile.force_battery_mode {
        profile.battery_fps_cap.or(profile.fps_cap)
    } else {
        profile.fps_cap
    }
    .unwrap_or(u32::MAX);
    let fps_target = refresh.map(|hz| hz.min(cap));
    RuntimeBudget {
        fps_target,
        vsync: true,
    }
}

/// Detect every display from an iterator of winit `MonitorHandle`s.
///
/// Decoupled from the `EventLoop` / `ActiveEventLoop` distinction so
/// callers can detect either *before* running the loop (with
/// `EventLoop::available_monitors()`) or *during* (with
/// `ActiveEventLoop::available_monitors()`). Both surfaces expose the
/// same `MonitorHandle` iterator + primary getter.
pub fn detect_displays<I>(
    monitors: I,
    primary: Option<winit::monitor::MonitorHandle>,
) -> Vec<Display>
where
    I: IntoIterator<Item = winit::monitor::MonitorHandle>,
{
    let primary_name = primary.as_ref().and_then(winit::monitor::MonitorHandle::name);
    monitors
        .into_iter()
        .enumerate()
        .map(|(idx, monitor)| {
            let monitor_name = monitor.name();
            let size = monitor.size();
            // refresh_rate_millihertz returns mHz — convert to Hz with
            // rounding so 59.94Hz reports as 60Hz, not 59.
            let refresh_hz = monitor
                .refresh_rate_millihertz()
                .map(|mhz| (mhz + 500) / 1000);
            let primary = match (primary_name.as_ref(), monitor_name.as_ref()) {
                (Some(p), Some(m)) => p == m,
                _ => idx == 0,
            };
            Display {
                name: monitor_name,
                size: (size.width, size.height),
                scale_factor: monitor.scale_factor(),
                refresh_hz,
                primary,
            }
        })
        .collect()
}

/// Detect GPU posture from a wgpu adapter.
#[must_use]
pub fn detect_gpu(adapter: &wgpu::Adapter) -> GpuPosture {
    let info = adapter.get_info();
    let limits = adapter.limits();
    GpuPosture {
        backend: format!("{:?}", info.backend),
        vendor: format!("0x{:04x}", info.vendor),
        device_name: info.name,
        device_type: format!("{:?}", info.device_type),
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
    }
}

/// Detect compile-time platform facts. Pure — no I/O.
#[must_use]
pub fn detect_platform() -> Platform {
    Platform {
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
    }
}

/// Compose the full posture from every available source.
///
/// `monitors` and `primary` are supplied by the caller from either
/// `EventLoop::available_monitors()` + `EventLoop::primary_monitor()`
/// (pre-run detection) or the equivalent methods on `ActiveEventLoop`
/// (in-loop detection). `gpu` is optional — pass `None` if the consumer
/// hasn't built an adapter yet.
pub fn detect_all<I>(
    monitors: I,
    primary: Option<winit::monitor::MonitorHandle>,
    gpu: Option<&wgpu::Adapter>,
) -> RuntimePosture
where
    I: IntoIterator<Item = winit::monitor::MonitorHandle>,
{
    let displays = detect_displays(monitors, primary);
    let high_refresh = displays
        .iter()
        .any(|d| d.refresh_hz.is_some_and(|hz| hz > 60));
    RuntimePosture {
        displays,
        gpu: gpu.map(detect_gpu),
        platform: detect_platform(),
        high_refresh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_posture(refresh: Option<u32>) -> RuntimePosture {
        RuntimePosture {
            displays: vec![Display {
                name: Some("test".into()),
                size: (1920, 1080),
                scale_factor: 2.0,
                refresh_hz: refresh,
                primary: true,
            }],
            gpu: None,
            platform: detect_platform(),
            high_refresh: refresh.is_some_and(|hz| hz > 60),
        }
    }

    #[test]
    fn effective_refresh_prefers_primary() {
        let posture = RuntimePosture {
            displays: vec![
                Display {
                    name: Some("ext".into()),
                    size: (3840, 2160),
                    scale_factor: 1.0,
                    refresh_hz: Some(60),
                    primary: false,
                },
                Display {
                    name: Some("builtin".into()),
                    size: (1920, 1080),
                    scale_factor: 2.0,
                    refresh_hz: Some(120),
                    primary: true,
                },
            ],
            gpu: None,
            platform: detect_platform(),
            high_refresh: true,
        };
        assert_eq!(posture.effective_refresh_hz(), Some(120));
    }

    #[test]
    fn effective_refresh_falls_back_to_max_when_no_primary_has_rate() {
        let posture = RuntimePosture {
            displays: vec![
                Display {
                    name: None,
                    size: (1920, 1080),
                    scale_factor: 1.0,
                    refresh_hz: None,
                    primary: true,
                },
                Display {
                    name: None,
                    size: (1920, 1080),
                    scale_factor: 1.0,
                    refresh_hz: Some(144),
                    primary: false,
                },
            ],
            gpu: None,
            platform: detect_platform(),
            high_refresh: true,
        };
        assert_eq!(posture.effective_refresh_hz(), Some(144));
    }

    #[test]
    fn recommend_caps_to_fps_cap() {
        let posture = make_posture(Some(120));
        let profile = RecommendationProfile {
            fps_cap: Some(60),
            battery_fps_cap: None,
            force_battery_mode: false,
        };
        let budget = recommend(&posture, &profile);
        assert_eq!(budget.fps_target, Some(60));
        assert!(budget.vsync);
    }

    #[test]
    fn recommend_uses_refresh_when_uncapped() {
        let posture = make_posture(Some(144));
        let budget = recommend(&posture, &RecommendationProfile::default());
        assert_eq!(budget.fps_target, Some(144));
    }

    #[test]
    fn recommend_returns_none_when_refresh_unknown() {
        let posture = make_posture(None);
        let budget = recommend(&posture, &RecommendationProfile::default());
        assert_eq!(budget.fps_target, None);
    }

    #[test]
    fn recommend_uses_battery_cap_when_forced() {
        let posture = make_posture(Some(120));
        let profile = RecommendationProfile {
            fps_cap: Some(120),
            battery_fps_cap: Some(60),
            force_battery_mode: true,
        };
        let budget = recommend(&posture, &profile);
        assert_eq!(budget.fps_target, Some(60));
    }

    #[test]
    fn recommend_falls_back_to_fps_cap_when_no_battery_cap() {
        let posture = make_posture(Some(120));
        let profile = RecommendationProfile {
            fps_cap: Some(90),
            battery_fps_cap: None,
            force_battery_mode: true,
        };
        let budget = recommend(&posture, &profile);
        assert_eq!(budget.fps_target, Some(90));
    }

    #[test]
    fn platform_is_compile_time_target() {
        let p = detect_platform();
        assert_eq!(p.os, std::env::consts::OS);
        assert_eq!(p.arch, std::env::consts::ARCH);
    }

    #[test]
    fn runtime_posture_serde_round_trip() {
        let original = RuntimePosture {
            displays: vec![Display {
                name: Some("primary".into()),
                size: (2880, 1864),
                scale_factor: 2.0,
                refresh_hz: Some(120),
                primary: true,
            }],
            gpu: Some(GpuPosture {
                backend: "Metal".into(),
                vendor: "0x106b".into(),
                device_name: "Apple M1 Pro".into(),
                device_type: "IntegratedGpu".into(),
                max_texture_dimension_2d: 16384,
            }),
            platform: Platform {
                os: "macos".into(),
                arch: "aarch64".into(),
            },
            high_refresh: true,
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: RuntimePosture = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, original);
    }

    #[test]
    fn high_refresh_flag_set_when_any_display_above_60() {
        // tests detect_all's flag computation by constructing the
        // posture directly.
        let posture = RuntimePosture {
            displays: vec![
                Display {
                    name: None,
                    size: (1920, 1080),
                    scale_factor: 1.0,
                    refresh_hz: Some(60),
                    primary: true,
                },
                Display {
                    name: None,
                    size: (3840, 2160),
                    scale_factor: 1.0,
                    refresh_hz: Some(144),
                    primary: false,
                },
            ],
            gpu: None,
            platform: detect_platform(),
            high_refresh: true,
        };
        assert!(posture.displays.iter().any(|d| d.refresh_hz.is_some_and(|hz| hz > 60)));
    }
}
