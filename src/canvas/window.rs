use std::collections::HashMap;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use serde_json::{Value, json};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId};

use super::protocol::{Frame, Refusal, Spec};
use super::server::{CanvasId, Report, Server, Sink};
use crate::context::GpuContext;
use crate::error::GarasuError;
use crate::frame::{FramePipeline, Readback};

const OPEN_WAIT: Duration = Duration::from_secs(4);

/// How [`run`] shows its canvases. With `verify`, each window's
/// surface is configured `RENDER_ATTACHMENT | COPY_SRC`, and after
/// each draw of a presented frame the server reads that surface
/// texture back and reports `verified`, `mismatch` or `unverified`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub verify: bool,
}

type Refused = Arc<Mutex<HashMap<CanvasId, Refusal>>>;

fn refused(map: &Refused) -> MutexGuard<'_, HashMap<CanvasId, Refusal>> {
    map.lock().unwrap_or_else(PoisonError::into_inner)
}

pub enum Command {
    Open {
        id: CanvasId,
        spec: Spec,
        answer: SyncSender<Result<(), Refusal>>,
    },
    Present(CanvasId, Frame),
    Close(CanvasId),
}

pub struct WindowSink {
    proxy: EventLoopProxy<Command>,
    refused: Refused,
}

fn display_gone() -> Refusal {
    Refusal::Unavailable("the display loop has stopped".into())
}

impl Sink for WindowSink {
    fn open(&self, id: CanvasId, spec: &Spec) -> Result<(), Refusal> {
        let (answer, opened) = sync_channel(1);
        self.proxy
            .send_event(Command::Open {
                id,
                spec: spec.clone(),
                answer,
            })
            .map_err(|_| display_gone())?;
        opened.recv_timeout(OPEN_WAIT).unwrap_or_else(|_| {
            Err(Refusal::Unavailable(
                "the display did not open a window in time".into(),
            ))
        })
    }

    fn present(&self, id: CanvasId, frame: Frame) -> Result<(), Refusal> {
        if let Some(refusal) = refused(&self.refused).get(&id) {
            return Err(refusal.clone());
        }
        self.proxy
            .send_event(Command::Present(id, frame))
            .map_err(|_| display_gone())
    }

    fn close(&self, id: CanvasId) {
        refused(&self.refused).remove(&id);
        let _ = self.proxy.send_event(Command::Close(id));
    }
}

struct Shown {
    canvas: CanvasId,
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    usage: wgpu::TextureUsages,
    frame: FramePipeline,
    presented: u64,
    sent: Option<Sent>,
}

struct Sent {
    number: u64,
    frame: Frame,
}

struct Canvases {
    gpu: Option<GpuContext>,
    shown: HashMap<WindowId, Shown>,
    windows: HashMap<CanvasId, WindowId>,
    refused: Refused,
    options: Options,
    report: Report,
}

pub fn run(server: Server, options: Options) -> Result<(), GarasuError> {
    let event_loop = EventLoop::<Command>::with_user_event()
        .build()
        .map_err(|e| GarasuError::Window(e.to_string()))?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let refused = Refused::default();
    let sink = Arc::new(WindowSink {
        proxy: event_loop.create_proxy(),
        refused: Arc::clone(&refused),
    });
    let report = server.report();
    let _running = server.spawn(sink)?;
    let mut canvases = Canvases {
        gpu: None,
        shown: HashMap::new(),
        windows: HashMap::new(),
        refused,
        options,
        report,
    };
    event_loop
        .run_app(&mut canvases)
        .map_err(|e| GarasuError::Window(e.to_string()))
}

fn unavailable(e: impl std::fmt::Display) -> Refusal {
    Refusal::Unavailable(e.to_string())
}

impl Canvases {
    fn open(
        &mut self,
        event_loop: &ActiveEventLoop,
        id: CanvasId,
        spec: &Spec,
    ) -> Result<(), Refusal> {
        let attributes = Window::default_attributes()
            .with_title(&spec.title)
            .with_inner_size(LogicalSize::new(spec.width, spec.height));
        let window = Arc::new(event_loop.create_window(attributes).map_err(unavailable)?);
        let surface = if let Some(gpu) = &self.gpu {
            gpu.instance
                .create_surface(Arc::clone(&window))
                .map_err(unavailable)?
        } else {
            let instance = GpuContext::instance();
            let surface = instance
                .create_surface(Arc::clone(&window))
                .map_err(unavailable)?;
            let gpu = pollster::block_on(GpuContext::new_for_surface(
                instance,
                &surface,
                wgpu::PowerPreference::LowPower,
            ))
            .map_err(unavailable)?;
            self.gpu = Some(gpu);
            surface
        };
        let gpu = self.gpu.as_ref().ok_or_else(|| unavailable("no GPU"))?;
        let caps = surface.get_capabilities(&gpu.adapter);
        if caps.formats.is_empty() {
            return Err(unavailable("the GPU cannot present to this window"));
        }
        let readable = caps.usages.contains(wgpu::TextureUsages::COPY_SRC);
        let usage = if self.options.verify && readable {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        let size = window.inner_size();
        let format = gpu.configure_surface_with_usage(
            &surface,
            size.width.max(1),
            size.height.max(1),
            usage,
        );
        let frame = FramePipeline::new(&gpu.device, format, spec.width, spec.height);
        window.request_redraw();
        self.windows.insert(id, window.id());
        self.shown.insert(
            window.id(),
            Shown {
                canvas: id,
                window,
                surface,
                usage,
                frame,
                presented: 0,
                sent: None,
            },
        );
        Ok(())
    }

    fn present(&mut self, id: CanvasId, frame: Frame) {
        let (Some(gpu), Some(window)) = (&self.gpu, self.windows.get(&id)) else {
            return;
        };
        let Some(shown) = self.shown.get_mut(window) else {
            return;
        };
        shown.presented += 1;
        match shown.frame.upload_rgb8(
            &gpu.device,
            &gpu.queue,
            frame.width,
            frame.height,
            &frame.rgb,
        ) {
            Ok(()) => {
                if self.options.verify {
                    shown.sent = Some(Sent {
                        number: shown.presented,
                        frame,
                    });
                }
                shown.window.request_redraw();
            }
            Err(e) => {
                tracing::warn!(target: "garasu::canvas", canvas = id, error = %e, "frame upload failed");
            }
        }
    }

    fn close(&mut self, id: CanvasId) {
        if let Some(window) = self.windows.remove(&id) {
            self.shown.remove(&window);
        }
    }

    fn reconfigure(&self, window: WindowId) {
        if let (Some(gpu), Some(shown)) = (&self.gpu, self.shown.get(&window)) {
            let size = shown.window.inner_size();
            gpu.configure_surface_with_usage(
                &shown.surface,
                size.width.max(1),
                size.height.max(1),
                shown.usage,
            );
            shown.window.request_redraw();
        }
    }

    fn draw(&self, window: WindowId) {
        let (Some(gpu), Some(shown)) = (&self.gpu, self.shown.get(&window)) else {
            return;
        };
        let texture = match shown.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.reconfigure(window);
                return;
            }
            Err(e) => {
                tracing::warn!(target: "garasu::canvas", error = %e, "no surface texture");
                return;
            }
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("garasu_canvas_encoder"),
            });
        let target = texture.texture.size();
        shown
            .frame
            .render(&mut encoder, &view, (target.width, target.height));
        gpu.queue.submit(std::iter::once(encoder.finish()));
        let verdict = shown
            .sent
            .as_ref()
            .map(|sent| verdict(gpu, shown, &texture.texture, sent));
        shown.window.pre_present_notify();
        texture.present();
        if let Some(event) = verdict {
            self.report.emit(&event);
        }
    }
}

fn verdict(gpu: &GpuContext, shown: &Shown, texture: &wgpu::Texture, sent: &Sent) -> Value {
    let unverified = |reason: String| {
        json!({
            "event": "unverified",
            "canvas": shown.canvas,
            "frame": sent.number,
            "reason": reason,
        })
    };
    if !shown.usage.contains(wgpu::TextureUsages::COPY_SRC) {
        return unverified("this backend cannot copy from a surface texture".into());
    }
    let checked = Readback::read(gpu, texture).and_then(|readback| {
        let frame = &sent.frame;
        readback
            .verify(frame.width, frame.height, &frame.rgb)
            .map(|verdict| (readback, verdict))
    });
    let (readback, verdict) = match checked {
        Ok(checked) => checked,
        Err(e) => return unverified(e.to_string()),
    };
    let mut event = json!({
        "event": "verified",
        "canvas": shown.canvas,
        "frame": sent.number,
        "blake3": verdict.blake3.to_hex().to_string(),
        "surface": {
            "width": readback.width(),
            "height": readback.height(),
            "format": format!("{:?}", readback.format()),
            "blake3": verdict.surface.to_hex().to_string(),
        },
    });
    if let Some(mismatch) = verdict.mismatch {
        event["event"] = json!("mismatch");
        event["x"] = json!(mismatch.x);
        event["y"] = json!(mismatch.y);
        event["want"] = json!(mismatch.want);
        event["got"] = json!(mismatch.got);
    }
    event
}

impl ApplicationHandler<Command> for Canvases {
    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {}

    fn user_event(&mut self, event_loop: &ActiveEventLoop, command: Command) {
        match command {
            Command::Open { id, spec, answer } => {
                let opened = self.open(event_loop, id, &spec);
                if answer.send(opened).is_err() {
                    self.close(id);
                }
            }
            Command::Present(id, frame) => self.present(id, frame),
            Command::Close(id) => self.close(id),
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        window: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                if let Some(shown) = self.shown.remove(&window) {
                    self.windows.remove(&shown.canvas);
                    refused(&self.refused).insert(shown.canvas, Refusal::WindowClosed);
                }
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                self.reconfigure(window);
            }
            WindowEvent::RedrawRequested => self.draw(window),
            _ => {}
        }
    }
}
