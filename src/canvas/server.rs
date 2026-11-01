use std::fs;
use std::io::{self, Write};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;

use serde_json::{Value, json};

use super::connection::Connection;
use super::protocol::{Frame, PROTOCOL, Refusal, Spec};
use crate::error::GarasuError;

pub type CanvasId = u64;

pub trait Sink: Send + Sync {
    fn open(&self, id: CanvasId, spec: &Spec) -> Result<(), Refusal>;
    fn present(&self, id: CanvasId, frame: Frame) -> Result<(), Refusal>;
    fn close(&self, id: CanvasId);
}

#[derive(Clone, Default)]
pub(crate) struct Report(Option<Arc<Mutex<Box<dyn Write + Send>>>>);

impl Report {
    pub(crate) fn emit(&self, event: &Value) {
        if let Some(out) = &self.0 {
            let mut out = out.lock().unwrap_or_else(PoisonError::into_inner);
            let _ = writeln!(out, "{event}").and_then(|()| out.flush());
        }
    }
}

pub struct Server {
    listener: UnixListener,
    path: PathBuf,
    report: Report,
}

fn in_use(path: &Path, why: &str) -> GarasuError {
    GarasuError::Io(io::Error::new(
        io::ErrorKind::AddrInUse,
        format!("{}: {why}", path.display()),
    ))
}

impl Server {
    pub fn bind(path: impl Into<PathBuf>) -> Result<Server, GarasuError> {
        let path = path.into();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty())
            && !dir.exists()
        {
            fs::create_dir_all(dir)?;
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
        }
        if let Ok(meta) = fs::symlink_metadata(&path) {
            if !meta.file_type().is_socket() {
                return Err(in_use(&path, "the path exists and is not a socket"));
            }
            if UnixStream::connect(&path).is_ok() {
                return Err(in_use(&path, "a canvas server already listens here"));
            }
            fs::remove_file(&path)?;
        }
        let listener = UnixListener::bind(&path)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
        Ok(Server {
            listener,
            path,
            report: Report::default(),
        })
    }

    #[must_use]
    pub fn report_to(mut self, out: impl Write + Send + 'static) -> Server {
        self.report = Report(Some(Arc::new(Mutex::new(Box::new(out)))));
        self
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn serve(self, sink: &Arc<dyn Sink>) {
        let stop = AtomicBool::new(false);
        self.accept(sink, &stop);
    }

    pub fn spawn(self, sink: Arc<dyn Sink>) -> Result<Running, GarasuError> {
        let path = self.path.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("garasu-canvas-accept".into())
            .spawn(move || self.accept(&sink, &flag))?;
        Ok(Running {
            path,
            stop,
            thread: Some(thread),
        })
    }

    fn accept(self, sink: &Arc<dyn Sink>, stop: &AtomicBool) {
        let ids = Arc::new(AtomicU64::new(1));
        self.report.emit(&json!({
            "event": "listen",
            "socket": self.path.display().to_string(),
            "protocol": PROTOCOL,
        }));
        for stream in self.listener.incoming() {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            let stream = match stream {
                Ok(stream) => stream,
                Err(e) => {
                    tracing::warn!(target: "garasu::canvas", error = %e, "accept failed");
                    continue;
                }
            };
            let (sink, ids, report) = (Arc::clone(sink), Arc::clone(&ids), self.report.clone());
            let spawned = std::thread::Builder::new()
                .name("garasu-canvas-connection".into())
                .spawn(move || {
                    if let Some(connection) = Connection::new(stream, &*sink, &ids, &report) {
                        connection.run();
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(target: "garasu::canvas", error = %e, "no thread for a connection");
            }
        }
    }
}

pub struct Running {
    path: PathBuf,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Running {
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = UnixStream::connect(&self.path);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let _ = fs::remove_file(&self.path);
    }
}
