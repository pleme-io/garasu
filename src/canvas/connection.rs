use std::io::{BufRead, BufReader, Read, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::json;

use super::protocol::{Header, MAX_FRAME, MAX_HEADER, Refusal, Spec, accepted};
use super::server::{CanvasId, Report, Sink};

const LINGER: Duration = Duration::from_secs(5);

struct Held {
    id: CanvasId,
    spec: Spec,
    presented: u64,
}

enum Next {
    Continue,
    End,
}

pub(crate) struct Connection<'a> {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    sink: &'a dyn Sink,
    ids: &'a AtomicU64,
    report: &'a Report,
    canvas: Option<Held>,
}

impl<'a> Connection<'a> {
    pub(crate) fn new(
        stream: UnixStream,
        sink: &'a dyn Sink,
        ids: &'a AtomicU64,
        report: &'a Report,
    ) -> Option<Connection<'a>> {
        let reader = BufReader::new(stream.try_clone().ok()?);
        Some(Connection {
            reader,
            writer: stream,
            sink,
            ids,
            report,
            canvas: None,
        })
    }

    pub(crate) fn run(mut self) {
        loop {
            let next = match self.step() {
                Ok(next) => next,
                Err(refusal) => self.refuse(&refusal),
            };
            if matches!(next, Next::End) {
                break;
            }
        }
        self.free();
    }

    fn free(&mut self) {
        if let Some(held) = self.canvas.take() {
            self.sink.close(held.id);
            self.report
                .emit(&json!({"event": "close", "canvas": held.id, "presented": held.presented}));
        }
    }

    fn header(&mut self) -> Result<Option<Header>, Refusal> {
        let mut line = Vec::new();
        let limit = u64::try_from(MAX_HEADER).unwrap_or(u64::MAX) + 1;
        match self
            .reader
            .by_ref()
            .take(limit)
            .read_until(b'\n', &mut line)
        {
            Ok(0) | Err(_) => Ok(None),
            Ok(_) if line.last() == Some(&b'\n') => {
                line.pop();
                Header::parse(&line).map(Some)
            }
            Ok(n) if n > MAX_HEADER => Err(Refusal::HeaderTooLong),
            Ok(_) => Ok(None),
        }
    }

    fn step(&mut self) -> Result<Next, Refusal> {
        let Some(header) = self.header()? else {
            return Ok(Next::End);
        };
        if self.canvas.is_none() && header.op != "open" {
            return Err(Refusal::NotOpen(header.op));
        }
        let allowed = match (&self.canvas, header.op.as_str()) {
            (Some(held), "present") => held.spec.frame_bytes(),
            _ => 0,
        };
        if header.bytes > allowed {
            return Err(Refusal::OversizedFrame {
                op: header.op,
                allowed,
                got: header.bytes,
            });
        }
        let mut payload = vec![0u8; usize::try_from(header.bytes).unwrap_or(0)];
        if self.reader.read_exact(&mut payload).is_err() {
            return Ok(Next::End);
        }
        match header.op.as_str() {
            "open" if self.canvas.is_some() => Err(Refusal::AlreadyOpen),
            "open" => {
                let spec = header.open()?;
                let id = self.ids.fetch_add(1, Ordering::SeqCst);
                self.sink.open(id, &spec)?;
                self.report.emit(&json!({
                    "event": "open",
                    "canvas": id,
                    "width": spec.width,
                    "height": spec.height,
                    "title": spec.title,
                }));
                self.canvas = Some(Held {
                    id,
                    spec,
                    presented: 0,
                });
                Ok(self.answer())
            }
            "present" => {
                let sink = self.sink;
                let report = self.report;
                let Some(held) = self.canvas.as_mut() else {
                    return Err(Refusal::NotOpen(header.op));
                };
                let frame = header.present(&held.spec, payload)?;
                let hash = blake3::hash(&frame.rgb).to_hex().to_string();
                sink.present(held.id, frame)?;
                held.presented += 1;
                report.emit(&json!({
                    "event": "present",
                    "canvas": held.id,
                    "frame": held.presented,
                    "blake3": hash,
                }));
                Ok(self.answer())
            }
            "close" => {
                self.free();
                self.answer();
                Ok(Next::End)
            }
            other => Err(Refusal::UnknownOp(other.to_string())),
        }
    }

    fn answer(&mut self) -> Next {
        self.write(&accepted())
    }

    fn write(&mut self, line: &str) -> Next {
        match self
            .writer
            .write_all(line.as_bytes())
            .and_then(|()| self.writer.flush())
        {
            Ok(()) => Next::Continue,
            Err(_) => Next::End,
        }
    }

    fn refuse(&mut self, refusal: &Refusal) -> Next {
        self.report.emit(&json!({
            "event": "refuse",
            "canvas": self.canvas.as_ref().map(|held| held.id),
            "refusal": refusal.name(),
            "error": refusal.to_string(),
        }));
        if !refusal.closes() {
            return self.write(&refusal.reply());
        }
        self.free();
        self.write(&refusal.reply());
        self.linger();
        Next::End
    }

    fn linger(&mut self) {
        let _ = self.writer.shutdown(Shutdown::Write);
        let deadline = Instant::now() + LINGER;
        let mut left = MAX_FRAME + u64::try_from(MAX_HEADER).unwrap_or(0);
        let mut buf = vec![0u8; 64 * 1024];
        while left > 0 {
            let Some(wait) = deadline
                .checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero())
            else {
                break;
            };
            if self.writer.set_read_timeout(Some(wait)).is_err() {
                break;
            }
            match self.reader.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => left = left.saturating_sub(n as u64),
            }
        }
    }
}
