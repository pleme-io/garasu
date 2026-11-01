use std::fmt;
use std::path::PathBuf;

use serde_json::{Value, json};

pub const PROTOCOL: &str = "garasu-canvas/1";
pub const SOCKET_VARIABLE: &str = "BLUE_CANVAS_SOCKET";
pub const MAX_SIDE: u32 = 4096;
pub const MAX_HEADER: usize = 64 * 1024;
pub const MAX_FRAME: u64 = MAX_SIDE as u64 * MAX_SIDE as u64 * 3;
pub const DEFAULT_TITLE: &str = "garasu-canvas";

#[must_use]
pub fn socket() -> PathBuf {
    if let Some(p) = std::env::var_os(SOCKET_VARIABLE) {
        return PathBuf::from(p);
    }
    std::env::var_os("XDG_RUNTIME_DIR")
        .map_or_else(std::env::temp_dir, PathBuf::from)
        .join("garasu")
        .join("canvas.sock")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub width: u32,
    pub height: u32,
    pub title: String,
}

impl Spec {
    #[must_use]
    pub fn frame_bytes(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height) * 3
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

impl Frame {
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let at = (usize::try_from(y).ok()? * usize::try_from(self.width).ok()?
            + usize::try_from(x).ok()?)
            * 3;
        self.rgb.get(at..at + 3).map(|p| [p[0], p[1], p[2]])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    HeaderTooLong,
    MalformedHeader(String),
    NotOpen(String),
    UnknownProtocol(String),
    BadSize {
        width: String,
        height: String,
    },
    OversizedFrame {
        op: String,
        allowed: u64,
        got: u64,
    },
    AlreadyOpen,
    UnknownOp(String),
    UnknownFormat(String),
    SizeMismatch {
        canvas: (u32, u32),
        got: (String, String),
    },
    ShortFrame {
        want: u64,
        got: u64,
    },
    WindowClosed,
    Unavailable(String),
}

impl Refusal {
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Refusal::HeaderTooLong => "header_too_long",
            Refusal::MalformedHeader(_) => "malformed_header",
            Refusal::NotOpen(_) => "not_open",
            Refusal::UnknownProtocol(_) => "unknown_protocol",
            Refusal::BadSize { .. } => "bad_size",
            Refusal::OversizedFrame { .. } => "oversized_frame",
            Refusal::AlreadyOpen => "already_open",
            Refusal::UnknownOp(_) => "unknown_op",
            Refusal::UnknownFormat(_) => "unknown_format",
            Refusal::SizeMismatch { .. } => "size_mismatch",
            Refusal::ShortFrame { .. } => "short_frame",
            Refusal::WindowClosed => "window_closed",
            Refusal::Unavailable(_) => "unavailable",
        }
    }

    #[must_use]
    pub fn closes(&self) -> bool {
        !matches!(
            self,
            Refusal::AlreadyOpen
                | Refusal::UnknownOp(_)
                | Refusal::UnknownFormat(_)
                | Refusal::SizeMismatch { .. }
                | Refusal::ShortFrame { .. }
        )
    }

    #[must_use]
    pub fn reply(&self) -> String {
        format!(
            "{}\n",
            json!({"ok": false, "error": self.to_string(), "refusal": self.name()})
        )
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: ", self.name())?;
        match self {
            Refusal::HeaderTooLong => write!(f, "a header line is at most {MAX_HEADER} bytes"),
            Refusal::MalformedHeader(why) | Refusal::Unavailable(why) => f.write_str(why),
            Refusal::NotOpen(op) => {
                write!(f, "the first message on a connection is open, got {op}")
            }
            Refusal::UnknownProtocol(got) => write!(f, "this server speaks {PROTOCOL}, got {got}"),
            Refusal::BadSize { width, height } => {
                write!(
                    f,
                    "a canvas side is 1 to {MAX_SIDE} pixels, got {width}x{height}"
                )
            }
            Refusal::OversizedFrame { op, allowed, got } => {
                write!(f, "{op} carries at most {allowed} payload bytes, got {got}")
            }
            Refusal::AlreadyOpen => f.write_str("this connection already holds a canvas"),
            Refusal::UnknownOp(op) => {
                write!(f, "this server knows open, present and close, got {op}")
            }
            Refusal::UnknownFormat(got) => write!(f, "a frame is rgb8, got {got}"),
            Refusal::SizeMismatch { canvas, got } => write!(
                f,
                "the canvas is {}x{}, the frame says {}x{}",
                canvas.0, canvas.1, got.0, got.1
            ),
            Refusal::ShortFrame { want, got } => {
                write!(f, "a frame for this canvas is {want} bytes, got {got}")
            }
            Refusal::WindowClosed => f.write_str("the canvas window was closed"),
        }
    }
}

impl std::error::Error for Refusal {}

#[must_use]
pub fn accepted() -> String {
    format!("{}\n", json!({"ok": true}))
}

#[derive(Clone, Debug)]
pub struct Header {
    pub op: String,
    pub bytes: u64,
    fields: Value,
}

fn shown(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "absent".into(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

fn side(v: Option<&Value>) -> Option<u32> {
    v.and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|n| (1..=MAX_SIDE).contains(n))
}

impl Header {
    pub fn parse(line: &[u8]) -> Result<Header, Refusal> {
        let fields: Value = serde_json::from_slice(line)
            .map_err(|e| Refusal::MalformedHeader(format!("the header is not JSON: {e}")))?;
        let op = fields
            .get("op")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Refusal::MalformedHeader("a header is a JSON object with a string op".into())
            })?
            .to_string();
        let bytes = match fields.get("bytes") {
            None | Some(Value::Null) => 0,
            Some(n) => n.as_u64().ok_or_else(|| {
                Refusal::MalformedHeader(format!("bytes is a whole number, got {n}"))
            })?,
        };
        Ok(Header { op, bytes, fields })
    }

    pub fn open(&self) -> Result<Spec, Refusal> {
        let protocol = self.fields.get("protocol");
        if protocol.and_then(Value::as_str) != Some(PROTOCOL) {
            return Err(Refusal::UnknownProtocol(shown(protocol)));
        }
        let (w, h) = (self.fields.get("width"), self.fields.get("height"));
        let (Some(width), Some(height)) = (side(w), side(h)) else {
            return Err(Refusal::BadSize {
                width: shown(w),
                height: shown(h),
            });
        };
        let title = match self.fields.get("title") {
            None | Some(Value::Null) => DEFAULT_TITLE.to_string(),
            Some(Value::String(t)) => t.clone(),
            Some(other) => {
                return Err(Refusal::MalformedHeader(format!(
                    "title is a string, got {other}"
                )));
            }
        };
        Ok(Spec {
            width,
            height,
            title,
        })
    }

    pub fn present(&self, spec: &Spec, rgb: Vec<u8>) -> Result<Frame, Refusal> {
        let format = self.fields.get("format");
        if format.and_then(Value::as_str) != Some("rgb8") {
            return Err(Refusal::UnknownFormat(shown(format)));
        }
        let (w, h) = (self.fields.get("width"), self.fields.get("height"));
        let same = |v: Option<&Value>, side: u32| v.and_then(Value::as_u64) == Some(side.into());
        if !same(w, spec.width) || !same(h, spec.height) {
            return Err(Refusal::SizeMismatch {
                canvas: (spec.width, spec.height),
                got: (shown(w), shown(h)),
            });
        }
        let got = rgb.len() as u64;
        if got != spec.frame_bytes() {
            return Err(Refusal::ShortFrame {
                want: spec.frame_bytes(),
                got,
            });
        }
        Ok(Frame {
            width: spec.width,
            height: spec.height,
            rgb,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(v: &Value) -> Header {
        Header::parse(v.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn an_accepted_message_is_answered_with_the_line_the_client_expects() {
        assert_eq!(accepted(), "{\"ok\":true}\n");
    }

    #[test]
    fn a_refusal_names_itself_in_the_error_the_client_shows_and_in_its_own_field() {
        let line = Refusal::WindowClosed.reply();
        let v: Value = serde_json::from_str(line.trim_end()).unwrap();
        assert_eq!(v["ok"], json!(false));
        assert_eq!(v["refusal"], json!("window_closed"));
        assert!(v["error"].as_str().unwrap().starts_with("window_closed: "));
        assert!(line.ends_with('\n'));
    }

    #[test]
    fn open_reads_the_protocol_the_size_and_the_title() {
        let spec = header(
            &json!({"op": "open", "protocol": PROTOCOL, "width": 3, "height": 2, "title": "probe"}),
        )
        .open()
        .unwrap();
        assert_eq!(
            spec,
            Spec {
                width: 3,
                height: 2,
                title: "probe".into()
            }
        );
        assert_eq!(spec.frame_bytes(), 18);
        let untitled =
            header(&json!({"op": "open", "protocol": PROTOCOL, "width": 1, "height": 1}))
                .open()
                .unwrap();
        assert_eq!(untitled.title, DEFAULT_TITLE);
    }

    #[test]
    fn open_refuses_another_protocol_and_a_side_outside_one_to_the_maximum() {
        let refusal = |v: Value| header(&v).open().unwrap_err().name();
        assert_eq!(
            refusal(json!({"op": "open", "protocol": "garasu-canvas/2", "width": 1, "height": 1})),
            "unknown_protocol"
        );
        assert_eq!(
            refusal(json!({"op": "open", "width": 1, "height": 1})),
            "unknown_protocol"
        );
        for (w, h) in [
            (json!(0), json!(1)),
            (json!(1), json!(MAX_SIDE + 1)),
            (json!(-3), json!(2)),
            (json!("4"), json!(4)),
            (Value::Null, json!(2)),
        ] {
            assert_eq!(
                refusal(json!({"op": "open", "protocol": PROTOCOL, "width": w, "height": h})),
                "bad_size"
            );
        }
        assert_eq!(
            refusal(
                json!({"op": "open", "protocol": PROTOCOL, "width": 2, "height": 2, "title": 7})
            ),
            "malformed_header"
        );
    }

    #[test]
    fn a_header_is_a_json_object_with_a_string_op_and_a_whole_byte_count() {
        for bad in [
            "",
            "not json",
            "[1]",
            "{\"op\": 3}",
            "{\"op\": \"present\", \"bytes\": -1}",
            "{\"op\": \"present\", \"bytes\": 1.5}",
        ] {
            assert_eq!(
                Header::parse(bad.as_bytes()).unwrap_err().name(),
                "malformed_header",
                "{bad}"
            );
        }
        let h = Header::parse(b"{\"op\":\"close\"}").unwrap();
        assert_eq!((h.op.as_str(), h.bytes), ("close", 0));
    }

    #[test]
    fn present_holds_a_frame_to_its_canvas() {
        let spec = Spec {
            width: 2,
            height: 1,
            title: String::new(),
        };
        let ok = header(
            &json!({"op": "present", "format": "rgb8", "width": 2, "height": 1, "bytes": 6}),
        );
        let frame = ok.present(&spec, vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(frame.pixel(1, 0), Some([4, 5, 6]));
        assert_eq!(frame.pixel(2, 0), None);
        assert_eq!(
            ok.present(&spec, vec![0; 3]).unwrap_err().name(),
            "short_frame"
        );
        let rgba = header(&json!({"op": "present", "format": "rgba8", "width": 2, "height": 1}));
        assert_eq!(
            rgba.present(&spec, vec![0; 6]).unwrap_err().name(),
            "unknown_format"
        );
        let wide = header(&json!({"op": "present", "format": "rgb8", "width": 3, "height": 1}));
        assert_eq!(
            wide.present(&spec, vec![0; 6]).unwrap_err().name(),
            "size_mismatch"
        );
    }

    #[test]
    fn framing_refusals_close_the_connection_and_message_refusals_keep_the_canvas() {
        let closes = [
            Refusal::HeaderTooLong,
            Refusal::MalformedHeader(String::new()),
            Refusal::NotOpen(String::new()),
            Refusal::UnknownProtocol(String::new()),
            Refusal::BadSize {
                width: String::new(),
                height: String::new(),
            },
            Refusal::OversizedFrame {
                op: String::new(),
                allowed: 0,
                got: 1,
            },
            Refusal::WindowClosed,
            Refusal::Unavailable(String::new()),
        ];
        let keeps = [
            Refusal::AlreadyOpen,
            Refusal::UnknownOp(String::new()),
            Refusal::UnknownFormat(String::new()),
            Refusal::SizeMismatch {
                canvas: (1, 1),
                got: (String::new(), String::new()),
            },
            Refusal::ShortFrame { want: 3, got: 0 },
        ];
        assert!(closes.iter().all(Refusal::closes));
        assert!(!keeps.iter().any(Refusal::closes));
    }
}
