#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use garasu::canvas::PROTOCOL;
use serde_json::{Value, json};

pub const BLUE_PROBE: [u8; 18] = [
    0, 0, 255, 255, 0, 0, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255,
];

pub struct Client {
    stream: UnixStream,
    replies: BufReader<UnixStream>,
}

impl Client {
    pub fn connect(path: &Path) -> Client {
        let stream = UnixStream::connect(path).expect("the server listens");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let replies = BufReader::new(stream.try_clone().unwrap());
        Client { stream, replies }
    }

    pub fn next_reply(&mut self) -> Option<Value> {
        let mut line = String::new();
        match self.replies.read_line(&mut line) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(serde_json::from_str(line.trim_end()).expect("a JSON reply")),
        }
    }

    pub fn send_raw(&mut self, bytes: &[u8]) -> Option<Value> {
        self.stream.write_all(bytes).ok()?;
        self.next_reply()
    }

    pub fn send(&mut self, header: &Value, payload: &[u8]) -> Option<Value> {
        let mut bytes = format!("{header}\n").into_bytes();
        bytes.extend_from_slice(payload);
        self.send_raw(&bytes)
    }

    pub fn open(&mut self, width: u32, height: u32, title: &str) -> Option<Value> {
        self.send(
            &json!({"op": "open", "protocol": PROTOCOL, "width": width, "height": height, "title": title}),
            &[],
        )
    }

    pub fn present(&mut self, width: u32, height: u32, rgb: &[u8]) -> Option<Value> {
        self.send(
            &json!({"op": "present", "format": "rgb8", "width": width, "height": height, "bytes": rgb.len()}),
            rgb,
        )
    }

    pub fn close(&mut self) -> Option<Value> {
        self.send(&json!({"op": "close"}), &[])
    }
}
pub fn rows(width: u32, height: u32, rgb: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| rgb(x, y))
        .collect()
}

#[allow(clippy::unnecessary_wraps)]
pub fn ok() -> Option<Value> {
    Some(json!({"ok": true}))
}

pub fn assert_refused(reply: Option<Value>, name: &str) {
    let reply = reply.unwrap_or_else(|| panic!("no reply where {name} was due"));
    assert_eq!(reply["ok"], json!(false), "{reply}");
    assert_eq!(reply["refusal"], json!(name), "{reply}");
    let error = reply["error"].as_str().expect("an error string");
    assert!(error.starts_with(&format!("{name}: ")), "{error}");
}
