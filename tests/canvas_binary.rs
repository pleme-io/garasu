#![cfg(unix)]

mod common;

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use common::{BLUE_PROBE, Client, assert_refused, ok};
use garasu::canvas::PROTOCOL;
use serde_json::{Value, json};

struct Running(Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn the_headless_binary_reports_each_canvas_event_as_one_json_line() {
    let dir = tempfile::Builder::new().prefix("gc").tempdir().unwrap();
    let path: PathBuf = dir.path().join("canvas.sock");
    let mut server = Running(
        Command::new(env!("CARGO_BIN_EXE_garasu-canvas"))
            .arg("--headless")
            .arg("--socket")
            .arg(&path)
            .stdout(Stdio::piped())
            .spawn()
            .expect("the binary starts"),
    );
    let mut events = BufReader::new(server.0.stdout.take().unwrap()).lines();
    let mut next =
        move || -> Value { serde_json::from_str(&events.next().unwrap().unwrap()).unwrap() };

    let listen = next();
    assert_eq!(listen["event"], json!("listen"));
    assert_eq!(listen["protocol"], json!(PROTOCOL));
    assert_eq!(listen["socket"], json!(path.display().to_string()));

    let mut client = Client::connect(&path);
    assert_eq!(client.open(3, 2, "bin"), ok());
    let open = next();
    assert_eq!(
        (
            open["event"].clone(),
            open["width"].clone(),
            open["height"].clone(),
            open["title"].clone()
        ),
        (json!("open"), json!(3), json!(2), json!("bin"))
    );
    let mut stranger = Client::connect(&path);
    assert_refused(stranger.send_raw(b"hello\n"), "malformed_header");
    let refused = next();
    assert_eq!(
        (
            refused["event"].clone(),
            refused["canvas"].clone(),
            refused["refusal"].clone()
        ),
        (json!("refuse"), Value::Null, json!("malformed_header"))
    );
    assert_eq!(client.present(3, 2, &BLUE_PROBE), ok());
    assert_refused(client.present(3, 2, &[0; 3]), "short_frame");
    assert_eq!(client.close(), ok());

    let present = next();
    assert_eq!(present["event"], json!("present"));
    assert_eq!(present["canvas"], open["canvas"]);
    assert_eq!(present["frame"], json!(1));
    assert_eq!(
        present["blake3"],
        json!(blake3::hash(&BLUE_PROBE).to_hex().to_string())
    );
    let refuse = next();
    assert_eq!(refuse["event"], json!("refuse"));
    assert_eq!(refuse["refusal"], json!("short_frame"));
    let close = next();
    assert_eq!(close["event"], json!("close"));
    assert_eq!(close["presented"], json!(1));
}
