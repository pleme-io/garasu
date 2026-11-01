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

#[cfg(feature = "gpu_tests")]
#[test]
fn a_verifying_window_reads_back_the_frame_it_was_sent() {
    use std::io::Read;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    const NAVY: [u8; 3] = [18, 52, 86];
    const RED: [u8; 3] = [255, 0, 0];
    let (width, height) = (320, 200);

    let dir = tempfile::Builder::new().prefix("gc").tempdir().unwrap();
    let path: PathBuf = dir.path().join("canvas.sock");
    let mut server = Running(
        Command::new(env!("CARGO_BIN_EXE_garasu-canvas"))
            .arg("--verify")
            .arg("--socket")
            .arg(&path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("the binary starts"),
    );
    let stdout = server.0.stdout.take().unwrap();
    let (sent, events) = mpsc::channel::<Value>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            eprintln!("{line}");
            if sent.send(serde_json::from_str(&line).unwrap()).is_err() {
                break;
            }
        }
    });
    let listen = events.recv_timeout(Duration::from_secs(20));
    if matches!(listen, Err(mpsc::RecvTimeoutError::Disconnected)) {
        let _ = server.0.wait();
        let mut stderr = String::new();
        let _ = server.0.stderr.take().unwrap().read_to_string(&mut stderr);
        assert!(
            stderr.contains("window error"),
            "the server stopped before it listened: {stderr}"
        );
        eprintln!("skipped: no display to open a window on: {stderr}");
        return;
    }
    let listen = listen.expect("the server listens within 20 s");
    assert_eq!(listen["event"], json!("listen"));

    let mut client = Client::connect(&path);
    let opened = client.open(width, height, "verify").expect("a reply");
    if opened["refusal"] == json!("unavailable") {
        eprintln!("skipped: the server can not show a window here: {opened}");
        return;
    }
    assert_eq!(Some(opened), ok());
    let rgb = common::rows(width, height, |x, y| {
        if (80..200).contains(&x) && (40..120).contains(&y) {
            RED
        } else {
            NAVY
        }
    });
    let hash = json!(blake3::hash(&rgb).to_hex().to_string());
    assert_eq!(client.present(width, height, &rgb), ok());

    let deadline = Instant::now() + Duration::from_secs(20);
    let (mut present, mut verdict) = (None, None);
    while present.is_none() || verdict.is_none() {
        let wait = deadline.saturating_duration_since(Instant::now());
        let event = events
            .recv_timeout(wait)
            .expect("the window reports a present and a verdict for frame 1");
        match event["event"].as_str() {
            Some("present") => present = Some(event),
            Some("verified" | "mismatch" | "unverified") if event["frame"] == json!(1) => {
                verdict = Some(event);
            }
            _ => {}
        }
    }
    let (present, verdict) = (present.unwrap(), verdict.unwrap());
    assert_eq!(present["blake3"], hash);
    assert_eq!(verdict["event"], json!("verified"), "{verdict}");
    assert_eq!(verdict["canvas"], present["canvas"]);
    assert_eq!(verdict["blake3"], hash, "{verdict}");
    let surface = &verdict["surface"];
    assert!(
        surface["width"].as_u64().unwrap() >= u64::from(width),
        "{verdict}"
    );
    assert!(
        surface["height"].as_u64().unwrap() >= u64::from(height),
        "{verdict}"
    );
    assert_eq!(client.close(), ok());
}
