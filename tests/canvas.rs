#![cfg(unix)]

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{BLUE_PROBE, Client, assert_refused, ok, rows};
use garasu::canvas::{HeadlessCanvases, PROTOCOL, Running, Server, Sink, Totals};
use serde_json::{Value, json};

struct Harness {
    running: Running,
    canvases: Arc<HeadlessCanvases>,
    _dir: tempfile::TempDir,
}

impl Harness {
    fn new() -> Harness {
        let dir = tempfile::Builder::new().prefix("gc").tempdir().unwrap();
        let canvases = Arc::new(HeadlessCanvases::default());
        let sink: Arc<dyn Sink> = canvases.clone();
        let running = Server::bind(dir.path().join("canvas.sock"))
            .unwrap()
            .spawn(sink)
            .unwrap();
        Harness {
            running,
            canvases,
            _dir: dir,
        }
    }

    fn connect(&self) -> Client {
        Client::connect(self.running.path())
    }
}

fn eventually(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn a_canvas_presented_twice_holds_the_second_frame_pixel_for_pixel() {
    let h = Harness::new();
    let mut client = h.connect();
    assert_eq!(client.open(3, 2, "probe"), ok());
    let first = rows(3, 2, |x, y| [u8::try_from(x * 80 + y).unwrap(), 9, 200]);
    assert_eq!(client.present(3, 2, &first), ok());
    assert_eq!(client.present(3, 2, &BLUE_PROBE), ok());

    let held = h.canvases.titled("probe").expect("the canvas is held");
    assert_eq!((held.spec.width, held.spec.height), (3, 2));
    assert_eq!(held.presented, 2);
    let last = held.last.expect("a presented frame");
    assert_eq!(last.rgb, BLUE_PROBE.to_vec());
    assert_ne!(last.rgb, first);
    assert_eq!(last.pixel(1, 0), Some([255, 0, 0]));
    assert_eq!(last.pixel(2, 1), Some([0, 0, 255]));
    assert_eq!(
        h.canvases.totals(),
        Totals {
            opened: 1,
            presented: 2,
            closed: 0
        }
    );
}

#[test]
fn a_malformed_handshake_is_refused_by_name_and_the_server_keeps_serving_other_clients() {
    let h = Harness::new();
    let mut keeper = h.connect();
    assert_eq!(keeper.open(2, 1, "keeper"), ok());

    let open = |protocol: &str, width: Value| {
        format!(
            "{}\n",
            json!({"op": "open", "protocol": protocol, "width": width, "height": 1})
        )
        .into_bytes()
    };
    let mut long = vec![b'a'; 70_000];
    long.push(b'\n');
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (b"not json\n".to_vec(), "malformed_header"),
        (b"{\"protocol\":\"garasu-canvas/1\"}\n".to_vec(), "malformed_header"),
        (
            b"{\"op\":\"present\",\"format\":\"rgb8\",\"width\":1,\"height\":1,\"bytes\":3}\n\x01\x02\x03"
                .to_vec(),
            "not_open",
        ),
        (open("garasu-canvas/2", json!(1)), "unknown_protocol"),
        (open(PROTOCOL, json!(0)), "bad_size"),
        (open(PROTOCOL, json!(4097)), "bad_size"),
        (long, "header_too_long"),
    ];
    for (bytes, name) in cases {
        let mut bad = h.connect();
        assert_refused(bad.send_raw(&bytes), name);
        assert_eq!(
            bad.next_reply(),
            None,
            "a refused handshake ends its connection"
        );
    }

    assert_eq!(h.canvases.totals().opened, 1);
    assert_eq!(keeper.present(2, 1, &[1, 2, 3, 4, 5, 6]), ok());
    let mut late = h.connect();
    assert_eq!(late.open(1, 1, "late"), ok());
    assert_eq!(h.canvases.live().len(), 2);
    assert_eq!(h.canvases.titled("keeper").unwrap().presented, 1);
}

#[test]
fn an_oversized_frame_is_refused_by_name_and_frees_only_its_own_canvas() {
    let h = Harness::new();
    let mut other = h.connect();
    assert_eq!(other.open(2, 1, "other"), ok());
    let mut big = h.connect();
    assert_eq!(big.open(3, 2, "big"), ok());

    let payload = vec![7u8; 1 << 20];
    let reply = big.send(
        &json!({"op": "present", "format": "rgb8", "width": 3, "height": 2, "bytes": payload.len()}),
        &payload,
    );
    let error = reply
        .as_ref()
        .map(|r| r["error"].to_string())
        .unwrap_or_default();
    assert_refused(reply, "oversized_frame");
    assert!(error.contains("at most 18 payload bytes"), "{error}");
    assert_eq!(
        big.next_reply(),
        None,
        "an oversized frame ends its connection"
    );
    assert!(h.canvases.titled("big").is_none(), "its canvas is freed");

    assert_eq!(other.present(2, 1, &[9; 6]), ok());
    let mut next = h.connect();
    assert_eq!(next.open(4, 4, "next"), ok());
    assert_eq!(h.canvases.live().len(), 2);
}

#[test]
fn a_refused_frame_that_fits_its_canvas_keeps_the_canvas_open() {
    let h = Harness::new();
    let mut client = h.connect();
    assert_eq!(client.open(3, 2, "kept"), ok());
    assert_refused(client.present(3, 2, &[0; 6]), "short_frame");
    assert_refused(client.present(2, 3, &BLUE_PROBE), "size_mismatch");
    assert_refused(
        client.send(
            &json!({"op": "present", "format": "rgba8", "width": 3, "height": 2, "bytes": 18}),
            &BLUE_PROBE,
        ),
        "unknown_format",
    );
    assert_refused(client.open(3, 2, "again"), "already_open");
    assert_refused(client.send(&json!({"op": "resize"}), &[]), "unknown_op");
    assert_eq!(client.present(3, 2, &BLUE_PROBE), ok());
    let held = h.canvases.titled("kept").unwrap();
    assert_eq!(held.presented, 1);
    assert_eq!(held.last.unwrap().rgb, BLUE_PROBE.to_vec());
}

#[test]
fn close_frees_the_canvas_and_so_does_a_client_that_leaves() {
    let h = Harness::new();
    let mut client = h.connect();
    assert_eq!(client.open(4, 4, "closing"), ok());
    assert_eq!(client.present(4, 4, &[5; 48]), ok());
    assert_eq!(h.canvases.live().len(), 1);
    assert_eq!(client.close(), ok());
    assert_eq!(client.next_reply(), None, "close ends the connection");
    assert_eq!(h.canvases.live().len(), 0);
    assert_eq!(
        h.canvases.totals(),
        Totals {
            opened: 1,
            presented: 1,
            closed: 1
        }
    );

    let mut leaving = h.connect();
    assert_eq!(leaving.open(2, 2, "leaving"), ok());
    assert!(h.canvases.titled("leaving").is_some());
    drop(leaving);
    eventually("a client that leaves frees its canvas", || {
        h.canvases.live().is_empty()
    });
    assert_eq!(h.canvases.totals().closed, 2);
}

#[test]
fn bind_replaces_a_stale_socket_and_refuses_a_live_one() {
    let dir = tempfile::Builder::new().prefix("gc").tempdir().unwrap();
    let path = dir.path().join("canvas.sock");
    drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
    assert!(path.exists(), "a stale socket file is left behind");
    let sink: Arc<dyn Sink> = Arc::new(HeadlessCanvases::default());
    let running = Server::bind(&path).unwrap().spawn(sink).unwrap();
    let refused = Server::bind(&path)
        .err()
        .expect("a live server keeps its socket");
    assert!(refused.to_string().contains("already listens"), "{refused}");
    assert_eq!(Client::connect(&path).open(1, 1, "live"), ok());
    drop(running);
    assert!(!path.exists());
}
