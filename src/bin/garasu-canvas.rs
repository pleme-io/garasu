use std::process::ExitCode;

#[cfg(unix)]
const USAGE: &str = "usage: garasu-canvas [--headless | --verify] [--socket PATH]

Serves garasu-canvas/1 on a Unix socket and shows each canvas in a window.
--headless  keep frames in memory, open no window
--verify    after each draw of a presented frame, read the window's surface
            texture back through the GPU, compare it with the frame and
            report verified, mismatch or unverified
--socket    the socket path (default: $BLUE_CANVAS_SOCKET, else
            $XDG_RUNTIME_DIR/garasu/canvas.sock, else $TMPDIR/garasu/canvas.sock)
";

#[cfg(unix)]
fn run(args: impl Iterator<Item = String>) -> Result<(), String> {
    use std::path::PathBuf;
    use std::sync::Arc;

    use garasu::canvas::{self, HeadlessCanvases, Server};

    let mut headless = false;
    let mut options = canvas::window::Options::default();
    let mut socket = None;
    let mut args = args;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--headless" => headless = true,
            "--verify" => options.verify = true,
            "--socket" => socket = Some(args.next().ok_or("--socket takes a path")?),
            "--help" | "-h" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    if headless && options.verify {
        return Err(format!(
            "--verify reads back a window, and --headless opens none\n{USAGE}"
        ));
    }
    let path = socket.map_or_else(canvas::socket, PathBuf::from);
    let server = Server::bind(&path)
        .map_err(|e| e.to_string())?
        .report_to(std::io::stdout());
    if headless {
        let canvases: Arc<dyn canvas::Sink> = Arc::new(HeadlessCanvases::default());
        server.serve(&canvases);
        Ok(())
    } else {
        canvas::window::run(server, options).map_err(|e| e.to_string())
    }
}

#[cfg(not(unix))]
fn run(_: impl Iterator<Item = String>) -> Result<(), String> {
    Err("garasu-canvas serves a Unix socket; this platform has none".into())
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("garasu-canvas: {e}");
            ExitCode::FAILURE
        }
    }
}
