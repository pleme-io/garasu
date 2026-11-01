# garasu-canvas/1

A client draws a frame on the CPU. It sends the frame to a canvas server on a
Unix socket. The server shows the frame in a window. The client links no GPU
stack. blue's `display` capability is the first client
(`crates/blue-lang-senses/src/display.rs` in blue). `garasu-canvas` is the
server.

The client's in-test server (`crates/blue-lang-senses/tests/display.rs` in
blue) is the reference for the wire format. This page adds the rules that the
reference leaves open: the handshake, framing bounds, errors and close.
`src/canvas/` holds the server code. `tests/canvas.rs` and
`tests/canvas_binary.rs` test it.

## The socket

The client and the server find the socket in the same order:

1. `$BLUE_CANVAS_SOCKET`
2. `$XDG_RUNTIME_DIR/garasu/canvas.sock`
3. `$TMPDIR/garasu/canvas.sock` (the platform temporary directory)

The server makes the socket directory with mode `0700` if it is missing. It
sets the socket to mode `0600`. A socket file with no listener is stale. The
server removes it and binds. If a server already listens on the path, the new
server does not start. A path that is not a socket is never removed.

## One connection is one canvas

A client opens one connection for each canvas. The server serves each
connection on its own thread. A refusal on one connection does not stop other
connections.

## Messages

Each message is one header line and then a payload:

- The header is one JSON object on one line. A `\n` ends the line. The line
  is at most 65536 bytes without the `\n`.
- `op` is a string. It is `open`, `present` or `close`.
- `bytes` is the payload length in bytes. If `bytes` is absent, the payload
  is empty.
- The payload follows the `\n` directly.

The server sends one reply line for each message:

- `{"ok":true}` when it accepts the message.
- `{"ok":false,"error":"<name>: <reason>","refusal":"<name>"}` when it refuses
  the message.

The error text starts with the refusal name. A client that shows only `error`
thus shows the name too. A program reads `refusal`.

### open

The first message on a connection must be `open`:

```json
{"op":"open","protocol":"garasu-canvas/1","width":320,"height":200,"title":"blue"}
```

- `protocol` must be `garasu-canvas/1`.
- `width` and `height` are whole numbers from 1 to 4096.
- `title` is a string. If it is absent, the title is `garasu-canvas`.
- `open` has no payload.

`ok` means the window is open (or, in headless mode, the canvas is held).

### present

```json
{"op":"present","format":"rgb8","width":320,"height":200,"bytes":192000}
```

The payload is the frame: `width * height` pixels, row by row from the top
left. Each pixel is three bytes: red, green, blue. These are sRGB values. The
server shows each value exactly.

- `format` must be `rgb8`.
- `width` and `height` must be the size from `open`.
- `bytes` must be `width * height * 3`.

`ok` means the server holds the frame. The window shows it at its next
redraw. The window keeps the frame's aspect ratio and fills the rest with
black. It scales with the nearest pixel.

### close

```json
{"op":"close"}
```

The server frees the canvas and closes the window. Then it replies `ok` and
closes the connection. A client that closes the connection without `close`
also frees its canvas.

## Refusals

A framing refusal closes the connection, because the server can not read the
rest of the stream safely. A message refusal keeps the canvas open. The server
reads the payload of a refused message only when it fits the bound of that
message.

| Refusal | When | Connection |
|---|---|---|
| `header_too_long` | the header line is longer than 65536 bytes | closes |
| `malformed_header` | the header is not a JSON object with a string `op`; `bytes` is not a whole number; `title` is not a string | closes |
| `not_open` | the first message is not `open` | closes |
| `unknown_protocol` | `protocol` is not `garasu-canvas/1` | closes |
| `bad_size` | a side is not a whole number from 1 to 4096 | closes |
| `oversized_frame` | `bytes` is more than the message can carry: `width * height * 3` for `present`, 0 for any other op | closes |
| `window_closed` | the user closed the window | closes |
| `unavailable` | the server can not open a window or reach the display | closes |
| `already_open` | a second `open` on a connection | stays |
| `unknown_op` | an `op` other than `open`, `present` and `close` | stays |
| `unknown_format` | `format` is not `rgb8` | stays |
| `size_mismatch` | `present` names a size other than the canvas size | stays |
| `short_frame` | `bytes` is less than `width * height * 3` | stays |

When a refusal closes the connection, the server:

1. frees the canvas,
2. sends the refusal line,
3. shuts down its write side,
4. reads and discards what the client still sends: at most one 4096 x 4096
   frame and one header, for at most 5 seconds,
5. closes the connection.

Step 4 lets a client that is still writing a large payload finish. The client
then reads the refusal and does not get a broken pipe.

## The server

```text
garasu-canvas [--headless] [--socket PATH]
```

- With no flag, the server opens one window for each canvas. It draws each
  frame through garasu's `FramePipeline`: one texture upload and one
  full-screen triangle.
- `--headless` opens no window and needs no GPU. The server keeps the last
  frame of each canvas in memory and counts the frames. Use it in CI.

The server writes one JSON line to standard output for each event:

```json
{"event":"listen","protocol":"garasu-canvas/1","socket":"/run/user/1000/garasu/canvas.sock"}
{"canvas":1,"event":"open","height":200,"title":"blue","width":320}
{"blake3":"…","canvas":1,"event":"present","frame":1}
{"canvas":1,"error":"short_frame: …","event":"refuse","refusal":"short_frame"}
{"canvas":1,"event":"close","presented":1}
```

`blake3` is the hash of the frame's rgb8 bytes. `canvas` is null on a refusal
that comes before `open`.

In Rust, `garasu::canvas::Server` binds the socket. It serves any
`garasu::canvas::Sink`. `HeadlessCanvases` is the headless sink, and
`canvas::window::run` is the windowed one.
