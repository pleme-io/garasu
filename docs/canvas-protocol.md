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
garasu-canvas [--headless | --verify] [--socket PATH]
```

- With no flag, the server opens one window for each canvas. It draws each
  frame through garasu's `FramePipeline`: one texture upload and one
  full-screen triangle.
- `--headless` opens no window and needs no GPU. The server keeps the last
  frame of each canvas in memory and counts the frames. Use it in CI.
- `--verify` shows windows as with no flag, and checks each window against
  the frames it shows. `--verify` and `--headless` together are refused.

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
`canvas::window::run(server, Options { verify })` is the windowed one.

### Verify

With `--verify`, the server configures each window's surface with the
`COPY_SRC` usage as well as `RENDER_ATTACHMENT`. Without the flag it asks for
`RENDER_ATTACHMENT` only. On Metal, a usage other than `RENDER_ATTACHMENT`
alone turns off the layer's `framebufferOnly`, so the drawable is readable.

After the server draws a presented frame into the surface texture, and before
it presents that texture, it copies the same texture into a buffer and maps
it. It reads the window's own surface texture through wgpu, which is the
texture it then presents. It does not capture the screen, so it needs no
Screen Recording permission. What the compositor does with the texture after
`present` is outside this check.

The server compares each pixel it read with what `FramePipeline` draws for
the frame at the window's size: the frame fitted to the window, nearest pixel,
black around it. Two kinds of pixel may show either of two colours, because
GPUs rasterize them differently: a pixel centre within 1/64 of a pixel of a
texel edge, and a pixel that a fractional edge of the fitted frame cuts. At a
whole-number scale and offset there are none of either.

The server then writes one more event for the draw:

```json
{"blake3":"…","canvas":1,"event":"verified","frame":1,"surface":{"blake3":"…","format":"Bgra8UnormSrgb","height":400,"width":640}}
{"blake3":"…","canvas":1,"event":"mismatch","frame":1,"got":[18,52,86],"surface":{…},"want":[255,0,0],"x":160,"y":80}
{"canvas":1,"event":"unverified","frame":1,"reason":"…"}
```

- `frame` is the number of the presented frame that the window showed.
- `blake3` is the hash of the frame as read back at its own size: one pixel
  from the middle of each frame pixel's area. When the window shows the frame
  at a scale of 1 or more, it equals the `blake3` of the `present` event.
- `surface` is the read-back texture: its size in pixels, its format and the
  hash of its pixels as rgb8.
- `mismatch` names the first pixel, row by row from the top left, that is not
  the frame. `x` and `y` are surface pixels. `want` is the frame's colour
  there and `got` is the colour read back.
- `unverified` means the server could not read the texture back. `reason`
  says why, for example a backend that can not copy from a surface texture.

The server checks each draw of a frame, so a resize checks the frame again at
the new size. It does not check a window before its first frame.

`garasu::frame::Readback` holds the read-back pixels and the comparison.
`garasu::headless::read_texture` does the copy and the map, for a surface
texture and for a `HeadlessTarget` alike. Two tests run with the `gpu_tests`
feature. `src/frame.rs` draws a frame into an offscreen `Bgra8UnormSrgb`
texture at four sizes, reads it back and expects `verified`, then draws the
frame upside down and expects the first pixel that differs.
`tests/canvas_binary.rs` starts `garasu-canvas --verify`, shows a red
rectangle in a real window and expects `verified` with the `present` hash.
When no display can take a window, it says so and skips.
