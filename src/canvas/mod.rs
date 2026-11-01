mod connection;
pub mod headless;
pub mod protocol;
pub mod server;
pub mod window;

pub use headless::{CanvasView, HeadlessCanvases, Totals};
pub use protocol::{Frame, MAX_SIDE, PROTOCOL, Refusal, SOCKET_VARIABLE, Spec, socket};
pub use server::{CanvasId, Running, Server, Sink};
