//! kentou (見当) — MOVED to [`mekuri::kentou`].
//!
//! ── ★ RETIRED IN PLACE, NOT DELETED ─────────────────────────────────────────
//!
//! The module is pure — its only import was `core::marker::PhantomData` — but
//! it lived here, in a crate that pulls wgpu, winit and glyphon. So the
//! consumer that needed it most could not take it: `omoya` is a CPU compositor
//! and must not acquire a GPU stack to ask a question about buffer identity.
//! It authored a duplicate instead, which is the cost this move removes.
//!
//! It now lives in `mekuri`, which is zero-dependency and `no_std`, already
//! owns the adjacent question ("is a frame owed, and the permission to draw
//! it"), and already lists `damage` and `compositor` among its keywords.
//!
//! ★ This is a RE-EXPORT, not a deletion (MODULARIZE, DON'T DELETE).
//! `garasu::kentou::Target` still resolves, to the same type, so nothing that
//! already depends on this path has to move.

pub use mekuri::kentou::*;
