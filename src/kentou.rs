//! **kentou (見当) — registration: proof of what a render target holds.**
//!
//! 見当 is the printer's word for *registration*, the alignment of plates so
//! impressions land on top of each other. When registration fails you do not
//! get a blank page — you get **ghosting and doubled images**, the previous
//! impression still visible beneath the current one.
//!
//! # The invariant
//!
//! **`LoadOp::Load` on a target whose identity is unknown is unsound.**
//!
//! wgpu hands you a `SurfaceTexture` with **no slot index and no buffer age**.
//! So for a swapchain image, "what does this already hold?" is not merely
//! unknown — it is *not askable*. Loading previous contents and painting only
//! what changed since the last frame is therefore a guess: the slot may be N
//! frames old, and everything that changed while it sat out is never painted.
//!
//! This module makes that guess **unrepresentable** rather than discouraged.
//! [`Target::load_preserving`] exists only for [`Known`]; on [`Unknown`] it is
//! an absent method — `E0599` at compile time, not a runtime check and not a
//! lint someone can allow.
//!
//! ```compile_fail
//! # use garasu::kentou::{Target, Unknown, Damage, Revision};
//! let mut t: Target<Unknown> = Target::surface(1920, 1080);
//! let d = Damage::everything(Revision::ORIGIN, 1920, 1080);
//! // Unknown contents: preserving them is not a thing you can ask for.
//! let _ = t.load_preserving(&d);
//! ```
//!
//! The same code on a target we allocated compiles, because there the question
//! has an answer:
//!
//! ```
//! # use garasu::kentou::{Target, Known, Damage, Revision};
//! let mut t: Target<Known> = Target::owned(1920, 1080, Revision::ORIGIN);
//! let d = Damage::everything(Revision::ORIGIN, 1920, 1080);
//! assert!(t.load_preserving(&d).is_ok());
//! ```
//!
//! # Scope, and what this is NOT
//!
//! kentou answers *into which target, covering what damage, and is that sound?*
//! It does **not** answer *should a frame be drawn at all* — that is
//! [`mekuri`](https://github.com/pleme-io/mekuri)'s page-turn decision, which
//! already ships `Cause`/`Ledger`/`Gate`/`Verdict`/`Pass` and which omoya
//! already consumes. The two compose: mekuri says `Draw`, kentou governs what
//! that draw may do. **This module introduces no second frame-necessity gate**
//! — an earlier draft did, which is the near-miss the compounding directive
//! warns about, recorded here so the boundary stays visible.
//!
//! Nor does it own pass topology within a frame; that is `engawa`.
//!
//! # Receipt
//!
//! Canonical: `theory/KENTOU.md`. Measured on plo 2026-08-28, mado's terminal
//! model held one non-blank row while the screen showed two stale prompts and
//! not the current one. **The mechanism recorded there is a hypothesis a
//! differential test did not confirm** — read §I before citing it. This module
//! seals the class regardless of which instance produced that frame, which is
//! the point of sealing a class rather than patching an instance.

use core::marker::PhantomData;

/// A version of the model being rendered.
///
/// Monotonic and opaque: not a timestamp, not a frame number. It answers
/// exactly one question — *is this the same content as before?* — and the
/// only way to advance it is [`Revision::next`], so it cannot move backwards.
///
/// A rewind is worse than it looks: consumers compare revisions for
/// **inequality**, so a counter that goes back makes two genuinely different
/// states compare equal, and the second one is never drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Revision(u64);

impl Revision {
    /// The revision every target and model starts at.
    pub const ORIGIN: Self = Self(0);

    /// The next revision. The only way to advance one.
    #[must_use]
    pub const fn next(self) -> Self {
        // Saturating, not wrapping. At u64 this is unreachable in practice
        // (585 years at 1 GHz), and wrapping would reintroduce the rewind
        // this type exists to forbid.
        Self(self.0.saturating_add(1))
    }

    /// The raw counter, for interop with an existing seqno.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Adopt an existing counter — for wiring kentou into code that already
    /// tracks a seqno, without demanding both move in lockstep on day one.
    #[must_use]
    pub const fn from_raw(v: u64) -> Self {
        Self(v)
    }
}

/// A rectangle of a target, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Region {
    /// A region covering a whole target of the given extent.
    #[must_use]
    pub const fn everything(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    /// True when this region covers every pixel of `width × height`.
    #[must_use]
    pub const fn covers_all(&self, width: u32, height: u32) -> bool {
        self.x == 0 && self.y == 0 && self.width >= width && self.height >= height
    }

    /// True when the region lies inside a target of the given extent.
    #[must_use]
    pub const fn fits_within(&self, width: u32, height: u32) -> bool {
        // Saturating so an overflowing region reads as out-of-bounds rather
        // than wrapping into a small in-bounds one.
        self.x.saturating_add(self.width) <= width
            && self.y.saturating_add(self.height) <= height
    }
}

/// What changed, **and since when**.
///
/// Damage is never "the changed region" on its own. A naked region invites
/// being applied to a target that is not at the baseline it was computed
/// against, which is the defect this whole module exists to remove — so the
/// baseline travels with it and [`Target::load_preserving`] checks it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Damage {
    base: Revision,
    region: Region,
}

impl Damage {
    /// Damage covering a whole target, relative to `base`.
    #[must_use]
    pub const fn everything(base: Revision, width: u32, height: u32) -> Self {
        Self {
            base,
            region: Region::everything(width, height),
        }
    }

    /// Damage covering `region`, relative to `base`.
    #[must_use]
    pub const fn since(base: Revision, region: Region) -> Self {
        Self { base, region }
    }

    /// The revision this damage was computed against.
    #[must_use]
    pub const fn base(&self) -> Revision {
        self.base
    }

    /// The region.
    #[must_use]
    pub const fn region(&self) -> Region {
        self.region
    }

    /// The union of two damages.
    ///
    /// Returns `None` when the bases differ: a union of "changed since 4" and
    /// "changed since 7" has no meaningful baseline, and picking one silently
    /// is how a conservative-looking merge loses the frames between.
    #[must_use]
    pub fn union(self, other: Self) -> Option<Self> {
        if self.base != other.base {
            return None;
        }
        let x = self.region.x.min(other.region.x);
        let y = self.region.y.min(other.region.y);
        let right = (self.region.x + self.region.width).max(other.region.x + other.region.width);
        let bottom = (self.region.y + self.region.height).max(other.region.y + other.region.height);
        Some(Self {
            base: self.base,
            region: Region {
                x,
                y,
                width: right - x,
                height: bottom - y,
            },
        })
    }
}

/// Sealed marker: whether a target's current contents are knowable.
mod sealed {
    pub trait Identity {}
}

/// Whether the contents of a target are knowable.
///
/// Sealed, so the set is closed: a downstream crate cannot introduce a third
/// identity that the safety argument was never made for.
pub trait Identity: sealed::Identity {}

/// We allocated this target and track its revision. Its contents are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Known;

/// This target was handed to us. Its contents are **not knowable** — a
/// swapchain image, where wgpu exposes neither slot index nor buffer age.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unknown;

impl sealed::Identity for Known {}
impl sealed::Identity for Unknown {}
impl Identity for Known {}
impl Identity for Unknown {}

/// Why a `load_preserving` was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Coverage {
    /// The damage was computed against an older revision than this target
    /// holds, so the frames between it and now are unaccounted for.
    #[error(
        "damage is relative to revision {damage_base}, but this target holds \
         revision {target}; the frames between are unaccounted for"
    )]
    StaleBaseline { damage_base: u64, target: u64 },

    /// The damage claims a region outside the target.
    #[error("damage region {region:?} does not fit within {width}x{height}")]
    OutOfBounds {
        region: Region,
        width: u32,
        height: u32,
    },
}

/// A render target, typed by whether its current contents are knowable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target<I: Identity> {
    width: u32,
    height: u32,
    revision: Revision,
    _identity: PhantomData<I>,
}

impl<I: Identity> Target<I> {
    /// Clear the target and paint every pixel.
    ///
    /// Available on **every** target, including one whose contents are
    /// unknown, because it does not depend on them: after a clear, the target
    /// holds exactly what this frame painted. This is the always-sound
    /// operation, and it is deliberately the one with no preconditions.
    pub fn clear_and_paint_all(&mut self, to: Revision) -> Painted {
        self.revision = to;
        Painted {
            revision: to,
            region: Region::everything(self.width, self.height),
        }
    }

    /// The extent.
    #[must_use]
    pub const fn extent(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The revision this target currently holds.
    #[must_use]
    pub const fn revision(&self) -> Revision {
        self.revision
    }
}

impl Target<Unknown> {
    /// A target handed to us — a swapchain image.
    ///
    /// There is no `revision` argument because there is nothing true to pass:
    /// wgpu will not say which slot this is or how old it is. The type records
    /// that ignorance rather than letting a caller invent a number.
    #[must_use]
    pub const fn surface(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            revision: Revision::ORIGIN,
            _identity: PhantomData,
        }
    }

    /// Adopt this target as `Known` by clearing it.
    ///
    /// **The only route from `Unknown` to `Known`**, and it is a route through
    /// a full paint — which is exactly the honest price. After a clear the
    /// contents are known because we just wrote all of them.
    pub fn adopt_by_clearing(self, to: Revision) -> Target<Known> {
        Target {
            width: self.width,
            height: self.height,
            revision: to,
            _identity: PhantomData,
        }
    }
}

impl Target<Known> {
    /// A target we allocated and track.
    #[must_use]
    pub const fn owned(width: u32, height: u32, revision: Revision) -> Self {
        Self {
            width,
            height,
            revision,
            _identity: PhantomData,
        }
    }

    /// Preserve existing contents and paint only `damage`.
    ///
    /// **Present only on `Target<Known>`.** On [`Unknown`] this method does not
    /// exist — see the module docs for the `compile_fail` proof. That absence
    /// is the seal: loading previous contents requires proof of *which*
    /// contents, and the proof is the type.
    ///
    /// Even here it can be refused, because knowing the target's revision is
    /// not the same as the damage matching it.
    pub fn load_preserving(&mut self, damage: &Damage) -> Result<Painted, Coverage> {
        if !damage.region.fits_within(self.width, self.height) {
            return Err(Coverage::OutOfBounds {
                region: damage.region,
                width: self.width,
                height: self.height,
            });
        }
        if damage.base != self.revision {
            return Err(Coverage::StaleBaseline {
                damage_base: damage.base.get(),
                target: self.revision.get(),
            });
        }
        self.revision = self.revision.next();
        Ok(Painted {
            revision: self.revision,
            region: damage.region,
        })
    }
}

/// Evidence that a target was painted this frame.
///
/// Produced only by [`Target::clear_and_paint_all`] and
/// [`Target::load_preserving`], so it cannot be manufactured — there is no
/// public constructor. A present path that demands one therefore cannot
/// present a target nothing painted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "a Painted that is dropped means a frame was painted and never presented"]
pub struct Painted {
    revision: Revision,
    region: Region,
}

impl Painted {
    /// The revision now on the target.
    #[must_use]
    pub const fn revision(&self) -> Revision {
        self.revision
    }

    /// What was painted.
    #[must_use]
    pub const fn region(&self) -> Region {
        self.region
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_target_becomes_known_only_by_being_cleared() {
        let surface = Target::<Unknown>::surface(800, 600);
        let known = surface.adopt_by_clearing(Revision::ORIGIN.next());
        assert_eq!(known.revision(), Revision::ORIGIN.next());
        assert_eq!(known.extent(), (800, 600));
    }

    #[test]
    fn a_clear_is_available_on_a_target_of_unknown_contents() {
        // The always-sound operation has no preconditions, which is the point:
        // the safe path must never be the harder one to reach.
        let mut surface = Target::<Unknown>::surface(64, 32);
        let painted = surface.clear_and_paint_all(Revision::ORIGIN.next());
        assert!(painted.region().covers_all(64, 32));
    }

    #[test]
    fn preserving_contents_requires_the_damage_to_match_the_target() {
        let r0 = Revision::ORIGIN;
        let mut t = Target::<Known>::owned(100, 100, r0.next().next());
        // Damage computed two revisions ago: the frames between are unknown.
        let stale = Damage::everything(r0, 100, 100);
        match t.load_preserving(&stale) {
            Err(Coverage::StaleBaseline { damage_base, target }) => {
                assert_eq!(damage_base, 0);
                assert_eq!(target, 2);
            }
            other => panic!("a stale baseline must be refused, got {other:?}"),
        }
    }

    #[test]
    fn preserving_contents_succeeds_when_the_baseline_matches() {
        let r = Revision::ORIGIN.next();
        let mut t = Target::<Known>::owned(100, 100, r);
        let d = Damage::everything(r, 100, 100);
        let painted = t.load_preserving(&d).expect("matching baseline");
        assert_eq!(painted.revision(), r.next());
        assert_eq!(t.revision(), r.next());
    }

    #[test]
    fn damage_outside_the_target_is_refused() {
        let r = Revision::ORIGIN;
        let mut t = Target::<Known>::owned(50, 50, r);
        let d = Damage::since(
            r,
            Region {
                x: 40,
                y: 0,
                width: 20,
                height: 10,
            },
        );
        assert!(matches!(
            t.load_preserving(&d),
            Err(Coverage::OutOfBounds { .. })
        ));
    }

    #[test]
    fn a_region_that_overflows_reads_as_out_of_bounds_not_as_wrapping() {
        let r = Revision::ORIGIN;
        let mut t = Target::<Known>::owned(50, 50, r);
        let d = Damage::since(
            r,
            Region {
                x: u32::MAX,
                y: 0,
                width: 8,
                height: 8,
            },
        );
        assert!(matches!(
            t.load_preserving(&d),
            Err(Coverage::OutOfBounds { .. })
        ));
    }

    #[test]
    fn a_revision_never_moves_backwards() {
        // Consumers compare for inequality, so a rewind would make two
        // different states compare equal and the second would never draw.
        let mut r = Revision::ORIGIN;
        let mut seen = Vec::new();
        for _ in 0..1000 {
            seen.push(r);
            r = r.next();
        }
        assert!(seen.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn a_union_of_damages_with_different_baselines_is_refused() {
        let a = Damage::everything(Revision::ORIGIN, 10, 10);
        let b = Damage::everything(Revision::ORIGIN.next(), 10, 10);
        assert!(
            a.union(b).is_none(),
            "merging across baselines silently loses the frames between"
        );
    }

    #[test]
    fn a_union_of_damages_sharing_a_baseline_covers_both() {
        let r = Revision::ORIGIN;
        let a = Damage::since(
            r,
            Region {
                x: 0,
                y: 0,
                width: 10,
                height: 10,
            },
        );
        let b = Damage::since(
            r,
            Region {
                x: 20,
                y: 5,
                width: 10,
                height: 10,
            },
        );
        let u = a.union(b).expect("same baseline");
        assert_eq!(u.region().x, 0);
        assert_eq!(u.region().width, 30);
        assert_eq!(u.region().height, 15);
    }
}
