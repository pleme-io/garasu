//! Persistent fontdb cache.
//!
//! cosmic-text's `FontSystem::new()` calls `fontdb::Database::
//! load_system_fonts()` which walks every system font directory
//! and parses each TTF/OTF file's header. On macOS that's ~200
//! files and ~150-200 ms on every cold start.
//!
//! This module persists the parsed face metadata to disk
//! ($XDG_CACHE_HOME/garasu/fontdb-vN.bin). On subsequent runs
//! we deserialize the cache (~5 ms) and reconstruct the
//! Database via `push_face_info` (~5-10 ms for ~200 entries),
//! skipping the per-file TTF parse entirely.
//!
//! Cache invalidation: a fingerprint is computed from the
//! mtimes of the standard font directories. If any dir has
//! changed (font installed / removed), the fingerprint mismatch
//! triggers a fresh scan.
//!
//! Schema version is bumped in `CACHE_SCHEMA_VERSION` whenever
//! the on-disk layout changes incompatibly — mismatched schema
//! also triggers a fresh scan.

use std::path::{Path, PathBuf};

use fontdb::{Database, FaceInfo, Language, Source, Stretch, Style, Weight, ID};
use serde::{Deserialize, Serialize};

/// Bump on any change to `CachedFace`'s layout.
const CACHE_SCHEMA_VERSION: u32 = 1;

/// Default cache file name; lives under
/// `$XDG_CACHE_HOME/garasu/` (or `~/Library/Caches/garasu/` on macOS).
const CACHE_FILENAME: &str = "fontdb-v1.bin";

#[derive(Serialize, Deserialize)]
struct CachedDb {
    schema_version: u32,
    fingerprint: u64,
    faces: Vec<CachedFace>,
}

/// On-disk projection of `fontdb::FaceInfo`. We drop the
/// per-family `Language` (default to English_UnitedStates on
/// reload) — terminal rendering doesn't rely on per-language
/// fallback, and serializing ttf_parser's giant Language enum
/// would couple us to its discriminants.
#[derive(Serialize, Deserialize)]
struct CachedFace {
    path: PathBuf,
    index: u32,
    families: Vec<String>,
    post_script_name: String,
    /// 0 = Normal, 1 = Italic, 2 = Oblique.
    style: u8,
    /// Weight as raw u16 (e.g., 400 = Regular, 700 = Bold).
    weight: u16,
    /// Stretch as the standard u8 (1..=9, see fontdb docs).
    stretch: u8,
    monospaced: bool,
}

impl CachedFace {
    fn from_face(info: &FaceInfo) -> Option<Self> {
        // We only cache File-backed faces. Binary/SharedFile
        // sources are runtime-only (consumer-injected fonts)
        // and don't survive a process restart anyway.
        let path = match &info.source {
            Source::File(p) => p.clone(),
            Source::SharedFile(p, _) => p.clone(),
            Source::Binary(_) => return None,
        };
        Some(Self {
            path,
            index: info.index,
            families: info.families.iter().map(|(n, _)| n.clone()).collect(),
            post_script_name: info.post_script_name.clone(),
            style: match info.style {
                Style::Normal => 0,
                Style::Italic => 1,
                Style::Oblique => 2,
            },
            weight: info.weight.0,
            stretch: match info.stretch {
                Stretch::UltraCondensed => 1,
                Stretch::ExtraCondensed => 2,
                Stretch::Condensed => 3,
                Stretch::SemiCondensed => 4,
                Stretch::Normal => 5,
                Stretch::SemiExpanded => 6,
                Stretch::Expanded => 7,
                Stretch::ExtraExpanded => 8,
                Stretch::UltraExpanded => 9,
            },
            monospaced: info.monospaced,
        })
    }

    fn into_face_info(self) -> FaceInfo {
        FaceInfo {
            id: ID::dummy(),  // overwritten by push_face_info
            source: Source::File(self.path),
            index: self.index,
            families: self
                .families
                .into_iter()
                .map(|n| (n, Language::English_UnitedStates))
                .collect(),
            post_script_name: self.post_script_name,
            style: match self.style {
                1 => Style::Italic,
                2 => Style::Oblique,
                _ => Style::Normal,
            },
            weight: Weight(self.weight),
            stretch: match self.stretch {
                1 => Stretch::UltraCondensed,
                2 => Stretch::ExtraCondensed,
                3 => Stretch::Condensed,
                4 => Stretch::SemiCondensed,
                6 => Stretch::SemiExpanded,
                7 => Stretch::Expanded,
                8 => Stretch::ExtraExpanded,
                9 => Stretch::UltraExpanded,
                _ => Stretch::Normal,
            },
            monospaced: self.monospaced,
        }
    }
}

/// Resolve the cache path under the user's cache directory.
/// Returns None when neither the XDG nor platform cache dir
/// is available (e.g., on a minimal sandbox).
fn cache_path() -> Option<PathBuf> {
    let base = dirs::cache_dir()?;
    Some(base.join("garasu").join(CACHE_FILENAME))
}

/// Compute a fingerprint over the standard font directories.
/// Changes when any dir's mtime advances (font added / removed
/// / updated). Cheap: stat() per dir, fold mtimes into a u64.
fn compute_fingerprint() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let dirs = font_dirs();
    let mut h = DefaultHasher::new();
    CACHE_SCHEMA_VERSION.hash(&mut h);
    for d in &dirs {
        if let Ok(meta) = std::fs::metadata(d) {
            d.hash(&mut h);
            if let Ok(mtime) = meta.modified() {
                if let Ok(dur) = mtime.duration_since(std::time::UNIX_EPOCH) {
                    dur.as_secs().hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

fn font_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    #[cfg(target_os = "macos")]
    {
        dirs.push(PathBuf::from("/System/Library/Fonts"));
        dirs.push(PathBuf::from("/Library/Fonts"));
        if let Some(home) = dirs::home_dir() {
            dirs.push(home.join("Library/Fonts"));
        }
    }
    #[cfg(target_os = "linux")]
    {
        dirs.push(PathBuf::from("/usr/share/fonts"));
        dirs.push(PathBuf::from("/usr/local/share/fonts"));
        if let Some(home) = dirs::home_dir() {
            dirs.push(home.join(".local/share/fonts"));
        }
    }
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".nix-profile/share/fonts"));
    }
    dirs
}

/// Try to load a Database from the on-disk cache. Returns
/// `Some(db)` only when the cache file exists AND the schema
/// version matches AND the fingerprint matches the current
/// font directories. Any I/O / deserialization / mismatch
/// failure returns None (caller falls back to a full scan).
pub fn try_load_cached_db() -> Option<Database> {
    let path = cache_path()?;
    let t_start = std::time::Instant::now();
    let bytes = std::fs::read(&path).ok()?;
    let cached: CachedDb = bincode::deserialize(&bytes).ok()?;
    if cached.schema_version != CACHE_SCHEMA_VERSION {
        return None;
    }
    if cached.fingerprint != compute_fingerprint() {
        return None;
    }
    let mut db = Database::new();
    for face in cached.faces {
        db.push_face_info(face.into_face_info());
    }
    tracing::info!(
        target: "garasu::font_cache",
        faces = db.len(),
        load_ms = t_start.elapsed().as_millis() as u64,
        path = %path.display(),
        "fontdb cache hit"
    );
    Some(db)
}

/// Persist the current Database to the cache so the next run
/// can skip the full scan. Best-effort — any failure is logged
/// but not propagated.
pub fn save_cache(db: &Database) {
    let Some(path) = cache_path() else { return };
    let t_start = std::time::Instant::now();
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            tracing::warn!(target: "garasu::font_cache", error = %e, "create cache dir failed");
            return;
        }
    }
    let faces: Vec<CachedFace> = db
        .faces()
        .filter_map(CachedFace::from_face)
        .collect();
    let cached = CachedDb {
        schema_version: CACHE_SCHEMA_VERSION,
        fingerprint: compute_fingerprint(),
        faces,
    };
    let bytes = match bincode::serialize(&cached) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(target: "garasu::font_cache", error = %e, "serialize failed");
            return;
        }
    };
    let n_faces = cached.faces.len();
    if let Err(e) = atomic_write(&path, &bytes) {
        tracing::warn!(target: "garasu::font_cache", error = %e, "write failed");
        return;
    }
    tracing::info!(
        target: "garasu::font_cache",
        faces = n_faces,
        bytes = bytes.len(),
        save_ms = t_start.elapsed().as_millis() as u64,
        "fontdb cache saved"
    );
}

/// Write `bytes` to `path` via a temp file + rename so a
/// crashing process never leaves a half-written cache.
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension(format!(
        "tmp.{}",
        std::process::id()
    ));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}
