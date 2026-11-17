//! Runtime selection of SIMD implementations.
//!
//! Every vectorised kernel lives in its own module with one file per
//! instruction set (`scalar.rs`, `sse41.rs`, `avx2.rs`, `avx512.rs`,
//! `neon.rs`); its `mod.rs` picks one with [`dispatch!`](crate::dispatch).
//! The scalar version is always kept, and every SIMD version gives exactly
//! the same results as it. The SIMD files use `std::arch` intrinsics
//! directly (`#![allow(unsafe_code)]`, a `// SAFETY:` comment per block).
//!
//! [`level`] is the instruction set used: the best one the CPU supports,
//! detected once. It can be lowered (to compare levels, or to work around
//! a problem):
//! - for the process, with [`set_level`] or the `DICOM_JPEG_SIMD`
//!   environment variable (`scalar`, `sse41`, `avx2`, `avx512`, `neon`);
//! - for the current thread, with [`with_level`], which takes precedence.
//!
//! A level the CPU does not support is never used: the best supported
//! level below it is taken instead.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

/// An instruction set level, in increasing order of capability on x86.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    /// Portable code (still auto-vectorised for the compilation target,
    /// e.g. SSE2 on x86-64).
    Scalar,
    /// x86 SSE4.1.
    Sse41,
    /// x86 AVX2 with FMA (FMA is not used, to stay bit-exact).
    Avx2,
    /// x86 AVX-512 F, BW, DQ and VL.
    Avx512,
    /// AArch64 Advanced SIMD.
    Neon,
}

impl Level {
    /// All levels.
    pub const ALL: [Level; 5] = [
        Level::Scalar,
        Level::Sse41,
        Level::Avx2,
        Level::Avx512,
        Level::Neon,
    ];

    /// Lower-case name, as accepted by `DICOM_JPEG_SIMD`.
    pub const fn name(self) -> &'static str {
        match self {
            Level::Scalar => "scalar",
            Level::Sse41 => "sse41",
            Level::Avx2 => "avx2",
            Level::Avx512 => "avx512",
            Level::Neon => "neon",
        }
    }

    /// Parse a [`Level::name`].
    pub fn from_name(name: &str) -> Option<Level> {
        Level::ALL.into_iter().find(|l| l.name() == name)
    }

    /// Whether the CPU running this code supports the level.
    pub fn is_supported(self) -> bool {
        match self {
            Level::Scalar => true,
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Sse41 => std::arch::is_x86_feature_detected!("sse4.1"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx2 => {
                Level::Sse41.is_supported()
                    && std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Level::Avx512 => {
                Level::Avx2.is_supported()
                    && std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw")
                    && std::arch::is_x86_feature_detected!("avx512dq")
                    && std::arch::is_x86_feature_detected!("avx512vl")
            }
            #[cfg(target_arch = "aarch64")]
            Level::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// The levels the CPU supports, lowest first.
    pub fn supported() -> impl Iterator<Item = Level> {
        Level::ALL.into_iter().filter(|l| l.is_supported())
    }

    /// The best supported level.
    pub fn best() -> Level {
        Level::supported().last().unwrap_or(Level::Scalar)
    }
}

impl Level {
    /// `self` if supported, else the best supported level below it.
    pub fn clamp_to_supported(self) -> Level {
        Level::supported()
            .filter(|&l| l <= self)
            .last()
            .unwrap_or(Level::Scalar)
    }
}

/// [`set_level`]: 0 for none, else 1 + the index in [`Level::ALL`].
static PROCESS_OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// The level selected for the process: [`set_level`], else the level
/// named by `DICOM_JPEG_SIMD`, else the best supported one.
fn process_level() -> Level {
    match PROCESS_OVERRIDE.load(Ordering::Relaxed) {
        0 => {}
        i => return Level::ALL[i as usize - 1],
    }
    static DEFAULT: OnceLock<Level> = OnceLock::new();
    *DEFAULT.get_or_init(|| {
        std::env::var("DICOM_JPEG_SIMD")
            .ok()
            .and_then(|v| Level::from_name(v.trim()))
            .map(Level::clamp_to_supported)
            .unwrap_or_else(Level::best)
    })
}

/// Use `level` (or the best supported level below it) for the whole
/// process from now on; `None` restores the default. Returns the level
/// that is used. [`with_level`] still takes precedence on its thread.
pub fn set_level(level: Option<Level>) -> Level {
    let v = level.map_or(0, |l| {
        let l = l.clamp_to_supported();
        1 + Level::ALL.iter().position(|&x| x == l).expect("listed") as u8
    });
    PROCESS_OVERRIDE.store(v, Ordering::Relaxed);
    process_level()
}

thread_local! {
    static OVERRIDE: Cell<Option<Level>> = const { Cell::new(None) };
}

/// The level kernels should use on this thread.
///
/// Read it once per image or row and pass it to [`dispatch!`](crate::dispatch),
/// rather than per sample.
#[inline]
pub fn level() -> Level {
    OVERRIDE.with(Cell::get).unwrap_or_else(process_level)
}

/// Run `f` with kernels on this thread using `level` (or the best
/// supported level below it, if the CPU lacks it).
pub fn with_level<R>(level: Level, f: impl FnOnce() -> R) -> R {
    let level = level.clamp_to_supported();
    let previous = OVERRIDE.with(|o| o.replace(Some(level)));
    struct Restore(Option<Level>);
    impl Drop for Restore {
        fn drop(&mut self) {
            OVERRIDE.with(|o| o.set(self.0));
        }
    }
    let _restore = Restore(previous);
    f()
}

/// Call `$f` from the per-level module that matches `$level`
/// (`avx512::$f`, `avx2::$f`, `sse41::$f`, `neon::$f`, or `scalar::$f`),
/// relative to the calling module.
///
/// The SIMD versions are `#[target_feature]` functions; this macro holds
/// the only `unsafe` needed to call them, justified by [`Level`] values
/// coming from runtime detection ([`level`], [`with_level`]).
#[macro_export]
macro_rules! dispatch {
    ($level:expr => $f:ident($($arg:expr),* $(,)?)) => {{
        match $level {
            #[cfg(target_arch = "x86_64")]
            $crate::simd::Level::Avx512 => {
                // SAFETY: `Level::Avx512` is only selected when the CPU
                // supports AVX-512 F/BW/DQ/VL (and AVX2, FMA, SSE4.1).
                #[allow(unsafe_code)]
                let r = unsafe { avx512::$f($($arg),*) };
                r
            }
            #[cfg(target_arch = "x86_64")]
            $crate::simd::Level::Avx2 => {
                // SAFETY: `Level::Avx2` is only selected when the CPU
                // supports AVX2, FMA and SSE4.1.
                #[allow(unsafe_code)]
                let r = unsafe { avx2::$f($($arg),*) };
                r
            }
            #[cfg(target_arch = "x86_64")]
            $crate::simd::Level::Sse41 => {
                // SAFETY: `Level::Sse41` is only selected when the CPU
                // supports SSE4.1.
                #[allow(unsafe_code)]
                let r = unsafe { sse41::$f($($arg),*) };
                r
            }
            #[cfg(target_arch = "aarch64")]
            $crate::simd::Level::Neon => neon::$f($($arg),*),
            _ => scalar::$f($($arg),*),
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_are_ordered_and_named() {
        for l in Level::ALL {
            assert_eq!(Level::from_name(l.name()), Some(l));
        }
        assert!(Level::Scalar.is_supported());
        assert!(Level::supported().all(|l| l.is_supported()));
    }

    #[test]
    fn with_level_overrides_and_restores() {
        let outer = level();
        with_level(Level::Scalar, || {
            assert_eq!(level(), Level::Scalar);
            with_level(Level::Avx512, || assert!(level().is_supported()));
            assert_eq!(level(), Level::Scalar);
        });
        assert_eq!(level(), outer);
    }

    #[test]
    fn set_level_clamps_and_resets() {
        let best = Level::best();
        assert_eq!(set_level(Some(Level::Scalar)), Level::Scalar);
        // The thread override still wins.
        with_level(best, || assert_eq!(level(), best));
        assert_eq!(set_level(None), process_level());
    }
}
