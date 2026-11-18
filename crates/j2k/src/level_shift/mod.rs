//! The DC level shift and clamp of decoded samples (T.800 G.1): integer
//! samples `clamp(v + shift, lo, hi)`, and float (9/7) samples
//! `clamp(round(v) + shift, lo, hi)` with rounding half away from zero.
//!
//! The SIMD versions clamp before shifting (`clamp(v, lo - shift, hi -
//! shift) + shift`), and for floats before rounding: rounding is monotonic
//! and keeps the (integer, f32-exact) bounds, so the results are identical
//! to the scalar ones, saturation and NaN (which rounds to 0) included.

use dicom_jpeg_common::dispatch;
use dicom_jpeg_common::simd::Level;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse41;

/// The bounds of a component: samples in `lo..=hi`, stored as `v + shift`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Range {
    pub(crate) lo: i32,
    pub(crate) hi: i32,
    pub(crate) shift: i32,
}

impl Range {
    /// The clamp bounds before the shift: `lo - shift` and `hi - shift`.
    fn unshifted(self) -> Option<(i32, i32)> {
        Some((
            self.lo.checked_sub(self.shift)?,
            self.hi.checked_sub(self.shift)?,
        ))
    }
}

/// `v = clamp(v + shift, lo, hi)` (saturating add).
pub(crate) fn shift_int(level: Level, data: &mut [i32], r: Range) {
    match r.unshifted() {
        Some((a, b)) => dispatch!(level => shift_int(data, a, b, r.shift)),
        None => scalar::shift_int_exact(data, r),
    }
}

/// `dst = clamp(round(src) + shift, lo, hi)` (saturating add).
pub(crate) fn shift_float(level: Level, src: &[f32], dst: &mut [i32], r: Range) {
    assert_eq!(src.len(), dst.len());
    // f32 holds every integer up to 2^24: the bounds must be exact.
    const EXACT: i32 = 1 << 24;
    match r.unshifted() {
        Some((a, b)) if (-EXACT..=EXACT).contains(&a) && (-EXACT..=EXACT).contains(&b) => {
            dispatch!(level => shift_float(src, dst, a, b, r.shift))
        }
        _ => scalar::shift_float_exact(src, dst, r),
    }
}

/// The kernels, for the benchmarks.
#[cfg(feature = "kernels")]
pub(crate) mod kernels {
    use super::*;

    /// Level shift and clamp of integer samples of `bits` bits (unsigned).
    pub fn level_shift_int(level: Level, data: &mut [i32], bits: u32) {
        shift_int(level, data, unsigned(bits));
    }

    /// Rounding, level shift and clamp of float samples of `bits` bits
    /// (unsigned).
    pub fn level_shift_float(level: Level, src: &[f32], dst: &mut [i32], bits: u32) {
        shift_float(level, src, dst, unsigned(bits));
    }

    fn unsigned(bits: u32) -> Range {
        Range {
            lo: 0,
            hi: (1 << bits) - 1,
            shift: 1 << (bits - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges() -> Vec<Range> {
        let mut v = Vec::new();
        for prec in [1u32, 8, 12, 16, 24, 25, 26, 31] {
            let half = 1i64 << (prec - 1);
            let unsigned = Range {
                lo: 0,
                hi: ((1i64 << prec) - 1).min(i32::MAX as i64) as i32,
                shift: half as i32,
            };
            let signed = Range {
                lo: -half as i32,
                hi: (half - 1) as i32,
                shift: 0,
            };
            v.extend([unsigned, signed]);
        }
        v
    }

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 11u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        // Halves, exact integers, the bounds, huge values and non-finite.
        let mut floats: Vec<f32> = (0..2000)
            .map(|i| match i % 5 {
                0 => (rnd() % 70000) as f32 / 2.0 - 17500.0,
                1 => (rnd() as i32) as f32,
                2 => f32::from_bits(rnd()),
                3 => (rnd() % 1000) as f32 * 0.37 - 185.0,
                _ => (rnd() % 600) as f32 - 300.0 + 0.5,
            })
            .collect();
        floats.extend([
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.5,
            0.5,
            -0.0,
            2.5e9,
            -2.5e9,
            16_777_216.0,
            -16_777_217.0,
        ]);
        let ints: Vec<i32> = (0..2000)
            .map(|i| {
                if i % 3 == 0 {
                    rnd() as i32
                } else {
                    (rnd() % 70000) as i32 - 35000
                }
            })
            .collect();
        for r in ranges() {
            let mut want_f = vec![0; floats.len()];
            scalar::shift_float_exact(&floats, &mut want_f, r);
            let mut want_i = ints.clone();
            scalar::shift_int_exact(&mut want_i, r);
            for level in Level::supported() {
                for len in [floats.len(), 13, 1] {
                    let mut got = vec![0; len];
                    shift_float(level, &floats[..len], &mut got, r);
                    assert_eq!(got, want_f[..len], "{level:?} float {r:?}");
                }
                let mut got = ints.clone();
                shift_int(level, &mut got, r);
                assert_eq!(got, want_i, "{level:?} int {r:?}");
            }
        }
    }
}
