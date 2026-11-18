//! The squared quantisation errors of the encoder's rate search: for a
//! magnitude `a` and a step `d`, `q = trunc(a / d)` and the mid-point
//! reconstruction `rec = (q + 0.5) d` (0 for `q = 0`) give `(a - rec)^2`.
//!
//! The SIMD versions compute the same IEEE operations per lane (division,
//! truncation, products; no FMA), so every term is identical to the scalar
//! one; the caller adds them up in order.

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

/// `trunc(x)` as `x as i64 as f64` computes it: values from 2^63 up
/// saturate to 2^63.
const SATURATE: f64 = 9_223_372_036_854_775_808.0;

/// `out[i]` = the squared error of `a[i]` (a non-negative magnitude) with
/// step `d > 0`.
pub(crate) fn error_terms(level: Level, a: &[f64], d: f64, out: &mut [f64]) {
    assert_eq!(a.len(), out.len());
    dispatch!(level => error_terms(a, d, out));
}

/// The sum of the squared errors of `a` with step `d`, added in order.
pub(crate) fn error_sum(level: Level, a: &[f64], d: f64) -> f64 {
    const CHUNK: usize = 512;
    let mut terms = [0f64; CHUNK];
    let mut err = 0.0;
    for a in a.chunks(CHUNK) {
        let t = &mut terms[..a.len()];
        error_terms(level, a, d, t);
        for &t in t.iter() {
            err += t;
        }
    }
    err
}

/// The kernel, for the benchmarks.
#[cfg(feature = "kernels")]
pub(crate) mod kernels {
    use super::*;

    /// The squared quantisation errors of the magnitudes `a` with step `d`.
    pub fn rate_error_terms(level: Level, a: &[f64], d: f64, out: &mut [f64]) {
        error_terms(level, a, d, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 41u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        // Magnitudes as from f32 coefficients: small, large, exact
        // multiples of the step, and extremes.
        let mut a: Vec<f64> = (0..2000)
            .map(|i| match i % 4 {
                0 => (rnd() % 100_000) as f32 as f64 / 7.0,
                1 => (rnd() % 64) as f64 * 1.5,
                2 => f32::from_bits(rnd() & 0x7FFF_FFFF).min(f32::MAX) as f64,
                _ => (rnd() % 1000) as f32 as f64,
            })
            .collect();
        a.extend([0.0, 1e-30, 3.0e38, 0.75, 0.7499999]);
        for d in [1.5, 0.001, 1e-30, 37.25, 1e6] {
            let mut want = vec![0f64; a.len()];
            scalar::error_terms(&a, d, &mut want);
            for level in Level::supported() {
                for len in [a.len(), 3] {
                    let mut got = vec![0f64; len];
                    error_terms(level, &a[..len], d, &mut got);
                    let same = got
                        .iter()
                        .zip(&want)
                        .all(|(g, w)| g.to_bits() == w.to_bits());
                    assert!(same, "{level:?} d {d}");
                }
            }
        }
    }
}
