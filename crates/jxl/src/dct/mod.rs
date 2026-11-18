//! DCT-II / DCT-III of power-of-two sizes (up to 256) with the JPEG XL
//! normalisation: the forward transform averages (DC = mean), the inverse
//! does not scale. Both use Lee's O(n log n) recursion, as libjxl does
//! (`scalar.rs`).
//!
//! The inverse 2-D transform of blocks whose sides are multiples of 8 (all
//! VarDCT blocks of 8 x 8 and up) has SIMD versions: they run the same
//! recursion on one vector per row of a group of columns, so every lane
//! computes exactly the scalar steps, in the same order (no FMA), and the
//! result is bit-identical.

use dicom_jpeg_common::dispatch;
use dicom_jpeg_common::simd::{self, Level};

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse41;

/// The inverse 2-D DCT, for the benchmarks.
#[cfg(feature = "kernels")]
pub(crate) mod kernels {
    use super::*;

    /// The inverse 2-D DCT of VarDCT blocks at one SIMD level.
    pub struct Idct(Dct);

    impl std::fmt::Debug for Idct {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Idct")
                .field("level", &self.0.level)
                .finish_non_exhaustive()
        }
    }

    impl Idct {
        pub fn new(level: Level) -> Self {
            Idct(Dct::with_level(level))
        }

        /// The inverse transform of the contiguous h x w block `block`.
        pub fn inverse(&mut self, block: &mut [f32], w: usize, h: usize) {
            self.0.transform_2d(block, w, w, h, true);
        }
    }
}

pub(crate) const MAX_DCT: usize = 256;

/// The transform state: constants, scratch blocks and the SIMD level.
pub(crate) struct Dct {
    level: Level,
    /// Per log2 size n: 1 / (2 cos((2i + 1) pi / 2n)) for i < n / 2.
    mul: Vec<Vec<f32>>,
    /// Scratch blocks.
    a: Vec<f32>,
    b: Vec<f32>,
    tmp: Vec<f32>,
}

impl Dct {
    pub(crate) fn new() -> Self {
        Self::with_level(simd::level())
    }

    fn with_level(level: Level) -> Self {
        let mul = (0..=8)
            .map(|l| {
                let n = 1usize << l;
                (0..n / 2)
                    .map(|i| {
                        let a = (2 * i + 1) as f64 * std::f64::consts::PI / (2 * n) as f64;
                        (0.5 / a.cos()) as f32
                    })
                    .collect()
            })
            .collect();
        Dct {
            level,
            mul,
            a: vec![0.0; MAX_DCT * MAX_DCT],
            b: vec![0.0; MAX_DCT * MAX_DCT],
            tmp: vec![0.0; MAX_DCT * MAX_DCT],
        }
    }

    /// The JPEG XL 1-D transform of `v` in place.
    #[cfg(test)]
    fn transform_1d(&mut self, v: &mut [f32], inverse: bool) {
        scalar::transform_cols(&self.mul, v, &mut self.tmp, v.len(), 1, inverse);
    }

    /// 2D transform of a w x h region at `data[..]` with row stride `stride`.
    pub(crate) fn transform_2d(
        &mut self,
        data: &mut [f32],
        stride: usize,
        w: usize,
        h: usize,
        inverse: bool,
    ) {
        let (b, tmp) = (&mut self.b[..w * h], &mut self.tmp[..w * h]);
        if stride == w {
            // Contiguous: transform in place.
            transform(
                self.level,
                &self.mul,
                &mut data[..w * h],
                b,
                tmp,
                w,
                h,
                inverse,
            );
            return;
        }
        let a = &mut self.a[..w * h];
        for (y, row) in a.chunks_exact_mut(w).enumerate() {
            row.copy_from_slice(&data[y * stride..][..w]);
        }
        transform(self.level, &self.mul, a, b, tmp, w, h, inverse);
        for (y, row) in a.chunks_exact(w).enumerate() {
            data[y * stride..][..w].copy_from_slice(row);
        }
    }
}

/// 2D transform of the h x w block `a`; `b` and `tmp` are scratch.
#[allow(clippy::too_many_arguments)]
fn transform(
    level: Level,
    mul: &[Vec<f32>],
    a: &mut [f32],
    b: &mut [f32],
    tmp: &mut [f32],
    w: usize,
    h: usize,
    inverse: bool,
) {
    if inverse && w.is_multiple_of(8) && h.is_multiple_of(8) {
        dispatch!(level => idct_2d(mul, a, b, tmp, w, h));
    } else {
        scalar::transform_2d(mul, a, b, tmp, w, h, inverse);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The JPEG XL DCT pair evaluated directly in f64.
    fn reference(v: &[f32], inverse: bool) -> Vec<f64> {
        let n = v.len();
        let t = |k: usize, i: usize| {
            if k == 0 {
                1.0
            } else {
                std::f64::consts::SQRT_2
                    * ((2 * i + 1) as f64 * k as f64 * std::f64::consts::PI / (2 * n) as f64).cos()
            }
        };
        (0..n)
            .map(|o| {
                if inverse {
                    (0..n).map(|k| v[k] as f64 * t(k, o)).sum()
                } else {
                    (0..n).map(|i| v[i] as f64 * t(o, i)).sum::<f64>() / n as f64
                }
            })
            .collect()
    }

    #[test]
    fn fast_dct_matches_definition() {
        let mut dct = Dct::new();
        let mut seed = 1u32;
        for l in 0..=8 {
            let n = 1 << l;
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    (seed >> 8) as f32 / (1 << 23) as f32 - 1.0
                })
                .collect();
            for inverse in [false, true] {
                let mut got = v.clone();
                dct.transform_1d(&mut got, inverse);
                let want = reference(&v, inverse);
                let scale = if inverse { n as f64 } else { 1.0 };
                for (g, w) in got.iter().zip(&want) {
                    assert!(
                        (*g as f64 - w).abs() < 2e-5 * scale,
                        "n {n} inverse {inverse}"
                    );
                }
            }
            let mut rt = v.clone();
            dct.transform_1d(&mut rt, false);
            dct.transform_1d(&mut rt, true);
            for (a, b) in rt.iter().zip(&v) {
                assert!((a - b).abs() < 1e-4, "round trip n {n}");
            }
        }
    }

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 8) as f32 / (1 << 23) as f32 - 1.0
        };
        let sizes = [8, 16, 32, 64, 128, 256];
        let mut scalar = Dct::with_level(Level::Scalar);
        for level in Level::supported() {
            let mut dct = Dct::with_level(level);
            for &w in &sizes {
                for &h in &sizes {
                    // Coefficients of decreasing magnitude, as after
                    // dequantisation; contiguous (in place) and with a
                    // stride wider than the block (copied).
                    for stride in [w, w + 8] {
                        let data: Vec<f32> = (0..stride * h)
                            .map(|i| rnd() * 1000.0 / (1 + i % stride + i / stride) as f32)
                            .collect();
                        let (mut want, mut got) = (data.clone(), data);
                        scalar.transform_2d(&mut want, stride, w, h, true);
                        dct.transform_2d(&mut got, stride, w, h, true);
                        let same = want
                            .iter()
                            .zip(&got)
                            .all(|(a, b)| a.to_bits() == b.to_bits());
                        assert!(same, "{level:?} {w}x{h} stride {stride}");
                    }
                }
            }
        }
    }
}
