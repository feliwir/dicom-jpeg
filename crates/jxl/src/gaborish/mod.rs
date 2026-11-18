//! The Gaborish 3 x 3 convolution of one row (ISO/IEC 18181-1 Annex J.2).
//! The SIMD versions do the scalar operations in the same order (no FMA),
//! so the output is bit-identical.

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

/// `dst[x]` from columns `x ..= x + 2` of the padded rows `up`, `mid` and
/// `down` (at least `dst.len() + 2` long), with the weights
/// `k = [side, diagonal, normalisation]`.
pub(crate) fn row(
    level: Level,
    up: &[f32],
    mid: &[f32],
    down: &[f32],
    dst: &mut [f32],
    k: [f32; 3],
) {
    dispatch!(level => row(up, mid, down, dst, k));
}

/// The row kernel, for the benchmarks.
#[cfg(feature = "kernels")]
pub(crate) mod kernels {
    use super::*;

    /// One Gaborish row (see [`row`](super::row)).
    pub fn gaborish_row(
        level: Level,
        up: &[f32],
        mid: &[f32],
        down: &[f32],
        dst: &mut [f32],
        k: [f32; 3],
    ) {
        row(level, up, mid, down, dst, k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 5u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32 - 0.5
        };
        let k = [0.115, 0.061, 1.0 / (1.0 + 4.0 * 0.115 + 4.0 * 0.061)];
        for w in [1, 3, 4, 15, 16, 17, 64, 77] {
            let rows: [Vec<f32>; 3] = std::array::from_fn(|_| (0..w + 2).map(|_| rnd()).collect());
            let mut want = vec![0f32; w];
            row(Level::Scalar, &rows[0], &rows[1], &rows[2], &mut want, k);
            for level in Level::supported() {
                let mut got = vec![0f32; w];
                row(level, &rows[0], &rows[1], &rows[2], &mut got, k);
                let same = want
                    .iter()
                    .zip(&got)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "{level:?} width {w}");
            }
        }
    }
}
