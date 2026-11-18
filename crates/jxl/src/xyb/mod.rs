//! XYB to linear sRGB (ISO/IEC 18181-1 Annex L.2), in place. The SIMD
//! versions do the scalar operations in the same order (no FMA), so the
//! output is bit-identical.

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

/// The constants of the conversion.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Xyb {
    /// Cube roots of the opsin biases.
    pub(crate) cbrt_bias: [f32; 3],
    /// The opsin biases.
    pub(crate) bias: [f32; 3],
    /// 255 / intensity target.
    pub(crate) itscale: f32,
    /// The opsin inverse matrix, row-major.
    pub(crate) matrix: [f32; 9],
}

/// X, Y, B -> linear R, G, B of the samples of `planes` (equal lengths).
pub(crate) fn to_linear(level: Level, planes: [&mut [f32]; 3], k: &Xyb) {
    dispatch!(level => to_linear(planes, k));
}

/// The conversion, for the benchmarks.
#[cfg(feature = "kernels")]
pub(crate) mod kernels {
    use super::*;

    /// XYB to linear RGB with the default opsin parameters.
    pub fn xyb_to_linear(level: Level, planes: [&mut [f32]; 3]) {
        let opsin = crate::headers::OpsinInverse::default();
        let k = Xyb {
            cbrt_bias: opsin.bias.map(f32::cbrt),
            bias: opsin.bias,
            itscale: 1.0,
            matrix: opsin.matrix,
        };
        to_linear(level, planes, &k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 9u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32
        };
        let opsin = crate::headers::OpsinInverse::default();
        let k = Xyb {
            cbrt_bias: opsin.bias.map(f32::cbrt),
            bias: opsin.bias,
            itscale: 1.0,
            matrix: opsin.matrix,
        };
        for n in [1, 5, 16, 33, 100] {
            // X is small and signed, Y and B in 0..1.
            let planes: [Vec<f32>; 3] = std::array::from_fn(|c| {
                (0..n)
                    .map(|_| if c == 0 { rnd() * 0.06 - 0.03 } else { rnd() })
                    .collect()
            });
            let mut want = planes.clone();
            let [a, b, c] = &mut want;
            to_linear(Level::Scalar, [a, b, c], &k);
            for level in Level::supported() {
                let mut got = planes.clone();
                let [a, b, c] = &mut got;
                to_linear(level, [a, b, c], &k);
                for (w, g) in want.iter().zip(&got) {
                    let same = w.iter().zip(g).all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(same, "{level:?} n {n}");
                }
            }
        }
    }
}
