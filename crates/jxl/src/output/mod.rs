//! Conversion of float samples to stored integer samples:
//! `clamp(round(v * scale), 0, top)`, rounding half away from zero.
//!
//! The SIMD versions clamp in float before rounding (rounding is monotonic
//! and keeps the integer bounds, which are exact in f32), so they give the
//! scalar results exactly, NaN (which rounds to 0) and infinities included.

use dicom_jpeg_common::dispatch;
use dicom_jpeg_common::float::round_i32;
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

/// `dst[i] = clamp(round(src[i] * scale), 0, top)`.
pub(crate) fn to_u16(level: Level, src: &[f32], scale: f32, top: u16, dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    dispatch!(level => to_u16(src, scale, top, dst));
}

/// The scalar conversion of one sample.
#[inline(always)]
fn convert(v: f32, scale: f32, top: u16) -> u16 {
    round_i32(v * scale).clamp(0, top as i32) as u16
}

/// The conversions, for the benchmarks.
#[cfg(feature = "kernels")]
pub(crate) mod kernels {
    use super::*;

    /// `dst[i] = clamp(round(src[i] * scale), 0, top)`.
    pub fn float_to_u16(level: Level, src: &[f32], scale: f32, top: u16, dst: &mut [u16]) {
        to_u16(level, src, scale, top, dst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 31u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        // In range (halves included), out of range, and non-finite.
        let mut src: Vec<f32> = (0..3000)
            .map(|i| match i % 4 {
                0 => (rnd() % 1_000_000) as f32 / 1_000_000.0,
                1 => (rnd() % 512) as f32 / 510.0,
                2 => f32::from_bits(rnd()),
                _ => (rnd() % 600) as f32 / 300.0 - 0.5,
            })
            .collect();
        src.extend([
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.5 / 255.0,
            1.0,
        ]);
        for (scale, top) in [
            (255.0, 255u16),
            (4095.0, 4095),
            (65535.0, 65535),
            (255.0, 65535),
        ] {
            let want: Vec<u16> = src.iter().map(|&v| convert(v, scale, top)).collect();
            for level in Level::supported() {
                for len in [src.len(), 7] {
                    let mut got = vec![0u16; len];
                    to_u16(level, &src[..len], scale, top, &mut got);
                    assert_eq!(got, want[..len], "{level:?} scale {scale} top {top}");
                }
            }
        }
    }
}
