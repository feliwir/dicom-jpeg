//! IJG libjpeg "fancy" chroma upsampling (jdsample.c): the triangle filter
//! for 2:1 horizontal (h2v1) and 2:1 horizontal and vertical (h2v2) factors.
//!
//! SIMD versions work in 16-bit lanes, which hold every intermediate for
//! samples of up to 12 bits; deeper samples use the scalar code.

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

/// Horizontal triangle filter of `src` (at least 2 samples) into `dst`
/// (`2 * src.len()` samples): output 2i gets 3/4 of input i and 1/4 of i-1
/// (rounding bias 1), output 2i+1 3/4 of i and 1/4 of i+1 (bias 2).
pub(crate) fn h2v1(level: Level, precision: u8, src: &[u16], dst: &mut [u16]) {
    let dst = &mut dst[..2 * src.len()];
    if precision > 12 {
        return scalar::h2v1(src, dst);
    }
    dispatch!(level => h2v1(src, dst));
}

/// h2v2 output row from the `near` and `far` input rows (at least 2
/// samples): column sums 3 * near + far, then the h2v1 weights 3:1 with
/// rounding biases 8 (even outputs) and 7 (odd outputs).
pub(crate) fn h2v2(level: Level, precision: u8, near: &[u16], far: &[u16], dst: &mut [u16]) {
    let far = &far[..near.len()];
    let dst = &mut dst[..2 * near.len()];
    if precision > 12 {
        return scalar::h2v2(near, far, dst);
    }
    dispatch!(level => h2v2(near, far, dst));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h2v1_matches_libjpeg_formula() {
        let mut dst = [0u16; 6];
        h2v1(Level::Scalar, 8, &[10, 20, 40], &mut dst);
        // [10, (30+20+2)>>2=13, (60+10+1)>>2=17, (60+40+2)>>2=25, (120+20+1)>>2=35, 40]
        assert_eq!(dst, [10, 13, 17, 25, 35, 40]);
    }

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 5u32;
        for precision in [8u8, 12] {
            let max = (1u32 << precision) - 1;
            for w in [
                2usize, 3, 9, 16, 17, 18, 33, 34, 35, 64, 65, 66, 67, 200, 513,
            ] {
                let mut rnd = || {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    ((seed >> 8) % (max + 1)) as u16
                };
                let a: Vec<u16> = (0..w).map(|_| rnd()).collect();
                let b: Vec<u16> = (0..w).map(|_| rnd()).collect();
                let (mut w1, mut w2) = (vec![0u16; 2 * w], vec![0u16; 2 * w]);
                scalar::h2v1(&a, &mut w1);
                scalar::h2v2(&a, &b, &mut w2);
                for level in Level::supported() {
                    let (mut g1, mut g2) = (vec![0u16; 2 * w], vec![0u16; 2 * w]);
                    h2v1(level, precision, &a, &mut g1);
                    h2v2(level, precision, &a, &b, &mut g2);
                    assert_eq!(g1, w1, "h2v1 {level:?} w {w}");
                    assert_eq!(g2, w2, "h2v2 {level:?} w {w}");
                }
            }
        }
    }
}
