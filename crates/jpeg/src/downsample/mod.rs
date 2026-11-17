//! IJG libjpeg chroma downsampling (jcsample.c): averages of 2 (h2v1) or 4
//! (h2v2) samples with alternating rounding biases (0, 1 and 1, 2).
//!
//! SIMD versions add in 16-bit lanes, which holds the sums for samples of up
//! to 12 bits; deeper samples use the scalar code.

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

/// `out[j]` = average of `row[2j]` and `row[2j + 1]` (bias `j % 2`).
pub fn h2v1(level: Level, precision: u8, row: &[u16], out: &mut [u16]) {
    let row = &row[..2 * out.len()];
    if precision > 12 {
        return scalar::h2v1(row, out);
    }
    dispatch!(level => h2v1(row, out));
}

/// `out[j]` = average of the 2 x 2 samples at column `2j` of rows `a` and `b`
/// (bias `1 + j % 2`).
pub fn h2v2(level: Level, precision: u8, a: &[u16], b: &[u16], out: &mut [u16]) {
    let (a, b) = (&a[..2 * out.len()], &b[..2 * out.len()]);
    if precision > 12 {
        return scalar::h2v2(a, b, out);
    }
    dispatch!(level => h2v2(a, b, out));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 23u32;
        for precision in [8u8, 12] {
            let max = (1u32 << precision) - 1;
            for n in [1usize, 3, 4, 7, 8, 9, 16, 17, 31, 32, 33, 100] {
                let mut rnd = || {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    ((seed >> 8) % (max + 1)) as u16
                };
                let a: Vec<u16> = (0..2 * n).map(|_| rnd()).collect();
                let b: Vec<u16> = (0..2 * n).map(|_| rnd()).collect();
                let (mut w1, mut w2) = (vec![0u16; n], vec![0u16; n]);
                scalar::h2v1(&a, &mut w1);
                scalar::h2v2(&a, &b, &mut w2);
                for level in Level::supported() {
                    let (mut g1, mut g2) = (vec![0u16; n], vec![0u16; n]);
                    h2v1(level, precision, &a, &mut g1);
                    h2v2(level, precision, &a, &b, &mut g2);
                    assert_eq!(g1, w1, "h2v1 {level:?} n {n}");
                    assert_eq!(g2, w2, "h2v2 {level:?} n {n}");
                }
            }
        }
    }
}
