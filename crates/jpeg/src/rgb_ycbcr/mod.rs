//! RGB -> YCbCr with the fixed-point arithmetic of IJG libjpeg (jccolor.c),
//! in place on three planar rows.

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

const SCALEBITS: u32 = 16;
const HALF: i32 = 1 << (SCALEBITS - 1);

const fn fix(x: f64) -> i32 {
    (x * 65536.0 + 0.5) as i32
}

const YR: i32 = fix(0.29900);
const YG: i32 = fix(0.58700);
const YB: i32 = fix(0.11400);
const CBR: i32 = fix(0.16874);
const CBG: i32 = fix(0.33126);
const CBB: i32 = fix(0.5);
const CRR: i32 = fix(0.5);
const CRG: i32 = fix(0.41869);
const CRB: i32 = fix(0.08131);

/// Replace the R, G, B samples of `a`, `b`, `c` (of `precision` bits, all of
/// the same length) by Y, Cb, Cr.
pub(crate) fn rgb_to_ycbcr(level: Level, precision: u8, [a, b, c]: [&mut [u16]; 3]) {
    let n = a.len();
    let (b, c) = (&mut b[..n], &mut c[..n]);
    // The weighted sums need 16 + precision + 1 bits: SIMD covers <= 12.
    if precision > 12 {
        return scalar::rgb_to_ycbcr(precision, a, b, c);
    }
    dispatch!(level => rgb_to_ycbcr(precision, a, b, c));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        for precision in [8u8, 12] {
            let max = (1u32 << precision) - 1;
            let n = 1003;
            let mut seed = 17u32;
            let mut rnd = || {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                match seed >> 30 {
                    0 => 0,
                    1 => max as u16,
                    _ => ((seed >> 8) % (max + 1)) as u16,
                }
            };
            let p: [Vec<u16>; 3] = std::array::from_fn(|_| (0..n).map(|_| rnd()).collect());
            let mut want = p.clone();
            let [x, y, z] = &mut want;
            rgb_to_ycbcr(Level::Scalar, precision, [x, y, z]);
            for level in Level::supported() {
                for len in [n, 31, 16, 1] {
                    let mut got: [Vec<u16>; 3] = std::array::from_fn(|k| p[k][..len].to_vec());
                    let [x, y, z] = &mut got;
                    rgb_to_ycbcr(level, precision, [x, y, z]);
                    for k in 0..3 {
                        assert_eq!(
                            got[k],
                            want[k][..len],
                            "{level:?} {precision}-bit len {len}"
                        );
                    }
                }
            }
        }
    }
}
