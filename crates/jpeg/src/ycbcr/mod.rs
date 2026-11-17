//! YCbCr -> RGB with the fixed-point arithmetic of IJG libjpeg (jdcolor.c),
//! one row at a time into three planar rows.

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
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);

const fn fix(x: f64) -> i32 {
    (x * (1u64 << SCALEBITS) as f64 + 0.5) as i32
}

const FR: i32 = fix(1.40200);
const FB: i32 = fix(1.77200);
const FGR: i32 = fix(0.71414);
const FGB: i32 = fix(0.34414);

/// Convert the samples of `y`, `cb`, `cr` (of `precision` <= 16 bits) to
/// `r`, `g`, `b` (all of the same length).
pub fn ycbcr_to_rgb(
    level: Level,
    [y, cb, cr]: [&[u16]; 3],
    precision: u8,
    [r, g, b]: [&mut [u16]; 3],
) {
    let n = y.len();
    let (cb, cr, r, g, b) = (&cb[..n], &cr[..n], &mut r[..n], &mut g[..n], &mut b[..n]);
    // The products need 16 + precision bits (signed): SIMD covers <= 12.
    if precision > 12 {
        return scalar::ycbcr_to_rgb(y, cb, cr, precision, r, g, b);
    }
    dispatch!(level => ycbcr_to_rgb(y, cb, cr, precision, r, g, b));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gray_maps_to_gray() {
        let (mut r, mut g, mut b) = ([0u16], [0u16], [0u16]);
        ycbcr_to_rgb(
            Level::Scalar,
            [&[77], &[128], &[128]],
            8,
            [&mut r, &mut g, &mut b],
        );
        assert_eq!((r, g, b), ([77], [77], [77]));
    }

    #[test]
    fn simd_matches_scalar() {
        for precision in [8u8, 12] {
            let max = (1u32 << precision) - 1;
            let n = 1003;
            let mut seed = 99u32;
            let mut rnd = || {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                ((seed >> 8) % (max + 1)) as u16
            };
            let p: [Vec<u16>; 3] = std::array::from_fn(|_| (0..n).map(|_| rnd()).collect());
            let src = [&p[0][..], &p[1][..], &p[2][..]];
            let mut want: [Vec<u16>; 3] = std::array::from_fn(|_| vec![0; n]);
            let [a, b, c] = &mut want;
            ycbcr_to_rgb(Level::Scalar, src, precision, [a, b, c]);
            for level in Level::supported() {
                for len in [n, 17, 8, 1] {
                    let mut got: [Vec<u16>; 3] = std::array::from_fn(|_| vec![0; len]);
                    let [a, b, c] = &mut got;
                    let src = [&p[0][..len], &p[1][..len], &p[2][..len]];
                    ycbcr_to_rgb(level, src, precision, [a, b, c]);
                    for k in 0..3 {
                        assert_eq!(got[k], want[k][..len], "{level:?} {precision}-bit");
                    }
                }
            }
        }
    }
}
