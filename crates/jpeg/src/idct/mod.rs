//! Accurate integer inverse DCT.
//!
//! This is the "ISLOW" algorithm of the IJG library (jidctint.c, Loeffler,
//! Ligtenberg & Moschytz), reproduced bit-exactly so decoded output matches
//! libjpeg 6b / DCMTK / libjpeg-turbo. As in libjpeg, 8-bit data is
//! transformed in 32-bit and 12-bit data in 64-bit arithmetic; valid data
//! never overflows, and corrupt coefficients wrap instead of panicking.
//!
//! The 8-bit transform has SIMD versions, which compute every column and
//! row in full and select libjpeg's shortcut results per lane, so they wrap
//! exactly like the scalar code. 12-bit data stays scalar (it needs 64-bit
//! multiplies).

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

const CONST_BITS: u32 = 13;

const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

/// Inverse DCT of one block of dequantized coefficients (natural order),
/// writing level-shifted, clamped samples of `precision` (8 or 12) bits.
#[cfg(test)]
pub(crate) fn idct_block(coef: &[i32; 64], precision: u8, out: &mut [u16], stride: usize) {
    scalar::idct_block(coef, precision, out, stride);
}

/// Inverse DCT of each block of `coefs` into the 8 x 8 block of `out` at the
/// same index, with the implementation for `level`.
pub fn idct_blocks(level: Level, coefs: &[[i32; 64]], precision: u8, out: &mut [[u16; 64]]) {
    assert_eq!(coefs.len(), out.len());
    if precision > 8 {
        for (c, o) in coefs.iter().zip(out.iter_mut()) {
            scalar::idct_block(c, precision, o, 8);
        }
        return;
    }
    dispatch!(level => idct8(coefs, out));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Blocks that exercise the shortcuts, wrapping and clamping.
    fn blocks() -> Vec<[i32; 64]> {
        let mut seed = 0x1234_5678u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut v = Vec::new();
        for n in 0..300 {
            let mut b = [0i32; 64];
            match n % 6 {
                // DC only.
                0 => b[0] = (rnd() % 4096) as i32 - 2048,
                // Some all-zero columns and rows.
                1 => {
                    for _ in 0..5 {
                        b[(rnd() % 64) as usize] = (rnd() % 2048) as i32 - 1024;
                    }
                }
                // Dense, valid range.
                2 => b = std::array::from_fn(|_| (rnd() % 1024) as i32 - 512),
                // Huge (corrupt) values that wrap.
                3 => b = std::array::from_fn(|_| rnd() as i32),
                4 => b[0] = rnd() as i32,
                // First row / column only.
                _ => {
                    for k in 0..8 {
                        b[if n % 2 == 0 { k } else { k * 8 }] = (rnd() % 4000) as i32 - 2000;
                    }
                }
            }
            v.push(b);
        }
        v
    }

    #[test]
    fn simd_matches_scalar() {
        let coefs = blocks();
        let mut want = vec![[0u16; 64]; coefs.len()];
        idct_blocks(Level::Scalar, &coefs, 8, &mut want);
        for level in Level::supported() {
            // Odd and even batch lengths.
            for len in [coefs.len(), coefs.len() - 1, 1] {
                let mut got = vec![[0u16; 64]; len];
                idct_blocks(level, &coefs[..len], 8, &mut got);
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g, w, "{level:?}, block {i}: {:?}", coefs[i]);
                }
            }
        }
    }
}
