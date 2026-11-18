//! Reordering quantised blocks into zig-zag order for entropy coding, with
//! a mask of the non-zero coefficients (bit k for zig-zag position k).
//!
//! The coefficients are clamped to the 12-bit range by the quantiser, so
//! they fit in i16 and the mask can be taken after narrowing. AVX-512 does
//! the reorder with two word permutes; the other levels gather the
//! coefficients through the table and build the mask with vector compares.

use dicom_jpeg_common::dispatch;
use dicom_jpeg_common::simd::Level;

use crate::decoder::ZIGZAG;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse41;

/// `zz[i]` = block `coefs[i]` in zig-zag order (narrowed to i16) and
/// `nonzero[i]` its mask of non-zero coefficients, for each block.
/// Coefficients must be within the i16 range.
pub fn zigzag(level: Level, coefs: &[[i32; 64]], zz: &mut [[i16; 64]], nonzero: &mut [u64]) {
    assert!(zz.len() == coefs.len() && nonzero.len() == coefs.len());
    debug_assert!(coefs.iter().flatten().all(|&v| i16::try_from(v).is_ok()));
    dispatch!(level => zigzag(coefs, zz, nonzero));
}

/// The mask of the non-zero entries of `zz` (bit k for entry k).
#[inline]
fn mask_of(zz: &[i16; 64]) -> u64 {
    zz.iter()
        .enumerate()
        .fold(0, |m, (k, &v)| m | ((v != 0) as u64) << k)
}

/// The scalar gather of one block into zig-zag order.
#[inline(always)]
fn gather(coefs: &[i32; 64], zz: &mut [i16; 64]) {
    for (z, &n) in zz.iter_mut().zip(&ZIGZAG) {
        *z = coefs[n] as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 21u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        // Sparse and dense blocks over the whole 12-bit range.
        let coefs: Vec<[i32; 64]> = (0..200)
            .map(|b| {
                std::array::from_fn(|_| {
                    let x = rnd();
                    if x % 8 < b % 9 {
                        (x >> 8) as i32 % 16384
                    } else {
                        0
                    }
                })
            })
            .collect();
        for len in [coefs.len(), 1] {
            let (mut want_z, mut want_m) = (vec![[0i16; 64]; len], vec![0u64; len]);
            zigzag(Level::Scalar, &coefs[..len], &mut want_z, &mut want_m);
            for (c, (z, &m)) in coefs.iter().zip(want_z.iter().zip(&want_m)) {
                for (k, &n) in ZIGZAG.iter().enumerate() {
                    assert_eq!(z[k] as i32, c[n]);
                    assert_eq!(m >> k & 1 == 1, c[n] != 0);
                }
            }
            for level in Level::supported() {
                let (mut z, mut m) = (vec![[0i16; 64]; len], vec![0u64; len]);
                zigzag(level, &coefs[..len], &mut z, &mut m);
                assert_eq!(z, want_z, "{level:?}");
                assert_eq!(m, want_m, "{level:?}");
            }
        }
    }
}
