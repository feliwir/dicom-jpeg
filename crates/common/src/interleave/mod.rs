//! Interleaving three planes into pixels and back (RGB / YCbCr rows).

use crate::dispatch;
use crate::simd::Level;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse41;

/// `out[3i + k] = plane_k[i] as u8` (truncating) for the `out.len() / 3`
/// pixels of `out`.
pub fn interleave3_u8(level: Level, a: &[u16], b: &[u16], c: &[u16], out: &mut [u8]) {
    let n = out.len() / 3;
    let (a, b, c) = (&a[..n], &b[..n], &c[..n]);
    dispatch!(level => interleave3_u8(a, b, c, &mut out[..3 * n]));
}

/// `out[3i + k] = plane_k[i]` for the `out.len() / 3` pixels of `out`.
pub fn interleave3_u16(level: Level, a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
    let n = out.len() / 3;
    let (a, b, c) = (&a[..n], &b[..n], &c[..n]);
    dispatch!(level => interleave3_u16(a, b, c, &mut out[..3 * n]));
}

/// `plane_k[i] = src[3i + k]` for the `src.len() / 3` pixels of `src`.
pub fn deinterleave3_u8(level: Level, src: &[u8], a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let n = src.len() / 3;
    let (a, b, c) = (&mut a[..n], &mut b[..n], &mut c[..n]);
    dispatch!(level => deinterleave3_u8(&src[..3 * n], a, b, c));
}

/// `plane_k[i] = src[3i + k]` for the `src.len() / 3` pixels of `src`.
pub fn deinterleave3_u16(level: Level, src: &[u16], a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let n = src.len() / 3;
    let (a, b, c) = (&mut a[..n], &mut b[..n], &mut c[..n]);
    dispatch!(level => deinterleave3_u16(&src[..3 * n], a, b, c));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        for n in [
            0usize, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 200, 1001,
        ] {
            let p: [Vec<u16>; 3] = std::array::from_fn(|k| {
                (0..n).map(|i| (i * 7 + k * 1000) as u16 ^ 0x5A5A).collect()
            });
            let mut w8 = vec![0u8; 3 * n];
            let mut w16 = vec![0u16; 3 * n];
            scalar::interleave3_u8(&p[0], &p[1], &p[2], &mut w8);
            scalar::interleave3_u16(&p[0], &p[1], &p[2], &mut w16);
            assert_eq!(w16[3 * n.saturating_sub(1)..].first(), p[0].last());
            for level in Level::supported() {
                let mut g8 = vec![0u8; 3 * n];
                let mut g16 = vec![0u16; 3 * n];
                interleave3_u8(level, &p[0], &p[1], &p[2], &mut g8);
                interleave3_u16(level, &p[0], &p[1], &p[2], &mut g16);
                assert_eq!(g8, w8, "{level:?} n {n}");
                assert_eq!(g16, w16, "{level:?} n {n}");
                let mut d: [Vec<u16>; 3] = std::array::from_fn(|_| vec![0u16; n]);
                let [d0, d1, d2] = &mut d;
                deinterleave3_u16(level, &w16, d0, d1, d2);
                assert_eq!(d, p, "{level:?} n {n}");
                let [d0, d1, d2] = &mut d;
                deinterleave3_u8(level, &w8, d0, d1, d2);
                let low: [Vec<u16>; 3] =
                    std::array::from_fn(|k| p[k].iter().map(|&v| v & 0xFF).collect());
                assert_eq!(d, low, "{level:?} n {n}");
            }
        }
    }
}
