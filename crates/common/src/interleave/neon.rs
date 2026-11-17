//! NEON interleaving with the structure loads and stores `vld3` / `vst3`.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(super) fn interleave3_u8(a: &[u16], b: &[u16], c: &[u16], out: &mut [u8]) {
    let n = out.len() / 3 / 16 * 16;
    for i in (0..n).step_by(16) {
        // SAFETY: `i + 16 <= n` samples remain in each plane, and 48 bytes
        // in `out` from `3 * i`.
        unsafe {
            let bytes = |p: &[u16]| {
                // Truncating narrow, as the scalar `as u8`.
                vcombine_u8(
                    vmovn_u16(vld1q_u16(p.as_ptr())),
                    vmovn_u16(vld1q_u16(p[8..].as_ptr())),
                )
            };
            let v = uint8x16x3_t(bytes(&a[i..]), bytes(&b[i..]), bytes(&c[i..]));
            vst3q_u8(out[3 * i..].as_mut_ptr(), v);
        }
    }
    super::scalar::interleave3_u8(&a[n..], &b[n..], &c[n..], &mut out[3 * n..]);
}

#[target_feature(enable = "neon")]
pub(super) fn interleave3_u16(a: &[u16], b: &[u16], c: &[u16], out: &mut [u16]) {
    let n = out.len() / 3 / 8 * 8;
    for i in (0..n).step_by(8) {
        // SAFETY: 8 samples remain in each plane from `i`, and 24 in `out`
        // from `3 * i`.
        unsafe {
            let v = uint16x8x3_t(
                vld1q_u16(a[i..].as_ptr()),
                vld1q_u16(b[i..].as_ptr()),
                vld1q_u16(c[i..].as_ptr()),
            );
            vst3q_u16(out[3 * i..].as_mut_ptr(), v);
        }
    }
    super::scalar::interleave3_u16(&a[n..], &b[n..], &c[n..], &mut out[3 * n..]);
}

#[target_feature(enable = "neon")]
pub(super) fn deinterleave3_u8(src: &[u8], a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let n = src.len() / 3 / 16 * 16;
    for i in (0..n).step_by(16) {
        // SAFETY: 48 bytes remain in `src` from `3 * i`, and 16 samples in
        // each plane from `i`.
        unsafe {
            let v = vld3q_u8(src[3 * i..].as_ptr());
            for (d, x) in [&mut a[i..], &mut b[i..], &mut c[i..]]
                .into_iter()
                .zip([v.0, v.1, v.2])
            {
                vst1q_u16(d.as_mut_ptr(), vmovl_u8(vget_low_u8(x)));
                vst1q_u16(d[8..].as_mut_ptr(), vmovl_u8(vget_high_u8(x)));
            }
        }
    }
    super::scalar::deinterleave3_u8(&src[3 * n..], &mut a[n..], &mut b[n..], &mut c[n..]);
}

#[target_feature(enable = "neon")]
pub(super) fn deinterleave3_u16(src: &[u16], a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let n = src.len() / 3 / 8 * 8;
    for i in (0..n).step_by(8) {
        // SAFETY: 24 samples remain in `src` from `3 * i`, and 8 in each
        // plane from `i`.
        unsafe {
            let v = vld3q_u16(src[3 * i..].as_ptr());
            vst1q_u16(a[i..].as_mut_ptr(), v.0);
            vst1q_u16(b[i..].as_mut_ptr(), v.1);
            vst1q_u16(c[i..].as_mut_ptr(), v.2);
        }
    }
    super::scalar::deinterleave3_u16(&src[3 * n..], &mut a[n..], &mut b[n..], &mut c[n..]);
}
