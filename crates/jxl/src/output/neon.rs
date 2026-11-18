//! NEON float to integer conversion: 4 samples per vector.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

/// Lanes per vector.
const L: usize = 4;

#[target_feature(enable = "neon")]
pub(super) fn to_u16(src: &[f32], scale: f32, top: u16, dst: &mut [u16]) {
    let (vtop, zero, half) = (vdupq_n_f32(top as f32), vdupq_n_f32(0.0), vdupq_n_f32(0.5));
    let (s, s_rest) = src.as_chunks::<L>();
    let (d, d_rest) = dst.as_chunks_mut::<L>();
    for (s, d) in s.iter().zip(d) {
        // SAFETY: `s` holds L f32 and `d` L u16.
        unsafe {
            let x = vmulq_n_f32(vld1q_f32(s.as_ptr()), scale);
            // NaN -> 0 (x == x is false only for NaN), then clamp to the
            // (exact) bounds.
            let x = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), vceqq_f32(x, x)));
            let x = vminq_f32(vmaxq_f32(x, zero), vtop);
            // Round half up (x >= 0): truncate, then add 1 where the exact
            // fraction is at least 0.5 (the mask is -1 there).
            let t = vcvtq_s32_f32(x);
            let up = vreinterpretq_s32_u32(vcgeq_f32(vsubq_f32(x, vcvtq_f32_s32(t)), half));
            let r = vsubq_s32(t, up);
            // 0..=65535: the unsigned saturating narrowing is exact.
            vst1_u16(d.as_mut_ptr(), vqmovun_s32(r));
        }
    }
    super::scalar::to_u16(s_rest, scale, top, d_rest);
}
