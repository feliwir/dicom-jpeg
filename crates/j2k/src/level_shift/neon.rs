//! NEON level shift: 4 samples per vector.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

/// Lanes per vector.
const L: usize = 4;

#[target_feature(enable = "neon")]
pub(super) fn shift_int(data: &mut [i32], a: i32, b: i32, shift: i32) {
    let (va, vb, vs) = (vdupq_n_s32(a), vdupq_n_s32(b), vdupq_n_s32(shift));
    let (chunks, rest) = data.as_chunks_mut::<L>();
    for c in chunks {
        // SAFETY: `c` holds L i32.
        unsafe {
            let v = vminq_s32(vmaxq_s32(vld1q_s32(c.as_ptr()), va), vb);
            vst1q_s32(c.as_mut_ptr(), vaddq_s32(v, vs));
        }
    }
    super::scalar::shift_int(rest, a, b, shift);
}

#[target_feature(enable = "neon")]
pub(super) fn shift_float(src: &[f32], dst: &mut [i32], a: i32, b: i32, shift: i32) {
    let (fa, fb) = (vdupq_n_f32(a as f32), vdupq_n_f32(b as f32));
    let (half, neg_half) = (vdupq_n_f32(0.5), vdupq_n_f32(-0.5));
    let vs = vdupq_n_s32(shift);
    let (s, s_rest) = src.as_chunks::<L>();
    let (d, d_rest) = dst.as_chunks_mut::<L>();
    for (s, d) in s.iter().zip(d) {
        // SAFETY: `s` and `d` hold L elements.
        unsafe {
            let v = vld1q_f32(s.as_ptr());
            // NaN -> 0 (v == v is false only for NaN), then clamp (exact
            // bounds).
            let v = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v), vceqq_f32(v, v)));
            let v = vminq_f32(vmaxq_f32(v, fa), fb);
            // Round half away from zero: truncate, then step by the
            // (exact) fraction. Masks are -1 where true.
            let t = vcvtq_s32_f32(v);
            let f = vsubq_f32(v, vcvtq_f32_s32(t));
            let up = vreinterpretq_s32_u32(vcgeq_f32(f, half));
            let down = vreinterpretq_s32_u32(vcleq_f32(f, neg_half));
            let r = vaddq_s32(vsubq_s32(t, up), down);
            vst1q_s32(d.as_mut_ptr(), vaddq_s32(r, vs));
        }
    }
    super::scalar::shift_float(s_rest, d_rest, a, b, shift);
}
