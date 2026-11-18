//! NEON squared quantisation errors: 2 f64 per vector (no FMA).
#![allow(unsafe_code)]

use std::arch::aarch64::*;

use super::SATURATE;

/// Lanes per vector.
const L: usize = 2;

#[target_feature(enable = "neon")]
pub(super) fn error_terms(a: &[f64], d: f64, out: &mut [f64]) {
    let (vd, half, sat) = (vdupq_n_f64(d), vdupq_n_f64(0.5), vdupq_n_f64(SATURATE));
    let (s, s_rest) = a.as_chunks::<L>();
    let (o, o_rest) = out.as_chunks_mut::<L>();
    for (s, o) in s.iter().zip(o) {
        // SAFETY: `s` and `o` hold L f64.
        unsafe {
            let a = vld1q_f64(s.as_ptr());
            let x = vdivq_f64(a, vd);
            // `x as i64 as f64`: truncation, saturating at 2^63.
            let q = vminq_f64(vrndq_f64(x), sat);
            let rec = vmulq_f64(vaddq_f64(q, half), vd);
            // 0 where q is 0.
            let nz = vmvnq_u32(vreinterpretq_u32_u64(vceqzq_f64(q)));
            let rec = vreinterpretq_f64_u32(vandq_u32(vreinterpretq_u32_f64(rec), nz));
            let e = vsubq_f64(a, rec);
            vst1q_f64(o.as_mut_ptr(), vmulq_f64(e, e));
        }
    }
    super::scalar::error_terms(s_rest, d, o_rest);
}
