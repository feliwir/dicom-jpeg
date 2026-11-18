//! NEON Gaborish rows: 4 lanes (FMA is not used, to match the scalar rounding).
#![allow(unsafe_code)]

use std::arch::aarch64::*;

/// Lanes per vector.
const L: usize = 4;

/// `scalar::row`, L outputs at a time.
#[target_feature(enable = "neon")]
pub(super) fn row(up: &[f32], mid: &[f32], down: &[f32], dst: &mut [f32], k: [f32; 3]) {
    let n = dst.len() / L * L;
    assert!(up.len() >= n + 2 && mid.len() >= n + 2 && down.len() >= n + 2);
    let (w0, w1, norm) = (vdupq_n_f32(k[0]), vdupq_n_f32(k[1]), vdupq_n_f32(k[2]));
    for x in (0..n).step_by(L) {
        // SAFETY: x + 2 + L <= n + 2 <= the length of each input row
        // (asserted), and x + L <= n <= dst.len().
        unsafe {
            let (u, m, d) = (
                up.as_ptr().add(x),
                mid.as_ptr().add(x),
                down.as_ptr().add(x),
            );
            let side = vaddq_f32(
                vaddq_f32(
                    vaddq_f32(vld1q_f32(m), vld1q_f32(m.add(2))),
                    vld1q_f32(u.add(1)),
                ),
                vld1q_f32(d.add(1)),
            );
            let diag = vaddq_f32(
                vaddq_f32(vaddq_f32(vld1q_f32(u), vld1q_f32(u.add(2))), vld1q_f32(d)),
                vld1q_f32(d.add(2)),
            );
            let v = vaddq_f32(
                vaddq_f32(vld1q_f32(m.add(1)), vmulq_f32(side, w0)),
                vmulq_f32(diag, w1),
            );
            vst1q_f32(dst.as_mut_ptr().add(x), vmulq_f32(v, norm));
        }
    }
    super::scalar::row(&up[n..], &mid[n..], &down[n..], &mut dst[n..], k);
}
