//! NEON XYB to linear RGB: 4 pixels at a time (FMA is not used, to match the scalar rounding).
#![allow(unsafe_code)]

use std::arch::aarch64::*;

use super::Xyb;

/// Lanes per vector.
const L: usize = 4;

/// `scalar::to_linear`, L pixels at a time.
#[target_feature(enable = "neon")]
pub(super) fn to_linear(planes: [&mut [f32]; 3], k: &Xyb) {
    let [p0, p1, p2] = planes;
    let n = p0.len().min(p1.len()).min(p2.len()) / L * L;
    let cb = k.cbrt_bias.map(|v| vdupq_n_f32(v));
    let bias = k.bias.map(|v| vdupq_n_f32(v));
    let it = vdupq_n_f32(k.itscale);
    let m = k.matrix.map(|v| vdupq_n_f32(v));
    for i in (0..n).step_by(L) {
        // SAFETY: i + L <= n <= the length of each plane.
        unsafe {
            let (x, y, b) = (
                vld1q_f32(p0.as_ptr().add(i)),
                vld1q_f32(p1.as_ptr().add(i)),
                vld1q_f32(p2.as_ptr().add(i)),
            );
            let l = vsubq_f32(vaddq_f32(y, x), cb[0]);
            let mm = vsubq_f32(vsubq_f32(y, x), cb[1]);
            let s = vsubq_f32(b, cb[2]);
            let mixed = [
                vmulq_f32(vaddq_f32(vmulq_f32(vmulq_f32(l, l), l), bias[0]), it),
                vmulq_f32(vaddq_f32(vmulq_f32(vmulq_f32(mm, mm), mm), bias[1]), it),
                vmulq_f32(vaddq_f32(vmulq_f32(vmulq_f32(s, s), s), bias[2]), it),
            ];
            let out = [p0.as_mut_ptr(), p1.as_mut_ptr(), p2.as_mut_ptr()];
            for (c, o) in out.into_iter().enumerate() {
                let v = vaddq_f32(
                    vaddq_f32(
                        vmulq_f32(m[c * 3], mixed[0]),
                        vmulq_f32(m[c * 3 + 1], mixed[1]),
                    ),
                    vmulq_f32(m[c * 3 + 2], mixed[2]),
                );
                vst1q_f32(o.add(i), v);
            }
        }
    }
    super::scalar::to_linear([&mut p0[n..], &mut p1[n..], &mut p2[n..]], k);
}
