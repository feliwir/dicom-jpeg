//! NEON RGB -> YCbCr, 4 samples (one vector of 4 x i32) per step.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

use super::*;

/// Y, Cb, Cr of one vector of R, G, B (i32 lanes).
#[inline]
#[target_feature(enable = "neon")]
fn convert(r: int32x4_t, g: int32x4_t, b: int32x4_t, offset: i32) -> [int32x4_t; 3] {
    let mul = |a, c| vmulq_n_s32(a, c);
    let y = vaddq_s32(
        vaddq_s32(vaddq_s32(mul(r, YR), mul(g, YG)), mul(b, YB)),
        vdupq_n_s32(HALF),
    );
    let bias = vdupq_n_s32(offset + HALF - 1);
    let cb = vaddq_s32(
        vaddq_s32(
            vsubq_s32(vsubq_s32(vdupq_n_s32(0), mul(r, CBR)), mul(g, CBG)),
            mul(b, CBB),
        ),
        bias,
    );
    let cr = vaddq_s32(
        vsubq_s32(vsubq_s32(mul(r, CRR), mul(g, CRG)), mul(b, CRB)),
        bias,
    );
    [y, cb, cr].map(|v| vshrq_n_s32::<16>(v))
}

#[target_feature(enable = "neon")]
pub(super) fn rgb_to_ycbcr(precision: u8, a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let offset = (1i32 << (precision - 1)) << SCALEBITS;
    let n = a.len() / 4 * 4;
    for i in (0..n).step_by(4) {
        // SAFETY: 4 samples remain from `i` in every row.
        let load =
            |s: &[u16]| unsafe { vreinterpretq_s32_u32(vmovl_u16(vld1_u16(s[i..i + 4].as_ptr()))) };
        let (r, g, bl) = (load(a), load(b), load(c));
        let [y, cb, cr] = convert(r, g, bl, offset);
        for (d, v) in [&mut *a, &mut *b, &mut *c].into_iter().zip([y, cb, cr]) {
            // Results are in 0..2^precision: the saturating narrow is exact.
            // SAFETY: 4 samples remain from `i`.
            unsafe { vst1_u16(d[i..i + 4].as_mut_ptr(), vqmovun_s32(v)) };
        }
    }
    super::scalar::rgb_to_ycbcr(precision, &mut a[n..], &mut b[n..], &mut c[n..]);
}
