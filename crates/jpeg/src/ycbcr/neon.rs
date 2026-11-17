//! NEON YCbCr -> RGB, 8 samples (two vectors of 4 x i32) per step.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

use super::*;

/// One vector of 4 samples: (r, g, b) as i32, clamped.
#[inline]
#[target_feature(enable = "neon")]
fn convert(
    y: int32x4_t,
    cb: int32x4_t,
    cr: int32x4_t,
    center: int32x4_t,
    max: int32x4_t,
) -> [int32x4_t; 3] {
    let cbv = vsubq_s32(cb, center);
    let crv = vsubq_s32(cr, center);
    let half = vdupq_n_s32(ONE_HALF);
    let r = vshrq_n_s32::<16>(vaddq_s32(vmulq_n_s32(crv, FR), half));
    let g = vshrq_n_s32::<16>(vsubq_s32(
        vaddq_s32(vmulq_n_s32(cbv, -FGB), half),
        vmulq_n_s32(crv, FGR),
    ));
    let b = vshrq_n_s32::<16>(vaddq_s32(vmulq_n_s32(cbv, FB), half));
    let clamp = |v| vminq_s32(vmaxq_s32(vaddq_s32(y, v), vdupq_n_s32(0)), max);
    [clamp(r), clamp(g), clamp(b)]
}

#[target_feature(enable = "neon")]
pub(super) fn ycbcr_to_rgb(
    y: &[u16],
    cb: &[u16],
    cr: &[u16],
    precision: u8,
    r: &mut [u16],
    g: &mut [u16],
    b: &mut [u16],
) {
    let max = vdupq_n_s32((1 << precision) - 1);
    let center = vdupq_n_s32(1 << (precision - 1));
    let n = y.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        // SAFETY: 8 samples remain from `i` in every row.
        let (yv, cbv, crv) = unsafe {
            (
                vld1q_u16(y[i..].as_ptr()),
                vld1q_u16(cb[i..].as_ptr()),
                vld1q_u16(cr[i..].as_ptr()),
            )
        };
        let lo = |v| vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v)));
        let hi = |v| vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v)));
        let a = convert(lo(yv), lo(cbv), lo(crv), center, max);
        let c = convert(hi(yv), hi(cbv), hi(crv), center, max);
        for (k, d) in [&mut *r, &mut *g, &mut *b].into_iter().enumerate() {
            // Values are in 0..=max: the saturating narrow is exact.
            let v = vcombine_u16(vqmovun_s32(a[k]), vqmovun_s32(c[k]));
            // SAFETY: 8 samples remain from `i`.
            unsafe { vst1q_u16(d[i..].as_mut_ptr(), v) };
        }
    }
    super::scalar::ycbcr_to_rgb(
        &y[n..],
        &cb[n..],
        &cr[n..],
        precision,
        &mut r[n..],
        &mut g[n..],
        &mut b[n..],
    );
}
