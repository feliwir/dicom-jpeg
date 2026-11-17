//! NEON lifting kernels: 4 lanes (FMA is not used, to match the scalar
//! rounding).
#![allow(unsafe_code)]

use std::arch::aarch64::*;

/// Lanes per vector.
const L: usize = 4;

#[target_feature(enable = "neon")]
pub(super) fn update53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    let n = cur.len() / L * L;
    for i in (0..n).step_by(L) {
        // SAFETY: L elements remain from `i` in all three slices.
        unsafe {
            let (va, vb) = (
                vld1q_s32(a[i..i + L].as_ptr()),
                vld1q_s32(b[i..i + L].as_ptr()),
            );
            let c = cur[i..i + L].as_mut_ptr();
            let s = vshrq_n_s32::<2>(vaddq_s32(vaddq_s32(va, vb), vdupq_n_s32(2)));
            vst1q_s32(c, vsubq_s32(vld1q_s32(c), s));
        }
    }
    super::scalar::update53(&mut cur[n..], &a[n..], &b[n..]);
}

#[target_feature(enable = "neon")]
pub(super) fn predict53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    let n = cur.len() / L * L;
    for i in (0..n).step_by(L) {
        // SAFETY: as in `update53`.
        unsafe {
            let (va, vb) = (
                vld1q_s32(a[i..i + L].as_ptr()),
                vld1q_s32(b[i..i + L].as_ptr()),
            );
            let c = cur[i..i + L].as_mut_ptr();
            vst1q_s32(
                c,
                vaddq_s32(vld1q_s32(c), vshrq_n_s32::<1>(vaddq_s32(va, vb))),
            );
        }
    }
    super::scalar::predict53(&mut cur[n..], &a[n..], &b[n..]);
}

#[cfg(feature = "encode")]
#[target_feature(enable = "neon")]
pub(super) fn apredict53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    let n = cur.len() / L * L;
    for i in (0..n).step_by(L) {
        // SAFETY: as in `update53`.
        unsafe {
            let (va, vb) = (
                vld1q_s32(a[i..i + L].as_ptr()),
                vld1q_s32(b[i..i + L].as_ptr()),
            );
            let c = cur[i..i + L].as_mut_ptr();
            vst1q_s32(
                c,
                vsubq_s32(vld1q_s32(c), vshrq_n_s32::<1>(vaddq_s32(va, vb))),
            );
        }
    }
    super::scalar::apredict53(&mut cur[n..], &a[n..], &b[n..]);
}

#[cfg(feature = "encode")]
#[target_feature(enable = "neon")]
pub(super) fn aupdate53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    let n = cur.len() / L * L;
    for i in (0..n).step_by(L) {
        // SAFETY: as in `update53`.
        unsafe {
            let (va, vb) = (
                vld1q_s32(a[i..i + L].as_ptr()),
                vld1q_s32(b[i..i + L].as_ptr()),
            );
            let c = cur[i..i + L].as_mut_ptr();
            let s = vshrq_n_s32::<2>(vaddq_s32(vaddq_s32(va, vb), vdupq_n_s32(2)));
            vst1q_s32(c, vaddq_s32(vld1q_s32(c), s));
        }
    }
    super::scalar::aupdate53(&mut cur[n..], &a[n..], &b[n..]);
}

#[target_feature(enable = "neon")]
pub(super) fn lift97(cur: &mut [f32], a: &[f32], b: &[f32], k: f32) {
    let n = cur.len() / L * L;
    for i in (0..n).step_by(L) {
        // SAFETY: as in `update53`.
        unsafe {
            let (va, vb) = (
                vld1q_f32(a[i..i + L].as_ptr()),
                vld1q_f32(b[i..i + L].as_ptr()),
            );
            let c = cur[i..i + L].as_mut_ptr();
            let t = vmulq_n_f32(vaddq_f32(va, vb), k);
            vst1q_f32(c, vsubq_f32(vld1q_f32(c), t));
        }
    }
    super::scalar::lift97(&mut cur[n..], &a[n..], &b[n..], k);
}

#[target_feature(enable = "neon")]
pub(super) fn scale97(cur: &mut [f32], k: f32) {
    let n = cur.len() / L * L;
    for i in (0..n).step_by(L) {
        // SAFETY: L elements remain from `i`.
        unsafe {
            let c = cur[i..i + L].as_mut_ptr();
            vst1q_f32(c, vmulq_n_f32(vld1q_f32(c), k));
        }
    }
    super::scalar::scale97(&mut cur[n..], k);
}
