//! NEON fancy upsampling: 8 inputs (16 outputs) per step in 16-bit lanes,
//! stored interleaved with `vst2`.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

use super::scalar;

/// Lanes per vector.
const L: usize = 8;

#[inline]
fn load(s: &[u16]) -> uint16x8_t {
    // SAFETY: callers pass slices of at least L samples.
    unsafe { vld1q_u16(s[..L].as_ptr()) }
}

#[inline]
fn store_pairs(d: &mut [u16], even: uint16x8_t, odd: uint16x8_t) {
    // SAFETY: `d` holds 2 * L samples.
    unsafe { vst2q_u16(d[..2 * L].as_mut_ptr(), uint16x8x2_t(even, odd)) }
}

#[inline]
fn times3(x: uint16x8_t) -> uint16x8_t {
    vaddq_u16(vaddq_u16(x, x), x)
}

pub(super) fn h2v1(src: &[u16], dst: &mut [u16]) {
    let w = src.len();
    scalar::h2v1_edges(src, dst);
    let nv = w.saturating_sub(2) / L * L;
    for i in (1..1 + nv).step_by(L) {
        let (prev, cur, next) = (load(&src[i - 1..]), load(&src[i..]), load(&src[i + 1..]));
        let c = times3(cur);
        let even = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(c, prev), vdupq_n_u16(1)));
        let odd = vshrq_n_u16::<2>(vaddq_u16(vaddq_u16(c, next), vdupq_n_u16(2)));
        store_pairs(&mut dst[2 * i..], even, odd);
    }
    scalar::h2v1_inner(&src[nv..], &mut dst[2 + 2 * nv..2 * w - 2]);
}

pub(super) fn h2v2(near: &[u16], far: &[u16], dst: &mut [u16]) {
    let w = near.len();
    scalar::h2v2_edges(near, far, dst);
    let nv = w.saturating_sub(2) / L * L;
    let cs = |i: usize| vaddq_u16(times3(load(&near[i..])), load(&far[i..]));
    for i in (1..1 + nv).step_by(L) {
        let (prev, cur, next) = (cs(i - 1), cs(i), cs(i + 1));
        let c = times3(cur);
        let even = vshrq_n_u16::<4>(vaddq_u16(vaddq_u16(c, prev), vdupq_n_u16(8)));
        let odd = vshrq_n_u16::<4>(vaddq_u16(vaddq_u16(c, next), vdupq_n_u16(7)));
        store_pairs(&mut dst[2 * i..], even, odd);
    }
    scalar::h2v2_inner(&near[nv..], &far[nv..], &mut dst[2 + 2 * nv..2 * w - 2]);
}
