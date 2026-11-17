//! NEON downsampling: 8 outputs per step with pairwise widening adds.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

/// Outputs per step (inputs: twice as many).
const L: usize = 8;

#[inline]
#[target_feature(enable = "neon")]
fn load(s: &[u16]) -> uint16x8_t {
    // SAFETY: callers pass at least L samples.
    unsafe { vld1q_u16(s[..L].as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store(d: &mut [u16], lo: uint32x4_t, hi: uint32x4_t) {
    // Values are below 2^16: the narrowing is exact.
    // SAFETY: `d` holds at least L samples.
    unsafe {
        vst1q_u16(
            d[..L].as_mut_ptr(),
            vcombine_u16(vmovn_u32(lo), vmovn_u32(hi)),
        )
    };
}

/// Per-lane rounding biases (even, odd outputs).
#[inline]
#[target_feature(enable = "neon")]
fn bias(even: u32, odd: u32) -> uint32x4_t {
    let b = [even, odd, even, odd];
    // SAFETY: `b` holds 4 values.
    unsafe { vld1q_u32(b.as_ptr()) }
}

#[target_feature(enable = "neon")]
pub(super) fn h2v1(row: &[u16], out: &mut [u16]) {
    let n = out.len() / L * L;
    let b = bias(0, 1);
    for j in (0..n).step_by(L) {
        let lo = vpaddlq_u16(load(&row[2 * j..]));
        let hi = vpaddlq_u16(load(&row[2 * j + L..]));
        let f = |s| vshrq_n_u32::<1>(vaddq_u32(s, b));
        store(&mut out[j..], f(lo), f(hi));
    }
    super::scalar::h2v1(&row[2 * n..], &mut out[n..]);
}

#[target_feature(enable = "neon")]
pub(super) fn h2v2(a: &[u16], b: &[u16], out: &mut [u16]) {
    let n = out.len() / L * L;
    let bs = bias(1, 2);
    for j in (0..n).step_by(L) {
        let col = |o: usize| vaddq_u16(load(&a[2 * j + o..]), load(&b[2 * j + o..]));
        let (lo, hi) = (vpaddlq_u16(col(0)), vpaddlq_u16(col(L)));
        let f = |s| vshrq_n_u32::<2>(vaddq_u32(s, bs));
        store(&mut out[j..], f(lo), f(hi));
    }
    super::scalar::h2v2(&a[2 * n..], &b[2 * n..], &mut out[n..]);
}
