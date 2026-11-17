//! SSE4.1 downsampling: 8 outputs per step.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Outputs per step (inputs: twice as many).
const L: usize = 8;

#[inline]
#[target_feature(enable = "sse4.1")]
fn load(s: &[u16]) -> __m128i {
    // SAFETY: callers pass at least L samples.
    unsafe { _mm_loadu_si128(s[..L].as_ptr().cast()) }
}

/// Sums of adjacent pairs of `x` and `y` (16-bit lanes) as i32: L lanes.
#[inline]
#[target_feature(enable = "sse4.1")]
fn pair_sums(x: __m128i, y: __m128i) -> (__m128i, __m128i) {
    let ones = _mm_set1_epi16(1);
    (_mm_madd_epi16(x, ones), _mm_madd_epi16(y, ones))
}

/// Store the i32 lanes of `lo` and `hi` (values < 2^16) as L u16.
#[inline]
#[target_feature(enable = "sse4.1")]
fn store(d: &mut [u16], lo: __m128i, hi: __m128i) {
    let v = _mm_packus_epi32(lo, hi);
    // SAFETY: `d` holds at least L samples.
    unsafe { _mm_storeu_si128(d[..L].as_mut_ptr().cast(), v) };
}

#[target_feature(enable = "sse4.1")]
pub(super) fn h2v1(row: &[u16], out: &mut [u16]) {
    let n = out.len() / L * L;
    // Bias 0 for even, 1 for odd outputs.
    let bias = _mm_set1_epi64x(1 << 32);
    for j in (0..n).step_by(L) {
        let (lo, hi) = pair_sums(load(&row[2 * j..]), load(&row[2 * j + L..]));
        let f = |s| _mm_srli_epi32::<1>(_mm_add_epi32(s, bias));
        store(&mut out[j..], f(lo), f(hi));
    }
    super::scalar::h2v1(&row[2 * n..], &mut out[n..]);
}

#[target_feature(enable = "sse4.1")]
pub(super) fn h2v2(a: &[u16], b: &[u16], out: &mut [u16]) {
    let n = out.len() / L * L;
    // Bias 1 for even, 2 for odd outputs.
    let bias = _mm_set1_epi64x(2 << 32 | 1);
    for j in (0..n).step_by(L) {
        let col = |o: usize| _mm_add_epi16(load(&a[2 * j + o..]), load(&b[2 * j + o..]));
        let (lo, hi) = pair_sums(col(0), col(L));
        let f = |s| _mm_srli_epi32::<2>(_mm_add_epi32(s, bias));
        store(&mut out[j..], f(lo), f(hi));
    }
    super::scalar::h2v2(&a[2 * n..], &b[2 * n..], &mut out[n..]);
}
