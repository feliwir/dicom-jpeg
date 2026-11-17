//! AVX-512 downsampling: 32 outputs per step.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Outputs per step (inputs: twice as many).
const L: usize = 32;

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn load(s: &[u16]) -> __m512i {
    // SAFETY: callers pass at least L samples.
    unsafe { _mm512_loadu_si512(s[..L].as_ptr().cast()) }
}

/// Sums of adjacent pairs of `x` and `y` (16-bit lanes) as i32: L lanes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn pair_sums(x: __m512i, y: __m512i) -> (__m512i, __m512i) {
    let ones = _mm512_set1_epi16(1);
    (_mm512_madd_epi16(x, ones), _mm512_madd_epi16(y, ones))
}

/// Store the i32 lanes of `lo` and `hi` (values < 2^16) as L u16.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn store(d: &mut [u16], lo: __m512i, hi: __m512i) {
    // SAFETY: `d` holds at least L samples.
    unsafe {
        _mm256_storeu_si256(d[..16].as_mut_ptr().cast(), _mm512_cvtepi32_epi16(lo));
        _mm256_storeu_si256(d[16..L].as_mut_ptr().cast(), _mm512_cvtepi32_epi16(hi));
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn h2v1(row: &[u16], out: &mut [u16]) {
    let n = out.len() / L * L;
    // Bias 0 for even, 1 for odd outputs.
    let bias = _mm512_set1_epi64(1 << 32);
    for j in (0..n).step_by(L) {
        let (lo, hi) = pair_sums(load(&row[2 * j..]), load(&row[2 * j + L..]));
        let f = |s| _mm512_srli_epi32::<1>(_mm512_add_epi32(s, bias));
        store(&mut out[j..], f(lo), f(hi));
    }
    super::sse41::h2v1(&row[2 * n..], &mut out[n..]);
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn h2v2(a: &[u16], b: &[u16], out: &mut [u16]) {
    let n = out.len() / L * L;
    // Bias 1 for even, 2 for odd outputs.
    let bias = _mm512_set1_epi64(2 << 32 | 1);
    for j in (0..n).step_by(L) {
        let col = |o: usize| _mm512_add_epi16(load(&a[2 * j + o..]), load(&b[2 * j + o..]));
        let (lo, hi) = pair_sums(col(0), col(L));
        let f = |s| _mm512_srli_epi32::<2>(_mm512_add_epi32(s, bias));
        store(&mut out[j..], f(lo), f(hi));
    }
    super::sse41::h2v2(&a[2 * n..], &b[2 * n..], &mut out[n..]);
}
