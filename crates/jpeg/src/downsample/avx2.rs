//! AVX2 downsampling: 16 outputs per step.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Outputs per step (inputs: twice as many).
const L: usize = 16;

#[inline]
#[target_feature(enable = "avx2")]
fn load(s: &[u16]) -> __m256i {
    // SAFETY: callers pass at least L samples.
    unsafe { _mm256_loadu_si256(s[..L].as_ptr().cast()) }
}

/// Sums of adjacent pairs of `x` and `y` (16-bit lanes) as i32: L lanes.
#[inline]
#[target_feature(enable = "avx2")]
fn pair_sums(x: __m256i, y: __m256i) -> (__m256i, __m256i) {
    let ones = _mm256_set1_epi16(1);
    (_mm256_madd_epi16(x, ones), _mm256_madd_epi16(y, ones))
}

/// Store the i32 lanes of `lo` and `hi` (values < 2^16) as L u16.
#[inline]
#[target_feature(enable = "avx2")]
fn store(d: &mut [u16], lo: __m256i, hi: __m256i) {
    // packus works per 128-bit lane: [lo0..4, hi0..4, lo4..8, hi4..8].
    let v = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi32(lo, hi));
    // SAFETY: `d` holds at least L samples.
    unsafe { _mm256_storeu_si256(d[..L].as_mut_ptr().cast(), v) };
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn h2v1(row: &[u16], out: &mut [u16]) {
    let n = out.len() / L * L;
    // Bias 0 for even, 1 for odd outputs.
    let bias = _mm256_set1_epi64x(1 << 32);
    for j in (0..n).step_by(L) {
        let (lo, hi) = pair_sums(load(&row[2 * j..]), load(&row[2 * j + L..]));
        let f = |s| _mm256_srli_epi32::<1>(_mm256_add_epi32(s, bias));
        store(&mut out[j..], f(lo), f(hi));
    }
    super::sse41::h2v1(&row[2 * n..], &mut out[n..]);
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn h2v2(a: &[u16], b: &[u16], out: &mut [u16]) {
    let n = out.len() / L * L;
    // Bias 1 for even, 2 for odd outputs.
    let bias = _mm256_set1_epi64x(2 << 32 | 1);
    for j in (0..n).step_by(L) {
        let col = |o: usize| _mm256_add_epi16(load(&a[2 * j + o..]), load(&b[2 * j + o..]));
        let (lo, hi) = pair_sums(col(0), col(L));
        let f = |s| _mm256_srli_epi32::<2>(_mm256_add_epi32(s, bias));
        store(&mut out[j..], f(lo), f(hi));
    }
    super::sse41::h2v2(&a[2 * n..], &b[2 * n..], &mut out[n..]);
}
