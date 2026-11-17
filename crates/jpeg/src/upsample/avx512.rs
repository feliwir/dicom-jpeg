//! AVX-512 fancy upsampling: 32 inputs (64 outputs) per step in 16-bit lanes.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::scalar;

/// Lanes per vector.
const L: usize = 32;

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn load(s: &[u16]) -> __m512i {
    // SAFETY: callers pass slices of at least L samples.
    unsafe { _mm512_loadu_si512(s[..L].as_ptr().cast()) }
}

/// Store the outputs `even` (2i) and `odd` (2i+1) interleaved to `d`
/// (2 * L samples).
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn store_pairs(d: &mut [u16], even: __m512i, odd: __m512i) {
    let lo = _mm512_unpacklo_epi16(even, odd);
    let hi = _mm512_unpackhi_epi16(even, odd);
    // unpack works per 128-bit lane: [L0 H0 L1 H1], [L2 H2 L3 H3].
    let a = _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), hi);
    let b = _mm512_permutex2var_epi64(lo, _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15), hi);
    // SAFETY: `d` holds 2 * L samples.
    unsafe {
        _mm512_storeu_si512(d[..L].as_mut_ptr().cast(), a);
        _mm512_storeu_si512(d[L..2 * L].as_mut_ptr().cast(), b);
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn times3(x: __m512i) -> __m512i {
    _mm512_add_epi16(_mm512_add_epi16(x, x), x)
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn h2v1(src: &[u16], dst: &mut [u16]) {
    let w = src.len();
    scalar::h2v1_edges(src, dst);
    // Inputs 1..w-1 have both neighbours; L of them per step.
    let nv = w.saturating_sub(2) / L * L;
    let (one, two) = (_mm512_set1_epi16(1), _mm512_set1_epi16(2));
    for i in (1..1 + nv).step_by(L) {
        let (prev, cur, next) = (load(&src[i - 1..]), load(&src[i..]), load(&src[i + 1..]));
        let c = times3(cur);
        let even = _mm512_srli_epi16::<2>(_mm512_add_epi16(_mm512_add_epi16(c, prev), one));
        let odd = _mm512_srli_epi16::<2>(_mm512_add_epi16(_mm512_add_epi16(c, next), two));
        store_pairs(&mut dst[2 * i..], even, odd);
    }
    scalar::h2v1_inner(&src[nv..], &mut dst[2 + 2 * nv..2 * w - 2]);
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn h2v2(near: &[u16], far: &[u16], dst: &mut [u16]) {
    let w = near.len();
    scalar::h2v2_edges(near, far, dst);
    let nv = w.saturating_sub(2) / L * L;
    let (seven, eight) = (_mm512_set1_epi16(7), _mm512_set1_epi16(8));
    let cs = |i: usize| _mm512_add_epi16(times3(load(&near[i..])), load(&far[i..]));
    for i in (1..1 + nv).step_by(L) {
        let (prev, cur, next) = (cs(i - 1), cs(i), cs(i + 1));
        let c = times3(cur);
        let even = _mm512_srli_epi16::<4>(_mm512_add_epi16(_mm512_add_epi16(c, prev), eight));
        let odd = _mm512_srli_epi16::<4>(_mm512_add_epi16(_mm512_add_epi16(c, next), seven));
        store_pairs(&mut dst[2 * i..], even, odd);
    }
    scalar::h2v2_inner(&near[nv..], &far[nv..], &mut dst[2 + 2 * nv..2 * w - 2]);
}
