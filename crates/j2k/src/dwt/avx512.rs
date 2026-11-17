//! AVX-512 lifting kernels: 16 lanes (FMA is not used, to match the scalar rounding).
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 16;

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn update53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    let n = cur.len() / L * L;
    let two = _mm512_set1_epi32(2);
    for i in (0..n).step_by(L) {
        // SAFETY: L elements remain from `i` in all three slices (callers
        // pass equal lengths).
        unsafe {
            let va = _mm512_loadu_si512(a[i..i + L].as_ptr().cast());
            let vb = _mm512_loadu_si512(b[i..i + L].as_ptr().cast());
            let c = cur[i..i + L].as_mut_ptr().cast();
            let s = _mm512_srai_epi32::<2>(_mm512_add_epi32(_mm512_add_epi32(va, vb), two));
            _mm512_storeu_si512(c, _mm512_sub_epi32(_mm512_loadu_si512(c), s));
        }
    }
    super::scalar::update53(&mut cur[n..], &a[n..], &b[n..]);
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn predict53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    let n = cur.len() / L * L;
    for i in (0..n).step_by(L) {
        // SAFETY: as in `update53`.
        unsafe {
            let va = _mm512_loadu_si512(a[i..i + L].as_ptr().cast());
            let vb = _mm512_loadu_si512(b[i..i + L].as_ptr().cast());
            let c = cur[i..i + L].as_mut_ptr().cast();
            let s = _mm512_srai_epi32::<1>(_mm512_add_epi32(va, vb));
            _mm512_storeu_si512(c, _mm512_add_epi32(_mm512_loadu_si512(c), s));
        }
    }
    super::scalar::predict53(&mut cur[n..], &a[n..], &b[n..]);
}

#[cfg(feature = "encode")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn apredict53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    let n = cur.len() / L * L;
    for i in (0..n).step_by(L) {
        // SAFETY: as in `update53`.
        unsafe {
            let va = _mm512_loadu_si512(a[i..i + L].as_ptr().cast());
            let vb = _mm512_loadu_si512(b[i..i + L].as_ptr().cast());
            let c = cur[i..i + L].as_mut_ptr().cast();
            let s = _mm512_srai_epi32::<1>(_mm512_add_epi32(va, vb));
            _mm512_storeu_si512(c, _mm512_sub_epi32(_mm512_loadu_si512(c), s));
        }
    }
    super::scalar::apredict53(&mut cur[n..], &a[n..], &b[n..]);
}

#[cfg(feature = "encode")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn aupdate53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    let n = cur.len() / L * L;
    let two = _mm512_set1_epi32(2);
    for i in (0..n).step_by(L) {
        // SAFETY: as in `update53`.
        unsafe {
            let va = _mm512_loadu_si512(a[i..i + L].as_ptr().cast());
            let vb = _mm512_loadu_si512(b[i..i + L].as_ptr().cast());
            let c = cur[i..i + L].as_mut_ptr().cast();
            let s = _mm512_srai_epi32::<2>(_mm512_add_epi32(_mm512_add_epi32(va, vb), two));
            _mm512_storeu_si512(c, _mm512_add_epi32(_mm512_loadu_si512(c), s));
        }
    }
    super::scalar::aupdate53(&mut cur[n..], &a[n..], &b[n..]);
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn lift97(cur: &mut [f32], a: &[f32], b: &[f32], k: f32) {
    let n = cur.len() / L * L;
    let vk = _mm512_set1_ps(k);
    for i in (0..n).step_by(L) {
        // SAFETY: as in `update53`.
        unsafe {
            let va = _mm512_loadu_ps(a[i..i + L].as_ptr());
            let vb = _mm512_loadu_ps(b[i..i + L].as_ptr());
            let c = cur[i..i + L].as_mut_ptr();
            // c - k * (a + b), rounded step by step (no FMA), as in scalar.
            let t = _mm512_mul_ps(vk, _mm512_add_ps(va, vb));
            _mm512_storeu_ps(c, _mm512_sub_ps(_mm512_loadu_ps(c), t));
        }
    }
    super::scalar::lift97(&mut cur[n..], &a[n..], &b[n..], k);
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn scale97(cur: &mut [f32], k: f32) {
    let n = cur.len() / L * L;
    let vk = _mm512_set1_ps(k);
    for i in (0..n).step_by(L) {
        // SAFETY: L elements remain from `i`.
        unsafe {
            let c = cur[i..i + L].as_mut_ptr();
            _mm512_storeu_ps(c, _mm512_mul_ps(_mm512_loadu_ps(c), vk));
        }
    }
    super::scalar::scale97(&mut cur[n..], k);
}
