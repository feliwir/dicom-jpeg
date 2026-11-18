//! AVX2 lifting kernels: 8 lanes (FMA is not used, to match the scalar rounding).
//!
//! Each kernel walks whole vectors with `as_chunks` (no per-iteration
//! bounds checks) and finishes the remainder with the scalar kernel.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 8;

#[target_feature(enable = "avx2,fma")]
pub(super) fn update53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    // x -= (a + b + 2) >> 2
    let two = _mm256_set1_epi32(2);
    let (c, c_rest) = cur.as_chunks_mut::<L>();
    let ((a, a_rest), (b, b_rest)) = (a.as_chunks::<L>(), b.as_chunks::<L>());
    for ((c, a), b) in c.iter_mut().zip(a).zip(b) {
        // SAFETY: every chunk holds L elements; unaligned loads and stores
        // are allowed.
        unsafe {
            let (va, vb) = (
                _mm256_loadu_si256(a.as_ptr().cast()),
                _mm256_loadu_si256(b.as_ptr().cast()),
            );
            let s = _mm256_srai_epi32::<2>(_mm256_add_epi32(_mm256_add_epi32(va, vb), two));
            _mm256_storeu_si256(
                c.as_mut_ptr().cast(),
                _mm256_sub_epi32(_mm256_loadu_si256(c.as_ptr().cast()), s),
            );
        }
    }
    super::scalar::update53(c_rest, a_rest, b_rest);
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn predict53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    // x += (a + b) >> 1
    let (c, c_rest) = cur.as_chunks_mut::<L>();
    let ((a, a_rest), (b, b_rest)) = (a.as_chunks::<L>(), b.as_chunks::<L>());
    for ((c, a), b) in c.iter_mut().zip(a).zip(b) {
        // SAFETY: as in `update53`.
        unsafe {
            let (va, vb) = (
                _mm256_loadu_si256(a.as_ptr().cast()),
                _mm256_loadu_si256(b.as_ptr().cast()),
            );
            let s = _mm256_srai_epi32::<1>(_mm256_add_epi32(va, vb));
            _mm256_storeu_si256(
                c.as_mut_ptr().cast(),
                _mm256_add_epi32(_mm256_loadu_si256(c.as_ptr().cast()), s),
            );
        }
    }
    super::scalar::predict53(c_rest, a_rest, b_rest);
}

#[cfg(feature = "encode")]
#[target_feature(enable = "avx2,fma")]
pub(super) fn apredict53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    // x -= (a + b) >> 1
    let (c, c_rest) = cur.as_chunks_mut::<L>();
    let ((a, a_rest), (b, b_rest)) = (a.as_chunks::<L>(), b.as_chunks::<L>());
    for ((c, a), b) in c.iter_mut().zip(a).zip(b) {
        // SAFETY: as in `update53`.
        unsafe {
            let (va, vb) = (
                _mm256_loadu_si256(a.as_ptr().cast()),
                _mm256_loadu_si256(b.as_ptr().cast()),
            );
            let s = _mm256_srai_epi32::<1>(_mm256_add_epi32(va, vb));
            _mm256_storeu_si256(
                c.as_mut_ptr().cast(),
                _mm256_sub_epi32(_mm256_loadu_si256(c.as_ptr().cast()), s),
            );
        }
    }
    super::scalar::apredict53(c_rest, a_rest, b_rest);
}

#[cfg(feature = "encode")]
#[target_feature(enable = "avx2,fma")]
pub(super) fn aupdate53(cur: &mut [i32], a: &[i32], b: &[i32]) {
    // x += (a + b + 2) >> 2
    let two = _mm256_set1_epi32(2);
    let (c, c_rest) = cur.as_chunks_mut::<L>();
    let ((a, a_rest), (b, b_rest)) = (a.as_chunks::<L>(), b.as_chunks::<L>());
    for ((c, a), b) in c.iter_mut().zip(a).zip(b) {
        // SAFETY: as in `update53`.
        unsafe {
            let (va, vb) = (
                _mm256_loadu_si256(a.as_ptr().cast()),
                _mm256_loadu_si256(b.as_ptr().cast()),
            );
            let s = _mm256_srai_epi32::<2>(_mm256_add_epi32(_mm256_add_epi32(va, vb), two));
            _mm256_storeu_si256(
                c.as_mut_ptr().cast(),
                _mm256_add_epi32(_mm256_loadu_si256(c.as_ptr().cast()), s),
            );
        }
    }
    super::scalar::aupdate53(c_rest, a_rest, b_rest);
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn lift97(cur: &mut [f32], a: &[f32], b: &[f32], k: f32) {
    // x -= k * (a + b), rounded step by step (no FMA), as in scalar
    let vk = _mm256_set1_ps(k);
    let (c, c_rest) = cur.as_chunks_mut::<L>();
    let ((a, a_rest), (b, b_rest)) = (a.as_chunks::<L>(), b.as_chunks::<L>());
    for ((c, a), b) in c.iter_mut().zip(a).zip(b) {
        // SAFETY: as in `update53`.
        unsafe {
            let (va, vb) = (_mm256_loadu_ps(a.as_ptr()), _mm256_loadu_ps(b.as_ptr()));
            let t = _mm256_mul_ps(vk, _mm256_add_ps(va, vb));
            _mm256_storeu_ps(
                c.as_mut_ptr(),
                _mm256_sub_ps(_mm256_loadu_ps(c.as_ptr()), t),
            );
        }
    }
    super::scalar::lift97(c_rest, a_rest, b_rest, k);
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn scale97(cur: &mut [f32], k: f32) {
    let vk = _mm256_set1_ps(k);
    let (c, c_rest) = cur.as_chunks_mut::<L>();
    for c in c {
        // SAFETY: every chunk holds L f32.
        unsafe {
            _mm256_storeu_ps(
                c.as_mut_ptr(),
                _mm256_mul_ps(_mm256_loadu_ps(c.as_ptr()), vk),
            )
        };
    }
    super::scalar::scale97(c_rest, k);
}
