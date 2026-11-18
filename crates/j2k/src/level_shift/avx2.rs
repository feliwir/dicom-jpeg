//! AVX2 level shift: 8 samples per vector.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 8;

#[target_feature(enable = "avx2,fma")]
pub(super) fn shift_int(data: &mut [i32], a: i32, b: i32, shift: i32) {
    let (va, vb, vs) = (
        _mm256_set1_epi32(a),
        _mm256_set1_epi32(b),
        _mm256_set1_epi32(shift),
    );
    let (chunks, rest) = data.as_chunks_mut::<L>();
    for c in chunks {
        // SAFETY: `c` holds L i32; unaligned loads and stores are allowed.
        unsafe {
            let v = _mm256_loadu_si256(c.as_ptr().cast());
            let v = _mm256_min_epi32(_mm256_max_epi32(v, va), vb);
            _mm256_storeu_si256(c.as_mut_ptr().cast(), _mm256_add_epi32(v, vs));
        }
    }
    super::scalar::shift_int(rest, a, b, shift);
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn shift_float(src: &[f32], dst: &mut [i32], a: i32, b: i32, shift: i32) {
    let (fa, fb) = (_mm256_set1_ps(a as f32), _mm256_set1_ps(b as f32));
    let (half, neg_half) = (_mm256_set1_ps(0.5), _mm256_set1_ps(-0.5));
    let vs = _mm256_set1_epi32(shift);
    let (s, s_rest) = src.as_chunks::<L>();
    let (d, d_rest) = dst.as_chunks_mut::<L>();
    for (s, d) in s.iter().zip(d) {
        // SAFETY: `s` and `d` hold L elements; unaligned loads and stores
        // are allowed.
        unsafe {
            let v = _mm256_loadu_ps(s.as_ptr());
            // NaN -> 0, then clamp (exact bounds).
            let v = _mm256_and_ps(v, _mm256_cmp_ps::<_CMP_ORD_Q>(v, v));
            let v = _mm256_min_ps(_mm256_max_ps(v, fa), fb);
            // Round half away from zero: truncate, then step by the
            // (exact) fraction. Masks are -1 where true.
            let t = _mm256_cvttps_epi32(v);
            let f = _mm256_sub_ps(v, _mm256_cvtepi32_ps(t));
            let up = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GE_OQ>(f, half));
            let down = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OQ>(f, neg_half));
            let r = _mm256_add_epi32(_mm256_sub_epi32(t, up), down);
            _mm256_storeu_si256(d.as_mut_ptr().cast(), _mm256_add_epi32(r, vs));
        }
    }
    super::scalar::shift_float(s_rest, d_rest, a, b, shift);
}
