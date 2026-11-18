//! SSE4.1 level shift: 4 samples per vector.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 4;

#[target_feature(enable = "sse4.1")]
pub(super) fn shift_int(data: &mut [i32], a: i32, b: i32, shift: i32) {
    let (va, vb, vs) = (_mm_set1_epi32(a), _mm_set1_epi32(b), _mm_set1_epi32(shift));
    let (chunks, rest) = data.as_chunks_mut::<L>();
    for c in chunks {
        // SAFETY: `c` holds L i32; unaligned loads and stores are allowed.
        unsafe {
            let v = _mm_loadu_si128(c.as_ptr().cast());
            let v = _mm_min_epi32(_mm_max_epi32(v, va), vb);
            _mm_storeu_si128(c.as_mut_ptr().cast(), _mm_add_epi32(v, vs));
        }
    }
    super::scalar::shift_int(rest, a, b, shift);
}

#[target_feature(enable = "sse4.1")]
pub(super) fn shift_float(src: &[f32], dst: &mut [i32], a: i32, b: i32, shift: i32) {
    let (fa, fb) = (_mm_set1_ps(a as f32), _mm_set1_ps(b as f32));
    let (half, neg_half) = (_mm_set1_ps(0.5), _mm_set1_ps(-0.5));
    let vs = _mm_set1_epi32(shift);
    let (s, s_rest) = src.as_chunks::<L>();
    let (d, d_rest) = dst.as_chunks_mut::<L>();
    for (s, d) in s.iter().zip(d) {
        // SAFETY: `s` and `d` hold L elements; unaligned loads and stores
        // are allowed.
        unsafe {
            let v = _mm_loadu_ps(s.as_ptr());
            // NaN -> 0, then clamp (exact bounds).
            let v = _mm_and_ps(v, _mm_cmpord_ps(v, v));
            let v = _mm_min_ps(_mm_max_ps(v, fa), fb);
            // Round half away from zero: truncate, then step by the
            // (exact) fraction. Masks are -1 where true.
            let t = _mm_cvttps_epi32(v);
            let f = _mm_sub_ps(v, _mm_cvtepi32_ps(t));
            let up = _mm_castps_si128(_mm_cmpge_ps(f, half));
            let down = _mm_castps_si128(_mm_cmple_ps(f, neg_half));
            let r = _mm_add_epi32(_mm_sub_epi32(t, up), down);
            _mm_storeu_si128(d.as_mut_ptr().cast(), _mm_add_epi32(r, vs));
        }
    }
    super::scalar::shift_float(s_rest, d_rest, a, b, shift);
}
