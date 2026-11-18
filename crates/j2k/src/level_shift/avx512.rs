//! AVX-512 level shift: 16 samples per vector.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 16;

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn shift_int(data: &mut [i32], a: i32, b: i32, shift: i32) {
    let (va, vb, vs) = (
        _mm512_set1_epi32(a),
        _mm512_set1_epi32(b),
        _mm512_set1_epi32(shift),
    );
    let (chunks, rest) = data.as_chunks_mut::<L>();
    for c in chunks {
        // SAFETY: `c` holds L i32; unaligned loads and stores are allowed.
        unsafe {
            let v = _mm512_loadu_si512(c.as_ptr().cast());
            let v = _mm512_min_epi32(_mm512_max_epi32(v, va), vb);
            _mm512_storeu_si512(c.as_mut_ptr().cast(), _mm512_add_epi32(v, vs));
        }
    }
    super::scalar::shift_int(rest, a, b, shift);
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn shift_float(src: &[f32], dst: &mut [i32], a: i32, b: i32, shift: i32) {
    let (fa, fb) = (_mm512_set1_ps(a as f32), _mm512_set1_ps(b as f32));
    let (half, neg_half) = (_mm512_set1_ps(0.5), _mm512_set1_ps(-0.5));
    let (one, vs) = (_mm512_set1_epi32(1), _mm512_set1_epi32(shift));
    let (s, s_rest) = src.as_chunks::<L>();
    let (d, d_rest) = dst.as_chunks_mut::<L>();
    for (s, d) in s.iter().zip(d) {
        // SAFETY: `s` and `d` hold L elements; unaligned loads and stores
        // are allowed.
        unsafe {
            let v = _mm512_loadu_ps(s.as_ptr());
            // NaN -> 0, then clamp (exact bounds).
            let v = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask::<_CMP_ORD_Q>(v, v), v);
            let v = _mm512_min_ps(_mm512_max_ps(v, fa), fb);
            // Round half away from zero: truncate, then step by the
            // (exact) fraction.
            let t = _mm512_cvttps_epi32(v);
            let f = _mm512_sub_ps(v, _mm512_cvtepi32_ps(t));
            let up = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(f, half);
            let down = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(f, neg_half);
            let r = _mm512_mask_add_epi32(t, up, t, one);
            let r = _mm512_mask_sub_epi32(r, down, r, one);
            _mm512_storeu_si512(d.as_mut_ptr().cast(), _mm512_add_epi32(r, vs));
        }
    }
    super::scalar::shift_float(s_rest, d_rest, a, b, shift);
}
