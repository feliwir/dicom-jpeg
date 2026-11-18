//! SSE4.1 float to integer conversion: 4 samples per vector.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 4;

#[target_feature(enable = "sse4.1")]
pub(super) fn to_u16(src: &[f32], scale: f32, top: u16, dst: &mut [u16]) {
    let (vs, vtop) = (_mm_set1_ps(scale), _mm_set1_ps(top as f32));
    let (zero, half) = (_mm_setzero_ps(), _mm_set1_ps(0.5));
    let (s, s_rest) = src.as_chunks::<L>();
    let (d, d_rest) = dst.as_chunks_mut::<L>();
    for (s, d) in s.iter().zip(d) {
        // SAFETY: `s` holds L f32 and `d` L u16; unaligned loads and stores
        // are allowed.
        unsafe {
            let x = _mm_mul_ps(_mm_loadu_ps(s.as_ptr()), vs);
            // NaN -> 0, then clamp to the (exact) bounds.
            let x = _mm_and_ps(x, _mm_cmpord_ps(x, x));
            let x = _mm_min_ps(_mm_max_ps(x, zero), vtop);
            // Round half up (x >= 0): truncate, then add 1 where the exact
            // fraction is at least 0.5 (the mask is -1 there).
            let t = _mm_cvttps_epi32(x);
            let up = _mm_castps_si128(_mm_cmpge_ps(_mm_sub_ps(x, _mm_cvtepi32_ps(t)), half));
            let r = _mm_sub_epi32(t, up);
            // 0..=65535: the unsigned saturating pack is exact.
            let p = _mm_packus_epi32(r, r);
            _mm_storel_epi64(d.as_mut_ptr().cast(), p);
        }
    }
    super::scalar::to_u16(s_rest, scale, top, d_rest);
}
