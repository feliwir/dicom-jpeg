//! AVX-512 float to integer conversion: 16 samples per vector.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 16;

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn to_u16(src: &[f32], scale: f32, top: u16, dst: &mut [u16]) {
    let (vs, vtop) = (_mm512_set1_ps(scale), _mm512_set1_ps(top as f32));
    let (zero, half, one) = (
        _mm512_setzero_ps(),
        _mm512_set1_ps(0.5),
        _mm512_set1_epi32(1),
    );
    let (s, s_rest) = src.as_chunks::<L>();
    let (d, d_rest) = dst.as_chunks_mut::<L>();
    for (s, d) in s.iter().zip(d) {
        // SAFETY: `s` holds L f32 and `d` L u16; unaligned loads and stores
        // are allowed.
        unsafe {
            let x = _mm512_mul_ps(_mm512_loadu_ps(s.as_ptr()), vs);
            // NaN -> 0, then clamp to the (exact) bounds.
            let x = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask::<_CMP_ORD_Q>(x, x), x);
            let x = _mm512_min_ps(_mm512_max_ps(x, zero), vtop);
            // Round half up (x >= 0): truncate, then add 1 where the exact
            // fraction is at least 0.5.
            let t = _mm512_cvttps_epi32(x);
            let up =
                _mm512_cmp_ps_mask::<_CMP_GE_OQ>(_mm512_sub_ps(x, _mm512_cvtepi32_ps(t)), half);
            let r = _mm512_mask_add_epi32(t, up, t, one);
            // 0..=65535: truncating to 16 bits is exact.
            _mm256_storeu_si256(d.as_mut_ptr().cast(), _mm512_cvtepi32_epi16(r));
        }
    }
    super::scalar::to_u16(s_rest, scale, top, d_rest);
}
