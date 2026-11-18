//! AVX2 float to integer conversion: 8 samples per vector.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 8;

#[target_feature(enable = "avx2,fma")]
pub(super) fn to_u16(src: &[f32], scale: f32, top: u16, dst: &mut [u16]) {
    let (vs, vtop) = (_mm256_set1_ps(scale), _mm256_set1_ps(top as f32));
    let (zero, half) = (_mm256_setzero_ps(), _mm256_set1_ps(0.5));
    let (s, s_rest) = src.as_chunks::<L>();
    let (d, d_rest) = dst.as_chunks_mut::<L>();
    for (s, d) in s.iter().zip(d) {
        // SAFETY: `s` holds L f32 and `d` L u16; unaligned loads and stores
        // are allowed.
        unsafe {
            let x = _mm256_mul_ps(_mm256_loadu_ps(s.as_ptr()), vs);
            // NaN -> 0, then clamp to the (exact) bounds.
            let x = _mm256_and_ps(x, _mm256_cmp_ps::<_CMP_ORD_Q>(x, x));
            let x = _mm256_min_ps(_mm256_max_ps(x, zero), vtop);
            // Round half up (x >= 0): truncate, then add 1 where the exact
            // fraction is at least 0.5 (the mask is -1 there).
            let t = _mm256_cvttps_epi32(x);
            let up = _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GE_OQ>(
                _mm256_sub_ps(x, _mm256_cvtepi32_ps(t)),
                half,
            ));
            let r = _mm256_sub_epi32(t, up);
            // 0..=65535: the unsigned saturating pack is exact; it works per
            // 128-bit lane, so take the low half after reordering.
            let p = _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi32(r, r));
            _mm_storeu_si128(d.as_mut_ptr().cast(), _mm256_castsi256_si128(p));
        }
    }
    super::scalar::to_u16(s_rest, scale, top, d_rest);
}
