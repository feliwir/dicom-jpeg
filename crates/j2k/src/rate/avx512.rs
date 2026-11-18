//! AVX-512 squared quantisation errors: 8 f64 per vector (no FMA).
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::SATURATE;

/// Lanes per vector.
const L: usize = 8;

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn error_terms(a: &[f64], d: f64, out: &mut [f64]) {
    let (vd, half, zero, sat) = (
        _mm512_set1_pd(d),
        _mm512_set1_pd(0.5),
        _mm512_setzero_pd(),
        _mm512_set1_pd(SATURATE),
    );
    let (s, s_rest) = a.as_chunks::<L>();
    let (o, o_rest) = out.as_chunks_mut::<L>();
    for (s, o) in s.iter().zip(o) {
        // SAFETY: `s` and `o` hold L f64; unaligned loads and stores are
        // allowed.
        unsafe {
            let a = _mm512_loadu_pd(s.as_ptr());
            let x = _mm512_div_pd(a, vd);
            // `x as i64 as f64`: truncation, saturating at 2^63.
            let q = _mm512_min_pd(
                _mm512_roundscale_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(x),
                sat,
            );
            let rec = _mm512_mul_pd(_mm512_add_pd(q, half), vd);
            let rec = _mm512_maskz_mov_pd(_mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(q, zero), rec);
            let e = _mm512_sub_pd(a, rec);
            _mm512_storeu_pd(o.as_mut_ptr(), _mm512_mul_pd(e, e));
        }
    }
    super::scalar::error_terms(s_rest, d, o_rest);
}
