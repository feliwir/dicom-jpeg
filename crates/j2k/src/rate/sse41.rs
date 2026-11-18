//! SSE4.1 squared quantisation errors: 2 f64 per vector.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::SATURATE;

/// Lanes per vector.
const L: usize = 2;

#[target_feature(enable = "sse4.1")]
pub(super) fn error_terms(a: &[f64], d: f64, out: &mut [f64]) {
    let (vd, half, zero, sat) = (
        _mm_set1_pd(d),
        _mm_set1_pd(0.5),
        _mm_setzero_pd(),
        _mm_set1_pd(SATURATE),
    );
    let (s, s_rest) = a.as_chunks::<L>();
    let (o, o_rest) = out.as_chunks_mut::<L>();
    for (s, o) in s.iter().zip(o) {
        // SAFETY: `s` and `o` hold L f64; unaligned loads and stores are
        // allowed.
        unsafe {
            let a = _mm_loadu_pd(s.as_ptr());
            let x = _mm_div_pd(a, vd);
            // `x as i64 as f64`: truncation, saturating at 2^63.
            let q = _mm_min_pd(
                _mm_round_pd::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(x),
                sat,
            );
            let rec = _mm_mul_pd(_mm_add_pd(q, half), vd);
            let rec = _mm_andnot_pd(_mm_cmpeq_pd(q, zero), rec);
            let e = _mm_sub_pd(a, rec);
            _mm_storeu_pd(o.as_mut_ptr(), _mm_mul_pd(e, e));
        }
    }
    super::scalar::error_terms(s_rest, d, o_rest);
}
