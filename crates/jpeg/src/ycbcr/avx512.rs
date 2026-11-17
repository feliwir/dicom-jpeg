//! AVX-512 YCbCr -> RGB, 16 samples (one vector of 16 x i32) per step.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn ycbcr_to_rgb(
    y: &[u16],
    cb: &[u16],
    cr: &[u16],
    precision: u8,
    r: &mut [u16],
    g: &mut [u16],
    b: &mut [u16],
) {
    let max = _mm512_set1_epi32((1 << precision) - 1);
    let center = _mm512_set1_epi32(1 << (precision - 1));
    let half = _mm512_set1_epi32(ONE_HALF);
    let mul = |a, k| _mm512_mullo_epi32(a, _mm512_set1_epi32(k));
    let n = y.len() / 16 * 16;
    for i in (0..n).step_by(16) {
        // SAFETY: 16 samples remain from `i` in every row.
        let load = |s: &[u16]| unsafe {
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(s[i..].as_ptr().cast()))
        };
        let (yv, cbv, crv) = (load(y), load(cb), load(cr));
        let cbv = _mm512_sub_epi32(cbv, center);
        let crv = _mm512_sub_epi32(crv, center);
        let rv = _mm512_srai_epi32::<16>(_mm512_add_epi32(mul(crv, FR), half));
        let gv = _mm512_srai_epi32::<16>(_mm512_sub_epi32(
            _mm512_add_epi32(mul(cbv, -FGB), half),
            mul(crv, FGR),
        ));
        let bv = _mm512_srai_epi32::<16>(_mm512_add_epi32(mul(cbv, FB), half));
        for (d, v) in [&mut *r, &mut *g, &mut *b].into_iter().zip([rv, gv, bv]) {
            let v = _mm512_min_epi32(
                _mm512_max_epi32(_mm512_add_epi32(yv, v), _mm512_setzero_si512()),
                max,
            );
            // Values are in 0..=max: truncating to 16 bits is exact.
            // SAFETY: 16 samples remain from `i`.
            unsafe { _mm256_storeu_si256(d[i..].as_mut_ptr().cast(), _mm512_cvtepi32_epi16(v)) };
        }
    }
    super::sse41::ycbcr_to_rgb(
        &y[n..],
        &cb[n..],
        &cr[n..],
        precision,
        &mut r[n..],
        &mut g[n..],
        &mut b[n..],
    );
}
