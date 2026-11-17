//! SSE4.1 YCbCr -> RGB, 8 samples (two vectors of 4 x i32) per step.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

/// One vector of 4 samples: (r, g, b) as i32, clamped.
#[inline]
#[target_feature(enable = "sse4.1")]
fn convert(y: __m128i, cb: __m128i, cr: __m128i, center: __m128i, max: __m128i) -> [__m128i; 3] {
    let cbv = _mm_sub_epi32(cb, center);
    let crv = _mm_sub_epi32(cr, center);
    let half = _mm_set1_epi32(ONE_HALF);
    let mul = |a, k| _mm_mullo_epi32(a, _mm_set1_epi32(k));
    let r = _mm_srai_epi32::<16>(_mm_add_epi32(mul(crv, FR), half));
    let g = _mm_srai_epi32::<16>(_mm_sub_epi32(
        _mm_add_epi32(mul(cbv, -FGB), half),
        mul(crv, FGR),
    ));
    let b = _mm_srai_epi32::<16>(_mm_add_epi32(mul(cbv, FB), half));
    let clamp = |v| _mm_min_epi32(_mm_max_epi32(_mm_add_epi32(y, v), _mm_setzero_si128()), max);
    [clamp(r), clamp(g), clamp(b)]
}

#[target_feature(enable = "sse4.1")]
pub(super) fn ycbcr_to_rgb(
    y: &[u16],
    cb: &[u16],
    cr: &[u16],
    precision: u8,
    r: &mut [u16],
    g: &mut [u16],
    b: &mut [u16],
) {
    let max = _mm_set1_epi32((1 << precision) - 1);
    let center = _mm_set1_epi32(1 << (precision - 1));
    let n = y.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        // SAFETY: 8 samples remain from `i` in every row (`i + 8 <= n`).
        let (yv, cbv, crv) = unsafe {
            (
                _mm_loadu_si128(y[i..].as_ptr().cast()),
                _mm_loadu_si128(cb[i..].as_ptr().cast()),
                _mm_loadu_si128(cr[i..].as_ptr().cast()),
            )
        };
        let lo = |v| _mm_cvtepu16_epi32(v);
        let hi = |v| _mm_cvtepu16_epi32(_mm_srli_si128::<8>(v));
        let a = convert(lo(yv), lo(cbv), lo(crv), center, max);
        let c = convert(hi(yv), hi(cbv), hi(crv), center, max);
        for (k, d) in [&mut *r, &mut *g, &mut *b].into_iter().enumerate() {
            // Values are in 0..=max, so the saturating pack is exact.
            let v = _mm_packus_epi32(a[k], c[k]);
            // SAFETY: 8 samples remain from `i`.
            unsafe { _mm_storeu_si128(d[i..].as_mut_ptr().cast(), v) };
        }
    }
    super::scalar::ycbcr_to_rgb(
        &y[n..],
        &cb[n..],
        &cr[n..],
        precision,
        &mut r[n..],
        &mut g[n..],
        &mut b[n..],
    );
}
