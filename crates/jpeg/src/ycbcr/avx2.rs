//! AVX2 YCbCr -> RGB, 16 samples (two vectors of 8 x i32) per step.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

/// One vector of 8 samples: (r, g, b) as i32, clamped.
#[inline]
#[target_feature(enable = "avx2")]
fn convert(y: __m256i, cb: __m256i, cr: __m256i, center: __m256i, max: __m256i) -> [__m256i; 3] {
    let cbv = _mm256_sub_epi32(cb, center);
    let crv = _mm256_sub_epi32(cr, center);
    let half = _mm256_set1_epi32(ONE_HALF);
    let mul = |a, k| _mm256_mullo_epi32(a, _mm256_set1_epi32(k));
    let r = _mm256_srai_epi32::<16>(_mm256_add_epi32(mul(crv, FR), half));
    let g = _mm256_srai_epi32::<16>(_mm256_sub_epi32(
        _mm256_add_epi32(mul(cbv, -FGB), half),
        mul(crv, FGR),
    ));
    let b = _mm256_srai_epi32::<16>(_mm256_add_epi32(mul(cbv, FB), half));
    let clamp = |v| {
        _mm256_min_epi32(
            _mm256_max_epi32(_mm256_add_epi32(y, v), _mm256_setzero_si256()),
            max,
        )
    };
    [clamp(r), clamp(g), clamp(b)]
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn ycbcr_to_rgb(
    y: &[u16],
    cb: &[u16],
    cr: &[u16],
    precision: u8,
    r: &mut [u16],
    g: &mut [u16],
    b: &mut [u16],
) {
    let max = _mm256_set1_epi32((1 << precision) - 1);
    let center = _mm256_set1_epi32(1 << (precision - 1));
    let n = y.len() / 16 * 16;
    for i in (0..n).step_by(16) {
        // SAFETY: 16 samples (two halves of 8) remain from `i` in every row.
        let load = |s: &[u16], o: usize| unsafe {
            _mm256_cvtepu16_epi32(_mm_loadu_si128(s[i + o..].as_ptr().cast()))
        };
        let a = convert(load(y, 0), load(cb, 0), load(cr, 0), center, max);
        let c = convert(load(y, 8), load(cb, 8), load(cr, 8), center, max);
        for (k, d) in [&mut *r, &mut *g, &mut *b].into_iter().enumerate() {
            // packus works per 128-bit lane: [a0..4, c0..4, a4..8, c4..8].
            let v = _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi32(a[k], c[k]));
            // SAFETY: 16 samples remain from `i`.
            unsafe { _mm256_storeu_si256(d[i..].as_mut_ptr().cast(), v) };
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
