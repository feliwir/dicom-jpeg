//! SSE4.1 RGB -> YCbCr, 4 samples (one vector of 4 x i32) per step.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

/// Y, Cb, Cr of one vector of R, G, B (i32 lanes).
#[inline]
#[target_feature(enable = "sse4.1")]
fn convert(r: __m128i, g: __m128i, b: __m128i, offset: i32) -> [__m128i; 3] {
    let k = |v| _mm_set1_epi32(v);
    let mul = |a, c| _mm_mullo_epi32(a, k(c));
    let add = |a, b| _mm_add_epi32(a, b);
    let sub = |a, b| _mm_sub_epi32(a, b);
    let y = add(add(add(mul(r, YR), mul(g, YG)), mul(b, YB)), k(HALF));
    let bias = k(offset + HALF - 1);
    let cb = add(
        add(
            sub(sub(_mm_setzero_si128(), mul(r, CBR)), mul(g, CBG)),
            mul(b, CBB),
        ),
        bias,
    );
    let cr = add(sub(sub(mul(r, CRR), mul(g, CRG)), mul(b, CRB)), bias);
    [y, cb, cr].map(|v| _mm_srai_epi32::<16>(v))
}

#[target_feature(enable = "sse4.1")]
pub(super) fn rgb_to_ycbcr(precision: u8, a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let offset = (1i32 << (precision - 1)) << SCALEBITS;
    let n = a.len() / 4 * 4;
    for i in (0..n).step_by(4) {
        // SAFETY: 4 samples remain from `i` in every row (`i + 4 <= n`).
        let load =
            |s: &[u16]| unsafe { _mm_cvtepu16_epi32(_mm_loadl_epi64(s[i..i + 4].as_ptr().cast())) };
        let (r, g, bl) = (load(a), load(b), load(c));
        let [y, cb, cr] = convert(r, g, bl, offset);
        for (d, v) in [&mut *a, &mut *b, &mut *c].into_iter().zip([y, cb, cr]) {
            // Results are in 0..2^precision: the saturating pack is exact.
            let v = _mm_packus_epi32(v, v);
            // SAFETY: 4 samples remain from `i`.
            unsafe { _mm_storel_epi64(d[i..i + 4].as_mut_ptr().cast(), v) };
        }
    }
    super::scalar::rgb_to_ycbcr(precision, &mut a[n..], &mut b[n..], &mut c[n..]);
}
