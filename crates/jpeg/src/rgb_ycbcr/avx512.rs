//! AVX-512 RGB -> YCbCr, 16 samples (one vector of 16 x i32) per step.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

/// Y, Cb, Cr of one vector of R, G, B (i32 lanes).
#[inline]
#[target_feature(enable = "avx512f")]
fn convert(r: __m512i, g: __m512i, b: __m512i, offset: i32) -> [__m512i; 3] {
    let k = |v| _mm512_set1_epi32(v);
    let mul = |a, c| _mm512_mullo_epi32(a, k(c));
    let add = |a, b| _mm512_add_epi32(a, b);
    let sub = |a, b| _mm512_sub_epi32(a, b);
    let y = add(add(add(mul(r, YR), mul(g, YG)), mul(b, YB)), k(HALF));
    let bias = k(offset + HALF - 1);
    let cb = add(
        add(
            sub(sub(_mm512_setzero_si512(), mul(r, CBR)), mul(g, CBG)),
            mul(b, CBB),
        ),
        bias,
    );
    let cr = add(sub(sub(mul(r, CRR), mul(g, CRG)), mul(b, CRB)), bias);
    [y, cb, cr].map(|v| _mm512_srai_epi32::<16>(v))
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn rgb_to_ycbcr(precision: u8, a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let offset = (1i32 << (precision - 1)) << SCALEBITS;
    let n = a.len() / 16 * 16;
    for i in (0..n).step_by(16) {
        // SAFETY: 16 samples remain from `i` in every row (`i + 16 <= n`).
        let load = |s: &[u16]| unsafe {
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(s[i..i + 16].as_ptr().cast()))
        };
        let (r, g, bl) = (load(a), load(b), load(c));
        let [y, cb, cr] = convert(r, g, bl, offset);
        for (d, v) in [&mut *a, &mut *b, &mut *c].into_iter().zip([y, cb, cr]) {
            // Results are in 0..2^precision: truncating to 16 bits is exact.
            // SAFETY: 16 samples remain from `i`.
            unsafe {
                _mm256_storeu_si256(d[i..i + 16].as_mut_ptr().cast(), _mm512_cvtepi32_epi16(v))
            };
        }
    }
    super::sse41::rgb_to_ycbcr(precision, &mut a[n..], &mut b[n..], &mut c[n..]);
}
