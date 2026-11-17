//! AVX2 RGB -> YCbCr, 8 samples (one vector of 8 x i32) per step.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

/// Y, Cb, Cr of one vector of R, G, B (i32 lanes).
#[inline]
#[target_feature(enable = "avx2")]
fn convert(r: __m256i, g: __m256i, b: __m256i, offset: i32) -> [__m256i; 3] {
    let k = |v| _mm256_set1_epi32(v);
    let mul = |a, c| _mm256_mullo_epi32(a, k(c));
    let add = |a, b| _mm256_add_epi32(a, b);
    let sub = |a, b| _mm256_sub_epi32(a, b);
    let y = add(add(add(mul(r, YR), mul(g, YG)), mul(b, YB)), k(HALF));
    let bias = k(offset + HALF - 1);
    let cb = add(
        add(
            sub(sub(_mm256_setzero_si256(), mul(r, CBR)), mul(g, CBG)),
            mul(b, CBB),
        ),
        bias,
    );
    let cr = add(sub(sub(mul(r, CRR), mul(g, CRG)), mul(b, CRB)), bias);
    [y, cb, cr].map(|v| _mm256_srai_epi32::<16>(v))
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn rgb_to_ycbcr(precision: u8, a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let offset = (1i32 << (precision - 1)) << SCALEBITS;
    let n = a.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        // SAFETY: 8 samples remain from `i` in every row (`i + 8 <= n`).
        let load = |s: &[u16]| unsafe {
            _mm256_cvtepu16_epi32(_mm_loadu_si128(s[i..i + 8].as_ptr().cast()))
        };
        let (r, g, bl) = (load(a), load(b), load(c));
        let [y, cb, cr] = convert(r, g, bl, offset);
        for (d, v) in [&mut *a, &mut *b, &mut *c].into_iter().zip([y, cb, cr]) {
            // Results are in 0..2^precision: the saturating pack is exact.
            let p = _mm256_packus_epi32(v, v);
            let v = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b1000>(p));
            // SAFETY: 8 samples remain from `i`.
            unsafe { _mm_storeu_si128(d[i..i + 8].as_mut_ptr().cast(), v) };
        }
    }
    super::sse41::rgb_to_ycbcr(precision, &mut a[n..], &mut b[n..], &mut c[n..]);
}
