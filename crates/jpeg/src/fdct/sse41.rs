//! SSE4.1 8-bit forward DCT and quantisation: each row of a block is two
//! vectors of 4 x i32 (the left and right half), transposed as four 4 x 4
//! blocks.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

/// `(x + 2^(N-1)) >> N`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn descale<const N: i32>(x: __m128i) -> __m128i {
    _mm_srai_epi32::<N>(_mm_add_epi32(x, _mm_set1_epi32(1 << (N - 1))))
}

/// The 1-D forward DCT of `d` (each lane independently): outputs 0 and 4
/// before their pass-specific scaling, the others before descaling.
#[inline]
#[target_feature(enable = "sse4.1")]
fn fdct_1d(d: &[__m128i; 8]) -> [__m128i; 8] {
    let add = |a, b| _mm_add_epi32(a, b);
    let sub = |a, b| _mm_sub_epi32(a, b);
    let tmp0 = add(d[0], d[7]);
    let tmp7 = sub(d[0], d[7]);
    let tmp1 = add(d[1], d[6]);
    let tmp6 = sub(d[1], d[6]);
    let tmp2 = add(d[2], d[5]);
    let tmp5 = sub(d[2], d[5]);
    let tmp3 = add(d[3], d[4]);
    let tmp4 = sub(d[3], d[4]);
    let tmp10 = add(tmp0, tmp3);
    let tmp13 = sub(tmp0, tmp3);
    let tmp11 = add(tmp1, tmp2);
    let tmp12 = sub(tmp1, tmp2);
    let z1 = _mm_mullo_epi32(add(tmp12, tmp13), _mm_set1_epi32(FIX_0_541196100));
    let o2 = add(z1, _mm_mullo_epi32(tmp13, _mm_set1_epi32(FIX_0_765366865)));
    let o6 = add(z1, _mm_mullo_epi32(tmp12, _mm_set1_epi32(-FIX_1_847759065)));
    let z1 = add(tmp4, tmp7);
    let z2 = add(tmp5, tmp6);
    let z3 = add(tmp4, tmp6);
    let z4 = add(tmp5, tmp7);
    let z5 = _mm_mullo_epi32(add(z3, z4), _mm_set1_epi32(FIX_1_175875602));
    let t4 = _mm_mullo_epi32(tmp4, _mm_set1_epi32(FIX_0_298631336));
    let t5 = _mm_mullo_epi32(tmp5, _mm_set1_epi32(FIX_2_053119869));
    let t6 = _mm_mullo_epi32(tmp6, _mm_set1_epi32(FIX_3_072711026));
    let t7 = _mm_mullo_epi32(tmp7, _mm_set1_epi32(FIX_1_501321110));
    let z1 = _mm_mullo_epi32(z1, _mm_set1_epi32(-FIX_0_899976223));
    let z2 = _mm_mullo_epi32(z2, _mm_set1_epi32(-FIX_2_562915447));
    let z3 = add(_mm_mullo_epi32(z3, _mm_set1_epi32(-FIX_1_961570560)), z5);
    let z4 = add(_mm_mullo_epi32(z4, _mm_set1_epi32(-FIX_0_390180644)), z5);
    [
        add(tmp10, tmp11),
        add(add(t7, z1), z4),
        o2,
        add(add(t6, z2), z3),
        sub(tmp10, tmp11),
        add(add(t5, z2), z4),
        o6,
        add(add(t4, z1), z3),
    ]
}

/// Transpose a 4 x 4 matrix of i32 held as 4 row vectors.
#[inline]
#[target_feature(enable = "sse4.1")]
fn transpose4(r: &[__m128i]) -> [__m128i; 4] {
    let t0 = _mm_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm_unpacklo_epi32(r[2], r[3]);
    let t2 = _mm_unpackhi_epi32(r[0], r[1]);
    let t3 = _mm_unpackhi_epi32(r[2], r[3]);
    [
        _mm_unpacklo_epi64(t0, t1),
        _mm_unpackhi_epi64(t0, t1),
        _mm_unpacklo_epi64(t2, t3),
        _mm_unpackhi_epi64(t2, t3),
    ]
}

/// Transpose an 8 x 8 matrix given as the left (columns 0..4) and right
/// (columns 4..8) halves of its rows; returns the transpose in that form.
#[inline]
#[target_feature(enable = "sse4.1")]
fn transpose(l: &[__m128i; 8], r: &[__m128i; 8]) -> ([__m128i; 8], [__m128i; 8]) {
    let (a, b, c, d) = (
        transpose4(&l[..4]),
        transpose4(&r[..4]),
        transpose4(&l[4..]),
        transpose4(&r[4..]),
    );
    (
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]],
        [c[0], c[1], c[2], c[3], d[0], d[1], d[2], d[3]],
    )
}

/// Pass 1 outputs (scaled by 2^2) of one half.
#[inline]
#[target_feature(enable = "sse4.1")]
fn pass1(t: &[__m128i; 8]) -> [__m128i; 8] {
    let o = fdct_1d(t);
    let mut w = [_mm_setzero_si128(); 8];
    for (k, w) in w.iter_mut().enumerate() {
        *w = if k % 4 == 0 {
            _mm_slli_epi32::<2>(o[k])
        } else {
            descale::<{ (CONST_BITS - 2) as i32 }>(o[k])
        };
    }
    w
}

/// Pass 2 outputs of one half.
#[inline]
#[target_feature(enable = "sse4.1")]
fn pass2(t: &[__m128i; 8]) -> [__m128i; 8] {
    let o = fdct_1d(t);
    let mut w = [_mm_setzero_si128(); 8];
    for (k, w) in w.iter_mut().enumerate() {
        *w = if k % 4 == 0 {
            descale::<2>(o[k])
        } else {
            descale::<{ (CONST_BITS + 2) as i32 }>(o[k])
        };
    }
    w
}

/// Quantise 4 coefficients with table entries `k..k + 4`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn quantize(t: &QuantTable, k: usize, c: __m128i) -> __m128i {
    // SAFETY: `k + 4 <= 64`.
    let (d, half) = unsafe {
        (
            _mm_loadu_ps(t.divf[k..k + 4].as_ptr()),
            _mm_loadu_si128(t.half[k..k + 4].as_ptr().cast()),
        )
    };
    let a = _mm_cvtepi32_ps(_mm_add_epi32(_mm_abs_epi32(c), half));
    let v = _mm_sign_epi32(_mm_cvttps_epi32(_mm_div_ps(a, d)), c);
    let max = _mm_set1_epi32(t.max);
    _mm_max_epi32(
        _mm_min_epi32(v, max),
        _mm_sub_epi32(_mm_setzero_si128(), max),
    )
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn block(src: &[u16], stride: usize, t: &QuantTable, out: &mut [i32; 64]) {
    let center = _mm_set1_epi32(128);
    let (mut l, mut r) = ([_mm_setzero_si128(); 8], [_mm_setzero_si128(); 8]);
    for k in 0..8 {
        // SAFETY: each row of the block holds 8 samples.
        let s = unsafe { _mm_loadu_si128(src[k * stride..k * stride + 8].as_ptr().cast()) };
        l[k] = _mm_sub_epi32(_mm_cvtepu16_epi32(s), center);
        r[k] = _mm_sub_epi32(_mm_cvtepu16_epi32(_mm_srli_si128::<8>(s)), center);
    }
    // Rows: lanes are rows 0..4 (left) and 4..8 (right).
    let (tl, tr) = transpose(&l, &r);
    // Columns: back to lanes as columns.
    let (ul, ur) = transpose(&pass1(&tl), &pass1(&tr));
    let (vl, vr) = (pass2(&ul), pass2(&ur));
    for (k, d) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (row, half) = (k / 2, k % 2);
        let v = quantize(t, 4 * k, if half == 0 { vl[row] } else { vr[row] });
        // SAFETY: `d` holds 4 i32.
        unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), v) };
    }
}

#[target_feature(enable = "sse4.1")]
pub(super) fn fdct_quantize8(src: &[u16], stride: usize, t: &QuantTable, out: &mut [[i32; 64]]) {
    for (i, o) in out.iter_mut().enumerate() {
        block(&src[8 * i..], stride, t, o);
    }
}
