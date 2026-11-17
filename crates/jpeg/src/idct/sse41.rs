//! SSE4.1 8-bit IDCT: each row of a block is two vectors of 4 x i32 (the
//! left and right half), transposed as four 4 x 4 blocks.

#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

#[inline]
#[target_feature(enable = "sse4.1")]
fn mul(a: __m128i, k: i32) -> __m128i {
    _mm_mullo_epi32(a, _mm_set1_epi32(k))
}

/// `(x + 2^(N-1)) >> N`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn descale<const N: i32>(x: __m128i) -> __m128i {
    _mm_srai_epi32::<N>(_mm_add_epi32(x, _mm_set1_epi32(1 << (N - 1))))
}

/// The 1-D IDCT of `i[0..8]` (each lane independently), before descaling.
#[inline]
#[target_feature(enable = "sse4.1")]
fn idct_1d(i: &[__m128i; 8]) -> [__m128i; 8] {
    let add = |a, b| _mm_add_epi32(a, b);
    let sub = |a, b| _mm_sub_epi32(a, b);
    // Even part
    let (z2, z3) = (i[2], i[6]);
    let z1 = mul(add(z2, z3), FIX_0_541196100);
    let tmp2 = add(z1, mul(z3, -FIX_1_847759065));
    let tmp3 = add(z1, mul(z2, FIX_0_765366865));
    let tmp0 = _mm_slli_epi32::<{ CONST_BITS as i32 }>(add(i[0], i[4]));
    let tmp1 = _mm_slli_epi32::<{ CONST_BITS as i32 }>(sub(i[0], i[4]));
    let tmp10 = add(tmp0, tmp3);
    let tmp13 = sub(tmp0, tmp3);
    let tmp11 = add(tmp1, tmp2);
    let tmp12 = sub(tmp1, tmp2);
    // Odd part
    let (t0, t1, t2, t3) = (i[7], i[5], i[3], i[1]);
    let z1 = add(t0, t3);
    let z2 = add(t1, t2);
    let z3 = add(t0, t2);
    let z4 = add(t1, t3);
    let z5 = mul(add(z3, z4), FIX_1_175875602);
    let t0 = mul(t0, FIX_0_298631336);
    let t1 = mul(t1, FIX_2_053119869);
    let t2 = mul(t2, FIX_3_072711026);
    let t3 = mul(t3, FIX_1_501321110);
    let z1 = mul(z1, -FIX_0_899976223);
    let z2 = mul(z2, -FIX_2_562915447);
    let z3 = add(mul(z3, -FIX_1_961570560), z5);
    let z4 = add(mul(z4, -FIX_0_390180644), z5);
    let t0 = add(add(t0, z1), z3);
    let t1 = add(add(t1, z2), z4);
    let t2 = add(add(t2, z2), z3);
    let t3 = add(add(t3, z1), z4);
    [
        add(tmp10, t3),
        add(tmp11, t2),
        add(tmp12, t1),
        add(tmp13, t0),
        sub(tmp13, t0),
        sub(tmp12, t1),
        sub(tmp11, t2),
        sub(tmp10, t3),
    ]
}

/// Lanes where `i[1..8]` are all zero (all ones), else zero.
#[inline]
#[target_feature(enable = "sse4.1")]
fn ac_zero(i: &[__m128i; 8]) -> __m128i {
    let mut ac = i[1];
    for v in &i[2..] {
        ac = _mm_or_si128(ac, *v);
    }
    _mm_cmpeq_epi32(ac, _mm_setzero_si128())
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
/// (columns 4..8) halves of its 8 rows. Returns the transpose in the same
/// form.
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

/// The column pass on one half: descaled outputs, or the DC shortcut.
#[inline]
#[target_feature(enable = "sse4.1")]
fn pass1(rows: &[__m128i; 8]) -> [__m128i; 8] {
    let o = idct_1d(rows);
    let zero = ac_zero(rows);
    let dc = _mm_slli_epi32::<2>(rows[0]);
    let mut ws = [_mm_setzero_si128(); 8];
    for (w, &v) in ws.iter_mut().zip(&o) {
        *w = _mm_blendv_epi8(descale::<{ (CONST_BITS - 2) as i32 }>(v), dc, zero);
    }
    ws
}

/// The row pass on one half: level-shifted, clamped samples.
#[inline]
#[target_feature(enable = "sse4.1")]
fn pass2(t: &[__m128i; 8]) -> [__m128i; 8] {
    let o = idct_1d(t);
    let zero = ac_zero(t);
    let dc = descale::<5>(t[0]);
    let (center, max) = (_mm_set1_epi32(128), _mm_set1_epi32(255));
    let mut out = [_mm_setzero_si128(); 8];
    for (c, &v) in out.iter_mut().zip(&o) {
        let v = _mm_blendv_epi8(descale::<{ (CONST_BITS + 2 + 3) as i32 }>(v), dc, zero);
        let v = _mm_add_epi32(v, center);
        *c = _mm_min_epi32(_mm_max_epi32(v, _mm_setzero_si128()), max);
    }
    out
}

#[inline]
#[target_feature(enable = "sse4.1")]
fn block(coef: &[i32; 64], out: &mut [u16; 64]) {
    let src = coef.as_chunks::<4>().0;
    let (mut l, mut r) = ([_mm_setzero_si128(); 8], [_mm_setzero_si128(); 8]);
    for k in 0..8 {
        // SAFETY: each chunk holds 4 i32 (16 bytes); unaligned loads are
        // allowed.
        unsafe {
            l[k] = _mm_loadu_si128(src[2 * k].as_ptr().cast());
            r[k] = _mm_loadu_si128(src[2 * k + 1].as_ptr().cast());
        }
    }
    let (l, r) = (pass1(&l), pass1(&r));
    // Rows: lanes are rows 0..4 (left) and 4..8 (right).
    let (tl, tr) = transpose(&l, &r);
    let (cl, cr) = transpose(&pass2(&tl), &pass2(&tr));
    for (k, d) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let v = _mm_packus_epi32(cl[k], cr[k]);
        // SAFETY: `d` holds 8 u16 (16 bytes); unaligned stores are allowed.
        unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), v) };
    }
}

#[target_feature(enable = "sse4.1")]
pub(super) fn idct8(coefs: &[[i32; 64]], out: &mut [[u16; 64]]) {
    for (c, o) in coefs.iter().zip(out.iter_mut()) {
        block(c, o);
    }
}

/// 12-bit IDCT: the scalar code. libjpeg's 12-bit transform needs 64-bit
/// arithmetic, and SSE4.1 has no 64-bit multiply; with an exact emulation
/// (two 32 x 32 -> 64-bit multiplies per product) the vector version was no
/// faster than the scalar one, which skips all-zero rows and columns.
#[target_feature(enable = "sse4.1")]
pub(super) fn idct12(coefs: &[[i32; 64]], out: &mut [[u16; 64]]) {
    super::scalar::idct12(coefs, out);
}
