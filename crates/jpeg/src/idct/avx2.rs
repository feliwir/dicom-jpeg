//! AVX2 8-bit IDCT: one row of a block per vector (8 x i32 lanes).

#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

#[inline]
#[target_feature(enable = "avx2")]
fn mul(a: __m256i, k: i32) -> __m256i {
    _mm256_mullo_epi32(a, _mm256_set1_epi32(k))
}

/// `(x + 2^(N-1)) >> N`.
#[inline]
#[target_feature(enable = "avx2")]
fn descale<const N: i32>(x: __m256i) -> __m256i {
    _mm256_srai_epi32::<N>(_mm256_add_epi32(x, _mm256_set1_epi32(1 << (N - 1))))
}

/// The 1-D IDCT of `i[0..8]` (each lane independently), before descaling.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn idct_1d(i: &[__m256i; 8]) -> [__m256i; 8] {
    let add = |a, b| _mm256_add_epi32(a, b);
    let sub = |a, b| _mm256_sub_epi32(a, b);
    // Even part
    let (z2, z3) = (i[2], i[6]);
    let z1 = mul(add(z2, z3), FIX_0_541196100);
    let tmp2 = add(z1, mul(z3, -FIX_1_847759065));
    let tmp3 = add(z1, mul(z2, FIX_0_765366865));
    let tmp0 = _mm256_slli_epi32::<{ CONST_BITS as i32 }>(add(i[0], i[4]));
    let tmp1 = _mm256_slli_epi32::<{ CONST_BITS as i32 }>(sub(i[0], i[4]));
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
#[target_feature(enable = "avx2")]
fn ac_zero(i: &[__m256i; 8]) -> __m256i {
    let mut ac = i[1];
    for v in &i[2..] {
        ac = _mm256_or_si256(ac, *v);
    }
    _mm256_cmpeq_epi32(ac, _mm256_setzero_si256())
}

/// Transpose an 8 x 8 matrix of i32 held as 8 row vectors.
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn transpose(r: &[__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);
    [
        _mm256_permute2x128_si256::<0x20>(u0, u4),
        _mm256_permute2x128_si256::<0x20>(u1, u5),
        _mm256_permute2x128_si256::<0x20>(u2, u6),
        _mm256_permute2x128_si256::<0x20>(u3, u7),
        _mm256_permute2x128_si256::<0x31>(u0, u4),
        _mm256_permute2x128_si256::<0x31>(u1, u5),
        _mm256_permute2x128_si256::<0x31>(u2, u6),
        _mm256_permute2x128_si256::<0x31>(u3, u7),
    ]
}

/// One block (PASS1_BITS = 2, level shift 128, range 0..=255).
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn block(coef: &[i32; 64], out: &mut [u16; 64]) {
    let src = coef.as_chunks::<8>().0;
    let mut rows = [_mm256_setzero_si256(); 8];
    for (r, s) in rows.iter_mut().zip(src) {
        // SAFETY: `s` holds 8 i32 (32 bytes); unaligned loads are allowed.
        *r = unsafe { _mm256_loadu_si256(s.as_ptr().cast()) };
    }
    // Columns: lanes are columns, vectors rows.
    let o = idct_1d(&rows);
    let zero = ac_zero(&rows);
    let dc = _mm256_slli_epi32::<2>(rows[0]);
    let mut ws = [_mm256_setzero_si256(); 8];
    for (w, &v) in ws.iter_mut().zip(&o) {
        *w = _mm256_blendv_epi8(descale::<{ (CONST_BITS - 2) as i32 }>(v), dc, zero);
    }
    // Rows: transpose so that lanes are rows.
    let t = transpose(&ws);
    let o = idct_1d(&t);
    let zero = ac_zero(&t);
    let dc = descale::<5>(t[0]);
    let center = _mm256_set1_epi32(128);
    let max = _mm256_set1_epi32(255);
    let mut cols = [_mm256_setzero_si256(); 8];
    for (c, &v) in cols.iter_mut().zip(&o) {
        let v = _mm256_blendv_epi8(descale::<{ (CONST_BITS + 2 + 3) as i32 }>(v), dc, zero);
        let v = _mm256_add_epi32(v, center);
        *c = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max);
    }
    let rows = transpose(&cols);
    let dst = out.as_chunks_mut::<16>().0;
    for (d, pair) in dst.iter_mut().zip(rows.as_chunks::<2>().0) {
        // [r0 0..4, r1 0..4, r0 4..8, r1 4..8] -> [r0, r1]
        let p = _mm256_packus_epi32(pair[0], pair[1]);
        let v = _mm256_permute4x64_epi64::<0b11_01_10_00>(p);
        // SAFETY: `d` holds 16 u16 (32 bytes); unaligned stores are allowed.
        unsafe { _mm256_storeu_si256(d.as_mut_ptr().cast(), v) };
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn idct8(coefs: &[[i32; 64]], out: &mut [[u16; 64]]) {
    for (c, o) in coefs.iter().zip(out.iter_mut()) {
        block(c, o);
    }
}
