//! AVX-512 8-bit IDCT: two blocks at once, each vector holding the same row
//! of both (16 x i32 lanes). An odd last block goes through the AVX2 code.

#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

#[inline]
#[target_feature(enable = "avx512f")]
fn mul(a: __m512i, k: i32) -> __m512i {
    _mm512_mullo_epi32(a, _mm512_set1_epi32(k))
}

/// `(x + 2^(N-1)) >> N`.
#[inline]
#[target_feature(enable = "avx512f")]
fn descale<const N: u32>(x: __m512i) -> __m512i {
    _mm512_srai_epi32::<N>(_mm512_add_epi32(x, _mm512_set1_epi32(1 << (N - 1))))
}

/// The 1-D IDCT of `i[0..8]` (each lane independently), before descaling.
#[inline]
#[target_feature(enable = "avx512f")]
fn idct_1d(i: &[__m512i; 8]) -> [__m512i; 8] {
    let add = |a, b| _mm512_add_epi32(a, b);
    let sub = |a, b| _mm512_sub_epi32(a, b);
    // Even part
    let (z2, z3) = (i[2], i[6]);
    let z1 = mul(add(z2, z3), FIX_0_541196100);
    let tmp2 = add(z1, mul(z3, -FIX_1_847759065));
    let tmp3 = add(z1, mul(z2, FIX_0_765366865));
    let tmp0 = _mm512_slli_epi32::<CONST_BITS>(add(i[0], i[4]));
    let tmp1 = _mm512_slli_epi32::<CONST_BITS>(sub(i[0], i[4]));
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

/// Lanes where `i[1..8]` are all zero.
#[inline]
#[target_feature(enable = "avx512f")]
fn ac_zero(i: &[__m512i; 8]) -> __mmask16 {
    let mut ac = i[1];
    for v in &i[2..] {
        ac = _mm512_or_si512(ac, *v);
    }
    _mm512_cmpeq_epi32_mask(ac, _mm512_setzero_si512())
}

/// Transpose the two 8 x 8 matrices (one per 256-bit half) held as 8 row
/// vectors.
#[inline]
#[target_feature(enable = "avx512f")]
fn transpose(r: &[__m512i; 8]) -> [__m512i; 8] {
    let t0 = _mm512_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm512_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm512_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm512_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm512_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm512_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm512_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm512_unpackhi_epi32(r[6], r[7]);
    let u0 = _mm512_unpacklo_epi64(t0, t2);
    let u1 = _mm512_unpackhi_epi64(t0, t2);
    let u2 = _mm512_unpacklo_epi64(t1, t3);
    let u3 = _mm512_unpackhi_epi64(t1, t3);
    let u4 = _mm512_unpacklo_epi64(t4, t6);
    let u5 = _mm512_unpackhi_epi64(t4, t6);
    let u6 = _mm512_unpacklo_epi64(t5, t7);
    let u7 = _mm512_unpackhi_epi64(t5, t7);
    // Per 256-bit half: the low (or high) 128 bits of both inputs, as
    // AVX2's permute2x128 0x20 (0x31).
    let lo = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    let hi = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    let p = |a, idx, b| _mm512_permutex2var_epi64(a, idx, b);
    [
        p(u0, lo, u4),
        p(u1, lo, u5),
        p(u2, lo, u6),
        p(u3, lo, u7),
        p(u0, hi, u4),
        p(u1, hi, u5),
        p(u2, hi, u6),
        p(u3, hi, u7),
    ]
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
fn pair(a: &[i32; 64], b: &[i32; 64], oa: &mut [u16; 64], ob: &mut [u16; 64]) {
    let (sa, sb) = (a.as_chunks::<8>().0, b.as_chunks::<8>().0);
    let mut rows = [_mm512_setzero_si512(); 8];
    for k in 0..8 {
        // SAFETY: each row holds 8 i32 (32 bytes); unaligned loads are
        // allowed.
        let (ra, rb) = unsafe {
            (
                _mm256_loadu_si256(sa[k].as_ptr().cast()),
                _mm256_loadu_si256(sb[k].as_ptr().cast()),
            )
        };
        rows[k] = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ra), rb);
    }
    // Columns: lanes are the columns of both blocks.
    let o = idct_1d(&rows);
    let zero = ac_zero(&rows);
    let dc = _mm512_slli_epi32::<2>(rows[0]);
    let mut ws = [_mm512_setzero_si512(); 8];
    for (w, &v) in ws.iter_mut().zip(&o) {
        *w = _mm512_mask_blend_epi32(zero, descale::<{ CONST_BITS - 2 }>(v), dc);
    }
    // Rows: lanes are the rows of both blocks.
    let t = transpose(&ws);
    let o = idct_1d(&t);
    let zero = ac_zero(&t);
    let dc = descale::<5>(t[0]);
    let center = _mm512_set1_epi32(128);
    let max = _mm512_set1_epi32(255);
    let mut cols = [_mm512_setzero_si512(); 8];
    for (c, &v) in cols.iter_mut().zip(&o) {
        let v = _mm512_mask_blend_epi32(zero, descale::<{ CONST_BITS + 2 + 3 }>(v), dc);
        let v = _mm512_add_epi32(v, center);
        *c = _mm512_min_epi32(_mm512_max_epi32(v, _mm512_setzero_si512()), max);
    }
    let rows = transpose(&cols);
    let (da, db) = (oa.as_chunks_mut::<8>().0, ob.as_chunks_mut::<8>().0);
    for k in 0..8 {
        // Values are in 0..=255, so truncating to 16 bits is exact.
        let v = _mm512_cvtepi32_epi16(rows[k]);
        // SAFETY: each output row holds 8 u16 (16 bytes); unaligned stores
        // are allowed.
        unsafe {
            _mm_storeu_si128(da[k].as_mut_ptr().cast(), _mm256_castsi256_si128(v));
            _mm_storeu_si128(db[k].as_mut_ptr().cast(), _mm256_extracti128_si256::<1>(v));
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn idct8(coefs: &[[i32; 64]], out: &mut [[u16; 64]]) {
    let (cp, ct) = coefs.as_chunks::<2>();
    let (op, ot) = out.as_chunks_mut::<2>();
    for (c, o) in cp.iter().zip(op.iter_mut()) {
        let [oa, ob] = o;
        pair(&c[0], &c[1], oa, ob);
    }
    for (c, o) in ct.iter().zip(ot.iter_mut()) {
        super::avx2::block(c, o);
    }
}
