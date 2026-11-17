//! AVX2 8-bit forward DCT and quantisation: one row of a block per vector
//! (8 x i32 lanes).
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

/// `(x + 2^(N-1)) >> N`.
#[inline]
#[target_feature(enable = "avx2")]
fn descale<const N: i32>(x: __m256i) -> __m256i {
    _mm256_srai_epi32::<N>(_mm256_add_epi32(x, _mm256_set1_epi32(1 << (N - 1))))
}

/// The 1-D forward DCT of `d` (each lane independently): outputs 0 and 4
/// before their pass-specific scaling, the others before descaling.
#[inline]
#[target_feature(enable = "avx2")]
fn fdct_1d(d: &[__m256i; 8]) -> [__m256i; 8] {
    let add = |a, b| _mm256_add_epi32(a, b);
    let sub = |a, b| _mm256_sub_epi32(a, b);
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
    let z1 = _mm256_mullo_epi32(add(tmp12, tmp13), _mm256_set1_epi32(FIX_0_541196100));
    let o2 = add(
        z1,
        _mm256_mullo_epi32(tmp13, _mm256_set1_epi32(FIX_0_765366865)),
    );
    let o6 = add(
        z1,
        _mm256_mullo_epi32(tmp12, _mm256_set1_epi32(-FIX_1_847759065)),
    );
    let z1 = add(tmp4, tmp7);
    let z2 = add(tmp5, tmp6);
    let z3 = add(tmp4, tmp6);
    let z4 = add(tmp5, tmp7);
    let z5 = _mm256_mullo_epi32(add(z3, z4), _mm256_set1_epi32(FIX_1_175875602));
    let t4 = _mm256_mullo_epi32(tmp4, _mm256_set1_epi32(FIX_0_298631336));
    let t5 = _mm256_mullo_epi32(tmp5, _mm256_set1_epi32(FIX_2_053119869));
    let t6 = _mm256_mullo_epi32(tmp6, _mm256_set1_epi32(FIX_3_072711026));
    let t7 = _mm256_mullo_epi32(tmp7, _mm256_set1_epi32(FIX_1_501321110));
    let z1 = _mm256_mullo_epi32(z1, _mm256_set1_epi32(-FIX_0_899976223));
    let z2 = _mm256_mullo_epi32(z2, _mm256_set1_epi32(-FIX_2_562915447));
    let z3 = add(
        _mm256_mullo_epi32(z3, _mm256_set1_epi32(-FIX_1_961570560)),
        z5,
    );
    let z4 = add(
        _mm256_mullo_epi32(z4, _mm256_set1_epi32(-FIX_0_390180644)),
        z5,
    );
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

/// Transpose an 8 x 8 matrix of i32 held as 8 row vectors.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose(r: &[__m256i; 8]) -> [__m256i; 8] {
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

/// Quantise one row of coefficients (row `r` of the table).
#[inline]
#[target_feature(enable = "avx2")]
fn quantize(t: &QuantTable, r: usize, c: __m256i) -> __m256i {
    // SAFETY: rows of the 64-entry tables hold 8 values.
    let (d, half) = unsafe {
        (
            _mm256_loadu_ps(t.divf[8 * r..].as_ptr()),
            _mm256_loadu_si256(t.half[8 * r..].as_ptr().cast()),
        )
    };
    let a = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_abs_epi32(c), half));
    let v = _mm256_sign_epi32(_mm256_cvttps_epi32(_mm256_div_ps(a, d)), c);
    let max = _mm256_set1_epi32(t.max);
    _mm256_max_epi32(
        _mm256_min_epi32(v, max),
        _mm256_sub_epi32(_mm256_setzero_si256(), max),
    )
}

/// One block at `src` (row stride `stride`).
#[inline]
#[target_feature(enable = "avx2")]
pub(super) fn block(src: &[u16], stride: usize, t: &QuantTable, out: &mut [i32; 64]) {
    let center = _mm256_set1_epi32(128);
    let mut rows = [_mm256_setzero_si256(); 8];
    for (r, v) in rows.iter_mut().enumerate() {
        // SAFETY: each row of the block holds 8 samples.
        let s = unsafe { _mm_loadu_si128(src[r * stride..r * stride + 8].as_ptr().cast()) };
        *v = _mm256_sub_epi32(_mm256_cvtepu16_epi32(s), center);
    }
    // Rows: transpose so that lanes are rows.
    let o = fdct_1d(&transpose(&rows));
    let mut ws = [_mm256_setzero_si256(); 8];
    for (k, w) in ws.iter_mut().enumerate() {
        *w = if k % 4 == 0 {
            _mm256_slli_epi32::<2>(o[k])
        } else {
            descale::<{ (CONST_BITS - 2) as i32 }>(o[k])
        };
    }
    // Columns: lanes are columns again.
    let o = fdct_1d(&transpose(&ws));
    let dst = out.as_chunks_mut::<8>().0;
    for (k, d) in dst.iter_mut().enumerate() {
        let v = if k % 4 == 0 {
            descale::<2>(o[k])
        } else {
            descale::<{ (CONST_BITS + 2) as i32 }>(o[k])
        };
        let q = quantize(t, k, v);
        // SAFETY: `d` holds 8 i32.
        unsafe { _mm256_storeu_si256(d.as_mut_ptr().cast(), q) };
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn fdct_quantize8(src: &[u16], stride: usize, t: &QuantTable, out: &mut [[i32; 64]]) {
    for (i, o) in out.iter_mut().enumerate() {
        block(&src[8 * i..], stride, t, o);
    }
}
