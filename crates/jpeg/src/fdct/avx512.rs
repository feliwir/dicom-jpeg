//! AVX-512 8-bit forward DCT and quantisation: two blocks at once, each
//! vector holding the same row of both (16 x i32 lanes). An odd last block
//! goes through the AVX2 code.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::*;

/// `(x + 2^(N-1)) >> N`.
#[inline]
#[target_feature(enable = "avx512f")]
fn descale<const N: u32>(x: __m512i) -> __m512i {
    _mm512_srai_epi32::<N>(_mm512_add_epi32(x, _mm512_set1_epi32(1 << (N - 1))))
}

/// The 1-D forward DCT of `d` (each lane independently): outputs 0 and 4
/// before their pass-specific scaling, the others before descaling.
#[inline]
#[target_feature(enable = "avx512f")]
fn fdct_1d(d: &[__m512i; 8]) -> [__m512i; 8] {
    let add = |a, b| _mm512_add_epi32(a, b);
    let sub = |a, b| _mm512_sub_epi32(a, b);
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
    let z1 = _mm512_mullo_epi32(add(tmp12, tmp13), _mm512_set1_epi32(FIX_0_541196100));
    let o2 = add(
        z1,
        _mm512_mullo_epi32(tmp13, _mm512_set1_epi32(FIX_0_765366865)),
    );
    let o6 = add(
        z1,
        _mm512_mullo_epi32(tmp12, _mm512_set1_epi32(-FIX_1_847759065)),
    );
    let z1 = add(tmp4, tmp7);
    let z2 = add(tmp5, tmp6);
    let z3 = add(tmp4, tmp6);
    let z4 = add(tmp5, tmp7);
    let z5 = _mm512_mullo_epi32(add(z3, z4), _mm512_set1_epi32(FIX_1_175875602));
    let t4 = _mm512_mullo_epi32(tmp4, _mm512_set1_epi32(FIX_0_298631336));
    let t5 = _mm512_mullo_epi32(tmp5, _mm512_set1_epi32(FIX_2_053119869));
    let t6 = _mm512_mullo_epi32(tmp6, _mm512_set1_epi32(FIX_3_072711026));
    let t7 = _mm512_mullo_epi32(tmp7, _mm512_set1_epi32(FIX_1_501321110));
    let z1 = _mm512_mullo_epi32(z1, _mm512_set1_epi32(-FIX_0_899976223));
    let z2 = _mm512_mullo_epi32(z2, _mm512_set1_epi32(-FIX_2_562915447));
    let z3 = add(
        _mm512_mullo_epi32(z3, _mm512_set1_epi32(-FIX_1_961570560)),
        z5,
    );
    let z4 = add(
        _mm512_mullo_epi32(z4, _mm512_set1_epi32(-FIX_0_390180644)),
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

/// Quantise row `r` of both blocks.
#[inline]
#[target_feature(enable = "avx512f,avx512dq")]
fn quantize(t: &QuantTable, r: usize, c: __m512i) -> __m512i {
    // SAFETY: rows of the 64-entry tables hold 8 values.
    let (d, half) = unsafe {
        (
            _mm512_broadcast_f32x8(_mm256_loadu_ps(t.divf[8 * r..].as_ptr())),
            _mm512_broadcast_i32x8(_mm256_loadu_si256(t.half[8 * r..].as_ptr().cast())),
        )
    };
    let a = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_abs_epi32(c), half));
    let q = _mm512_cvttps_epi32(_mm512_div_ps(a, d));
    let neg = _mm512_cmplt_epi32_mask(c, _mm512_setzero_si512());
    let v = _mm512_mask_sub_epi32(q, neg, _mm512_setzero_si512(), q);
    let max = _mm512_set1_epi32(t.max);
    _mm512_max_epi32(
        _mm512_min_epi32(v, max),
        _mm512_sub_epi32(_mm512_setzero_si512(), max),
    )
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
fn pair(src: &[u16], stride: usize, t: &QuantTable, oa: &mut [i32; 64], ob: &mut [i32; 64]) {
    let center = _mm512_set1_epi32(128);
    let mut rows = [_mm512_setzero_si512(); 8];
    for (r, v) in rows.iter_mut().enumerate() {
        let s = &src[r * stride..r * stride + 16];
        // SAFETY: row `r` of both blocks holds 16 samples.
        let s = unsafe { _mm256_loadu_si256(s.as_ptr().cast()) };
        *v = _mm512_sub_epi32(_mm512_cvtepu16_epi32(s), center);
    }
    let o = fdct_1d(&transpose(&rows));
    let mut ws = [_mm512_setzero_si512(); 8];
    for (k, w) in ws.iter_mut().enumerate() {
        *w = if k % 4 == 0 {
            _mm512_slli_epi32::<2>(o[k])
        } else {
            descale::<{ CONST_BITS - 2 }>(o[k])
        };
    }
    let o = fdct_1d(&transpose(&ws));
    let (da, db) = (oa.as_chunks_mut::<8>().0, ob.as_chunks_mut::<8>().0);
    for k in 0..8 {
        let v = if k % 4 == 0 {
            descale::<2>(o[k])
        } else {
            descale::<{ CONST_BITS + 2 }>(o[k])
        };
        let q = quantize(t, k, v);
        // SAFETY: each output row holds 8 i32.
        unsafe {
            _mm256_storeu_si256(da[k].as_mut_ptr().cast(), _mm512_castsi512_si256(q));
            _mm256_storeu_si256(db[k].as_mut_ptr().cast(), _mm512_extracti64x4_epi64::<1>(q));
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
pub(super) fn fdct_quantize8(src: &[u16], stride: usize, t: &QuantTable, out: &mut [[i32; 64]]) {
    let n = out.len() / 2 * 2;
    let (pairs, tail) = out.split_at_mut(n);
    for (i, p) in pairs.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let [oa, ob] = p;
        pair(&src[16 * i..], stride, t, oa, ob);
    }
    if let Some(o) = tail.first_mut() {
        super::avx2::block(&src[8 * n..], stride, t, o);
    }
}

// ---------------------------------------------------------------------------
// 12-bit: the same transform in 64-bit lanes (vpmullq / vpsraq), one block
// row of eight i64 per vector, then the 32-bit quantisation of `avx2.rs`.

mod wide {
    use std::arch::x86_64::*;

    use super::super::*;

    /// `(x + 2^(N-1)) >> N` (arithmetic).
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn descale<const N: u32>(x: __m512i) -> __m512i {
        _mm512_srai_epi64::<N>(_mm512_add_epi64(x, _mm512_set1_epi64(1 << (N - 1))))
    }

    /// The 1-D forward DCT in 64-bit lanes: outputs 0 and 4 before their
    /// pass-specific scaling, the others before descaling.
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq")]
    fn fdct_1d(d: &[__m512i; 8]) -> [__m512i; 8] {
        let add = |a, b| _mm512_add_epi64(a, b);
        let sub = |a, b| _mm512_sub_epi64(a, b);
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
        let z1 = _mm512_mullo_epi64(add(tmp12, tmp13), _mm512_set1_epi64(FIX_0_541196100 as i64));
        let o2 = add(
            z1,
            _mm512_mullo_epi64(tmp13, _mm512_set1_epi64(FIX_0_765366865 as i64)),
        );
        let o6 = add(
            z1,
            _mm512_mullo_epi64(tmp12, _mm512_set1_epi64(-FIX_1_847759065 as i64)),
        );
        let z1 = add(tmp4, tmp7);
        let z2 = add(tmp5, tmp6);
        let z3 = add(tmp4, tmp6);
        let z4 = add(tmp5, tmp7);
        let z5 = _mm512_mullo_epi64(add(z3, z4), _mm512_set1_epi64(FIX_1_175875602 as i64));
        let t4 = _mm512_mullo_epi64(tmp4, _mm512_set1_epi64(FIX_0_298631336 as i64));
        let t5 = _mm512_mullo_epi64(tmp5, _mm512_set1_epi64(FIX_2_053119869 as i64));
        let t6 = _mm512_mullo_epi64(tmp6, _mm512_set1_epi64(FIX_3_072711026 as i64));
        let t7 = _mm512_mullo_epi64(tmp7, _mm512_set1_epi64(FIX_1_501321110 as i64));
        let z1 = _mm512_mullo_epi64(z1, _mm512_set1_epi64(-FIX_0_899976223 as i64));
        let z2 = _mm512_mullo_epi64(z2, _mm512_set1_epi64(-FIX_2_562915447 as i64));
        let z3 = add(
            _mm512_mullo_epi64(z3, _mm512_set1_epi64(-FIX_1_961570560 as i64)),
            z5,
        );
        let z4 = add(
            _mm512_mullo_epi64(z4, _mm512_set1_epi64(-FIX_0_390180644 as i64)),
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

    /// Transpose an 8 x 8 matrix of i64 held as 8 row vectors.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn transpose(r: &[__m512i; 8]) -> [__m512i; 8] {
        let t: [__m512i; 8] = std::array::from_fn(|k| {
            let (a, b) = (r[k / 2 * 2], r[k / 2 * 2 + 1]);
            if k % 2 == 0 {
                _mm512_unpacklo_epi64(a, b)
            } else {
                _mm512_unpackhi_epi64(a, b)
            }
        });
        let even = |a, b| _mm512_shuffle_i64x2::<0x88>(a, b);
        let odd = |a, b| _mm512_shuffle_i64x2::<0xDD>(a, b);
        let (s0, s1) = (even(t[0], t[2]), odd(t[0], t[2]));
        let (s2, s3) = (even(t[1], t[3]), odd(t[1], t[3]));
        let (s4, s5) = (even(t[4], t[6]), odd(t[4], t[6]));
        let (s6, s7) = (even(t[5], t[7]), odd(t[5], t[7]));
        [
            even(s0, s4),
            even(s2, s6),
            even(s1, s5),
            even(s3, s7),
            odd(s0, s4),
            odd(s2, s6),
            odd(s1, s5),
            odd(s3, s7),
        ]
    }

    /// One 12-bit block at `src` (row stride `stride`): PASS1_BITS = 1.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2")]
    fn block(src: &[u16], stride: usize, t: &QuantTable, out: &mut [i32; 64]) {
        let center = _mm512_set1_epi64(2048);
        let mut rows = [_mm512_setzero_si512(); 8];
        for (r, v) in rows.iter_mut().enumerate() {
            // SAFETY: each row of the block holds 8 samples (16 bytes).
            let s = unsafe { _mm_loadu_si128(src[r * stride..r * stride + 8].as_ptr().cast()) };
            *v = _mm512_sub_epi64(_mm512_cvtepu16_epi64(s), center);
        }
        // Rows: transpose so that lanes are rows.
        let o = fdct_1d(&transpose(&rows));
        let ws: [__m512i; 8] = std::array::from_fn(|k| {
            if k % 4 == 0 {
                _mm512_slli_epi64::<1>(o[k])
            } else {
                descale::<{ CONST_BITS - 1 }>(o[k])
            }
        });
        // Columns: lanes are columns again.
        let o = fdct_1d(&transpose(&ws));
        let dst = out.as_chunks_mut::<8>().0;
        for (k, d) in dst.iter_mut().enumerate() {
            let v = if k % 4 == 0 {
                descale::<1>(o[k])
            } else {
                descale::<{ CONST_BITS + 1 }>(o[k])
            };
            // `as i32`, then the 8-lane quantisation.
            let q = super::super::avx2::quantize(t, k, _mm512_cvtepi64_epi32(v));
            // SAFETY: `d` holds 8 i32.
            unsafe { _mm256_storeu_si256(d.as_mut_ptr().cast(), q) };
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx2,fma")]
    pub(in super::super) fn fdct_quantize12(
        src: &[u16],
        stride: usize,
        t: &QuantTable,
        out: &mut [[i32; 64]],
    ) {
        for (i, o) in out.iter_mut().enumerate() {
            block(&src[8 * i..], stride, t, o);
        }
    }
}

pub(super) use wide::fdct_quantize12;
