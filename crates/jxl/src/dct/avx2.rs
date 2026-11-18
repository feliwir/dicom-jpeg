//! AVX2 inverse 2-D DCT: 8 columns per vector, 8 x 8 transposes. FMA is not
//! used, so every lane rounds exactly like the scalar code.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

/// Lanes per vector.
const L: usize = 8;

/// `scalar::dct3` of size 2 on one vector per row.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn dct3_2(_: &[Vec<f32>], x: &mut [__m256; 2]) {
    let q = _mm256_mul_ps(x[1], _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2));
    let p = x[0];
    x[0] = _mm256_add_ps(p, q);
    x[1] = _mm256_sub_ps(p, q);
}

/// `scalar::dct3` of each size on one vector per row: fixed-size arrays, so
/// the small sizes inline into each other and stay in registers.
macro_rules! dct3 {
    ($($name:ident($n:literal) => $half:ident;)*) => {$(
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        fn $name(mul: &[Vec<f32>], x: &mut [__m256; $n]) {
            const H: usize = $n / 2;
            let mut even = [_mm256_setzero_ps(); H];
            let mut odd = [_mm256_setzero_ps(); H];
            even[0] = x[0];
            odd[0] = x[1];
            for k in 1..H {
                even[k] = x[2 * k];
                odd[k] = _mm256_add_ps(x[2 * k + 1], x[2 * k - 1]);
            }
            $half(mul, &mut even);
            $half(mul, &mut odd);
            let m: &[f32; H] = mul[($n as usize).trailing_zeros() as usize][..H]
                .try_into()
                .expect("multiplier table");
            for i in 0..H {
                let o = _mm256_mul_ps(odd[i], _mm256_set1_ps(m[i]));
                x[i] = _mm256_add_ps(even[i], o);
                x[$n - 1 - i] = _mm256_sub_ps(even[i], o);
            }
        }
    )*};
}

dct3! {
    dct3_4(4) => dct3_2;
    dct3_8(8) => dct3_4;
    dct3_16(16) => dct3_8;
    dct3_32(32) => dct3_16;
    dct3_64(64) => dct3_32;
    dct3_128(128) => dct3_64;
    dct3_256(256) => dct3_128;
}

/// The inverse transform along the columns of an N x w block
/// (`scalar::transform_cols`), `L` columns at a time.
macro_rules! columns {
    ($($name:ident($n:literal) => $dct:ident;)*) => {$(
        #[target_feature(enable = "avx2,fma")]
        fn $name(mul: &[Vec<f32>], x: &mut [f32], w: usize) {
            assert!(w.is_multiple_of(L) && x.len() >= $n * w);
            let sqrt2 = _mm256_set1_ps(std::f32::consts::SQRT_2);
            for c0 in (0..w).step_by(L) {
                let mut v = [_mm256_setzero_ps(); $n];
                for (r, v) in v.iter_mut().enumerate() {
                    // SAFETY: r * w + c0 + L <= N * w <= x.len() (asserted).
                    *v = unsafe { _mm256_loadu_ps(x.as_ptr().add(r * w + c0)) };
                }
                for v in &mut v[1..] {
                    *v = _mm256_mul_ps(*v, sqrt2);
                }
                $dct(mul, &mut v);
                for (r, v) in v.iter().enumerate() {
                    // SAFETY: as for the loads.
                    unsafe { _mm256_storeu_ps(x.as_mut_ptr().add(r * w + c0), *v) };
                }
            }
        }
    )*};
}

columns! {
    columns_8(8) => dct3_8;
    columns_16(16) => dct3_16;
    columns_32(32) => dct3_32;
    columns_64(64) => dct3_64;
    columns_128(128) => dct3_128;
    columns_256(256) => dct3_256;
}

/// The columns of the n x w block `x`.
#[target_feature(enable = "avx2,fma")]
fn columns(mul: &[Vec<f32>], x: &mut [f32], n: usize, w: usize) {
    match n {
        8 => columns_8(mul, x, w),
        16 => columns_16(mul, x, w),
        32 => columns_32(mul, x, w),
        64 => columns_64(mul, x, w),
        128 => columns_128(mul, x, w),
        256 => columns_256(mul, x, w),
        _ => unreachable!("DCT size {n}"),
    }
}

/// Transpose an 8 x 8 matrix held as 8 row vectors.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn transpose8(r: &[__m256; 8]) -> [__m256; 8] {
    let t0 = _mm256_unpacklo_ps(r[0], r[1]);
    let t1 = _mm256_unpackhi_ps(r[0], r[1]);
    let t2 = _mm256_unpacklo_ps(r[2], r[3]);
    let t3 = _mm256_unpackhi_ps(r[2], r[3]);
    let t4 = _mm256_unpacklo_ps(r[4], r[5]);
    let t5 = _mm256_unpackhi_ps(r[4], r[5]);
    let t6 = _mm256_unpacklo_ps(r[6], r[7]);
    let t7 = _mm256_unpackhi_ps(r[6], r[7]);
    let u0 = _mm256_shuffle_ps::<0x44>(t0, t2);
    let u1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
    let u2 = _mm256_shuffle_ps::<0x44>(t1, t3);
    let u3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
    let u4 = _mm256_shuffle_ps::<0x44>(t4, t6);
    let u5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
    let u6 = _mm256_shuffle_ps::<0x44>(t5, t7);
    let u7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
    [
        _mm256_permute2f128_ps::<0x20>(u0, u4),
        _mm256_permute2f128_ps::<0x20>(u1, u5),
        _mm256_permute2f128_ps::<0x20>(u2, u6),
        _mm256_permute2f128_ps::<0x20>(u3, u7),
        _mm256_permute2f128_ps::<0x31>(u0, u4),
        _mm256_permute2f128_ps::<0x31>(u1, u5),
        _mm256_permute2f128_ps::<0x31>(u2, u6),
        _mm256_permute2f128_ps::<0x31>(u3, u7),
    ]
}

/// `dst` (w x h) = transpose of `src` (h x w), in 8 x 8 tiles.
#[target_feature(enable = "avx2,fma")]
fn transpose(src: &[f32], dst: &mut [f32], w: usize, h: usize) {
    assert!(w.is_multiple_of(8) && h.is_multiple_of(8));
    assert!(src.len() >= w * h && dst.len() >= w * h);
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            let mut r = [_mm256_setzero_ps(); 8];
            for (k, r) in r.iter_mut().enumerate() {
                // SAFETY: (by + k) * w + bx + 8 <= h * w <= src.len().
                *r = unsafe { _mm256_loadu_ps(src.as_ptr().add((by + k) * w + bx)) };
            }
            for (k, t) in transpose8(&r).iter().enumerate() {
                // SAFETY: (bx + k) * h + by + 8 <= w * h <= dst.len().
                unsafe { _mm256_storeu_ps(dst.as_mut_ptr().add((bx + k) * h + by), *t) };
            }
        }
    }
}

/// Inverse 2-D transform of the h x w block `a` (sides multiples of 8): the
/// rows (as the columns of the transpose), then the columns. `b` is scratch.
#[target_feature(enable = "avx2,fma")]
pub(super) fn idct_2d(
    mul: &[Vec<f32>],
    a: &mut [f32],
    b: &mut [f32],
    _tmp: &mut [f32],
    w: usize,
    h: usize,
) {
    transpose(a, b, w, h);
    columns(mul, b, w, h);
    transpose(b, a, h, w);
    columns(mul, a, h, w);
}
