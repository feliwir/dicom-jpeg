//! NEON inverse 2-D DCT: 4 columns per vector, 4 x 4 transposes. FMA is not
//! used, so every lane rounds exactly like the scalar code.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

/// Lanes per vector.
const L: usize = 4;

/// `scalar::dct3` of size 2 on one vector per row.
#[inline]
#[target_feature(enable = "neon")]
fn dct3_2(_: &[Vec<f32>], x: &mut [float32x4_t; 2]) {
    let q = vmulq_f32(x[1], vdupq_n_f32(std::f32::consts::FRAC_1_SQRT_2));
    let p = x[0];
    x[0] = vaddq_f32(p, q);
    x[1] = vsubq_f32(p, q);
}

/// `scalar::dct3` of each size on one vector per row: fixed-size arrays, so
/// the small sizes inline into each other and stay in registers.
macro_rules! dct3 {
    ($($name:ident($n:literal) => $half:ident;)*) => {$(
        #[inline]
        #[target_feature(enable = "neon")]
        fn $name(mul: &[Vec<f32>], x: &mut [float32x4_t; $n]) {
            const H: usize = $n / 2;
            let mut even = [vdupq_n_f32(0.0); H];
            let mut odd = [vdupq_n_f32(0.0); H];
            even[0] = x[0];
            odd[0] = x[1];
            for k in 1..H {
                even[k] = x[2 * k];
                odd[k] = vaddq_f32(x[2 * k + 1], x[2 * k - 1]);
            }
            $half(mul, &mut even);
            $half(mul, &mut odd);
            let m: &[f32; H] = mul[($n as usize).trailing_zeros() as usize][..H]
                .try_into()
                .expect("multiplier table");
            for i in 0..H {
                let o = vmulq_f32(odd[i], vdupq_n_f32(m[i]));
                x[i] = vaddq_f32(even[i], o);
                x[$n - 1 - i] = vsubq_f32(even[i], o);
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
        #[target_feature(enable = "neon")]
        fn $name(mul: &[Vec<f32>], x: &mut [f32], w: usize) {
            assert!(w.is_multiple_of(L) && x.len() >= $n * w);
            let sqrt2 = vdupq_n_f32(std::f32::consts::SQRT_2);
            for c0 in (0..w).step_by(L) {
                let mut v = [vdupq_n_f32(0.0); $n];
                for (r, v) in v.iter_mut().enumerate() {
                    // SAFETY: r * w + c0 + L <= N * w <= x.len() (asserted).
                    *v = unsafe { vld1q_f32(x.as_ptr().add(r * w + c0)) };
                }
                for v in &mut v[1..] {
                    *v = vmulq_f32(*v, sqrt2);
                }
                $dct(mul, &mut v);
                for (r, v) in v.iter().enumerate() {
                    // SAFETY: as for the loads.
                    unsafe { vst1q_f32(x.as_mut_ptr().add(r * w + c0), *v) };
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
#[target_feature(enable = "neon")]
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

/// Transpose a 4 x 4 matrix held as 4 row vectors.
#[inline]
#[target_feature(enable = "neon")]
fn transpose4(r: &[float32x4_t; 4]) -> [float32x4_t; 4] {
    let t01 = vtrnq_f32(r[0], r[1]);
    let t23 = vtrnq_f32(r[2], r[3]);
    [
        vcombine_f32(vget_low_f32(t01.0), vget_low_f32(t23.0)),
        vcombine_f32(vget_low_f32(t01.1), vget_low_f32(t23.1)),
        vcombine_f32(vget_high_f32(t01.0), vget_high_f32(t23.0)),
        vcombine_f32(vget_high_f32(t01.1), vget_high_f32(t23.1)),
    ]
}

/// `dst` (w x h) = transpose of `src` (h x w), in 4 x 4 tiles.
#[target_feature(enable = "neon")]
fn transpose(src: &[f32], dst: &mut [f32], w: usize, h: usize) {
    assert!(w.is_multiple_of(4) && h.is_multiple_of(4));
    assert!(src.len() >= w * h && dst.len() >= w * h);
    for by in (0..h).step_by(4) {
        for bx in (0..w).step_by(4) {
            let mut r = [vdupq_n_f32(0.0); 4];
            for (k, r) in r.iter_mut().enumerate() {
                // SAFETY: (by + k) * w + bx + 4 <= h * w <= src.len().
                *r = unsafe { vld1q_f32(src.as_ptr().add((by + k) * w + bx)) };
            }
            for (k, t) in transpose4(&r).iter().enumerate() {
                // SAFETY: (bx + k) * h + by + 4 <= w * h <= dst.len().
                unsafe { vst1q_f32(dst.as_mut_ptr().add((bx + k) * h + by), *t) };
            }
        }
    }
}

/// Inverse 2-D transform of the h x w block `a` (sides multiples of 8): the
/// rows (as the columns of the transpose), then the columns. `b` is scratch.
#[target_feature(enable = "neon")]
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
