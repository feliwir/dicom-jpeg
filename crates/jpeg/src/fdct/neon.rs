//! NEON 8-bit forward DCT and quantisation: each row of a block is two
//! vectors of 4 x i32 (the left and right half), transposed as four 4 x 4
//! blocks.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

use super::*;

/// `(x + 2^(N-1)) >> N`.
#[inline]
#[target_feature(enable = "neon")]
fn descale<const N: i32>(x: int32x4_t) -> int32x4_t {
    vshrq_n_s32::<N>(vaddq_s32(x, vdupq_n_s32(1 << (N - 1))))
}

/// The 1-D forward DCT of `d` (each lane independently): outputs 0 and 4
/// before their pass-specific scaling, the others before descaling.
#[inline]
#[target_feature(enable = "neon")]
fn fdct_1d(d: &[int32x4_t; 8]) -> [int32x4_t; 8] {
    let add = |a, b| vaddq_s32(a, b);
    let sub = |a, b| vsubq_s32(a, b);
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
    let z1 = vmulq_n_s32(add(tmp12, tmp13), FIX_0_541196100);
    let o2 = add(z1, vmulq_n_s32(tmp13, FIX_0_765366865));
    let o6 = add(z1, vmulq_n_s32(tmp12, -FIX_1_847759065));
    let z1 = add(tmp4, tmp7);
    let z2 = add(tmp5, tmp6);
    let z3 = add(tmp4, tmp6);
    let z4 = add(tmp5, tmp7);
    let z5 = vmulq_n_s32(add(z3, z4), FIX_1_175875602);
    let t4 = vmulq_n_s32(tmp4, FIX_0_298631336);
    let t5 = vmulq_n_s32(tmp5, FIX_2_053119869);
    let t6 = vmulq_n_s32(tmp6, FIX_3_072711026);
    let t7 = vmulq_n_s32(tmp7, FIX_1_501321110);
    let z1 = vmulq_n_s32(z1, -FIX_0_899976223);
    let z2 = vmulq_n_s32(z2, -FIX_2_562915447);
    let z3 = add(vmulq_n_s32(z3, -FIX_1_961570560), z5);
    let z4 = add(vmulq_n_s32(z4, -FIX_0_390180644), z5);
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
#[target_feature(enable = "neon")]
fn transpose4(r: &[int32x4_t]) -> [int32x4_t; 4] {
    let t01 = vtrnq_s32(r[0], r[1]);
    let t23 = vtrnq_s32(r[2], r[3]);
    [
        vcombine_s32(vget_low_s32(t01.0), vget_low_s32(t23.0)),
        vcombine_s32(vget_low_s32(t01.1), vget_low_s32(t23.1)),
        vcombine_s32(vget_high_s32(t01.0), vget_high_s32(t23.0)),
        vcombine_s32(vget_high_s32(t01.1), vget_high_s32(t23.1)),
    ]
}

/// Transpose an 8 x 8 matrix given as the left (columns 0..4) and right
/// (columns 4..8) halves of its rows; returns the transpose in that form.
#[inline]
#[target_feature(enable = "neon")]
fn transpose(l: &[int32x4_t; 8], r: &[int32x4_t; 8]) -> ([int32x4_t; 8], [int32x4_t; 8]) {
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

#[inline]
#[target_feature(enable = "neon")]
fn pass1(t: &[int32x4_t; 8]) -> [int32x4_t; 8] {
    let o = fdct_1d(t);
    std::array::from_fn(|k| {
        if k % 4 == 0 {
            vshlq_n_s32::<2>(o[k])
        } else {
            descale::<{ (CONST_BITS - 2) as i32 }>(o[k])
        }
    })
}

#[inline]
#[target_feature(enable = "neon")]
fn pass2(t: &[int32x4_t; 8]) -> [int32x4_t; 8] {
    let o = fdct_1d(t);
    std::array::from_fn(|k| {
        if k % 4 == 0 {
            descale::<2>(o[k])
        } else {
            descale::<{ (CONST_BITS + 2) as i32 }>(o[k])
        }
    })
}

/// Quantise 4 coefficients with table entries `k..k + 4`.
#[inline]
#[target_feature(enable = "neon")]
fn quantize(t: &QuantTable, k: usize, c: int32x4_t) -> int32x4_t {
    // SAFETY: `k + 4 <= 64`.
    let (d, half) = unsafe {
        (
            vld1q_f32(t.divf[k..k + 4].as_ptr()),
            vld1q_s32(t.half[k..k + 4].as_ptr()),
        )
    };
    let a = vcvtq_f32_s32(vaddq_s32(vabsq_s32(c), half));
    // Truncating conversion of the correctly rounded quotient.
    let q = vcvtq_s32_f32(vdivq_f32(a, d));
    let v = vbslq_s32(vcltq_s32(c, vdupq_n_s32(0)), vnegq_s32(q), q);
    let max = vdupq_n_s32(t.max);
    vmaxq_s32(vminq_s32(v, max), vnegq_s32(max))
}

#[target_feature(enable = "neon")]
fn block(src: &[u16], stride: usize, t: &QuantTable, out: &mut [i32; 64]) {
    let center = vdupq_n_s32(128);
    let (mut l, mut r) = ([vdupq_n_s32(0); 8], [vdupq_n_s32(0); 8]);
    for k in 0..8 {
        // SAFETY: each row of the block holds 8 samples.
        let s = unsafe { vld1q_u16(src[k * stride..k * stride + 8].as_ptr()) };
        l[k] = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s))), center);
        r[k] = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s))), center);
    }
    let (tl, tr) = transpose(&l, &r);
    let (ul, ur) = transpose(&pass1(&tl), &pass1(&tr));
    let (vl, vr) = (pass2(&ul), pass2(&ur));
    for (k, d) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (row, half) = (k / 2, k % 2);
        let v = quantize(t, 4 * k, if half == 0 { vl[row] } else { vr[row] });
        // SAFETY: `d` holds 4 i32.
        unsafe { vst1q_s32(d.as_mut_ptr(), v) };
    }
}

#[target_feature(enable = "neon")]
pub(super) fn fdct_quantize8(src: &[u16], stride: usize, t: &QuantTable, out: &mut [[i32; 64]]) {
    for (i, o) in out.iter_mut().enumerate() {
        block(&src[8 * i..], stride, t, o);
    }
}

/// 12-bit: the scalar transform and quantisation (it needs 64-bit
/// arithmetic, see `idct::sse41`).
#[target_feature(enable = "neon")]
pub(super) fn fdct_quantize12(src: &[u16], stride: usize, t: &QuantTable, out: &mut [[i32; 64]]) {
    super::scalar::fdct_quantize12(src, stride, t, out);
}
