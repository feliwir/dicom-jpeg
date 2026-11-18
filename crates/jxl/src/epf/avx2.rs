//! AVX2 EPF: one group of 8 pixels per vector. FMA is not used, so every
//! lane rounds exactly like the scalar code.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::{EpfStep, Step0, Step1, Step2, Window, neg_inv_sigma};

/// Lanes per vector.
const L: usize = 8;

/// The EPF of the L pixels from column `x` of the window `rows` (padded rows
/// y - 3 ..= y + 3; `x + L + 6` columns must exist), as `scalar::epf_lanes`.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn lanes<S: EpfStep>(rows: &Window, x: usize, nis: __m256, sc: &[__m256; 3]) -> [__m256; 3] {
    // Sample (x + dx, dy) of the window of channel c, for the L lanes.
    macro_rules! at {
        ($c:expr, $dx:expr, $dy:expr) => {{
            let row: &[f32] = rows[$c][$dy];
            debug_assert!(x + $dx + L <= row.len());
            // SAFETY: the caller asserted x + L + 6 <= row.len(), and
            // dx <= 6.
            unsafe { _mm256_loadu_ps(row.as_ptr().add(x + $dx)) }
        }};
    }
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let mut sum_w = one;
    let mut sums = [at!(0, 3, 3), at!(1, 3, 3), at!(2, 3, 3)];
    for &(kx, ky) in S::KERNEL {
        let mut d = zero;
        for (c, &sc) in sc.iter().enumerate() {
            let mut acc = zero;
            for &(ox, oy) in S::DIST {
                let diff = _mm256_sub_ps(at!(c, kx + ox - 3, ky + oy - 3), at!(c, ox, oy));
                acc = _mm256_add_ps(acc, _mm256_and_ps(diff, abs));
            }
            d = _mm256_add_ps(d, _mm256_mul_ps(sc, acc));
        }
        // max(NaN, 0) is 0, as for f32::max.
        let weight = _mm256_max_ps(_mm256_add_ps(one, _mm256_mul_ps(d, nis)), zero);
        sum_w = _mm256_add_ps(sum_w, weight);
        for (c, sum) in sums.iter_mut().enumerate() {
            *sum = _mm256_add_ps(*sum, _mm256_mul_ps(weight, at!(c, kx, ky)));
        }
    }
    sums.map(|s| _mm256_div_ps(s, sum_w))
}

/// `scalar::row`: the filtered groups of 8 pixels of one row.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn row<S: EpfStep>(
    rows: &Window,
    mut out: [&mut [f32]; 3],
    sigma: &[f32],
    y_border: bool,
    mul: [f32; 2],
    scale: [f32; 3],
) {
    let n = out[0].len() / 8 * 8;
    assert!(out.iter().all(|o| o.len() >= n) && sigma.len() >= n / 8);
    assert!(rows.iter().flatten().all(|r| r.len() >= n + 6));
    let sc = [
        _mm256_set1_ps(scale[0]),
        _mm256_set1_ps(scale[1]),
        _mm256_set1_ps(scale[2]),
    ];
    for (g, &s) in sigma[..n / 8].iter().enumerate() {
        // sigma is constant over the 8 columns of a block.
        if s < 0.3 {
            // Unfiltered: `out` already holds the input.
            continue;
        }
        let x0 = 8 * g;
        // Block edges (and rows next to a block edge) use the border factor.
        let (inner, edge) = (
            neg_inv_sigma(s, mul[y_border as usize]),
            neg_inv_sigma(s, mul[1]),
        );
        let nis = _mm256_setr_ps(edge, inner, inner, inner, inner, inner, inner, edge);
        let res = lanes::<S>(rows, x0, nis, &sc);
        for (o, r) in out.iter_mut().zip(res) {
            let dst = &mut o[x0..][..L];
            // SAFETY: `dst` holds L f32; unaligned stores are allowed.
            unsafe { _mm256_storeu_ps(dst.as_mut_ptr(), r) };
        }
    }
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn epf_row0(
    rows: &Window,
    out: [&mut [f32]; 3],
    sigma: &[f32],
    y_border: bool,
    mul: [f32; 2],
    scale: [f32; 3],
) {
    row::<Step0>(rows, out, sigma, y_border, mul, scale);
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn epf_row1(
    rows: &Window,
    out: [&mut [f32]; 3],
    sigma: &[f32],
    y_border: bool,
    mul: [f32; 2],
    scale: [f32; 3],
) {
    row::<Step1>(rows, out, sigma, y_border, mul, scale);
}

#[target_feature(enable = "avx2,fma")]
pub(super) fn epf_row2(
    rows: &Window,
    out: [&mut [f32]; 3],
    sigma: &[f32],
    y_border: bool,
    mul: [f32; 2],
    scale: [f32; 3],
) {
    row::<Step2>(rows, out, sigma, y_border, mul, scale);
}
