//! SSE4.1 EPF: a group of 8 pixels as two vectors of 4. Every lane rounds
//! exactly like the scalar code.
#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::{EpfStep, Step0, Step1, Step2, Window, neg_inv_sigma};

/// Lanes per vector.
const L: usize = 4;

/// The EPF of the L pixels from column `x` of the window `rows` (padded rows
/// y - 3 ..= y + 3; `x + L + 6` columns must exist), as `scalar::epf_lanes`.
#[inline]
#[target_feature(enable = "sse4.1")]
fn lanes<S: EpfStep>(rows: &Window, x: usize, nis: __m128, sc: &[__m128; 3]) -> [__m128; 3] {
    // Sample (x + dx, dy) of the window of channel c, for the L lanes.
    macro_rules! at {
        ($c:expr, $dx:expr, $dy:expr) => {{
            let row: &[f32] = rows[$c][$dy];
            debug_assert!(x + $dx + L <= row.len());
            // SAFETY: the caller asserted x + L + 6 <= row.len(), and
            // dx <= 6.
            unsafe { _mm_loadu_ps(row.as_ptr().add(x + $dx)) }
        }};
    }
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let abs = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF));
    let mut sum_w = one;
    let mut sums = [at!(0, 3, 3), at!(1, 3, 3), at!(2, 3, 3)];
    for &(kx, ky) in S::KERNEL {
        let mut d = zero;
        for (c, &sc) in sc.iter().enumerate() {
            let mut acc = zero;
            for &(ox, oy) in S::DIST {
                let diff = _mm_sub_ps(at!(c, kx + ox - 3, ky + oy - 3), at!(c, ox, oy));
                acc = _mm_add_ps(acc, _mm_and_ps(diff, abs));
            }
            d = _mm_add_ps(d, _mm_mul_ps(sc, acc));
        }
        // max(NaN, 0) is 0, as for f32::max.
        let weight = _mm_max_ps(_mm_add_ps(one, _mm_mul_ps(d, nis)), zero);
        sum_w = _mm_add_ps(sum_w, weight);
        for (c, sum) in sums.iter_mut().enumerate() {
            *sum = _mm_add_ps(*sum, _mm_mul_ps(weight, at!(c, kx, ky)));
        }
    }
    sums.map(|s| _mm_div_ps(s, sum_w))
}

/// `scalar::row`: the filtered groups of 8 pixels of one row.
#[inline]
#[target_feature(enable = "sse4.1")]
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
        _mm_set1_ps(scale[0]),
        _mm_set1_ps(scale[1]),
        _mm_set1_ps(scale[2]),
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
        // The group as two halves of L lanes.
        let halves = [
            (x0, _mm_setr_ps(edge, inner, inner, inner)),
            (x0 + L, _mm_setr_ps(inner, inner, inner, edge)),
        ];
        for (x, nis) in halves {
            let res = lanes::<S>(rows, x, nis, &sc);
            for (o, r) in out.iter_mut().zip(res) {
                let dst = &mut o[x..][..L];
                // SAFETY: `dst` holds L f32; unaligned stores are allowed.
                unsafe { _mm_storeu_ps(dst.as_mut_ptr(), r) };
            }
        }
    }
}

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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
