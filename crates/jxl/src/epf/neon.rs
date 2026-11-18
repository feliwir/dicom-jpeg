//! NEON EPF: a group of 8 pixels as two vectors of 4. FMA is not used, so
//! every lane rounds exactly like the scalar code.
#![allow(unsafe_code)]

use std::arch::aarch64::*;

use super::{EpfStep, Step0, Step1, Step2, Window, neg_inv_sigma};

/// Lanes per vector.
const L: usize = 4;

/// The EPF of the L pixels from column `x` of the window `rows` (padded rows
/// y - 3 ..= y + 3; `x + L + 6` columns must exist), as `scalar::epf_lanes`.
#[inline]
#[target_feature(enable = "neon")]
fn lanes<S: EpfStep>(
    rows: &Window,
    x: usize,
    nis: float32x4_t,
    sc: &[float32x4_t; 3],
) -> [float32x4_t; 3] {
    // Sample (x + dx, dy) of the window of channel c, for the L lanes.
    macro_rules! at {
        ($c:expr, $dx:expr, $dy:expr) => {{
            let row: &[f32] = rows[$c][$dy];
            debug_assert!(x + $dx + L <= row.len());
            // SAFETY: the caller asserted x + L + 6 <= row.len(), and
            // dx <= 6.
            unsafe { vld1q_f32(row.as_ptr().add(x + $dx)) }
        }};
    }
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let mut sum_w = one;
    let mut sums = [at!(0, 3, 3), at!(1, 3, 3), at!(2, 3, 3)];
    for &(kx, ky) in S::KERNEL {
        let mut d = zero;
        for (c, &sc) in sc.iter().enumerate() {
            let mut acc = zero;
            for &(ox, oy) in S::DIST {
                let diff = vsubq_f32(at!(c, kx + ox - 3, ky + oy - 3), at!(c, ox, oy));
                acc = vaddq_f32(acc, vabsq_f32(diff));
            }
            d = vaddq_f32(d, vmulq_f32(sc, acc));
        }
        // maxnm(NaN, 0) is 0, as for f32::max.
        let weight = vmaxnmq_f32(vaddq_f32(one, vmulq_f32(d, nis)), zero);
        sum_w = vaddq_f32(sum_w, weight);
        for (c, sum) in sums.iter_mut().enumerate() {
            *sum = vaddq_f32(*sum, vmulq_f32(weight, at!(c, kx, ky)));
        }
    }
    sums.map(|s| vdivq_f32(s, sum_w))
}

/// `scalar::row`: the filtered groups of 8 pixels of one row.
#[inline]
#[target_feature(enable = "neon")]
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
        vdupq_n_f32(scale[0]),
        vdupq_n_f32(scale[1]),
        vdupq_n_f32(scale[2]),
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
        let (lo, hi) = ([edge, inner, inner, inner], [inner, inner, inner, edge]);
        // SAFETY: `lo` and `hi` hold L f32.
        let halves = unsafe {
            [
                (x0, vld1q_f32(lo.as_ptr())),
                (x0 + L, vld1q_f32(hi.as_ptr())),
            ]
        };
        for (x, nis) in halves {
            let res = lanes::<S>(rows, x, nis, &sc);
            for (o, r) in out.iter_mut().zip(res) {
                let dst = &mut o[x..][..L];
                // SAFETY: `dst` holds L f32; unaligned stores are allowed.
                unsafe { vst1q_f32(dst.as_mut_ptr(), r) };
            }
        }
    }
}

#[target_feature(enable = "neon")]
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

#[target_feature(enable = "neon")]
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

#[target_feature(enable = "neon")]
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
