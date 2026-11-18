//! Portable EPF rows: 8 pixels at a time in fixed-size arrays, with the
//! same operations per lane (the SIMD versions repeat them per vector lane).

use super::{EpfStep, Step0, Step1, Step2, Window, neg_inv_sigma};

/// The filtered groups of 8 pixels of one row: `out` holds whole groups,
/// `sigma` one value per group, `rows` the padded window rows.
#[inline(always)]
fn row<S: EpfStep>(
    rows: &Window,
    out: [&mut [f32]; 3],
    sigma: &[f32],
    y_border: bool,
    mul: [f32; 2],
    scale: [f32; 3],
) {
    let [o0, o1, o2] = out;
    let groups = (o0.as_chunks_mut::<8>().0.iter_mut())
        .zip(o1.as_chunks_mut::<8>().0.iter_mut())
        .zip(o2.as_chunks_mut::<8>().0.iter_mut());
    for (g, ((o0, o1), o2)) in groups.enumerate() {
        // sigma is constant over the 8 columns of a block.
        let s = sigma[g];
        if s < 0.3 {
            // Unfiltered: `out` already holds the input.
            continue;
        }
        let x0 = 8 * g;
        let nis: [f32; 8] =
            std::array::from_fn(|l| neg_inv_sigma(s, mul[(y_border || l == 0 || l == 7) as usize]));
        // Columns x0 - 3 .. x0 + 11 of rows y - 3 ..= y + 3.
        let win: [[&[f32; 14]; 7]; 3] = std::array::from_fn(|c| {
            std::array::from_fn(|j| rows[c][j][x0..].first_chunk().expect("padded row"))
        });
        let [r0, r1, r2] = epf_lanes::<S>(&win, &nis, &scale);
        (*o0, *o1, *o2) = (r0, r1, r2);
    }
}

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

/// `epf_pixel` for 8 adjacent pixels at once (the same operations per
/// lane), from their 14 x 7 window per channel.
#[inline(always)]
fn epf_lanes<S: EpfStep>(
    win: &[[&[f32; 14]; 7]; 3],
    nis: &[f32; 8],
    scale: &[f32; 3],
) -> [[f32; 8]; 3] {
    let at = |c: usize, (dx, dy): (usize, usize)| -> [f32; 8] {
        win[c][dy][dx..dx + 8].try_into().expect("8 lanes")
    };
    let mut sum_w = [1.0f32; 8];
    let mut sums: [[f32; 8]; 3] = std::array::from_fn(|c| at(c, (3, 3)));
    for &(kx, ky) in S::KERNEL {
        let mut d = [0f32; 8];
        for (c, &sc) in scale.iter().enumerate() {
            let mut acc = [0f32; 8];
            for &(ox, oy) in S::DIST {
                let (a, b) = (at(c, (kx + ox - 3, ky + oy - 3)), at(c, (ox, oy)));
                for l in 0..8 {
                    acc[l] += (a[l] - b[l]).abs();
                }
            }
            for l in 0..8 {
                d[l] += sc * acc[l];
            }
        }
        let mut weight = [0f32; 8];
        for l in 0..8 {
            weight[l] = (1.0 + d[l] * nis[l]).max(0.0);
            sum_w[l] += weight[l];
        }
        for (c, sum) in sums.iter_mut().enumerate() {
            let v = at(c, (kx, ky));
            for l in 0..8 {
                sum[l] += weight[l] * v[l];
            }
        }
    }
    let mut out = [[0f32; 8]; 3];
    for c in 0..3 {
        for l in 0..8 {
            out[c][l] = sums[c][l] / sum_w[l];
        }
    }
    out
}
