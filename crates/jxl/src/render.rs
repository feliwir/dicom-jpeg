//! Restoration filters (Gaborish, edge-preserving filter) and the XYB to
//! RGB colour conversion (ISO/IEC 18181-1 Annex J and L).

use super::frame::{EpfParams, FrameHeader};
use super::headers::OpsinInverse;

/// Mirror an index into `0..len` (whole-sample symmetric).
#[inline]
fn mirror(mut i: isize, len: usize) -> usize {
    let len = len as isize;
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= len {
            i = 2 * len - i - 1;
        } else {
            return i as usize;
        }
    }
}

/// A copy of the `w` x `h` plane with a mirrored border of `b` samples
/// (row stride `w + 2b`).
fn padded(plane: &[f32], w: usize, h: usize, b: usize) -> Vec<f32> {
    let pw = w + 2 * b;
    let cols: Vec<usize> = (0..pw)
        .map(|x| mirror(x as isize - b as isize, w))
        .collect();
    let mut out = vec![0f32; pw * (h + 2 * b)];
    for (py, dst) in out.chunks_exact_mut(pw).enumerate() {
        let src = &plane[mirror(py as isize - b as isize, h) * w..][..w];
        dst[b..b + w].copy_from_slice(src);
        for x in (0..b).chain(b + w..pw) {
            dst[x] = src[cols[x]];
        }
    }
    out
}

/// Gaborish: a 3x3 convolution per channel with mirrored edges.
pub(crate) fn gaborish(planes: &mut [Vec<f32>; 3], w: usize, h: usize, weights: [[f32; 2]; 3]) {
    let pw = w + 2;
    for (c, plane) in planes.iter_mut().enumerate() {
        let [w0, w1] = weights[c];
        let norm = 1.0 / (1.0 + 4.0 * w0 + 4.0 * w1);
        let src = padded(plane, w, h, 1);
        for (y, dst) in plane[..w * h].chunks_exact_mut(w).enumerate() {
            let (up, mid, down) = (
                &src[y * pw..][..pw],
                &src[(y + 1) * pw..][..pw],
                &src[(y + 2) * pw..][..pw],
            );
            for (x, d) in dst.iter_mut().enumerate() {
                let side = mid[x] + mid[x + 2] + up[x + 1] + down[x + 1];
                let diag = up[x] + up[x + 2] + down[x] + down[x + 2];
                *d = (mid[x + 1] + side * w0 + diag * w1) * norm;
            }
        }
    }
}

const KERNEL_STEP0: [(isize, isize); 12] = [
    (0, -2),
    (-1, -1),
    (0, -1),
    (1, -1),
    (-2, 0),
    (-1, 0),
    (1, 0),
    (2, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (0, 2),
];
const KERNEL_STEP12: [(isize, isize); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
const DIST_STEP0: [(isize, isize); 5] = [(0, -1), (1, 0), (0, 0), (-1, 0), (0, 1)];
const DIST_STEP1: [(isize, isize); 5] = [(0, -1), (0, 0), (0, 1), (-1, 0), (1, 0)];
const DIST_STEP2: [(isize, isize); 1] = [(0, 0)];

/// Border of the padded planes: kernel reach (2) plus distance reach (1).
const EPF_BORDER: usize = 3;

/// One EPF step on `planes`, in place, with kernel taps `kernel` and the
/// distance taps `dist` around each.
#[allow(clippy::too_many_arguments)]
fn epf_step<const K: usize, const D: usize>(
    planes: &mut [Vec<f32>; 3],
    w: usize,
    h: usize,
    sigma: &[f32],
    sigma_stride: usize,
    params: &EpfParams,
    step_mul: f32,
    kernel: &[(isize, isize); K],
    dist: &[(isize, isize); D],
) {
    const B: usize = EPF_BORDER;
    let pw = w + 2 * B;
    let src: [Vec<f32>; 3] = std::array::from_fn(|c| padded(&planes[c], w, h, B));
    let off = |(dx, dy): (isize, isize)| dy * pw as isize + dx;
    let kernel: [isize; K] = kernel.map(off);
    let dist: [isize; D] = dist.map(off);
    let scale = params.channel_scale;
    let mul = [step_mul, step_mul * params.border_sad_mul];
    let [p0, p1, p2] = planes;
    let out = [&mut p0[..], &mut p1[..], &mut p2[..]];
    for y in 0..h {
        let y_border = (y + 1) & 0b110 == 0;
        let sigma_row = &sigma[(y / 8) * sigma_stride..];
        let row = (y + B) * pw + B;
        for x in 0..w {
            let s = sigma_row[x / 8];
            if s < 0.3 {
                // Unfiltered: `out` already holds the input.
                continue;
            }
            let border = y_border || matches!(x & 7, 0 | 7);
            let neg_inv_sigma =
                6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / s * mul[border as usize];
            let i = (row + x) as isize;
            let at = |c: usize, o: isize| src[c][(i + o) as usize];
            let mut sum_w = 1.0f32;
            let mut sums: [f32; 3] = std::array::from_fn(|c| at(c, 0));
            for &k in &kernel {
                let mut d = 0f32;
                for (c, &sc) in scale.iter().enumerate() {
                    let mut acc = 0f32;
                    for &o in &dist {
                        acc += (at(c, k + o) - at(c, o)).abs();
                    }
                    d += sc * acc;
                }
                let weight = (1.0 + d * neg_inv_sigma).max(0.0);
                sum_w += weight;
                for (c, sum) in sums.iter_mut().enumerate() {
                    *sum += weight * at(c, k);
                }
            }
            for c in 0..3 {
                out[c][y * w + x] = sums[c] / sum_w;
            }
        }
    }
}

/// Edge-preserving filter with a per-8x8-block sigma map.
pub(crate) fn epf(
    planes: &mut [Vec<f32>; 3],
    w: usize,
    h: usize,
    sigma: &[f32],
    sigma_stride: usize,
    fh: &FrameHeader,
) {
    let iters = fh.epf_iters;
    let p = &fh.epf;
    if iters == 3 {
        let mul = p.pass0_sigma_scale;
        epf_step(
            planes,
            w,
            h,
            sigma,
            sigma_stride,
            p,
            mul,
            &KERNEL_STEP0,
            &DIST_STEP0,
        );
    }
    if iters >= 1 {
        epf_step(
            planes,
            w,
            h,
            sigma,
            sigma_stride,
            p,
            1.0,
            &KERNEL_STEP12,
            &DIST_STEP1,
        );
    }
    if iters >= 2 {
        let mul = p.pass2_sigma_scale;
        epf_step(
            planes,
            w,
            h,
            sigma,
            sigma_stride,
            p,
            mul,
            &KERNEL_STEP12,
            &DIST_STEP2,
        );
    }
}

/// XYB to linear sRGB (in place: X, Y, B -> R, G, B).
pub(crate) fn xyb_to_linear(
    planes: &mut [Vec<f32>; 3],
    opsin: &OpsinInverse,
    intensity_target: f32,
) {
    let itscale = 255.0 / intensity_target;
    let cbrt_bias = opsin.bias.map(|b| b.cbrt());
    let m = &opsin.matrix;
    let n = planes[0].len();
    for i in 0..n {
        let (x, y, b) = (planes[0][i], planes[1][i], planes[2][i]);
        let l = y + x - cbrt_bias[0];
        let mm = y - x - cbrt_bias[1];
        let s = b - cbrt_bias[2];
        let mixed = [
            (l * l * l + opsin.bias[0]) * itscale,
            (mm * mm * mm + opsin.bias[1]) * itscale,
            (s * s * s + opsin.bias[2]) * itscale,
        ];
        for c in 0..3 {
            planes[c][i] = m[c * 3] * mixed[0] + m[c * 3 + 1] * mixed[1] + m[c * 3 + 2] * mixed[2];
        }
    }
}

/// Linear light to the encoded transfer function of the output colour space.
pub(crate) enum Transfer {
    Linear,
    Srgb,
    Bt709,
    Dci,
    /// Encoded = linear^gamma.
    Gamma(f32),
}

impl Transfer {
    pub(crate) fn encode(&self, v: f32) -> f32 {
        let v = v.clamp(0.0, 1.0);
        match *self {
            Transfer::Linear => v,
            Transfer::Srgb => {
                if v <= 0.003_130_8 {
                    12.92 * v
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }
            Transfer::Bt709 => {
                if v < 0.018 {
                    4.5 * v
                } else {
                    1.099 * v.powf(0.45) - 0.099
                }
            }
            Transfer::Dci => v.powf(1.0 / 2.6),
            Transfer::Gamma(g) => v.powf(g),
        }
    }
}
