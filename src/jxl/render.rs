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

/// Gaborish: a 3x3 convolution per channel with mirrored edges.
pub(crate) fn gaborish(planes: &mut [Vec<f32>; 3], w: usize, h: usize, weights: [[f32; 2]; 3]) {
    for (c, plane) in planes.iter_mut().enumerate() {
        let [w0, w1] = weights[c];
        let norm = 1.0 / (1.0 + 4.0 * w0 + 4.0 * w1);
        let src = plane.clone();
        for y in 0..h {
            let ym = mirror(y as isize - 1, h);
            let yp = mirror(y as isize + 1, h);
            for x in 0..w {
                let xm = mirror(x as isize - 1, w);
                let xp = mirror(x as isize + 1, w);
                let p = |xx: usize, yy: usize| src[yy * w + xx];
                let side = p(xm, y) + p(xp, y) + p(x, ym) + p(x, yp);
                let diag = p(xm, ym) + p(xp, ym) + p(xm, yp) + p(xp, yp);
                plane[y * w + x] = (p(x, y) + side * w0 + diag * w1) * norm;
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

fn epf_step(
    planes: &[Vec<f32>; 3],
    w: usize,
    h: usize,
    sigma: &[f32],
    sigma_stride: usize,
    params: &EpfParams,
    step: usize,
) -> [Vec<f32>; 3] {
    type Offsets = &'static [(isize, isize)];
    let (kernel, dist_off, step_mul): (Offsets, Offsets, f32) = match step {
        0 => (&KERNEL_STEP0, &DIST_STEP0, params.pass0_sigma_scale),
        1 => (&KERNEL_STEP12, &DIST_STEP1, 1.0),
        _ => (&KERNEL_STEP12, &DIST_STEP2, params.pass2_sigma_scale),
    };
    let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; w * h]);
    let at = |c: usize, x: isize, y: isize| planes[c][mirror(y, h) * w + mirror(x, w)];
    for y in 0..h {
        let y_border = (y + 1) & 0b110 == 0;
        for x in 0..w {
            let s = sigma[(y / 8) * sigma_stride + x / 8];
            if s < 0.3 {
                for c in 0..3 {
                    out[c][y * w + x] = planes[c][y * w + x];
                }
                continue;
            }
            let x_border = matches!(x & 7, 0 | 7);
            let sm = if y_border || x_border {
                step_mul * params.border_sad_mul
            } else {
                step_mul
            };
            let neg_inv_sigma = 6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / s * sm;
            let (xi, yi) = (x as isize, y as isize);
            let mut sum_w = 1.0f32;
            let mut sums: [f32; 3] = std::array::from_fn(|c| planes[c][y * w + x]);
            for &(kx, ky) in kernel {
                let mut dist = 0f32;
                for (c, &scale) in params.channel_scale.iter().enumerate() {
                    let mut acc = 0f32;
                    for &(ox, oy) in dist_off {
                        acc += (at(c, xi + kx + ox, yi + ky + oy) - at(c, xi + ox, yi + oy)).abs();
                    }
                    dist += scale * acc;
                }
                let weight = (1.0 + dist * neg_inv_sigma).max(0.0);
                sum_w += weight;
                for (c, sum) in sums.iter_mut().enumerate() {
                    *sum += weight * at(c, xi + kx, yi + ky);
                }
            }
            for c in 0..3 {
                out[c][y * w + x] = sums[c] / sum_w;
            }
        }
    }
    out
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
    if iters == 3 {
        *planes = epf_step(planes, w, h, sigma, sigma_stride, &fh.epf, 0);
    }
    if iters >= 1 {
        *planes = epf_step(planes, w, h, sigma, sigma_stride, &fh.epf, 1);
    }
    if iters >= 2 {
        *planes = epf_step(planes, w, h, sigma, sigma_stride, &fh.epf, 2);
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
