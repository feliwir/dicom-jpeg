//! The edge-preserving filter (ISO/IEC 18181-1 Annex J.3).
//!
//! Each step filters a row 8 pixels at a time: the 8 pixels of a block share
//! their sigma. The SIMD versions filter one such group per vector (two at
//! SSE4.1 / NEON) with the scalar operations in the same order (no FMA), so
//! the output is bit-identical; the last, partial group of a row is always
//! filtered per pixel.

use dicom_jpeg_common::dispatch;
use dicom_jpeg_common::simd::{self, Level};

use super::frame::{EpfParams, FrameHeader};
use super::render::padded;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse41;

/// One EPF step: the kernel taps, and the distance taps around each.
pub(crate) trait EpfStep {
    const KERNEL: &'static [(usize, usize)];
    const DIST: &'static [(usize, usize)];
}

// Offsets are (dx + 3, dy + 3), indices into a 7 x 7 window.
pub(crate) struct Step0;
impl EpfStep for Step0 {
    const KERNEL: &'static [(usize, usize)] = &[
        (3, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (1, 3),
        (2, 3),
        (4, 3),
        (5, 3),
        (2, 4),
        (3, 4),
        (4, 4),
        (3, 5),
    ];
    const DIST: &'static [(usize, usize)] = &[(3, 2), (4, 3), (3, 3), (2, 3), (3, 4)];
}

pub(crate) struct Step1;
impl EpfStep for Step1 {
    const KERNEL: &'static [(usize, usize)] = &[(3, 2), (3, 4), (2, 3), (4, 3)];
    const DIST: &'static [(usize, usize)] = &[(3, 2), (3, 3), (3, 4), (2, 3), (4, 3)];
}

pub(crate) struct Step2;
impl EpfStep for Step2 {
    const KERNEL: &'static [(usize, usize)] = Step1::KERNEL;
    const DIST: &'static [(usize, usize)] = &[(3, 3)];
}

/// The EPF steps, for the benchmarks.
#[cfg(feature = "kernels")]
pub(crate) mod kernels {
    use super::*;

    /// EPF step `step` (0, 1 or 2) of the `w` x `h` planes at one SIMD
    /// level, with the default parameters and a sigma per 8 x 8 block.
    pub fn epf(
        level: Level,
        step: usize,
        planes: &mut [Vec<f32>; 3],
        w: usize,
        h: usize,
        sigma: &[f32],
    ) {
        let p = EpfParams::default();
        let bw = w.div_ceil(8);
        match step {
            0 => epf_step::<Step0>(level, 0, planes, w, h, sigma, bw, &p, p.pass0_sigma_scale),
            1 => epf_step::<Step1>(level, 1, planes, w, h, sigma, bw, &p, 1.0),
            _ => epf_step::<Step2>(level, 2, planes, w, h, sigma, bw, &p, p.pass2_sigma_scale),
        }
    }
}

/// Border of the padded planes: kernel reach (2) plus distance reach (1).
const EPF_BORDER: usize = 3;

/// The weight factor of a block with sigma `s` (`mul` is the step's
/// multiplier, times the border factor on block edges).
#[inline]
fn neg_inv_sigma(s: f32, mul: f32) -> f32 {
    6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / s * mul
}

/// Rows `y - 3 ..= y + 3` of each padded channel.
type Window<'a> = [[&'a [f32]; 7]; 3];

/// One EPF step on `planes`, in place.
#[allow(clippy::too_many_arguments)]
fn epf_step<S: EpfStep>(
    level: Level,
    step: usize,
    planes: &mut [Vec<f32>; 3],
    w: usize,
    h: usize,
    sigma: &[f32],
    sigma_stride: usize,
    params: &EpfParams,
    step_mul: f32,
) {
    const B: usize = EPF_BORDER;
    let pw = w + 2 * B;
    let src: [Vec<f32>; 3] = std::array::from_fn(|c| padded(&planes[c], w, h, B));
    let scale = params.channel_scale;
    let mul = [step_mul, step_mul * params.border_sad_mul];
    let full = w / 8 * 8;
    for y in 0..h {
        let y_border = (y + 1) & 0b110 == 0;
        let sigma_row = &sigma[(y / 8) * sigma_stride..][..w.div_ceil(8)];
        let rows: Window =
            std::array::from_fn(|c| std::array::from_fn(|j| &src[c][(y + j) * pw..][..pw]));
        let [p0, p1, p2] = planes;
        let out = [
            &mut p0[y * w..][..full],
            &mut p1[y * w..][..full],
            &mut p2[y * w..][..full],
        ];
        match step {
            0 => dispatch!(level => epf_row0(&rows, out, sigma_row, y_border, mul, scale)),
            1 => dispatch!(level => epf_row1(&rows, out, sigma_row, y_border, mul, scale)),
            _ => dispatch!(level => epf_row2(&rows, out, sigma_row, y_border, mul, scale)),
        }
        // The last, partial group.
        let s = sigma_row[full / 8..].first().copied().unwrap_or(0.0);
        if s < 0.3 {
            // Unfiltered: `planes` already holds the input.
            continue;
        }
        for x in full..w {
            let nis = neg_inv_sigma(s, mul[(y_border || matches!(x & 7, 0 | 7)) as usize]);
            let win: [[&[f32; 7]; 7]; 3] = std::array::from_fn(|c| {
                std::array::from_fn(|j| rows[c][j][x..].first_chunk().expect("padded row"))
            });
            let out = epf_pixel::<S>(&win, nis, &scale);
            for (c, &o) in out.iter().enumerate() {
                planes[c][y * w + x] = o;
            }
        }
    }
}

/// The EPF of one pixel from its 7 x 7 window per channel.
#[inline(always)]
fn epf_pixel<S: EpfStep>(win: &[[&[f32; 7]; 7]; 3], nis: f32, scale: &[f32; 3]) -> [f32; 3] {
    let at = |c: usize, (dx, dy): (usize, usize)| win[c][dy][dx];
    let mut sum_w = 1.0f32;
    let mut sums: [f32; 3] = std::array::from_fn(|c| at(c, (3, 3)));
    for &(kx, ky) in S::KERNEL {
        let mut d = 0f32;
        for (c, &sc) in scale.iter().enumerate() {
            let mut acc = 0f32;
            for &(ox, oy) in S::DIST {
                acc += (at(c, (kx + ox - 3, ky + oy - 3)) - at(c, (ox, oy))).abs();
            }
            d += sc * acc;
        }
        let weight = (1.0 + d * nis).max(0.0);
        sum_w += weight;
        for (c, sum) in sums.iter_mut().enumerate() {
            *sum += weight * at(c, (kx, ky));
        }
    }
    sums.map(|s| s / sum_w)
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
    let level = simd::level();
    let iters = fh.epf_iters;
    let p = &fh.epf;
    let (s, ss) = (sigma, sigma_stride);
    if iters == 3 {
        epf_step::<Step0>(level, 0, planes, w, h, s, ss, p, p.pass0_sigma_scale);
    }
    if iters >= 1 {
        epf_step::<Step1>(level, 1, planes, w, h, s, ss, p, 1.0);
    }
    if iters >= 2 {
        epf_step::<Step2>(level, 2, planes, w, h, s, ss, p, p.pass2_sigma_scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 3u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32
        };
        let params = EpfParams::default();
        // Partial groups on the right, partial blocks at the bottom.
        let (w, h): (usize, usize) = (45, 21);
        let (bw, bh) = (w.div_ceil(8), h.div_ceil(8));
        let planes: [Vec<f32>; 3] = std::array::from_fn(|c| {
            // Smooth content with a little noise, so that the weights vary.
            (0..w * h)
                .map(|i| {
                    ((i % w) as f32 * 0.02 + (i / w) as f32 * 0.01) * (c + 1) as f32 + rnd() * 0.01
                })
                .collect()
        });
        // Some blocks below the 0.3 threshold (unfiltered).
        let sigma: Vec<f32> = (0..bw * bh).map(|_| rnd() * 3.0).collect();
        for step in 0..3 {
            let run = |level: Level| {
                let mut p = planes.clone();
                let args = (w, h, sigma.as_slice(), bw, &params);
                match step {
                    0 => epf_step::<Step0>(
                        level, 0, &mut p, args.0, args.1, args.2, args.3, args.4, 0.9,
                    ),
                    1 => epf_step::<Step1>(
                        level, 1, &mut p, args.0, args.1, args.2, args.3, args.4, 1.0,
                    ),
                    _ => epf_step::<Step2>(
                        level, 2, &mut p, args.0, args.1, args.2, args.3, args.4, 6.5,
                    ),
                }
                p
            };
            let want = run(Level::Scalar);
            assert_ne!(want, planes, "step {step} filters");
            for level in Level::supported() {
                let got = run(level);
                for c in 0..3 {
                    let same = want[c]
                        .iter()
                        .zip(&got[c])
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    assert!(same, "{level:?} step {step} channel {c}");
                }
            }
        }
    }
}
