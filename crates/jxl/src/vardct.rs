//! VarDCT frames (ISO/IEC 18181-1 Annex I): LF coefficients, HF metadata,
//! dequantisation matrices, coefficient orders, HF coefficient decoding and
//! the reconstruction of XYB samples (dequantisation, chroma from luma, LLF
//! coefficients and the inverse transforms).
//!
//! Coefficients of a varblock live in its own spatial area of a frame-wide
//! grid: position (x, y) holds horizontal frequency x and vertical
//! frequency y.

use super::bits::{BitReader, bits, invalid, unpack_signed, val};
use super::entropy::{Histograms, SymbolReader, decode_context_map, read_permutation};
use super::frame::{FLAG_SKIP_ADAPTIVE_LF_SMOOTHING, FrameHeader};
use super::modular::{self, Channel, MaConfig};
use dicom_jpeg_common::Result;

// ---------------------------------------------------------------------------
// Transform types

/// Per DctSelect: (blocks wide, blocks high, dequant param index, order id).
const TRANSFORMS: [(usize, usize, usize, usize); 27] = [
    (1, 1, 0, 0),     // DCT8
    (1, 1, 1, 1),     // Hornuss
    (1, 1, 2, 1),     // DCT2
    (1, 1, 3, 1),     // DCT4
    (2, 2, 4, 2),     // DCT16
    (4, 4, 5, 3),     // DCT32
    (1, 2, 6, 4),     // DCT16x8
    (2, 1, 6, 4),     // DCT8x16
    (1, 4, 7, 5),     // DCT32x8
    (4, 1, 7, 5),     // DCT8x32
    (2, 4, 8, 6),     // DCT32x16
    (4, 2, 8, 6),     // DCT16x32
    (1, 1, 9, 1),     // DCT4x8
    (1, 1, 9, 1),     // DCT8x4
    (1, 1, 10, 1),    // AFV0
    (1, 1, 10, 1),    // AFV1
    (1, 1, 10, 1),    // AFV2
    (1, 1, 10, 1),    // AFV3
    (8, 8, 11, 7),    // DCT64
    (4, 8, 12, 8),    // DCT64x32
    (8, 4, 12, 8),    // DCT32x64
    (16, 16, 13, 9),  // DCT128
    (8, 16, 14, 10),  // DCT128x64
    (16, 8, 14, 10),  // DCT64x128
    (32, 32, 15, 11), // DCT256
    (16, 32, 16, 12), // DCT256x128
    (32, 16, 16, 12), // DCT128x256
];

const HORNUSS: usize = 1;
const DCT2: usize = 2;
const DCT4: usize = 3;
const DCT4X8: usize = 12;
const DCT8X4: usize = 13;
const AFV0: usize = 14;

/// Whether the natural order / dequant matrix are stored transposed.
fn need_transpose(t: usize) -> bool {
    if matches!(t, HORNUSS | DCT2 | DCT4 | DCT4X8 | DCT8X4) || (AFV0..AFV0 + 4).contains(&t) {
        return false;
    }
    let (w, h, _, _) = TRANSFORMS[t];
    h >= w
}

/// Dequant matrix size (width, height) per parameter index; width >= height.
const MATRIX_SIZE: [(usize, usize); 17] = [
    (8, 8),
    (8, 8),
    (8, 8),
    (8, 8),
    (16, 16),
    (32, 32),
    (16, 8),
    (32, 8),
    (32, 16),
    (8, 8),
    (8, 8),
    (64, 64),
    (64, 32),
    (128, 128),
    (128, 64),
    (256, 256),
    (256, 128),
];

/// Coefficient order block sizes (width, height), width >= height.
const ORDER_SIZE: [(usize, usize); 13] = [
    (8, 8),
    (8, 8),
    (16, 16),
    (32, 32),
    (16, 8),
    (32, 8),
    (32, 16),
    (64, 64),
    (64, 32),
    (128, 128),
    (128, 64),
    (256, 256),
    (256, 128),
];

/// Natural coefficient order (x, y) of an order id.
fn natural_order(order_id: usize) -> Vec<(u16, u16)> {
    let (bw, bh) = ORDER_SIZE[order_id];
    let y_scale = bw / bh;
    let (lbw, lbh) = (bw / 8, bh / 8);
    let mut out = Vec::with_capacity(bw * bh);
    for idx in 0..lbw * lbh {
        out.push(((idx % lbw) as u16, (idx / lbw) as u16));
    }
    for dist in 1..2 * bw {
        let margin = dist.saturating_sub(bw);
        for order in margin..dist - margin {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            if (x < lbw && y < lbw) || y % y_scale != 0 {
                continue;
            }
            out.push((x as u16, (y / y_scale) as u16));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dequantisation matrices

enum MatrixParams {
    Hornuss([[f32; 3]; 3]),
    Dct2([[f32; 6]; 3]),
    Dct4([[f32; 2]; 3], [Vec<f32>; 3]),
    Dct4x8([[f32; 1]; 3], [Vec<f32>; 3]),
    Afv([[f32; 9]; 3], [Vec<f32>; 3], [Vec<f32>; 3]),
    Dct([Vec<f32>; 3]),
    Raw(f32, [Vec<i32>; 3]),
}

const SEQ_A: [f32; 7] = [
    -1.025,
    -0.78,
    -0.65012,
    -0.19041574,
    -0.20819396,
    -0.421064,
    -0.32733846,
];
const SEQ_B: [f32; 7] = [
    -0.30419582,
    -0.36330363,
    -0.3566038,
    -0.34430745,
    -0.33699593,
    -0.30180866,
    -0.27321684,
];
const SEQ_C: [f32; 7] = [-1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5];
const DCT4X8_PARAMS: [[f32; 4]; 3] = [
    [2198.0505, -0.96269625, -0.7619425, -0.65511405],
    [764.36554, -0.926302, -0.967523, -0.2784529],
    [527.10754, -1.4594386, -1.4500821, -1.5843723],
];
const DCT4_PARAMS: [[f32; 4]; 3] = [
    [2200.0, 0.0, 0.0, 0.0],
    [392.0, 0.0, 0.0, 0.0],
    [112.0, -0.25, -0.25, -0.5],
];

fn seq(a: f32, b: f32, c: f32) -> MatrixParams {
    let mk = |first: f32, rest: &[f32; 7]| {
        let mut v = vec![first];
        v.extend_from_slice(rest);
        v
    };
    MatrixParams::Dct([mk(a, &SEQ_A), mk(b, &SEQ_B), mk(c, &SEQ_C)])
}

#[rustfmt::skip]
fn default_params(idx: usize) -> MatrixParams {
    use MatrixParams::*;
    match idx {
        0 => Dct([
            vec![3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
            vec![560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
            vec![512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
        ]),
        1 => Hornuss([[280.0, 3160.0, 3160.0], [60.0, 864.0, 864.0], [18.0, 200.0, 200.0]]),
        2 => Dct2([
            [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
            [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
            [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
        ]),
        3 => Dct4([[1.0; 2]; 3], DCT4_PARAMS.map(|v| v.to_vec())),
        4 => Dct([
            vec![8996.873, -1.3000778, -0.4942453, -0.43909377, -0.6350102, -0.9017726, -1.6162099],
            vec![3191.4836, -0.67424583, -0.80745816, -0.4492584, -0.3586544, -0.3132239, -0.37615025],
            vec![1157.504, -2.0531423, -1.4, -0.5068713, -0.4270873, -1.4856834, -4.920914],
        ]),
        5 => Dct([
            vec![15718.408, -1.025, -0.98, -0.9012, -0.4, -0.48819396, -0.421064, -0.27],
            vec![7305.7637, -0.8041958, -0.76330364, -0.5566038, -0.49785304, -0.43699592, -0.40180868, -0.27321684],
            vec![3803.5317, -3.0607336, -2.041327, -2.023565, -0.54953897, -0.4, -0.4, -0.3],
        ]),
        6 => Dct([
            vec![7240.7734, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5],
            vec![1448.1547, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2],
            vec![506.85413, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6],
        ]),
        7 => Dct([
            vec![16283.249, -1.7812846, -1.6309059, -1.0382179, -0.85, -0.7, -0.9, -1.2360638],
            vec![5089.1577, -0.3200494, -0.3536285, -0.3034, -0.61, -0.5, -0.5, -0.6],
            vec![3397.7761, -0.32132736, -0.3450762, -0.7034, -0.9, -1.0, -1.0, -1.1754606],
        ]),
        8 => Dct([
            vec![13844.971, -0.971138, -0.658, -0.42026, -0.22712, -0.2206, -0.226, -0.6],
            vec![4798.964, -0.6112531, -0.8377079, -0.7901486, -0.26927274, -0.38272768, -0.22924222, -0.20719099],
            vec![1807.2369, -1.2, -1.2, -0.7, -0.7, -0.7, -0.4, -0.5],
        ]),
        9 => Dct4x8([[1.0]; 3], DCT4X8_PARAMS.map(|v| v.to_vec())),
        10 => Afv(
            [
                [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
                [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
                [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
            ],
            DCT4X8_PARAMS.map(|v| v.to_vec()),
            DCT4_PARAMS.map(|v| v.to_vec()),
        ),
        11 => seq(23966.166, 8380.191, 4493.024),
        12 => seq(15358.898, 5597.3604, 2919.9617),
        13 => seq(47932.332, 16760.383, 8986.048),
        14 => seq(30717.797, 11194.721, 5839.9233),
        15 => seq(95864.664, 33520.766, 17972.096),
        _ => seq(61435.594, 24209.441, 12979.847),
    }
}

fn read_fixed<const N: usize>(br: &mut BitReader) -> Result<[[f32; N]; 3]> {
    let mut out = [[0f32; N]; 3];
    for v in out.iter_mut().flatten() {
        *v = br.f16()?;
    }
    Ok(out)
}

fn read_dct_params(br: &mut BitReader) -> Result<[Vec<f32>; 3]> {
    let n = br.read(4)? as usize + 1;
    let mut p = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    for v in p.iter_mut().flatten() {
        *v = br.f16()?;
    }
    for c in &mut p {
        c[0] *= 64.0;
    }
    Ok(p)
}

fn read_matrix_params(
    br: &mut BitReader,
    idx: usize,
    stream_id: u32,
    global: Option<&MaConfig>,
    alloc_limit: u64,
) -> Result<MatrixParams> {
    use MatrixParams::*;
    let mode = br.read(3)?;
    if (1..=5).contains(&mode) && !matches!(idx, 0 | 1 | 2 | 3 | 9 | 10) {
        return Err(invalid("dequant matrix mode needs an 8x8 transform"));
    }
    Ok(match mode {
        0 => default_params(idx),
        1 => Hornuss(read_fixed(br)?),
        2 => Dct2(read_fixed(br)?),
        3 => Dct4(read_fixed(br)?, read_dct_params(br)?),
        4 => Dct4x8(read_fixed(br)?, read_dct_params(br)?),
        5 => {
            let mut p = read_fixed::<9>(br)?;
            for c in &mut p {
                for v in &mut c[..6] {
                    *v *= 64.0;
                }
            }
            Afv(p, read_dct_params(br)?, read_dct_params(br)?)
        }
        6 => Dct(read_dct_params(br)?),
        _ => {
            let (w, h) = MATRIX_SIZE[idx];
            let denominator = br.f16()?;
            let mut img = modular::Image {
                channels: (0..3).map(|_| Channel::new(w, h, 0, 0)).collect(),
                bit_depth: 8,
                ..Default::default()
            };
            modular::decode_stream(
                br,
                &mut img,
                stream_id,
                global,
                usize::MAX,
                true,
                alloc_limit,
            )?;
            let mut ch = img.channels.into_iter().map(|c| c.data);
            Raw(
                denominator,
                [
                    ch.next().unwrap_or_default(),
                    ch.next().unwrap_or_default(),
                    ch.next().unwrap_or_default(),
                ],
            )
        }
    })
}

fn interpolate(pos: f32, max: f32, bands: &[f32]) -> f32 {
    if bands.len() == 1 {
        return bands[0];
    }
    let scaled = pos * (bands.len() - 1) as f32 / max;
    let i = (scaled as usize).min(bands.len() - 2);
    let frac = scaled - i as f32;
    let (a, b) = (bands[i], bands[i + 1]);
    a * (b / a).powf(frac)
}

fn band_mult(x: f32) -> f32 {
    if x > 0.0 { 1.0 + x } else { 1.0 / (1.0 - x) }
}

fn quant_weights(params: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let mut bands = Vec::with_capacity(params.len());
    let mut last = params[0];
    if last <= 0.0 {
        return Err(invalid("non-positive dequant band"));
    }
    bands.push(last);
    for &p in &params[1..] {
        last *= band_mult(p);
        if last <= 0.0 {
            return Err(invalid("non-positive dequant band"));
        }
        bands.push(last);
    }
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 / (width - 1) as f32;
            let dy = y as f32 / (height - 1) as f32;
            let d = (dx * dx + dy * dy).sqrt();
            out.push(interpolate(d, std::f32::consts::SQRT_2 + 1e-6, &bands));
        }
    }
    Ok(out)
}

/// Multipliers (reciprocal weights) per channel, width x height row-major.
fn build_matrix(params: MatrixParams, idx: usize) -> Result<[Vec<f32>; 3]> {
    use MatrixParams::*;
    let (w, h) = MATRIX_SIZE[idx];
    let raw = matches!(params, Raw(..));
    let mut m: [Vec<f32>; 3] = match params {
        Dct(p) => [
            quant_weights(&p[0], w, h)?,
            quant_weights(&p[1], w, h)?,
            quant_weights(&p[2], w, h)?,
        ],
        Hornuss(p) => p.map(|p| {
            let mut r = vec![p[0]; 64];
            r[0] = 1.0;
            r[1] = p[1];
            r[8] = p[1];
            r[9] = p[2];
            r
        }),
        Dct2(p) => p.map(|p| {
            let mut r = vec![0f32; 64];
            r[0] = 1.0;
            for (i, v) in p.into_iter().enumerate() {
                let dim = 1usize << (i / 2);
                if i % 2 == 0 {
                    for y in 0..dim {
                        for x in dim..dim * 2 {
                            r[y * 8 + x] = v;
                            r[x * 8 + y] = v;
                        }
                    }
                } else {
                    for y in dim..dim * 2 {
                        for x in dim..dim * 2 {
                            r[y * 8 + x] = v;
                        }
                    }
                }
            }
            r
        }),
        Dct4(p, d) => {
            let mut out = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..3 {
                let mat = quant_weights(&d[c], 4, 4)?;
                let mut r = vec![0f32; 64];
                for y in 0..4 {
                    for x in 0..4 {
                        let v = mat[y * 4 + x];
                        r[y * 16 + x * 2] = v;
                        r[y * 16 + x * 2 + 1] = v;
                        r[(y * 2 + 1) * 8 + x * 2] = v;
                        r[(y * 2 + 1) * 8 + x * 2 + 1] = v;
                    }
                }
                r[1] /= p[c][0];
                r[8] /= p[c][0];
                r[9] /= p[c][1];
                out[c] = r;
            }
            out
        }
        Dct4x8(p, d) => {
            let mut out = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..3 {
                let mat = quant_weights(&d[c], 8, 4)?;
                let mut r: Vec<f32> = mat
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .flat_map(|row| [row, row])
                    .flatten()
                    .copied()
                    .collect();
                r[8] /= p[c][0];
                out[c] = r;
            }
            out
        }
        Afv(p, d, d4) => {
            const FREQS: [f32; 16] = [
                0.0, 0.0, 0.8517779, 5.3777843, 0.0, 0.0, 4.734748, 5.4492455, 1.659827, 4.0,
                7.275749, 10.423227, 2.6629324, 7.6306577, 8.962389, 12.971662,
            ];
            let (lo, hi) = (FREQS[2], FREQS[15]);
            let mut out = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..3 {
                let params = p[c];
                let w48 = quant_weights(&d[c], 8, 4)?;
                let w44 = quant_weights(&d4[c], 4, 4)?;
                let mut bands = [params[5], 0.0, 0.0, 0.0];
                for k in 1..4 {
                    bands[k] = bands[k - 1] * band_mult(params[5 + k]);
                }
                let mut r = vec![0f32; 64];
                for y in 0..4 {
                    for x in 0..4 {
                        r[16 * y + 2 * x] = match (x, y) {
                            (0, 0) => 1.0,
                            (0, 1) => params[2],
                            (1, 0) => params[3],
                            (1, 1) => params[4],
                            (x, y) => interpolate(FREQS[y * 4 + x] - lo, hi - lo + 1e-6, &bands),
                        };
                    }
                }
                for y in 0..4 {
                    for x in 0..8 {
                        r[16 * y + 8 + x] = if x == 0 && y == 0 {
                            params[0]
                        } else {
                            w48[y * 8 + x]
                        };
                    }
                    for x in 0..4 {
                        r[16 * y + 2 * x + 1] = if x == 0 && y == 0 {
                            params[1]
                        } else {
                            w44[y * 4 + x]
                        };
                    }
                }
                out[c] = r;
            }
            out
        }
        Raw(den, ch) => ch.map(|c| c.into_iter().map(|v| v as f32 * den).collect()),
    };
    for c in &mut m {
        if c.len() != w * h {
            return Err(invalid("dequant matrix has the wrong size"));
        }
        for v in c.iter_mut() {
            if !raw {
                *v = 1.0 / *v;
            }
            if !(*v > 0.0 && *v < 1e8) {
                return Err(invalid("dequant matrix entry out of range"));
            }
        }
    }
    Ok(m)
}

// ---------------------------------------------------------------------------
// Frame-global VarDCT parameters (LfGlobal)

pub(crate) struct LfGlobalVarDct {
    pub(crate) global_scale: u32,
    pub(crate) quant_lf: u32,
    qf_thresholds: Vec<u32>,
    lf_thresholds: [Vec<i32>; 3],
    block_ctx_map: Vec<u8>,
    num_block_clusters: usize,
    colour_factor: u32,
    base_corr_x: f32,
    base_corr_b: f32,
    x_factor_lf: u32,
    b_factor_lf: u32,
}

impl LfGlobalVarDct {
    pub(crate) fn read(br: &mut BitReader) -> Result<Self> {
        let global_scale = br.u32([bits(11, 1), bits(11, 2049), bits(12, 4097), bits(16, 8193)])?;
        let quant_lf = br.u32([val(16), bits(5, 1), bits(8, 1), bits(16, 1)])?;
        let mut qf_thresholds = Vec::new();
        let mut lf_thresholds: [Vec<i32>; 3] = Default::default();
        let (block_ctx_map, num_block_clusters) = if br.read_bool()? {
            (
                vec![
                    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14,
                    14, 14, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
                ],
                15,
            )
        } else {
            let mut bsize = 1usize;
            for t in &mut lf_thresholds {
                let n = br.read(4)?;
                bsize *= n as usize + 1;
                for _ in 0..n {
                    let v = br.u32([bits(4, 0), bits(8, 16), bits(16, 272), bits(32, 65808)])?;
                    t.push(unpack_signed(v));
                }
            }
            let n = br.read(4)?;
            bsize *= n as usize + 1;
            for _ in 0..n {
                qf_thresholds.push(1 + br.u32([bits(2, 0), bits(3, 4), bits(5, 12), bits(8, 44)])?);
            }
            if bsize * 39 > 39 * 64 {
                return Err(invalid("HF block context too large"));
            }
            let map = decode_context_map(br, bsize * 39)?;
            let n = map.iter().copied().max().unwrap_or(0) as usize + 1;
            if n > 16 {
                return Err(invalid("too many HF block clusters"));
            }
            (map, n)
        };
        let (mut colour_factor, mut base_corr_x, mut base_corr_b, mut x_factor_lf, mut b_factor_lf) =
            (84, 0.0, 1.0, 128, 128);
        if !br.read_bool()? {
            colour_factor = br.u32([val(84), val(256), bits(8, 2), bits(16, 258)])?;
            base_corr_x = br.f16()?;
            base_corr_b = br.f16()?;
            x_factor_lf = br.read(8)?;
            b_factor_lf = br.read(8)?;
            if base_corr_x.abs() > 4.0 || base_corr_b.abs() > 4.0 {
                return Err(invalid("base correlation out of range"));
            }
        }
        Ok(LfGlobalVarDct {
            global_scale,
            quant_lf,
            qf_thresholds,
            lf_thresholds,
            block_ctx_map,
            num_block_clusters,
            colour_factor,
            base_corr_x,
            base_corr_b,
            x_factor_lf,
            b_factor_lf,
        })
    }
}

// ---------------------------------------------------------------------------
// LF groups

#[derive(Debug, Clone, Copy)]
enum Block {
    Free,
    Occupied,
    Data { transform: usize, hf_mul: i32 },
}

/// Decoded data of one LF group.
pub(crate) struct LfGroup {
    /// Origin in 8x8 blocks.
    bx0: usize,
    by0: usize,
    bw: usize,
    bh: usize,
    /// Dequantised LF samples, XYB, bw x bh.
    lf: [Vec<f32>; 3],
    /// Quantised LF values (XYB) for the block contexts.
    lf_quant: [Vec<i32>; 3],
    x_from_y: Vec<i32>,
    b_from_y: Vec<i32>,
    /// Width of the colour correlation maps (in 64x64 tiles).
    cfl_w: usize,
    blocks: Vec<Block>,
    /// EPF sigma per block.
    pub(crate) sigma: Vec<f32>,
}

/// Parse the VarDCT part of an LF group section.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_lf_group(
    br: &mut BitReader,
    fh: &FrameHeader,
    lg: &LfGlobalVarDct,
    m_lf: [f32; 3],
    idx: u32,
    (px0, py0, pw, ph): (usize, usize, usize, usize),
    global: Option<&MaConfig>,
    alloc_limit: u64,
    after_lf: &mut dyn FnMut(&mut BitReader) -> Result<()>,
) -> Result<LfGroup> {
    let num_lf = fh.num_lf_groups();
    let (bw, bh) = (pw.div_ceil(8), ph.div_ceil(8));
    // LF coefficients.
    let extra_precision = br.read(2)?;
    let mut img = modular::Image {
        channels: (0..3).map(|_| Channel::new(bw, bh, 0, 0)).collect(),
        bit_depth: 8,
        ..Default::default()
    };
    modular::decode_stream(br, &mut img, 1 + idx, global, usize::MAX, true, alloc_limit)?;
    if img.channels.len() != 3 {
        return Err(invalid("LF image changed its channel count"));
    }
    let scale_inv = lg.global_scale as f64 * lg.quant_lf as f64;
    let mut lf: [Vec<f32>; 3] = Default::default();
    let mut lf_quant: [Vec<i32>; 3] = Default::default();
    // Modular channels are Y, X, B.
    for (c, src) in [(0usize, 1usize), (1, 0), (2, 2)] {
        let ch = &img.channels[src];
        let scale = (m_lf[c] as f64 * (1u32 << (9 - extra_precision)) as f64 / scale_inv) as f32;
        lf[c] = ch.data.iter().map(|&q| q as f32 * scale).collect();
        lf_quant[c] = ch.data.clone();
    }
    // Modular LF group (extra channels).
    after_lf(br)?;
    // HF metadata.
    let nb_blocks = 1 + br.read((bw * bh).next_power_of_two().trailing_zeros())? as usize;
    let cfl_w = pw.div_ceil(64);
    let cfl_h = ph.div_ceil(64);
    let mut meta = modular::Image {
        channels: vec![
            Channel::new(cfl_w, cfl_h, 0, 0),
            Channel::new(cfl_w, cfl_h, 0, 0),
            Channel::new(nb_blocks, 2, 0, 0),
            Channel::new(bw, bh, 0, 0),
        ],
        bit_depth: 8,
        ..Default::default()
    };
    modular::decode_stream(
        br,
        &mut meta,
        1 + 2 * num_lf + idx,
        global,
        usize::MAX,
        true,
        alloc_limit,
    )?;
    if meta.channels.len() != 4 {
        return Err(invalid("HF metadata changed its channel count"));
    }
    let mut ch = meta.channels.into_iter();
    let x_from_y = ch.next().expect("4 channels").data;
    let b_from_y = ch.next().expect("4 channels").data;
    let info = ch.next().expect("4 channels");
    let sharpness = ch.next().expect("4 channels").data;
    let epf_scale = fh.epf.quant_mul * 65536.0 / lg.global_scale as f32;
    let mut blocks = vec![Block::Free; bw * bh];
    let mut sigma = vec![0f32; bw * bh];
    let mut data_idx = 0usize;
    for y in 0..bh {
        let mut x = 0;
        while x < bw {
            if !matches!(blocks[y * bw + x], Block::Free) {
                x += 1;
                continue;
            }
            if data_idx >= nb_blocks {
                return Err(invalid("block info does not fill the LF group"));
            }
            let t = info.data[data_idx];
            if !(0..27).contains(&t) {
                return Err(invalid("invalid transform type"));
            }
            let transform = t as usize;
            let hf_mul = info.data[nb_blocks + data_idx]
                .checked_add(1)
                .ok_or_else(|| invalid("HfMul overflows"))?;
            if hf_mul <= 0 {
                return Err(invalid("non-positive HfMul"));
            }
            let (dw, dh, _, _) = TRANSFORMS[transform];
            if x % 32 + dw > 32 || y % 32 + dh > 32 || x + dw > bw || y + dh > bh {
                return Err(invalid("varblock crosses a group border"));
            }
            for dy in 0..dh {
                for dx in 0..dw {
                    let i = (y + dy) * bw + x + dx;
                    if !matches!(blocks[i], Block::Free) {
                        return Err(invalid("varblocks overlap"));
                    }
                    blocks[i] = if dx == 0 && dy == 0 {
                        Block::Data { transform, hf_mul }
                    } else {
                        Block::Occupied
                    };
                    let s = sharpness[i];
                    if !(0..8).contains(&s) {
                        return Err(invalid("invalid EPF sharpness"));
                    }
                    sigma[i] = epf_scale / hf_mul as f32 * fh.epf.sharp_lut[s as usize];
                }
            }
            data_idx += 1;
            x += dw;
        }
    }
    Ok(LfGroup {
        bx0: px0 / 8,
        by0: py0 / 8,
        bw,
        bh,
        lf,
        lf_quant,
        x_from_y,
        b_from_y,
        cfl_w,
        blocks,
        sigma,
    })
}

impl LfGroup {
    /// Copy this group's EPF sigma into a frame-wide block map.
    pub(crate) fn copy_sigma(&self, out: &mut [f32], stride: usize) {
        for y in 0..self.bh {
            let dst = (self.by0 + y) * stride + self.bx0;
            out[dst..dst + self.bw].copy_from_slice(&self.sigma[y * self.bw..(y + 1) * self.bw]);
        }
    }
}

// ---------------------------------------------------------------------------
// HfGlobal

/// Dequantisation multipliers of one param index, per channel.
struct Matrices {
    normal: [Vec<f32>; 3],
    /// For transposed transforms.
    transposed: [Vec<f32>; 3],
}

impl Matrices {
    fn build(p: MatrixParams, i: usize) -> Result<Self> {
        let normal = build_matrix(p, i)?;
        let (w, h) = MATRIX_SIZE[i];
        let transposed = std::array::from_fn(|c| {
            let src = &normal[c];
            // Transposed: width h, height w.
            (0..w * h).map(|k| src[(k % h) * w + k / h]).collect()
        });
        Ok(Matrices { normal, transposed })
    }

    /// The default matrices of param index `i`, built on first use.
    fn default(i: usize) -> &'static Matrices {
        static CACHE: [std::sync::OnceLock<Matrices>; 17] =
            [const { std::sync::OnceLock::new() }; 17];
        CACHE[i].get_or_init(|| {
            Matrices::build(default_params(i), i).expect("default matrices are valid")
        })
    }
}

pub(crate) struct HfGlobal {
    /// Signalled dequantisation matrices per param index; `None` for the
    /// defaults.
    custom: Option<Vec<Matrices>>,
    num_hf_presets: u32,
    passes: Vec<HfPass>,
}

struct HfPass {
    /// Coefficient order per order id and channel (empty = natural).
    orders: Vec<[Vec<(u16, u16)>; 3]>,
    histograms: Histograms,
}

pub(crate) fn read_hf_global(
    br: &mut BitReader,
    fh: &FrameHeader,
    lg: &LfGlobalVarDct,
    global: Option<&MaConfig>,
    alloc_limit: u64,
) -> Result<HfGlobal> {
    let num_lf = fh.num_lf_groups();
    let custom = if br.read_bool()? {
        None
    } else {
        Some(
            (0..17)
                .map(|i| {
                    let p =
                        read_matrix_params(br, i, 1 + 3 * num_lf + i as u32, global, alloc_limit)?;
                    Matrices::build(p, i)
                })
                .collect::<Result<_>>()?,
        )
    };
    let num_groups = fh.num_groups();
    let num_hf_presets = 1 + br.read(num_groups.next_power_of_two().trailing_zeros())?;
    let mut passes = Vec::with_capacity(fh.passes.num_passes as usize);
    for _ in 0..fh.passes.num_passes {
        let used = br.u32([val(0x5F), val(0x13), val(0), bits(13, 0)])?;
        let mut orders: Vec<[Vec<(u16, u16)>; 3]> = (0..13).map(|_| Default::default()).collect();
        if used != 0 {
            let h = Histograms::decode(br, 8)?;
            let mut reader = SymbolReader::new(&h, br, 0)?;
            for (o, slot) in orders.iter_mut().enumerate() {
                if used >> o & 1 == 0 {
                    continue;
                }
                let nat = natural_order(o);
                let size = nat.len() as u32;
                for c in slot.iter_mut() {
                    let perm = read_permutation(br, &h, &mut reader, size, size / 64)?;
                    *c = perm.into_iter().map(|p| nat[p as usize]).collect();
                }
            }
            reader.finish()?;
        }
        let ctx = 495 * num_hf_presets as usize * lg.num_block_clusters;
        let histograms = Histograms::decode(br, ctx)?;
        passes.push(HfPass { orders, histograms });
    }
    Ok(HfGlobal {
        custom,
        num_hf_presets,
        passes,
    })
}

// ---------------------------------------------------------------------------
// HF coefficients (pass groups)

const COEFF_FREQ_CONTEXT: [u32; 63] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
    20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27,
    27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];
const COEFF_NUM_NONZERO_CONTEXT: [u32; 63] = [
    0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152, 180,
    180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206,
];

/// Frame-wide integer coefficient grids (XYB) with their row stride.
pub(crate) struct Coefficients {
    pub(crate) stride: usize,
    pub(crate) data: [Vec<i32>; 3],
}

/// Decode the HF coefficients of pass group `(gx, gy)` (in 256-pixel
/// groups) into `coeffs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_hf_coefficients(
    br: &mut BitReader,
    fh: &FrameHeader,
    lg: &LfGlobalVarDct,
    hg: &HfGlobal,
    lf_groups: &[LfGroup],
    pass: usize,
    (gx, gy): (usize, usize),
    coeffs: &mut Coefficients,
) -> Result<()> {
    let hfp = br.read(hg.num_hf_presets.next_power_of_two().trailing_zeros())?;
    if hfp >= hg.num_hf_presets {
        return Err(invalid("HF preset out of range"));
    }
    let hp = &hg.passes[pass];
    let h = &hp.histograms;
    let nbc = lg.num_block_clusters;
    let ctx_offset = 495 * nbc * hfp as usize;
    let shift = pass_shift(fh, pass);
    let (lgx_count, _) = fh.lf_groups();
    let lf_idx_mul = lg
        .lf_thresholds
        .iter()
        .map(|t| t.len() + 1)
        .product::<usize>();
    let hf_idx_mul = lg.qf_thresholds.len() + 1;
    let mut reader = SymbolReader::new(h, br, 0)?;

    // Blocks of the group (32x32 blocks of the frame).
    let (frame_bw, frame_bh) = {
        let (w, h) = fh.sample_size();
        ((w as usize).div_ceil(8), (h as usize).div_ceil(8))
    };
    let bx0 = gx * 32;
    let by0 = gy * 32;
    let gbw = 32.min(frame_bw - bx0);
    let gbh = 32.min(frame_bh - by0);
    let lgi = (gy / 8) * lgx_count as usize + gx / 8;
    let group = &lf_groups[lgi];
    let mut nonzeros = vec![[0u32; 3]; gbw * gbh];
    for y in 0..gbh {
        for x in 0..gbw {
            let lx = bx0 + x - group.bx0;
            let ly = by0 + y - group.by0;
            let Block::Data { transform, hf_mul } = group.blocks[ly * group.bw + lx] else {
                continue;
            };
            let (w8, h8, _, order_id) = TRANSFORMS[transform];
            let num_blocks = (w8 * h8) as u32;
            let log_nb = num_blocks.trailing_zeros();
            // LF index (X, B, Y order of thresholds).
            let mut lf_idx = 0usize;
            for c in [0usize, 2, 1] {
                let t = &lg.lf_thresholds[c];
                lf_idx *= t.len() + 1;
                let q = group.lf_quant[c][ly * group.bw + lx];
                lf_idx += t.iter().filter(|&&th| q > th).count();
            }
            let hf_idx = lg
                .qf_thresholds
                .iter()
                .filter(|&&t| hf_mul > t as i32)
                .count();
            for (cy, c) in [1usize, 0, 2].into_iter().enumerate() {
                let ch_idx = cy * 13 + order_id;
                let idx = (ch_idx * hf_idx_mul + hf_idx) * lf_idx_mul + lf_idx;
                let block_ctx = *lg
                    .block_ctx_map
                    .get(idx)
                    .ok_or_else(|| invalid("block context out of range"))?
                    as usize;
                let predicted = match (x, y) {
                    (0, 0) => 32,
                    (_, 0) => nonzeros[y * gbw + x - 1][c],
                    (0, _) => nonzeros[(y - 1) * gbw + x][c],
                    _ => (nonzeros[(y - 1) * gbw + x][c] + nonzeros[y * gbw + x - 1][c] + 1) >> 1,
                };
                let nz_ctx_idx = if predicted >= 8 {
                    4 + predicted / 2
                } else {
                    predicted
                } as usize;
                let nz_ctx = ctx_offset + block_ctx + nz_ctx_idx * nbc;
                let mut nz = reader.read(h, br, nz_ctx)?;
                if nz > (63 << log_nb) {
                    return Err(invalid("too many non-zero coefficients"));
                }
                let nz_val = (nz + num_blocks - 1) >> log_nb;
                for dy in 0..h8 {
                    for dx in 0..w8 {
                        if y + dy < gbh && x + dx < gbw {
                            nonzeros[(y + dy) * gbw + x + dx][c] = nz_val;
                        }
                    }
                }
                if nz == 0 {
                    continue;
                }
                let order: &[(u16, u16)] = if hp.orders[order_id][c].is_empty() {
                    &natural_cache(order_id)
                } else {
                    &hp.orders[order_id][c]
                };
                let tr = need_transpose(transform);
                let base = ctx_offset + block_ctx * 458 + 37 * nbc;
                let mut prev = (nz <= num_blocks * 4) as u32;
                let grid = &mut coeffs.data[c];
                for (i, &(ox, oy)) in order[num_blocks as usize..].iter().enumerate() {
                    let k = ((nz - 1) >> log_nb) as usize;
                    let ctx = base
                        + ((COEFF_NUM_NONZERO_CONTEXT[k] + COEFF_FREQ_CONTEXT[i >> log_nb]) * 2
                            + prev) as usize;
                    if ctx - base >= 458 {
                        return Err(invalid("HF coefficient context out of range"));
                    }
                    let u = reader.read(h, br, ctx)?;
                    if u == 0 {
                        prev = 0;
                        continue;
                    }
                    let v = unpack_signed(u).wrapping_shl(shift);
                    let (dx, dy) = if tr { (oy, ox) } else { (ox, oy) };
                    let pos =
                        ((by0 + y) * 8 + dy as usize) * coeffs.stride + (bx0 + x) * 8 + dx as usize;
                    grid[pos] = grid[pos].wrapping_add(v);
                    prev = 1;
                    nz -= 1;
                    if nz == 0 {
                        break;
                    }
                }
                if nz != 0 {
                    return Err(invalid("non-zero count not reached"));
                }
            }
        }
    }
    reader.finish()
}

fn natural_cache(order_id: usize) -> Vec<(u16, u16)> {
    natural_order(order_id)
}

/// Coefficient shift of a pass.
fn pass_shift(fh: &FrameHeader, pass: usize) -> u32 {
    fh.passes.shift.get(pass).copied().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Reconstruction

/// Scale factors for LLF coefficients (`ScaleF`).
#[allow(clippy::excessive_precision)]
const SCALE_F: [f32; 32] = [
    1.0000000000000000,
    0.9996047255830407,
    0.9984194528776054,
    0.9964458326264695,
    0.9936866130906366,
    0.9901456355893141,
    0.9858278282666936,
    0.9807391980963174,
    0.9748868211368796,
    0.9682788310563117,
    0.9609244059440204,
    0.9528337534340876,
    0.9440180941651672,
    0.9344896436056892,
    0.9242615922757944,
    0.9133480844001980,
    0.9017641950288744,
    0.8895259056651056,
    0.8766500784429904,
    0.8631544288990163,
    0.8490574973847023,
    0.8343786191696513,
    0.8191378932865928,
    0.8033561501721485,
    0.7870549181591013,
    0.7702563888779096,
    0.7529833816270532,
    0.7352593067735488,
    0.7171081282466044,
    0.6985543251889097,
    0.6796228528314652,
    0.6603391026591464,
];

fn scale_f(c: usize, logb: usize) -> f32 {
    SCALE_F[c << logb]
}

/// DCT-II / DCT-III of power-of-two sizes (up to 256) with the JPEG XL
/// normalisation: the forward transform averages (DC = mean), the inverse
/// does not scale. Both use Lee's O(n log n) recursion, as libjxl does, on
/// all columns of a block at once (`w` lanes), so every step is a loop over
/// whole rows; rows are transformed as the columns of the transpose.
struct Dct {
    /// Per log2 size n: 1 / (2 cos((2i + 1) pi / 2n)) for i < n / 2.
    mul: Vec<Vec<f32>>,
    /// Scratch blocks.
    a: Vec<f32>,
    b: Vec<f32>,
    tmp: Vec<f32>,
}

const MAX_DCT: usize = 256;

/// Rows `i` and `j > i` of `x` (`w` wide), mutably.
#[inline]
fn two_rows(x: &mut [f32], w: usize, i: usize, j: usize) -> (&mut [f32], &mut [f32]) {
    let (lo, hi) = x.split_at_mut(j * w);
    (&mut lo[i * w..][..w], &mut hi[..w])
}

/// Column-wise y_i = sum_k x_k cos((2i + 1) k pi / 2n) of the n x w block
/// `x`, in place; `tmp` is scratch of the same size.
fn dct3(mul: &[Vec<f32>], x: &mut [f32], tmp: &mut [f32], n: usize, w: usize) {
    match n {
        2 => dct3_n::<2>(mul, x, tmp, w),
        4 => dct3_n::<4>(mul, x, tmp, w),
        8 => dct3_n::<8>(mul, x, tmp, w),
        16 => dct3_n::<16>(mul, x, tmp, w),
        32 => dct3_n::<32>(mul, x, tmp, w),
        64 => dct3_n::<64>(mul, x, tmp, w),
        128 => dct3_n::<128>(mul, x, tmp, w),
        256 => dct3_n::<256>(mul, x, tmp, w),
        _ => {}
    }
}

/// Column-wise z_k = sum_i x_i cos((2i + 1) k pi / 2n) (the transpose of
/// `dct3`).
fn dct2(mul: &[Vec<f32>], x: &mut [f32], tmp: &mut [f32], n: usize, w: usize) {
    match n {
        2 => dct2_n::<2>(mul, x, tmp, w),
        4 => dct2_n::<4>(mul, x, tmp, w),
        8 => dct2_n::<8>(mul, x, tmp, w),
        16 => dct2_n::<16>(mul, x, tmp, w),
        32 => dct2_n::<32>(mul, x, tmp, w),
        64 => dct2_n::<64>(mul, x, tmp, w),
        128 => dct2_n::<128>(mul, x, tmp, w),
        256 => dct2_n::<256>(mul, x, tmp, w),
        _ => {}
    }
}

/// `dct3` for size `N`; the recursion halves go through `dct3`, which
/// dispatches to the specialisation of size `N / 2`.
#[inline(always)]
fn dct3_n<const N: usize>(mul: &[Vec<f32>], x: &mut [f32], tmp: &mut [f32], w: usize) {
    if N == 2 {
        let (r0, r1) = x[..2 * w].split_at_mut(w);
        for (a, b) in r0.iter_mut().zip(r1) {
            let (p, q) = (*a, *b * std::f32::consts::FRAC_1_SQRT_2);
            *a = p + q;
            *b = p - q;
        }
        return;
    }
    let h = N / 2;
    let (even, odd) = tmp[..N * w].split_at_mut(h * w);
    for k in 0..h {
        even[k * w..][..w].copy_from_slice(&x[2 * k * w..][..w]);
        let o = &mut odd[k * w..][..w];
        let cur = &x[(2 * k + 1) * w..][..w];
        if k == 0 {
            o.copy_from_slice(cur);
        } else {
            let prev = &x[(2 * k - 1) * w..][..w];
            for ((o, &c), &p) in o.iter_mut().zip(cur).zip(prev) {
                *o = c + p;
            }
        }
    }
    let (s0, s1) = x[..N * w].split_at_mut(h * w);
    dct3(mul, even, s0, h, w);
    dct3(mul, odd, s1, h, w);
    let m = &mul[N.trailing_zeros() as usize];
    for i in 0..h {
        let mi = m[i];
        let (e, o) = (&even[i * w..][..w], &odd[i * w..][..w]);
        let (xi, xj) = two_rows(x, w, i, N - 1 - i);
        for (((xi, xj), &e), &o) in xi.iter_mut().zip(xj).zip(e).zip(o) {
            let o = o * mi;
            *xi = e + o;
            *xj = e - o;
        }
    }
}

/// `dct2` for size `N`.
#[inline(always)]
fn dct2_n<const N: usize>(mul: &[Vec<f32>], x: &mut [f32], tmp: &mut [f32], w: usize) {
    if N == 2 {
        let (r0, r1) = x[..2 * w].split_at_mut(w);
        for (a, b) in r0.iter_mut().zip(r1) {
            let (p, q) = (*a, *b);
            *a = p + q;
            *b = (p - q) * std::f32::consts::FRAC_1_SQRT_2;
        }
        return;
    }
    let h = N / 2;
    let m = &mul[N.trailing_zeros() as usize];
    let (even, odd) = tmp[..N * w].split_at_mut(h * w);
    for i in 0..h {
        let mi = m[i];
        let (xi, xj) = two_rows(x, w, i, N - 1 - i);
        let (e, o) = (&mut even[i * w..][..w], &mut odd[i * w..][..w]);
        for (((e, o), &a), &b) in e.iter_mut().zip(o).zip(&*xi).zip(&*xj) {
            *e = a + b;
            *o = (a - b) * mi;
        }
    }
    let (s0, s1) = x[..N * w].split_at_mut(h * w);
    dct2(mul, even, s0, h, w);
    dct2(mul, odd, s1, h, w);
    for k in 0..h {
        x[2 * k * w..][..w].copy_from_slice(&even[k * w..][..w]);
        let dst = &mut x[(2 * k + 1) * w..][..w];
        if k + 1 < h {
            let (cur, next) = (&odd[k * w..][..w], &odd[(k + 1) * w..][..w]);
            for ((d, &c), &nx) in dst.iter_mut().zip(cur).zip(next) {
                *d = c + nx;
            }
        } else {
            dst.copy_from_slice(&odd[k * w..][..w]);
        }
    }
}

/// The JPEG XL transform along the columns of the n x w block `x`.
fn transform_cols(
    mul: &[Vec<f32>],
    x: &mut [f32],
    tmp: &mut [f32],
    n: usize,
    w: usize,
    inverse: bool,
) {
    if n <= 1 {
        return;
    }
    let sqrt2 = std::f32::consts::SQRT_2;
    let x = &mut x[..n * w];
    if inverse {
        // x_i = c_0 + sum_k sqrt(2) c_k cos(...)
        for c in &mut x[w..] {
            *c *= sqrt2;
        }
        dct3(mul, x, tmp, n, w);
    } else {
        dct2(mul, x, tmp, n, w);
        let inv = 1.0 / n as f32;
        for c in &mut x[..w] {
            *c *= inv;
        }
        for c in &mut x[w..] {
            *c *= sqrt2 * inv;
        }
    }
}

/// `dst` (w x h) = transpose of `src` (h x w).
fn transpose(src: &[f32], dst: &mut [f32], w: usize, h: usize) {
    for (y, row) in src[..w * h].chunks_exact(w).enumerate() {
        for (x, &v) in row.iter().enumerate() {
            dst[x * h + y] = v;
        }
    }
}

impl Dct {
    fn new() -> Self {
        let mul = (0..=8)
            .map(|l| {
                let n = 1usize << l;
                (0..n / 2)
                    .map(|i| {
                        let a = (2 * i + 1) as f64 * std::f64::consts::PI / (2 * n) as f64;
                        (0.5 / a.cos()) as f32
                    })
                    .collect()
            })
            .collect();
        Dct {
            mul,
            a: vec![0.0; MAX_DCT * MAX_DCT],
            b: vec![0.0; MAX_DCT * MAX_DCT],
            tmp: vec![0.0; MAX_DCT * MAX_DCT],
        }
    }

    /// The JPEG XL 1-D transform of `v` in place.
    #[cfg(test)]
    fn transform_1d(&mut self, v: &mut [f32], inverse: bool) {
        transform_cols(&self.mul, v, &mut self.tmp, v.len(), 1, inverse);
    }

    /// 2D transform of a w x h region at `data[..]` with row stride `stride`.
    fn transform_2d(&mut self, data: &mut [f32], stride: usize, w: usize, h: usize, inverse: bool) {
        let (a, b) = (&mut self.a[..w * h], &mut self.b[..w * h]);
        for (y, row) in a.chunks_exact_mut(w).enumerate() {
            row.copy_from_slice(&data[y * stride..][..w]);
        }
        // Rows first, as columns of the transpose, then the columns.
        transpose(a, b, w, h);
        transform_cols(&self.mul, b, &mut self.tmp, w, h, inverse);
        transpose(b, a, h, w);
        transform_cols(&self.mul, a, &mut self.tmp, h, w, inverse);
        for (y, row) in a.chunks_exact(w).enumerate() {
            data[y * stride..][..w].copy_from_slice(row);
        }
    }
}

/// An 8x8 block view helper: (x, y) -> index.
#[inline]
fn at(x: usize, y: usize) -> usize {
    y * 8 + x
}

fn aux_idct2(b: &mut [f32; 64], size: usize) {
    let n = size / 2;
    let mut s = [0f32; 64];
    for y in 0..n {
        for x in 0..n {
            let c00 = b[at(x, y)];
            let c01 = b[at(x + n, y)];
            let c10 = b[at(x, y + n)];
            let c11 = b[at(x + n, y + n)];
            s[at(2 * x, 2 * y)] = c00 + c01 + c10 + c11;
            s[at(2 * x + 1, 2 * y)] = c00 + c01 - c10 - c11;
            s[at(2 * x, 2 * y + 1)] = c00 - c01 + c10 - c11;
            s[at(2 * x + 1, 2 * y + 1)] = c00 - c01 - c10 + c11;
        }
    }
    for y in 0..size {
        for x in 0..size {
            b[at(x, y)] = s[at(x, y)];
        }
    }
}

fn transform_dct2(b: &mut [f32; 64]) {
    aux_idct2(b, 2);
    aux_idct2(b, 4);
    aux_idct2(b, 8);
}

fn transform_dct4(b: &mut [f32; 64], dct: &mut Dct) {
    aux_idct2(b, 2);
    let mut scratch = [0f32; 64];
    for y in 0..2 {
        for x in 0..2 {
            let sub = &mut scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    // Transposed placement.
                    sub[ix * 4 + iy] = b[at(x + ix * 2, y + iy * 2)];
                }
            }
            dct.transform_2d(sub, 4, 4, 4, true);
        }
    }
    for y in 0..2 {
        for x in 0..2 {
            let sub = &scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    b[at(x * 4 + ix, y * 4 + iy)] = sub[iy * 4 + ix];
                }
            }
        }
    }
}

fn transform_hornuss(b: &mut [f32; 64]) {
    aux_idct2(b, 2);
    let mut scratch = [0f32; 64];
    for y in 0..2 {
        for x in 0..2 {
            let sub = &mut scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    sub[iy * 4 + ix] = b[at(x + ix * 2, y + iy * 2)];
                }
            }
            let residual: f32 = sub[1..].iter().sum();
            let avg = sub[0] - residual / 16.0;
            sub[0] = sub[5];
            sub[5] = 0.0;
            for s in sub.iter_mut() {
                *s += avg;
            }
        }
    }
    for y in 0..2 {
        for x in 0..2 {
            let sub = &scratch[(y * 2 + x) * 16..][..16];
            for iy in 0..4 {
                for ix in 0..4 {
                    b[at(x * 4 + ix, y * 4 + iy)] = sub[iy * 4 + ix];
                }
            }
        }
    }
}

fn transform_dct4x8(b: &mut [f32; 64], transposed: bool, dct: &mut Dct) {
    let (c0, c1) = (b[at(0, 0)], b[at(0, 1)]);
    b[at(0, 0)] = c0 + c1;
    b[at(0, 1)] = c0 - c1;
    let mut scratch = [0f32; 64];
    for idx in 0..2 {
        let sub = &mut scratch[idx * 32..][..32];
        for iy in 0..4 {
            for ix in 0..8 {
                sub[iy * 8 + ix] = b[at(ix, iy * 2 + idx)];
            }
        }
        dct.transform_2d(sub, 8, 8, 4, true);
    }
    for y in 0..8 {
        for x in 0..8 {
            if transposed {
                b[at(y, x)] = scratch[y * 8 + x];
            } else {
                b[at(x, y)] = scratch[y * 8 + x];
            }
        }
    }
}

#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const AFV_BASIS: [[f32; 16]; 16] = [
    [0.25; 16],
    [0.876902929799142, 0.2206518106944235, -0.10140050393753763, -0.1014005039375375, 0.2206518106944236, -0.10140050393753777, -0.10140050393753772, -0.10140050393753763, -0.10140050393753758, -0.10140050393753769, -0.1014005039375375, -0.10140050393753768, -0.10140050393753768, -0.10140050393753759, -0.10140050393753763, -0.10140050393753741],
    [0.0, 0.0, 0.40670075830260755, 0.44444816619734445, 0.0, 0.0, 0.19574399372042936, 0.2929100136981264, -0.40670075830260716, -0.19574399372042872, 0.0, 0.11379074460448091, -0.44444816619734384, -0.29291001369812636, -0.1137907446044814, 0.0],
    [0.0, 0.0, -0.21255748058288748, 0.3085497062849767, 0.0, 0.4706702258572536, -0.1621205195722993, 0.0, -0.21255748058287047, -0.16212051957228327, -0.47067022585725277, -0.1464291867126764, 0.3085497062849487, 0.0, -0.14642918671266536, 0.4251149611657548],
    [0.0, -0.7071067811865474, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-0.4105377591765233, 0.6235485373547691, -0.06435071657946274, -0.06435071657946266, 0.6235485373547694, -0.06435071657946284, -0.0643507165794628, -0.06435071657946274, -0.06435071657946272, -0.06435071657946279, -0.06435071657946266, -0.06435071657946277, -0.06435071657946277, -0.06435071657946273, -0.06435071657946274, -0.0643507165794626],
    [0.0, 0.0, -0.4517556589999482, 0.15854503551840063, 0.0, -0.04038515160822202, 0.0074182263792423875, 0.39351034269210167, -0.45175565899994635, 0.007418226379244351, 0.1107416575309343, 0.08298163094882051, 0.15854503551839705, 0.3935103426921022, 0.0829816309488214, -0.45175565899994796],
    [0.0, 0.0, -0.304684750724869, 0.5112616136591823, 0.0, 0.0, -0.290480129728998, -0.06578701549142804, 0.304684750724884, 0.2904801297290076, 0.0, -0.23889773523344604, -0.5112616136592012, 0.06578701549142545, 0.23889773523345467, 0.0],
    [0.0, 0.0, 0.3017929516615495, 0.25792362796341184, 0.0, 0.16272340142866204, 0.09520022653475037, 0.0, 0.3017929516615503, 0.09520022653475055, -0.16272340142866173, -0.35312385449816297, 0.25792362796341295, 0.0, -0.3531238544981624, -0.6035859033230976],
    [0.0, 0.0, 0.40824829046386274, 0.0, 0.0, 0.0, 0.0, -0.4082482904638628, -0.4082482904638635, 0.0, 0.0, -0.40824829046386296, 0.0, 0.4082482904638634, 0.408248290463863, 0.0],
    [0.0, 0.0, 0.1747866975480809, 0.0812611176717539, 0.0, 0.0, -0.3675398009862027, -0.307882213957909, -0.17478669754808135, 0.3675398009862011, 0.0, 0.4826689115059883, -0.08126111767175039, 0.30788221395790305, -0.48266891150598584, 0.0],
    [0.0, 0.0, -0.21105601049335784, 0.18567180916109802, 0.0, 0.0, 0.49215859013738733, -0.38525013709251915, 0.21105601049335806, -0.49215859013738905, 0.0, 0.17419412659916217, -0.18567180916109904, 0.3852501370925211, -0.1741941265991621, 0.0],
    [0.0, 0.0, -0.14266084808807264, -0.3416446842253372, 0.0, 0.7367497537172237, 0.24627107722075148, -0.08574019035519306, -0.14266084808807344, 0.24627107722075137, 0.14883399227113567, -0.04768680350229251, -0.3416446842253373, -0.08574019035519267, -0.047686803502292804, -0.14266084808807242],
    [0.0, 0.0, -0.13813540350758585, 0.3302282550303788, 0.0, 0.08755115000587084, -0.07946706605909573, -0.4613374887461511, -0.13813540350758294, -0.07946706605910261, 0.49724647109535086, 0.12538059448563663, 0.3302282550303805, -0.4613374887461554, 0.12538059448564315, -0.13813540350758452],
    [0.0, 0.0, -0.17437602599651067, 0.0702790691196284, 0.0, -0.2921026642334881, 0.3623817333531167, 0.0, -0.1743760259965108, 0.36238173335311646, 0.29210266423348785, -0.4326608024727445, 0.07027906911962818, 0.0, -0.4326608024727457, 0.34875205199302267],
    [0.0, 0.0, 0.11354987314994337, -0.07417504595810355, 0.0, 0.19402893032594343, -0.435190496523228, 0.21918684838857466, 0.11354987314994257, -0.4351904965232251, 0.5550443808910661, -0.25468277124066463, -0.07417504595810233, 0.2191868483885728, -0.25468277124066413, 0.1135498731499429],
];

fn transform_afv(b: &mut [f32; 64], n: usize, dct: &mut Dct) {
    let flip_x = n % 2;
    let flip_y = n / 2;
    let mut coeff_afv = [0f32; 16];
    coeff_afv[0] = (b[at(0, 0)] + b[at(1, 0)] + b[at(0, 1)]) * 4.0;
    for (idx, v) in coeff_afv.iter_mut().enumerate().skip(1) {
        *v = b[at(2 * (idx % 4), 2 * (idx / 4))];
    }
    let mut samples = [0f32; 16];
    for (c, basis) in coeff_afv.iter().zip(&AFV_BASIS) {
        for (s, &bv) in samples.iter_mut().zip(basis) {
            *s += c * bv;
        }
    }
    let mut s44 = [0f32; 16];
    s44[0] = b[at(0, 0)] - b[at(1, 0)] + b[at(0, 1)];
    for iy in 0..4 {
        for ix in 0..4 {
            if ix | iy != 0 {
                s44[ix * 4 + iy] = b[at(2 * ix + 1, 2 * iy)];
            }
        }
    }
    dct.transform_2d(&mut s44, 4, 4, 4, true);
    let mut s48 = [0f32; 32];
    s48[0] = b[at(0, 0)] - b[at(0, 1)];
    for iy in 0..4 {
        for ix in 0..8 {
            if ix | iy != 0 {
                s48[iy * 8 + ix] = b[at(ix, 2 * iy + 1)];
            }
        }
    }
    dct.transform_2d(&mut s48, 8, 8, 4, true);
    for iy in 0..4 {
        let ay = if flip_y == 0 { iy } else { 3 - iy };
        for ix in 0..4 {
            let ax = if flip_x == 0 { ix } else { 3 - ix };
            b[at(flip_x * 4 + ix, flip_y * 4 + iy)] = samples[ay * 4 + ax];
        }
    }
    for iy in 0..4 {
        for ix in 0..4 {
            b[at((1 - flip_x) * 4 + ix, flip_y * 4 + iy)] = s44[iy * 4 + ix];
        }
    }
    for iy in 0..4 {
        for ix in 0..8 {
            b[at(ix, (1 - flip_y) * 4 + iy)] = s48[iy * 8 + ix];
        }
    }
}

/// Adaptive LF smoothing (I.2.3) over frame-wide XYB LF planes.
fn adaptive_lf_smoothing(lf: &mut [Vec<f32>; 3], w: usize, h: usize, lf_scale: [f32; 3]) {
    const SELF: f32 = 0.052262735;
    const SIDE: f32 = 0.2034514;
    const DIAG: f32 = 0.03348292;
    if w <= 2 || h <= 2 {
        return;
    }
    let src: [Vec<f32>; 3] = lf.clone();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut wa = [0f32; 3];
            let mut gap = 0.5f32;
            for c in 0..3 {
                let p = |dx: isize, dy: isize| {
                    src[c][(y as isize + dy) as usize * w + (x as isize + dx) as usize]
                };
                let s = p(0, 0);
                let side = p(-1, 0) + p(1, 0) + p(0, -1) + p(0, 1);
                let diag = p(-1, -1) + p(1, -1) + p(-1, 1) + p(1, 1);
                wa[c] = s * SELF + side * SIDE + diag * DIAG;
                gap = gap.max((wa[c] - s).abs() / lf_scale[c]);
            }
            let g = (3.0 - 4.0 * gap).max(0.0);
            for c in 0..3 {
                let s = src[c][y * w + x];
                lf[c][y * w + x] = (wa[c] - s) * g + s;
            }
        }
    }
}

/// Reconstruct XYB samples (frame size rounded up to 8) from the decoded
/// coefficients and LF groups.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reconstruct(
    fh: &FrameHeader,
    lg: &LfGlobalVarDct,
    hg: &HfGlobal,
    lf_groups: &[LfGroup],
    coeffs: Coefficients,
    m_lf: [f32; 3],
    quant_bias: [f32; 3],
    quant_bias_numerator: f32,
) -> Result<[Vec<f32>; 3]> {
    let stride = coeffs.stride;
    let (sw, sh) = fh.sample_size();
    let (bw, bh) = ((sw as usize).div_ceil(8), (sh as usize).div_ceil(8));
    // Frame-wide LF planes.
    let mut lf: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; bw * bh]);
    for g in lf_groups {
        for c in 0..3 {
            for y in 0..g.bh {
                let dst = (g.by0 + y) * bw + g.bx0;
                lf[c][dst..dst + g.bw].copy_from_slice(&g.lf[c][y * g.bw..(y + 1) * g.bw]);
            }
        }
    }
    let kx_lf = lg.base_corr_x + (lg.x_factor_lf as i32 - 128) as f32 / lg.colour_factor as f32;
    let kb_lf = lg.base_corr_b + (lg.b_factor_lf as i32 - 128) as f32 / lg.colour_factor as f32;
    for i in 0..bw * bh {
        let y = lf[1][i];
        lf[0][i] += kx_lf * y;
        lf[2][i] += kb_lf * y;
    }
    if fh.flags & FLAG_SKIP_ADAPTIVE_LF_SMOOTHING == 0 {
        let scale_inv = lg.global_scale as f64 * lg.quant_lf as f64;
        let s = m_lf.map(|m| (512.0 * m as f64 / scale_inv) as f32);
        adaptive_lf_smoothing(&mut lf, bw, bh, s);
    }

    let qm_scale = [
        0.8f32.powi(fh.x_qm_scale as i32 - 2),
        1.0,
        0.8f32.powi(fh.b_qm_scale as i32 - 2),
    ];
    let mut dct = Dct::new();
    let mut out: [Vec<f32>; 3] = coeffs
        .data
        .map(|d| d.into_iter().map(|v| v as f32).collect());
    let (lgx_count, _) = fh.lf_groups();
    // Dequantised coefficients of the current varblock, per channel.
    let mut deq: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0f32; MAX_DCT * MAX_DCT]);
    let mut llf = vec![0f32; 32 * 32];
    for g in lf_groups {
        for ly in 0..g.bh {
            for lx in 0..g.bw {
                let Block::Data { transform, hf_mul } = g.blocks[ly * g.bw + lx] else {
                    continue;
                };
                let (w8, h8, param, _) = TRANSFORMS[transform];
                let (pw, ph) = (w8 * 8, h8 * 8);
                let fx = (g.bx0 + lx) * 8;
                let fy = (g.by0 + ly) * 8;
                let tr = need_transpose(transform);
                // Colour correlation for this varblock's 64x64 tile.
                let tile = (ly / 8) * g.cfl_w + lx / 8;
                let kx = lg.base_corr_x + g.x_from_y[tile] as f32 / lg.colour_factor as f32;
                let kb = lg.base_corr_b + g.b_from_y[tile] as f32 / lg.colour_factor as f32;
                let mul_base = 65536.0 / (lg.global_scale as f32 * hf_mul as f32);
                // Dequantise all three channels first (CfL needs Y).
                for c in 0..3 {
                    let m = match &hg.custom {
                        Some(custom) => &custom[param],
                        None => Matrices::default(param),
                    };
                    let matrix = if tr { &m.transposed[c] } else { &m.normal[c] };
                    let mul = mul_base * qm_scale[c];
                    let bias = quant_bias[c];
                    let d = &mut deq[c][..pw * ph];
                    for ((drow, mrow), y) in d
                        .chunks_exact_mut(pw)
                        .zip(matrix.chunks_exact(pw))
                        .zip(fy..)
                    {
                        let src = &out[c][y * stride + fx..][..pw];
                        for ((d, &m), &q) in drow.iter_mut().zip(mrow).zip(src) {
                            let q = if q.abs() <= 1.0 {
                                q * bias
                            } else {
                                q - quant_bias_numerator / q
                            };
                            *d = q * m * mul;
                        }
                    }
                }
                {
                    let [d0, d1, d2] = &mut deq;
                    for ((x, b), &y) in d0[..pw * ph]
                        .iter_mut()
                        .zip(&mut d2[..pw * ph])
                        .zip(&d1[..pw * ph])
                    {
                        *x += kx * y;
                        *b += kb * y;
                    }
                }
                for c in 0..3 {
                    let b = &mut deq[c][..pw * ph];
                    // LLF coefficients from the LF image.
                    let bx = g.bx0 + lx;
                    let by = g.by0 + ly;
                    if w8 == 1 && h8 == 1 {
                        b[0] = lf[c][by * bw + bx];
                    } else {
                        let llf = &mut llf[..w8 * h8];
                        for y in 0..h8 {
                            for x in 0..w8 {
                                llf[y * w8 + x] = lf[c][(by + y) * bw + bx + x];
                            }
                        }
                        dct.transform_2d(llf, w8, w8, h8, false);
                        let (lw, lh) = (w8.trailing_zeros() as usize, h8.trailing_zeros() as usize);
                        for y in 0..h8 {
                            for x in 0..w8 {
                                b[y * pw + x] =
                                    llf[y * w8 + x] / (scale_f(y, 5 - lh) * scale_f(x, 5 - lw));
                            }
                        }
                    }
                    if pw == 8 && ph == 8 && transform != 0 {
                        let mut b8 = [0f32; 64];
                        b8.copy_from_slice(b);
                        match transform {
                            HORNUSS => transform_hornuss(&mut b8),
                            DCT2 => transform_dct2(&mut b8),
                            DCT4 => transform_dct4(&mut b8, &mut dct),
                            DCT4X8 => transform_dct4x8(&mut b8, false, &mut dct),
                            DCT8X4 => transform_dct4x8(&mut b8, true, &mut dct),
                            t => transform_afv(&mut b8, t - AFV0, &mut dct),
                        }
                        b.copy_from_slice(&b8);
                    } else {
                        dct.transform_2d(b, pw, pw, ph, true);
                    }
                    for y in 0..ph {
                        let dst = (fy + y) * stride + fx;
                        out[c][dst..dst + pw].copy_from_slice(&b[y * pw..(y + 1) * pw]);
                    }
                }
            }
        }
    }
    let _ = lgx_count;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The JPEG XL DCT pair evaluated directly in f64.
    fn reference(v: &[f32], inverse: bool) -> Vec<f64> {
        let n = v.len();
        let t = |k: usize, i: usize| {
            if k == 0 {
                1.0
            } else {
                std::f64::consts::SQRT_2
                    * ((2 * i + 1) as f64 * k as f64 * std::f64::consts::PI / (2 * n) as f64).cos()
            }
        };
        (0..n)
            .map(|o| {
                if inverse {
                    (0..n).map(|k| v[k] as f64 * t(k, o)).sum()
                } else {
                    (0..n).map(|i| v[i] as f64 * t(o, i)).sum::<f64>() / n as f64
                }
            })
            .collect()
    }

    #[test]
    fn fast_dct_matches_definition() {
        let mut dct = Dct::new();
        let mut seed = 1u32;
        for l in 0..=8 {
            let n = 1 << l;
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                    (seed >> 8) as f32 / (1 << 23) as f32 - 1.0
                })
                .collect();
            for inverse in [false, true] {
                let mut got = v.clone();
                dct.transform_1d(&mut got, inverse);
                let want = reference(&v, inverse);
                let scale = if inverse { n as f64 } else { 1.0 };
                for (g, w) in got.iter().zip(&want) {
                    assert!(
                        (*g as f64 - w).abs() < 2e-5 * scale,
                        "n {n} inverse {inverse}"
                    );
                }
            }
            let mut rt = v.clone();
            dct.transform_1d(&mut rt, false);
            dct.transform_1d(&mut rt, true);
            for (a, b) in rt.iter().zip(&v) {
                assert!((a - b).abs() < 1e-4, "round trip n {n}");
            }
        }
    }
}
