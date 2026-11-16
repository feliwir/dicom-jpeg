//! JPEG 2000 Part 1 encoder: one tile, one quality layer, 64x64 code-blocks,
//! one precinct per resolution. Lossless uses the 5/3 wavelet with RCT;
//! lossy uses the 9/7 wavelet with ICT and per-band step sizes scaled by the
//! synthesis gain of each sub-band, so quantization noise is spread evenly.
//! With `high_throughput` every code-block is coded as a single HTJ2K
//! cleanup pass (ISO/IEC 15444-15) instead of EBCOT.

use super::CODEC;
use super::codestream::{ComponentCoding, ComponentSiz, GlobalCoding, Quantization, TileParams};
use super::dwt::{Analyse, Plane, analyse_level, synth_level};
use super::ht::encode::encode as ht_encode;
use super::t1::Orient;
use super::t1_enc::{self, EncodedBlock};
use super::tagtree::TagTreeEncoder;
use super::tile::{TileComponent, packet_order};
use dicom_jpeg_common::color::{rct, ycbcr};
use dicom_jpeg_common::{AppliedTransform, ColorSpace, Encoded, Image};
use dicom_jpeg_common::{Error, Result};

/// JPEG 2000 specific encoder parameters.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EncoderOptions {
    /// Mathematically lossless coding (5/3 wavelet, RCT).
    pub reversible: bool,
    /// Quality 1..=100 for lossy coding (ignored when `reversible`).
    pub quality: u8,
    /// Number of wavelet decomposition levels (`None` = up to 5, limited by the image size).
    pub levels: Option<u8>,
    /// Progression order: 0 LRCP, 1 RLCP, 2 RPCL, 3 PCRL, 4 CPRL.
    pub progression: u8,
    /// Apply the inverse component transform signalling (RCT / ICT) to RGB input.
    pub color_transform: bool,
    /// Use the HTJ2K block coder (ISO/IEC 15444-15) instead of EBCOT; writes
    /// a CAP marker and code-block style 0x40.
    pub high_throughput: bool,
    /// Write a TLM marker. With a resolution-major progression (RLCP, RPCL)
    /// every resolution level also gets its own tile-part, so a decoder can
    /// stop after any resolution.
    pub tlm: bool,
    /// JPEG 2000 Part 2 wavelet-based multiple component transform
    /// (ITU-T T.801 J.3.2) along the component axis: 5/3 when `reversible`,
    /// 9/7 otherwise. Replaces RCT / ICT and writes the CBD, MCT, MCC and
    /// MCO marker segments.
    pub component_wavelet: Option<ComponentWavelet>,
}

/// Parameters of the Part 2 component-axis wavelet transform.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ComponentWavelet {
    /// Decomposition levels along the components (`None` = up to 5,
    /// limited by the component count).
    pub levels: Option<u8>,
    /// Interleave factor `k`: the transform runs over components
    /// 0, k, 2k, ..., then 1, k + 1, ..., so that interleaved samples (e.g.
    /// the R, G, B planes of consecutive frames) form smooth runs. 0 and 1
    /// keep the natural order.
    pub interleave: u16,
}

impl ComponentWavelet {
    /// A component wavelet transform with the given interleave factor.
    pub fn new(levels: Option<u8>, interleave: u16) -> Self {
        ComponentWavelet { levels, interleave }
    }
}

impl Default for EncoderOptions {
    fn default() -> Self {
        EncoderOptions {
            reversible: true,
            quality: 90,
            levels: None,
            progression: 0,
            color_transform: true,
            high_throughput: false,
            tlm: false,
            component_wavelet: None,
        }
    }
}

const GUARD_BITS_MAX: u32 = 7;

fn band_gain(o: Orient) -> u32 {
    match o {
        Orient::LL => 0,
        Orient::HL | Orient::LH => 1,
        Orient::HH => 2,
    }
}

/// L2 norm of the 9/7 synthesis basis function of a coefficient in the
/// sub-band at decomposition level `level` (1 = finest) with orientation `o`.
fn synthesis_norm97(level: u32, o: Orient) -> f64 {
    // Separable: compute the 1-D norm for each direction.
    fn norm1d(level: u32, high: bool) -> f64 {
        let n = 1usize << (level + 5);
        let mut cur: Plane<f32> = Plane::new(0, 0, (n >> level) as u32, 1);
        let mut hi_band: Plane<f32> = Plane::new(0, 0, (n >> level) as u32, 1);
        let mid = (n >> level) / 2;
        if high {
            hi_band.data[mid] = 1.0;
        } else {
            cur.data[mid] = 1.0;
        }
        let zero_rows = |w: u32| Plane::<f32>::new(0, 0, w, 0);
        for l in (1..=level).rev() {
            let w = (n >> (l - 1)) as u32;
            let hb = if l == level {
                hi_band.clone()
            } else {
                Plane::new(0, 0, w / 2, 1)
            };
            // Horizontal-only synthesis: LL = cur (low), HL = high band; no vertical bands.
            let out = synth_level(
                &cur,
                &hb,
                &zero_rows(w / 2),
                &zero_rows(w / 2),
                (0, 0, w, 1),
            );
            cur = out;
        }
        cur.data
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>()
            .sqrt()
    }
    let (hx, hy) = match o {
        Orient::LL => (false, false),
        Orient::HL => (true, false),
        Orient::LH => (false, true),
        Orient::HH => (true, true),
    };
    norm1d(level, hx) * norm1d(level, hy)
}

/// (exponent, mantissa) representing `delta` relative to 2^`range` (E.1.1.1).
fn encode_step(delta: f64, range: u32) -> (u8, u16) {
    let t = delta / 2f64.powi(range as i32);
    let mut e = (-t.log2()).ceil() as i32;
    // Ensure 1 <= t * 2^e < 2.
    while t * 2f64.powi(e) < 1.0 {
        e += 1;
    }
    while t * 2f64.powi(e) >= 2.0 {
        e -= 1;
    }
    let e = e.clamp(0, 31);
    let m = ((t * 2f64.powi(e) - 1.0) * 2048.0)
        .round()
        .clamp(0.0, 2047.0);
    (e as u8, m as u16)
}

fn step_value(e: u8, m: u16, range: u32) -> f64 {
    2f64.powi(range as i32 - e as i32) * (1.0 + m as f64 / 2048.0)
}

struct BandData {
    orient: Orient,
    coefs: Vec<i32>,
    w: usize,
    x0: u32,
    y0: u32,
}

/// Packet header bit writer with bit stuffing (B.10.1).
struct HeaderWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
    prev_ff: bool,
}

impl HeaderWriter {
    fn new() -> Self {
        HeaderWriter {
            out: Vec::new(),
            acc: 0,
            n: 0,
            prev_ff: false,
        }
    }

    fn bit(&mut self, b: bool) {
        self.acc = (self.acc << 1) | b as u32;
        self.n += 1;
        let width = if self.prev_ff { 7 } else { 8 };
        if self.n == width {
            let byte = self.acc as u8;
            self.out.push(byte);
            self.prev_ff = byte == 0xFF;
            self.acc = 0;
            self.n = 0;
        }
    }

    fn bits(&mut self, v: u32, n: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1 == 1);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        while self.n > 0 {
            self.bit(false);
        }
        if self.prev_ff {
            self.out.push(0);
        }
        self.out
    }
}

fn floor_log2(v: u32) -> u32 {
    31 - v.leading_zeros()
}

fn segment(out: &mut Vec<u8>, marker: u16, payload: &[u8]) {
    out.extend_from_slice(&marker.to_be_bytes());
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
}

/// Decompose one component plane into its sub-bands in codestream order
/// (LL, then HL, LH, HH for each resolution 1..=NL).
fn decompose<T: Analyse>(plane: Plane<T>, levels: u32) -> Vec<(Orient, Plane<T>)> {
    let mut detail: Vec<[Plane<T>; 3]> = Vec::new();
    let mut cur = plane;
    for _ in 0..levels {
        let [ll, hl, lh, hh] = analyse_level(&cur);
        detail.push([hl, lh, hh]);
        cur = ll;
    }
    let mut out = vec![(Orient::LL, cur)];
    for bands in detail.into_iter().rev() {
        for (p, o) in bands.into_iter().zip([Orient::HL, Orient::LH, Orient::HH]) {
            out.push((o, p));
        }
    }
    out
}

fn quantized<T: Copy>(bands: &[(Orient, Plane<T>)], q: &dyn Fn(T, usize) -> i32) -> Vec<BandData> {
    bands
        .iter()
        .enumerate()
        .map(|(i, (o, p))| BandData {
            orient: *o,
            coefs: p.data.iter().map(|&v| q(v, i)).collect(),
            w: p.w,
            x0: p.x0,
            y0: p.y0,
        })
        .collect()
}

/// Decomposition level and orientation of sub-band `index` in codestream order.
fn band_level(index: usize, levels: u32) -> (u32, Orient) {
    if index == 0 {
        (levels, Orient::LL)
    } else {
        let r = (index - 1) / 3 + 1;
        (
            levels - r as u32 + 1,
            [Orient::HL, Orient::LH, Orient::HH][(index - 1) % 3],
        )
    }
}

/// Target PSNR (dB, relative to the used dynamic range) for a quality setting.
fn target_psnr(quality: u8) -> f64 {
    26.0 + 0.23 * quality.clamp(1, 99) as f64
}

/// Encode with JPEG 2000 specific parameters.
pub fn encode_with(image: &Image, opts: &EncoderOptions) -> Result<Encoded> {
    if opts.high_throughput && !cfg!(feature = "htj2k") {
        return Err(Error::unsupported(
            CODEC,
            "HTJ2K (enable the htj2k feature)",
        ));
    }
    if opts.component_wavelet.is_some() && !cfg!(feature = "part2") {
        return Err(Error::unsupported(
            CODEC,
            "Part 2 transforms (enable the part2 feature)",
        ));
    }
    let info = &image.info;
    let (w, h) = (info.width, info.height);
    let nc = info.components as usize;
    if nc == 0 || nc > 16384 {
        return Err(Error::input("invalid component count"));
    }
    if opts.progression > 4 {
        return Err(Error::input("progression order must be 0..=4"));
    }
    let prec = info.bits_per_sample as u32;
    let levels = opts
        .levels
        .map(|l| l as u32)
        .unwrap_or_else(|| 5.min(31 - w.min(h).max(1).leading_zeros()));
    if levels > 32 {
        return Err(Error::input("too many decomposition levels"));
    }
    let shift = if info.signed { 0 } else { 1i32 << (prec - 1) };
    let mut planes: Vec<Vec<i32>> = image.planes();
    for v in planes.iter_mut().flatten() {
        *v -= shift;
    }
    // Part 2: the order of the transformed collection and its levels.
    let part2 = match &opts.component_wavelet {
        None => None,
        Some(cw) => {
            let k = (cw.interleave.max(1) as usize).min(nc);
            if !nc.is_multiple_of(k) {
                return Err(Error::input(
                    "component count is not a multiple of the interleave factor",
                ));
            }
            let order: Vec<usize> = (0..k).flat_map(|s| (s..nc).step_by(k)).collect();
            let max_levels = 31 - (nc as u32).leading_zeros();
            let levels = match cw.levels {
                Some(l) if l as u32 > max_levels => {
                    return Err(Error::input("too many component wavelet levels"));
                }
                Some(l) => l as u32,
                None => max_levels.min(5),
            };
            Some((order, levels))
        }
    };
    let mct =
        part2.is_none() && opts.color_transform && nc >= 3 && info.color_space == ColorSpace::Rgb;
    // Precision and signedness of the codestream components; with a
    // component transform they are signed and may need extra bits.
    let mut cs_prec = prec;
    let cs_signed = info.signed || part2.is_some();
    // Range needed to hold the signed values `lo..=hi` in two's complement.
    let signed_bits = |lo: i64, hi: i64| -> u32 {
        let m = hi.max(-lo - 1).max(0) as u64;
        65 - m.leading_zeros()
    };

    let reversible = opts.reversible;
    let guard: u32;
    let mut band_sets: Vec<Vec<BandData>> = Vec::with_capacity(nc);
    let mut quants: Vec<Quantization> = Vec::with_capacity(nc);

    if reversible {
        if let Some((order, cl)) = &part2 {
            planes = super::mct::forward_components(&planes, order, *cl);
            let (lo, hi) = planes.iter().flatten().fold((0i64, 0i64), |(a, b), &v| {
                (a.min(v as i64), b.max(v as i64))
            });
            cs_prec = cs_prec.max(signed_bits(lo, hi));
        }
        if mct {
            for i in 0..planes[0].len() {
                let [y, cb, cr] = rct::forward(planes[0][i], planes[1][i], planes[2][i]);
                planes[0][i] = y;
                planes[1][i] = cb;
                planes[2][i] = cr;
            }
        }
        guard = 2;
        for p in planes {
            let plane = Plane {
                x0: 0,
                y0: 0,
                w: w as usize,
                h: h as usize,
                data: p,
            };
            let bands = quantized(&decompose(plane, levels), &|v: i32, _| v);
            let steps = bands
                .iter()
                .map(|b| {
                    let max = b.coefs.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
                    let needed = 32 - max.leading_zeros();
                    // Mb = G + e - 1 must cover the magnitudes.
                    (
                        (needed + 1)
                            .saturating_sub(guard)
                            .max(cs_prec + band_gain(b.orient)) as u8,
                        0u16,
                    )
                })
                .collect();
            quants.push(Quantization {
                style: 0,
                guard_bits: guard as u8,
                steps,
            });
            band_sets.push(bands);
        }
    } else {
        let fplanes: Vec<Vec<f32>> = if let Some((order, cl)) = &part2 {
            let f: Vec<Vec<f32>> = planes
                .iter()
                .map(|p| p.iter().map(|&v| v as f32).collect())
                .collect();
            let f = super::mct::forward_components(&f, order, *cl);
            let (lo, hi) = f
                .iter()
                .flatten()
                .fold((0f32, 0f32), |(a, b), &v| (a.min(v), b.max(v)));
            // One extra bit for quantization overshoot.
            cs_prec = cs_prec.max(signed_bits(lo.floor() as i64, hi.ceil() as i64) + 1);
            f
        } else if mct {
            let mut f: Vec<Vec<f32>> = planes
                .iter()
                .map(|p| p.iter().map(|&v| v as f32).collect())
                .collect();
            for i in 0..f[0].len() {
                let [y, cb, cr] = ycbcr::forward(f[0][i], f[1][i], f[2][i]);
                f[0][i] = y;
                f[1][i] = cb;
                f[2][i] = cr;
            }
            f
        } else {
            planes
                .iter()
                .map(|p| p.iter().map(|&v| v as f32).collect())
                .collect()
        };
        // Dynamic range of the image samples the quality target refers to.
        let range = if part2.is_some() {
            let (lo, hi) = planes
                .iter()
                .flatten()
                .fold((i32::MAX, i32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            ((hi as f64) - (lo as f64)).max(1.0)
        } else {
            let (lo, hi) = fplanes
                .iter()
                .flatten()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            ((hi - lo) as f64).max(1.0)
        };
        let pixels = (w as f64 * h as f64 * nc as f64).max(1.0);
        let decomposed: Vec<Vec<(Orient, Plane<f32>)>> = fplanes
            .into_iter()
            .map(|p| {
                decompose(
                    Plane {
                        x0: 0,
                        y0: 0,
                        w: w as usize,
                        h: h as usize,
                        data: p,
                    },
                    levels,
                )
            })
            .collect();
        let nbands = 3 * levels as usize + 1;
        let norms: Vec<f64> = (0..nbands)
            .map(|i| {
                let (lvl, o) = band_level(i, levels);
                if levels == 0 {
                    1.0
                } else {
                    synthesis_norm97(lvl, o)
                }
            })
            .collect();
        let steps_for = |base: f64| -> Vec<(u8, u16, f64)> {
            (0..nbands)
                .map(|i| {
                    let (_, o) = band_level(i, levels);
                    let r = cs_prec + band_gain(o);
                    let (e, m) = encode_step(base / norms[i], r);
                    (e, m, step_value(e, m, r))
                })
                .collect()
        };
        // Predicted image-domain MSE for a base step (mid-point reconstruction;
        // every 3rd coefficient is sampled for speed).
        // Error energy of each component as seen in the output samples: with ICT
        // an error in Y, Cb or Cr reaches all three of R, G, B (sum of squared
        // inverse-transform coefficients of that column).
        let component_weights = part2
            .as_ref()
            .map(|(order, cl)| super::mct::synthesis_weights(order.len(), *cl));
        let weight = |c: usize| -> f64 {
            if let Some(wts) = &component_weights {
                return wts[c];
            }
            if mct && c < 3 {
                ycbcr::inverse_energy()[c]
            } else {
                1.0
            }
        };
        let mse_for = |base: f64| -> f64 {
            let steps = steps_for(base);
            let mut total = 0.0;
            for (c, comp) in decomposed.iter().enumerate() {
                let wc = weight(c);
                for (i, (_, p)) in comp.iter().enumerate() {
                    let d = steps[i].2;
                    let mut err = 0.0;
                    let mut n = 0usize;
                    for &v in p.data.iter().step_by(3) {
                        let a = v.abs() as f64;
                        let q = (a / d).floor();
                        let rec = if q == 0.0 { 0.0 } else { (q + 0.5) * d };
                        err += (a - rec) * (a - rec);
                        n += 1;
                    }
                    if n > 0 {
                        total += wc * err * (p.data.len() as f64 / n as f64) * norms[i] * norms[i];
                    }
                }
            }
            total / pixels
        };
        let target_mse = range * range / 10f64.powf(target_psnr(opts.quality) / 10.0);
        let (mut lo_b, mut hi_b) = (1e-4 * range, 64.0 * range);
        for _ in 0..24 {
            let mid = (lo_b * hi_b).sqrt();
            if mse_for(mid) > target_mse {
                hi_b = mid;
            } else {
                lo_b = mid;
            }
        }
        let steps = steps_for(lo_b);
        let mut max_guard = 1;
        let mut per_comp = Vec::with_capacity(nc);
        for comp in &decomposed {
            let bands = quantized(comp, &|v: f32, i| {
                let q = (v.abs() as f64 / steps[i].2).floor() as i32;
                if v < 0.0 { -q } else { q }
            });
            for (i, b) in bands.iter().enumerate() {
                let max = b.coefs.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
                let needed = 32 - max.leading_zeros();
                max_guard = max_guard.max((needed + 1).saturating_sub(steps[i].0 as u32));
            }
            per_comp.push(steps.iter().map(|&(e, m, _)| (e, m)).collect::<Vec<_>>());
            band_sets.push(bands);
        }
        if max_guard > GUARD_BITS_MAX {
            return Err(Error::unsupported(
                CODEC,
                "quantization range exceeds 7 guard bits",
            ));
        }
        guard = max_guard.max(1);
        for steps in per_comp {
            quants.push(Quantization {
                style: 2,
                guard_bits: guard as u8,
                steps,
            });
        }
    }

    // Geometry shared with the decoder.
    let coding = ComponentCoding {
        levels: levels as u8,
        xcb: 6,
        ycb: 6,
        cblk_style: if opts.high_throughput { 0x40 } else { 0 },
        reversible,
        precincts: vec![(15, 15); levels as usize + 1],
    };
    if cs_prec > 38 {
        return Err(Error::unsupported(
            CODEC,
            "component transform output exceeds 38 bits",
        ));
    }
    let siz = ComponentSiz {
        precision: cs_prec as u8,
        signed: cs_signed,
        dx: 1,
        dy: 1,
    };
    let tile = (0, 0, w, h);
    let mut max_blocks = u64::MAX;
    let comps: Vec<TileComponent> = (0..nc)
        .map(|_| TileComponent::new(tile, &siz, &coding, &mut max_blocks))
        .collect::<Result<_>>()?;
    let params = TileParams {
        global: GlobalCoding {
            sop: false,
            eph: false,
            progression: opts.progression,
            layers: 1,
            mct: mct as u8,
        },
        comps: vec![coding.clone(); nc],
        quant: quants.clone(),
        roi_shift: vec![0; nc],
        poc: Vec::new(),
    };
    let order = packet_order(tile, &comps, &params, u64::MAX)?;

    // Tier 1 for every code-block, then tier 2 in packet order.
    let mut body = Vec::new();
    // Byte offset in `body` where each resolution starts (for tile-parts).
    let mut res_starts: Vec<usize> = Vec::new();
    let resolution_major = matches!(opts.progression, 1 | 2);
    let mut last_res = None;
    for id in order {
        if last_res != Some(id.res) {
            last_res = Some(id.res);
            res_starts.push(body.len());
        }
        let (c, r, p) = (id.comp as usize, id.res as usize, id.precinct as usize);
        let res = &comps[c].resolutions[r];
        let bands = &band_sets[c];
        let q = &quants[c];
        let mut hw = HeaderWriter::new();
        let mut packet_body = Vec::new();
        // Encode the code-blocks of this precinct, band by band.
        let mut coded: Vec<(Vec<EncodedBlock>, usize, usize, Vec<u32>)> = Vec::new();
        for (bi, band) in res.bands.iter().enumerate() {
            let bd = &bands[band.step_index];
            let pb = &res.precincts[p][bi];
            let mb = q.guard_bits as u32 + q.steps[band.step_index].0 as u32 - 1;
            let mut blocks = Vec::with_capacity(pb.blocks.len());
            let mut zb = Vec::with_capacity(pb.blocks.len());
            for cb in &pb.blocks {
                let (bw, bh) = ((cb.x1 - cb.x0) as usize, (cb.y1 - cb.y0) as usize);
                let mut coefs = Vec::with_capacity(bw * bh);
                for y in cb.y0..cb.y1 {
                    let row = (y - bd.y0) as usize * bd.w;
                    for x in cb.x0..cb.x1 {
                        coefs.push(bd.coefs[row + (x - bd.x0) as usize]);
                    }
                }
                let enc = if opts.high_throughput {
                    ht_encode(bw, bh, &coefs).ok_or_else(|| {
                        Error::unsupported(CODEC, "code-block exceeds the HT segment limits")
                    })?
                } else {
                    t1_enc::encode(bw, bh, bd.orient, &coefs)
                };
                if enc.bitplanes > mb {
                    return Err(Error::invalid(
                        CODEC,
                        "internal: code-block exceeds Mb bit-planes",
                    ));
                }
                if opts.high_throughput {
                    // One cleanup pass at bit-plane 0: the signalled missing
                    // MSBs are Mb - 1 (T.814 B.1).
                    if mb > 30 {
                        return Err(Error::unsupported(
                            CODEC,
                            "HT code-blocks with more than 30 magnitude bit-planes",
                        ));
                    }
                    zb.push(mb - 1);
                } else {
                    zb.push(mb - enc.bitplanes);
                }
                blocks.push(enc);
            }
            coded.push((blocks, pb.cw, pb.ch, zb));
        }
        let any = coded.iter().any(|(b, ..)| b.iter().any(|e| e.passes > 0));
        hw.bit(any);
        if any {
            for (blocks, cw, ch, zb) in &coded {
                if blocks.is_empty() {
                    continue;
                }
                let incl: Vec<i32> = blocks
                    .iter()
                    .map(|b| if b.passes > 0 { 0 } else { 1 })
                    .collect();
                let zbv: Vec<i32> = blocks
                    .iter()
                    .zip(zb)
                    .map(|(b, &z)| if b.passes > 0 { z as i32 } else { i32::MAX / 2 })
                    .collect();
                let mut it = TagTreeEncoder::new(*cw, *ch, &incl);
                let mut zt = TagTreeEncoder::new(*cw, *ch, &zbv);
                for cy in 0..*ch {
                    for cx in 0..*cw {
                        let k = cy * cw + cx;
                        let b = &blocks[k];
                        let mut emit = |bit: bool| hw.bit(bit);
                        it.encode(cx, cy, 1, &mut emit);
                        if b.passes == 0 {
                            continue;
                        }
                        for t in 1..=zbv[k] + 1 {
                            zt.encode(cx, cy, t, &mut |bit| hw.bit(bit));
                        }
                        // Number of passes (Table B.4).
                        let n = b.passes;
                        match n {
                            1 => hw.bit(false),
                            2 => hw.bits(0b10, 2),
                            3..=5 => hw.bits((0b11 << 2) | (n - 3), 4),
                            6..=36 => hw.bits((0b1111 << 5) | (n - 6), 9),
                            _ => hw.bits((0b1_1111_1111 << 7) | (n - 37), 16),
                        }
                        // Lblock (initial 3) and the length.
                        let len = b.data.len() as u32;
                        let mut lblock = 3;
                        while len >= 1 << (lblock + floor_log2(n)) {
                            lblock += 1;
                        }
                        for _ in 3..lblock {
                            hw.bit(true);
                        }
                        hw.bit(false);
                        hw.bits(len, lblock + floor_log2(n));
                        packet_body.extend_from_slice(&b.data);
                    }
                }
            }
        }
        body.extend(hw.finish());
        body.extend(packet_body);
    }

    // Codestream.
    let mut out = vec![0xFF, 0x4F];
    let mut siz_seg = Vec::new();
    // Rsiz: Part 2 with the multiple component transformation capability.
    // Bit 14 signals the HTJ2K capabilities of the CAP marker (T.814 A.2).
    let mut rsiz: u16 = if part2.is_some() { 0x8100 } else { 0 };
    if opts.high_throughput {
        rsiz |= 0x4000;
    }
    siz_seg.extend_from_slice(&rsiz.to_be_bytes());
    for v in [w, h, 0, 0, w, h, 0, 0] {
        siz_seg.extend_from_slice(&v.to_be_bytes());
    }
    siz_seg.extend_from_slice(&(nc as u16).to_be_bytes());
    for _ in 0..nc {
        siz_seg.extend_from_slice(&[
            ((cs_prec - 1) as u8) | if cs_signed { 0x80 } else { 0 },
            1,
            1,
        ]);
    }
    segment(&mut out, 0xFF51, &siz_seg);
    if opts.high_throughput {
        // CAP: Pcap bit 15 (Part 15), Ccap15 = HT only, single HT set,
        // HTREV for the 5/3 wavelet and the MAGB bound.
        let magb = quants
            .iter()
            .flat_map(|q| {
                q.steps
                    .iter()
                    .map(|&(e, _)| q.guard_bits as u32 + e as u32 - 1)
            })
            .max()
            .unwrap_or(0);
        let bp = match magb {
            0..=8 => 0,
            9..=27 => magb - 8,
            28..=47 => 13 + (magb >> 2),
            _ => 31,
        };
        let ccap = (if reversible { 0x20u16 } else { 0 }) | bp as u16;
        let mut cap = 0x0002_0000u32.to_be_bytes().to_vec();
        cap.extend_from_slice(&ccap.to_be_bytes());
        segment(&mut out, 0xFF50, &cap);
    }
    let mut cod = vec![0u8, opts.progression];
    cod.extend_from_slice(&1u16.to_be_bytes());
    // SGcod MCT: 1 = RCT / ICT, 4 = wavelet-based multiple component transform.
    cod.push(if part2.is_some() { 4 } else { mct as u8 });
    cod.extend_from_slice(&[levels as u8, 4, 4, coding.cblk_style, reversible as u8]);
    segment(&mut out, 0xFF52, &cod);
    let quant_payload = |q: &Quantization| {
        let mut p = vec![(q.guard_bits << 5) | q.style];
        for &(e, m) in &q.steps {
            if q.style == 0 {
                p.push(e << 3);
            } else {
                p.extend_from_slice(&(((e as u16) << 11) | m).to_be_bytes());
            }
        }
        p
    };
    segment(&mut out, 0xFF5C, &quant_payload(&quants[0]));
    for (c, q) in quants.iter().enumerate().skip(1) {
        if q.steps != quants[0].steps || q.guard_bits != quants[0].guard_bits {
            let mut p = if nc < 257 {
                vec![c as u8]
            } else {
                (c as u16).to_be_bytes().to_vec()
            };
            p.extend(quant_payload(q));
            segment(&mut out, 0xFF5D, &p);
        }
    }
    if let Some((order, cl)) = &part2 {
        write_part2_markers(&mut out, order, *cl, prec, info.signed, reversible)?;
    }
    // Tile-parts: one per resolution for resolution-major orders with TLM.
    let mut cuts = if opts.tlm && resolution_major {
        res_starts
    } else {
        vec![0]
    };
    cuts.dedup();
    if cuts.is_empty() || cuts.len() > 255 {
        cuts = vec![0];
    }
    cuts.push(body.len());
    let parts: Vec<&[u8]> = cuts.windows(2).map(|c| &body[c[0]..c[1]]).collect();
    if opts.tlm {
        // Ztlm 0, Stlm: no tile indices (ST = 0), 32-bit lengths (SP = 1).
        let mut tlm = vec![0u8, 0x40];
        for part in &parts {
            tlm.extend_from_slice(&((14 + part.len()) as u32).to_be_bytes());
        }
        segment(&mut out, 0xFF55, &tlm);
    }
    for (i, part) in parts.iter().enumerate() {
        let psot = (12 + 2 + part.len()) as u32;
        out.extend_from_slice(&[0xFF, 0x90, 0, 10, 0, 0]);
        out.extend_from_slice(&psot.to_be_bytes());
        out.extend_from_slice(&[i as u8, parts.len() as u8, 0xFF, 0x93]);
        out.extend_from_slice(part);
    }
    out.extend_from_slice(&[0xFF, 0xD9]);

    let transform = match (mct, reversible) {
        (false, _) => AppliedTransform::None,
        (true, true) => AppliedTransform::Rct,
        (true, false) => AppliedTransform::Ict,
    };
    Ok(Encoded {
        data: out,
        transform,
        lossy: !reversible,
    })
}

/// Marker segments of a single-stage, single-collection wavelet-based
/// multiple component transform (T.801 A.3.6 - A.3.9): MCO, CBD, MCC and
/// the MCT offset array that undoes the DC level shift.
fn write_part2_markers(
    out: &mut Vec<u8>,
    order: &[usize],
    levels: u32,
    prec: u32,
    signed: bool,
    reversible: bool,
) -> Result<()> {
    let nc = order.len();
    // MCO: one stage using MCC index 1.
    segment(out, 0xFF77, &[1, 1]);
    // CBD: all reconstructed components share one bit depth.
    let mut cbd = (0x8000u16 | nc as u16).to_be_bytes().to_vec();
    cbd.push(((prec - 1) as u8) | if signed { 0x80 } else { 0 });
    segment(out, 0xFF78, &cbd);
    // MCC: one wavelet collection, C_k = k and W in transform order.
    let wide = nc > 256;
    let mut coll = vec![3u8];
    let count = (nc as u16) | if wide { 0x8000 } else { 0 };
    let index = |v: usize, d: &mut Vec<u8>| {
        if wide {
            d.extend_from_slice(&(v as u16).to_be_bytes());
        } else {
            d.push(v as u8);
        }
    };
    coll.extend_from_slice(&count.to_be_bytes());
    (0..nc).for_each(|k| index(k, &mut coll));
    coll.extend_from_slice(&count.to_be_bytes());
    order.iter().for_each(|&w| index(w, &mut coll));
    let offset_index: u32 = if signed { 0 } else { 1 };
    let kernel: u32 = if reversible { 1 } else { 0 };
    let t = (levels << 16) | (offset_index << 8) | kernel;
    coll.extend_from_slice(&t.to_be_bytes()[1..]);
    coll.extend_from_slice(&0u32.to_be_bytes());
    // Zmcc, Imcc (and Ymcc, Qmcc in the first segment).
    write_series(out, 0xFF75, &coll, |z, last| {
        let mut h = (z as u16).to_be_bytes().to_vec();
        h.push(1);
        if z == 0 {
            h.extend_from_slice(&(last as u16).to_be_bytes());
            h.extend_from_slice(&1u16.to_be_bytes());
        }
        h
    })?;
    if !signed {
        // MCT offset array (type 2, index 1): the DC level shift.
        let shift = 1i64 << (prec - 1);
        let (dtype, data): (u16, Vec<u8>) = if shift <= i16::MAX as i64 {
            (
                0,
                (0..nc).flat_map(|_| (shift as i16).to_be_bytes()).collect(),
            )
        } else {
            (
                1,
                (0..nc).flat_map(|_| (shift as i32).to_be_bytes()).collect(),
            )
        };
        let imct = (dtype << 10) | (2 << 8) | 1;
        write_series(out, 0xFF74, &data, |z, last| {
            let mut h = (z as u16).to_be_bytes().to_vec();
            h.extend_from_slice(&imct.to_be_bytes());
            if z == 0 {
                h.extend_from_slice(&(last as u16).to_be_bytes());
            }
            h
        })?;
    }
    Ok(())
}

/// Write `data` as a series of marker segments with index Z = 0, 1, ...;
/// `header(z, last_z)` builds each segment's leading fields.
fn write_series(
    out: &mut Vec<u8>,
    marker: u16,
    data: &[u8],
    header: impl Fn(usize, usize) -> Vec<u8>,
) -> Result<()> {
    const CHUNK: usize = 65_000;
    let chunks: Vec<&[u8]> = data.chunks(CHUNK).collect();
    let last = chunks.len().saturating_sub(1);
    if last > u16::MAX as usize {
        return Err(Error::unsupported(CODEC, "marker segment series too long"));
    }
    for (z, c) in chunks.iter().enumerate() {
        let mut p = header(z, last);
        p.extend_from_slice(c);
        segment(out, marker, &p);
    }
    Ok(())
}

#[cfg(test)]
mod norm_tests {
    use super::*;

    /// LL synthesis norms match the values published by OpenJPEG
    /// (dwt_norms_real); high-pass bands differ by the 2^gain factor that this
    /// implementation folds into the quantization step instead.
    #[test]
    fn synthesis_norms() {
        for (lvl, ll) in [(1, 1.965), (2, 4.177), (3, 8.403), (4, 16.90)] {
            let n = synthesis_norm97(lvl, Orient::LL);
            assert!((n - ll).abs() / ll < 0.02, "level {lvl}: {n} vs {ll}");
        }
    }
}
