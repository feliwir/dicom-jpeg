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
use crate::error::{Error, Result};
use crate::{AppliedTransform, ColorSpace, Encoded, Image, Samples};

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
    let samples: Vec<i32> = match &image.samples {
        Samples::U8(v) => v.iter().map(|&s| s as i32).collect(),
        Samples::U16(v) => v.iter().map(|&s| s as i32).collect(),
    };
    let mask = if prec >= 32 {
        -1
    } else {
        (1i64 << prec) as i32 - 1
    };
    let sign_extend = |v: i32| {
        if info.signed && prec < 32 && v & (1 << (prec - 1)) != 0 {
            v | !mask
        } else {
            v
        }
    };
    let shift = if info.signed { 0 } else { 1i32 << (prec - 1) };
    let mut planes: Vec<Vec<i32>> = (0..nc)
        .map(|c| {
            samples
                .iter()
                .skip(c)
                .step_by(nc)
                .map(|&v| sign_extend(v & mask) - shift)
                .collect()
        })
        .collect();
    let mct = opts.color_transform && nc >= 3 && info.color_space == ColorSpace::Rgb;

    let reversible = opts.reversible;
    let guard: u32;
    let mut band_sets: Vec<Vec<BandData>> = Vec::with_capacity(nc);
    let mut quants: Vec<Quantization> = Vec::with_capacity(nc);

    if reversible {
        if mct {
            for i in 0..planes[0].len() {
                let (r, g, b) = (planes[0][i], planes[1][i], planes[2][i]);
                planes[0][i] = (r + 2 * g + b) >> 2;
                planes[1][i] = b - g;
                planes[2][i] = r - g;
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
                            .max(prec + band_gain(b.orient)) as u8,
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
        let fplanes: Vec<Vec<f32>> = if mct {
            let mut f: Vec<Vec<f32>> = planes
                .iter()
                .map(|p| p.iter().map(|&v| v as f32).collect())
                .collect();
            for i in 0..f[0].len() {
                let (r, g, b) = (f[0][i], f[1][i], f[2][i]);
                f[0][i] = 0.299 * r + 0.587 * g + 0.114 * b;
                f[1][i] = -0.168_736 * r - 0.331_264 * g + 0.5 * b;
                f[2][i] = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
            }
            f
        } else {
            planes
                .iter()
                .map(|p| p.iter().map(|&v| v as f32).collect())
                .collect()
        };
        let (lo, hi) = fplanes
            .iter()
            .flatten()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let range = ((hi - lo) as f64).max(1.0);
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
                    let r = prec + band_gain(o);
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
        let weight = |c: usize| -> f64 {
            match (mct, c) {
                (true, 0) => 3.0,
                (true, 1) => 0.344_136f64.powi(2) + 1.772f64.powi(2),
                (true, 2) => 1.402f64.powi(2) + 0.714_136f64.powi(2),
                _ => 1.0,
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
    let siz = ComponentSiz {
        precision: prec as u8,
        signed: info.signed,
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
    siz_seg.extend_from_slice(&0u16.to_be_bytes());
    for v in [w, h, 0, 0, w, h, 0, 0] {
        siz_seg.extend_from_slice(&v.to_be_bytes());
    }
    siz_seg.extend_from_slice(&(nc as u16).to_be_bytes());
    for _ in 0..nc {
        siz_seg.extend_from_slice(&[
            ((prec - 1) as u8) | if info.signed { 0x80 } else { 0 },
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
    cod.push(mct as u8);
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
