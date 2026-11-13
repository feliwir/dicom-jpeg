//! Tile decoding: tier-2 / tier-1, dequantization, inverse DWT, inverse
//! component transform and DC level shift (T.800 Annexes B, E, F, G).

use super::CODEC;
use super::codestream::{Codestream, Quantization, TileParams};
use super::dwt::{Plane, synth_level};
use super::t1::{self, Orient};
use super::tile::{HeaderReader, TileComponent, ceil_div, decode_packet, packet_order};
use crate::Limits;
use crate::error::{Error, Result};

/// A decoded image component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    /// Width in samples (at the decoded resolution).
    pub width: u32,
    /// Height in samples (at the decoded resolution).
    pub height: u32,
    /// Horizontal sub-sampling factor (XRsiz).
    pub dx: u32,
    /// Vertical sub-sampling factor (YRsiz).
    pub dy: u32,
    /// Bit depth.
    pub precision: u8,
    /// Whether samples are signed.
    pub signed: bool,
    /// Samples, row-major.
    pub data: Vec<i32>,
}

/// JPEG 2000 specific decoding options.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DecoderOptions {
    /// Number of highest resolution levels to discard (0 = full resolution).
    pub reduce: u8,
    /// Resource limits.
    pub limits: Limits,
}

fn band_gain(o: Orient) -> u32 {
    match o {
        Orient::LL => 0,
        Orient::HL | Orient::LH => 1,
        Orient::HH => 2,
    }
}

/// (exponent, mantissa) of sub-band `index` (E.1.1), handling derived quantization.
fn step(q: &Quantization, index: usize, levels: u8) -> (u32, u32) {
    if q.style == 1 {
        let (e0, m0) = q.steps[0];
        // nb = NL for LL, NL - r + 1 for resolution r.
        let r = if index == 0 { 0 } else { (index - 1) / 3 + 1 };
        let nb = if index == 0 {
            levels as i32
        } else {
            levels as i32 - r as i32 + 1
        };
        let e = (e0 as i32 - levels as i32 + nb).max(0) as u32;
        (e, m0 as u32)
    } else {
        let (e, m) = q
            .steps
            .get(index)
            .copied()
            .unwrap_or(*q.steps.last().expect("non-empty"));
        (e as u32, m as u32)
    }
}

enum Coefficients {
    Int(Vec<Plane<i32>>),
    Float(Vec<Plane<f32>>),
}

/// Decode one tile into per-component sample planes (at the target resolution).
fn decode_tile(
    cs: &Codestream,
    t: usize,
    opts: &DecoderOptions,
) -> Result<(Vec<Plane<i32>>, TileParams)> {
    let siz = &cs.siz;
    let params = cs.tile_params(t)?;
    let ntx = siz.tiles_x() as usize;
    let (p, q) = ((t % ntx) as u32, (t / ntx) as u32);
    let tx0 = (siz.tx0 + p * siz.tw).max(siz.x0);
    let ty0 = (siz.ty0 + q * siz.th).max(siz.y0);
    let tx1 = siz
        .tx0
        .saturating_add((p + 1).saturating_mul(siz.tw))
        .min(siz.x1);
    let ty1 = siz
        .ty0
        .saturating_add((q + 1).saturating_mul(siz.th))
        .min(siz.y1);
    let tile = (tx0, ty0, tx1, ty1);

    let mut max_blocks: u64 = 1 << 24;
    let mut comps: Vec<TileComponent> = siz
        .comps
        .iter()
        .zip(&params.comps)
        .map(|(s, c)| TileComponent::new(tile, s, c, &mut max_blocks))
        .collect::<Result<_>>()?;
    for c in &comps {
        if (opts.reduce as usize) >= c.resolutions.len() {
            return Err(Error::input(
                "reduce exceeds the number of decomposition levels",
            ));
        }
    }

    // Tier 2.
    let td = &cs.tiles[t];
    let order = packet_order(tile, &comps, &params, 1 << 26)?;
    let packed = td.packed_headers.is_some();
    let header_data: &[u8] = td.packed_headers.as_deref().unwrap_or(&td.data);
    let mut headers = HeaderReader::new(header_data, 0);
    let mut body_pos = 0usize;
    for id in order {
        let c = id.comp as usize;
        let q = &params.quant[c];
        let levels = comps[c].coding.levels;
        let guard = q.guard_bits as u32;
        let mb = |si: usize| guard + step(q, si, levels).0 - 1;
        let ok = decode_packet(
            &mut comps[c],
            id,
            &mut headers,
            &td.data,
            &mut body_pos,
            packed,
            params.global.sop,
            params.global.eph,
            &mb,
        )?;
        if !ok {
            break; // truncated: decode what we have
        }
    }

    // Tier 1 + dequantization + inverse DWT per component.
    let mut out_planes = Vec::with_capacity(comps.len());
    let mut coeffs: Vec<Coefficients> = Vec::with_capacity(comps.len());
    for (c, tc) in comps.iter().enumerate() {
        let q = &params.quant[c];
        let prec = siz.comps[c].precision as u32;
        let roi = params.roi_shift[c] as u32;
        let reversible = tc.coding.reversible;
        let levels = tc.coding.levels;
        let target = tc.resolutions.len() - 1 - opts.reduce as usize;
        let mut bands_int: Vec<Vec<Plane<i32>>> = Vec::new();
        let mut bands_f: Vec<Vec<Plane<f32>>> = Vec::new();
        for res in &tc.resolutions[..=target] {
            let mut bi = Vec::new();
            let mut bf = Vec::new();
            for (b_idx, band) in res.bands.iter().enumerate() {
                let (expn, mant) = step(q, band.step_index, levels);
                let mb = q.guard_bits as u32 + expn - 1;
                let delta = 2f32.powi((prec + band_gain(band.orient)) as i32 - expn as i32)
                    * (1.0 + mant as f32 / 2048.0);
                let mut pi: Plane<i32> = Plane::new(band.x0, band.y0, band.x1, band.y1);
                let mut pf: Plane<f32> = if reversible {
                    Plane::new(0, 0, 0, 0)
                } else {
                    Plane::new(band.x0, band.y0, band.x1, band.y1)
                };
                let mut buf = Vec::new();
                for prec_bands in &res.precincts {
                    let pb = &prec_bands[b_idx];
                    for cb in &pb.blocks {
                        if !cb.included || cb.segments.is_empty() {
                            continue;
                        }
                        let (w, h) = ((cb.x1 - cb.x0) as usize, (cb.y1 - cb.y0) as usize);
                        buf.resize(w * h, 0);
                        let bitplanes = (mb + roi).saturating_sub(cb.zero_bitplanes);
                        t1::decode(
                            w,
                            h,
                            band.orient,
                            tc.coding.cblk_style,
                            bitplanes,
                            &cb.segments,
                            &mut buf,
                        )?;
                        for y in 0..h {
                            for x in 0..w {
                                let mut v = buf[y * w + x];
                                if roi > 0 {
                                    let m = v.unsigned_abs();
                                    if m >= 1 << (roi + 1) {
                                        let m = (m >> roi) as i32;
                                        v = if v < 0 { -m } else { m };
                                    }
                                }
                                let px = (cb.x0 - band.x0) as usize + x;
                                let py = (cb.y0 - band.y0) as usize + y;
                                if reversible {
                                    let m = v.unsigned_abs() as i32 >> 1;
                                    pi.data[py * pi.w + px] = if v < 0 { -m } else { m };
                                } else {
                                    pf.data[py * pf.w + px] = v as f32 * 0.5 * delta;
                                }
                            }
                        }
                    }
                }
                bi.push(pi);
                bf.push(pf);
            }
            bands_int.push(bi);
            bands_f.push(bf);
        }
        // Synthesis up to the target resolution.
        if reversible {
            let mut cur = bands_int[0].remove(0);
            for (res, b) in tc.resolutions[1..=target].iter().zip(&bands_int[1..]) {
                cur = synth_level(&cur, &b[0], &b[1], &b[2], (res.x0, res.y0, res.x1, res.y1));
            }
            coeffs.push(Coefficients::Int(vec![cur]));
        } else {
            let mut cur = bands_f[0].remove(0);
            for (res, b) in tc.resolutions[1..=target].iter().zip(&bands_f[1..]) {
                cur = synth_level(&cur, &b[0], &b[1], &b[2], (res.x0, res.y0, res.x1, res.y1));
            }
            coeffs.push(Coefficients::Float(vec![cur]));
        }
    }

    // Inverse component transform (G.2 / G.3).
    let same_size = |a: usize, b: usize| {
        let pa = match &coeffs[a] {
            Coefficients::Int(p) => (p[0].w, p[0].h),
            Coefficients::Float(p) => (p[0].w, p[0].h),
        };
        let pb = match &coeffs[b] {
            Coefficients::Int(p) => (p[0].w, p[0].h),
            Coefficients::Float(p) => (p[0].w, p[0].h),
        };
        pa == pb
    };
    if params.global.mct == 1 && coeffs.len() >= 3 {
        if !same_size(0, 1) || !same_size(0, 2) {
            return Err(Error::invalid(
                CODEC,
                "component transform on differently sized components",
            ));
        }
        let (a, rest) = coeffs.split_at_mut(1);
        let (b, c) = rest.split_at_mut(1);
        match (&mut a[0], &mut b[0], &mut c[0]) {
            (Coefficients::Int(y0), Coefficients::Int(y1), Coefficients::Int(y2)) => {
                for i in 0..y0[0].data.len() {
                    let (v0, v1, v2) = (y0[0].data[i], y1[0].data[i], y2[0].data[i]);
                    let g = v0 - ((v2 + v1) >> 2);
                    y0[0].data[i] = v2 + g;
                    y1[0].data[i] = g;
                    y2[0].data[i] = v1 + g;
                }
            }
            (Coefficients::Float(y0), Coefficients::Float(y1), Coefficients::Float(y2)) => {
                for i in 0..y0[0].data.len() {
                    let (y, cb, cr) = (y0[0].data[i], y1[0].data[i], y2[0].data[i]);
                    y0[0].data[i] = y + 1.402 * cr;
                    y1[0].data[i] = y - 0.344_136 * cb - 0.714_136 * cr;
                    y2[0].data[i] = y + 1.772 * cb;
                }
            }
            _ => {
                return Err(Error::invalid(
                    CODEC,
                    "component transform on mixed wavelet types",
                ));
            }
        }
    }

    // DC level shift and clamping (G.1).
    for (c, co) in coeffs.into_iter().enumerate() {
        let s = &siz.comps[c];
        let prec = s.precision as u32;
        let (lo, hi, shift) = if s.signed {
            (-(1i64 << (prec - 1)), (1i64 << (prec - 1)) - 1, 0i64)
        } else {
            (0, (1i64 << prec) - 1, 1i64 << (prec - 1))
        };
        let plane = match co {
            Coefficients::Int(mut p) => {
                let mut p = p.remove(0);
                for v in &mut p.data {
                    *v = (*v as i64 + shift).clamp(lo, hi) as i32;
                }
                p
            }
            Coefficients::Float(mut p) => {
                let p = p.remove(0);
                Plane {
                    x0: p.x0,
                    y0: p.y0,
                    w: p.w,
                    h: p.h,
                    data: p
                        .data
                        .iter()
                        .map(|&v| ((v as f64).round() as i64 + shift).clamp(lo, hi) as i32)
                        .collect(),
                }
            }
        };
        out_planes.push(plane);
    }
    Ok((out_planes, params))
}

/// Decode all components of a raw codestream.
pub(crate) fn decode_codestream(cs: &Codestream, opts: &DecoderOptions) -> Result<Vec<Component>> {
    let siz = &cs.siz;
    if cs.high_throughput {
        // HT code-blocks are detected per block; the rest of the pipeline is shared.
    }
    let r = opts.reduce as u32;
    let mut comps: Vec<Component> = Vec::with_capacity(siz.comps.len());
    let mut origins = Vec::with_capacity(siz.comps.len());
    for s in &siz.comps {
        let fx = (s.dx as u64) << r;
        let fy = (s.dy as u64) << r;
        let x0 = ceil_div(siz.x0 as u64, fx);
        let y0 = ceil_div(siz.y0 as u64, fy);
        let w = ceil_div(siz.x1 as u64, fx) - x0;
        let h = ceil_div(siz.y1 as u64, fy) - y0;
        opts.limits.check_alloc(w * h * 4)?;
        comps.push(Component {
            width: w as u32,
            height: h as u32,
            dx: s.dx,
            dy: s.dy,
            precision: s.precision,
            signed: s.signed,
            data: vec![0; (w * h) as usize],
        });
        origins.push((x0 as u32, y0 as u32));
    }
    let total: u64 = comps.iter().map(|c| c.width as u64 * c.height as u64).sum();
    if total > opts.limits.max_samples {
        return Err(Error::LimitExceeded {
            what: "samples",
            requested: total,
            limit: opts.limits.max_samples,
        });
    }
    let mut any = false;
    for t in 0..cs.tiles.len() {
        if !cs.tiles[t].seen {
            continue;
        }
        any = true;
        let (planes, _) = decode_tile(cs, t, opts)?;
        for (c, p) in planes.into_iter().enumerate() {
            let comp = &mut comps[c];
            let (ox, oy) = origins[c];
            for y in 0..p.h {
                let gy = (p.y0 as usize + y).wrapping_sub(oy as usize);
                if gy >= comp.height as usize {
                    continue;
                }
                for x in 0..p.w {
                    let gx = (p.x0 as usize + x).wrapping_sub(ox as usize);
                    if gx < comp.width as usize {
                        comp.data[gy * comp.width as usize + gx] = p.data[y * p.w + x];
                    }
                }
            }
        }
    }
    if !any {
        return Err(Error::invalid(CODEC, "no tile data"));
    }
    Ok(comps)
}
