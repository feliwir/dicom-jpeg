//! JPEG 2000 Part 2 multiple component transformations (ITU-T T.801 |
//! ISO/IEC 15444-2 Annex J): array-based decorrelation and dependency
//! transforms and the wavelet-based (5/3, 9/7) transform along the
//! component axis, organised in stages (MCO) of component collections (MCC)
//! with coefficient / offset arrays (MCT).

use super::CODEC;
use super::codestream::{Codestream, HeaderParams};
#[cfg(feature = "encode")]
use super::dwt::Analyse;
use super::dwt::{Plane, Sample};
use crate::error::{Error, Result};

fn invalid(msg: &'static str) -> Error {
    Error::invalid(CODEC, msg)
}

/// Collection transform kinds (Xmcc, Table A.35).
pub(crate) const DEPENDENCY: u8 = 0;
pub(crate) const DECORRELATION: u8 = 1;
pub(crate) const WAVELET: u8 = 3;

/// Array types (Imct bits 8..9, Table A.33).
const ARRAY_OFFSET: u8 = 2;

/// One component collection of a transform stage.
#[derive(Debug, Clone)]
pub(crate) struct Collection {
    pub(crate) kind: u8,
    pub(crate) inputs: Vec<usize>,
    pub(crate) outputs: Vec<usize>,
    /// Tmcc (Tables A.38 / A.39).
    pub(crate) t: u32,
    /// Omcc: component-axis origin of a wavelet collection.
    pub(crate) origin: u32,
}

/// The resolved inverse multiple component transform of one tile.
#[derive(Debug, Clone)]
pub(crate) struct Pipeline {
    pub(crate) stages: Vec<Vec<Collection>>,
    /// (Imct, values) with tile-header arrays taking precedence.
    arrays: Vec<(u16, Vec<f64>)>,
}

/// Whether a tile uses Part 2 multiple component transforms.
pub(crate) fn pipeline(cs: &Codestream, t: usize) -> Result<Option<Pipeline>> {
    let tile = cs.tiles[t].params.as_ref();
    let main = &cs.main;
    let mco = tile.and_then(|p| p.mco.as_ref()).or(main.mco.as_ref());
    let Some(mco) = mco.filter(|m| !m.is_empty()) else {
        return Ok(None);
    };
    let mut stages = Vec::with_capacity(mco.len());
    for &imcc in mco {
        let data = tile
            .and_then(|p| mcc_series(p, imcc))
            .or_else(|| mcc_series(main, imcc))
            .ok_or_else(|| invalid("MCO refers to a missing MCC marker"))?;
        stages.push(parse_collections(&data.0, data.1)?);
    }
    let mut arrays = Vec::new();
    for p in tile.into_iter().chain(std::iter::once(main)) {
        let mut ids: Vec<u16> = p.mct.iter().map(|m| m.0).collect();
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            if arrays.iter().any(|(i, _)| *i == id) {
                continue; // overridden by the tile header
            }
            let mut segs: Vec<&(u16, u16, Vec<u8>)> = p.mct.iter().filter(|m| m.0 == id).collect();
            segs.sort_by_key(|m| m.1);
            let bytes: Vec<u8> = segs.iter().flat_map(|m| m.2.iter().copied()).collect();
            arrays.push((id, array_values(id, &bytes)?));
        }
    }
    Ok(Some(Pipeline { stages, arrays }))
}

/// Concatenated MCC series `imcc` of a header: (data, Qmcc).
fn mcc_series(p: &HeaderParams, imcc: u8) -> Option<(Vec<u8>, u16)> {
    let mut segs: Vec<_> = p.mcc.iter().filter(|m| m.0 == imcc).collect();
    if segs.is_empty() {
        return None;
    }
    segs.sort_by_key(|m| m.1);
    let q = segs[0].2?;
    Some((segs.iter().flat_map(|m| m.3.iter().copied()).collect(), q))
}

fn array_values(imct: u16, bytes: &[u8]) -> Result<Vec<f64>> {
    let size = match (imct >> 10) & 3 {
        0 => 2,
        1 | 2 => 4,
        _ => 8,
    };
    if !bytes.len().is_multiple_of(size) {
        return Err(invalid(
            "MCT array length is not a multiple of its element size",
        ));
    }
    Ok(bytes
        .chunks_exact(size)
        .map(|c| match (imct >> 10) & 3 {
            0 => i16::from_be_bytes([c[0], c[1]]) as f64,
            1 => i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f64,
            2 => f32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f64,
            _ => f64::from_be_bytes(c.try_into().expect("8 bytes")),
        })
        .collect())
}

fn parse_collections(d: &[u8], q: u16) -> Result<Vec<Collection>> {
    let mut p = 0usize;
    let mut take = |n: usize| -> Result<u32> {
        let s = d.get(p..p + n).ok_or(Error::Truncated { codec: CODEC })?;
        p += n;
        Ok(s.iter().fold(0u32, |a, &b| (a << 8) | b as u32))
    };
    let mut out = Vec::with_capacity(q as usize);
    for _ in 0..q {
        let kind = (take(1)? & 3) as u8;
        if kind == 2 {
            return Err(invalid("reserved MCC transform type"));
        }
        let indices = |take: &mut dyn FnMut(usize) -> Result<u32>| -> Result<Vec<usize>> {
            let n = take(2)?;
            let width = if n & 0x8000 != 0 { 2 } else { 1 };
            let count = (n & 0x7FFF) as usize;
            if count == 0 || count > 16384 {
                return Err(invalid("invalid MCC component count"));
            }
            (0..count).map(|_| Ok(take(width)? as usize)).collect()
        };
        let inputs = indices(&mut take)?;
        let outputs = indices(&mut take)?;
        let t = take(3)?;
        let origin = if kind == WAVELET { take(4)? } else { 0 };
        if kind != DECORRELATION && inputs.len() != outputs.len() {
            return Err(invalid(
                "MCC collection needs as many outputs as inputs for this transform",
            ));
        }
        out.push(Collection {
            kind,
            inputs,
            outputs,
            t,
            origin,
        });
    }
    Ok(out)
}

impl Pipeline {
    /// Array `index` (1..=255) of array type `kind`; `None` for index 0.
    fn array(&self, index: u32, kind: u8, len: usize) -> Result<Option<&[f64]>> {
        if index == 0 {
            return Ok(None);
        }
        let a = self
            .arrays
            .iter()
            .find(|(id, _)| (id & 0xFF) as u32 == index && ((id >> 8) & 3) as u8 == kind)
            .map(|(_, v)| v.as_slice())
            .ok_or_else(|| invalid("MCC refers to a missing MCT array"))?;
        if a.len() < len {
            return Err(invalid("MCT array too short for its component collection"));
        }
        Ok(Some(&a[..len]))
    }

    /// Apply all inverse stages to the spatially reconstructed components;
    /// returns the reconstructed image components (`None` = null component).
    pub(crate) fn inverse(&self, comps: Vec<Plane<f64>>) -> Result<Vec<Option<Plane<f64>>>> {
        let mut cur: Vec<Option<Plane<f64>>> = comps.into_iter().map(Some).collect();
        for stage in &self.stages {
            let mut next: Vec<Option<Plane<f64>>> = Vec::new();
            for coll in stage {
                let dims = coll
                    .inputs
                    .iter()
                    .find_map(|&i| cur.get(i).and_then(|p| p.as_ref()))
                    .or_else(|| cur.iter().flatten().next())
                    .map(|p| (p.x0, p.y0, p.w, p.h))
                    .ok_or_else(|| invalid("multiple component transform without input"))?;
                let n = dims.2 * dims.3;
                let zeros = vec![0.0; n];
                let mut inputs: Vec<&[f64]> = Vec::with_capacity(coll.inputs.len());
                for &i in &coll.inputs {
                    match cur.get(i).and_then(|p| p.as_ref()) {
                        Some(p) if (p.w, p.h) == (dims.2, dims.3) => inputs.push(&p.data),
                        Some(_) => {
                            return Err(invalid(
                                "component collection with differently sized components",
                            ));
                        }
                        None => inputs.push(&zeros),
                    }
                }
                let results = self.apply(coll, &inputs, n)?;
                for (&o, data) in coll.outputs.iter().zip(results) {
                    if o >= 16384 {
                        return Err(invalid("MCC output component index out of range"));
                    }
                    if next.len() <= o {
                        next.resize(o + 1, None);
                    }
                    if next[o].is_some() {
                        return Err(invalid("MCC output component assigned twice"));
                    }
                    next[o] = Some(Plane {
                        x0: dims.0,
                        y0: dims.1,
                        w: dims.2,
                        h: dims.3,
                        data,
                    });
                }
            }
            cur = next;
        }
        Ok(cur)
    }

    fn apply(&self, coll: &Collection, c: &[&[f64]], n: usize) -> Result<Vec<Vec<f64>>> {
        let (ni, mo) = (c.len(), coll.outputs.len());
        let reversible = coll.t & 0x1_0000 != 0;
        let t_index = coll.t & 0xFF;
        let o_index = (coll.t >> 8) & 0xFF;
        let offsets = self.array(o_index, ARRAY_OFFSET, mo)?;
        let offset = |i: usize| offsets.map_or(0.0, |o| o[i]);
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(mo);
        match coll.kind {
            DECORRELATION if reversible => {
                return Err(Error::unsupported(
                    CODEC,
                    "reversible array-based decorrelation transform",
                ));
            }
            DECORRELATION => {
                // W_i = sum_j t_ij C_j + o_i (J-2).
                let m = self.array(t_index, DECORRELATION, mo * ni)?;
                for i in 0..mo {
                    let mut out = vec![offset(i); n];
                    for (j, cj) in c.iter().enumerate() {
                        let t = m.map_or((i == j) as u8 as f64, |m| m[i * ni + j]);
                        if t != 0.0 {
                            for (o, &v) in out.iter_mut().zip(cj.iter()) {
                                *o += t * v;
                            }
                        }
                    }
                    w.push(out);
                }
            }
            DEPENDENCY if reversible => {
                // W_j = floor((sum_{i<j} t_ji W_i + floor(t_jj / 2)) / t_jj) + C_j + o_j (J-15).
                let m = self.array(t_index, DEPENDENCY, (mo * (mo + 1) / 2).saturating_sub(1))?;
                for j in 0..mo {
                    let row = |i: usize| -> f64 {
                        match m {
                            None => (i == j) as u8 as f64,
                            Some(m) => m[j * (j + 1) / 2 - 1 + i],
                        }
                    };
                    let mut out: Vec<f64> = c[j].iter().map(|&v| v + offset(j)).collect();
                    if j > 0 {
                        let scale = row(j);
                        if scale <= 0.0 {
                            return Err(invalid("reversible dependency scale is not positive"));
                        }
                        let half = (scale / 2.0).floor();
                        for (s, o) in out.iter_mut().enumerate() {
                            let mut sum = half;
                            for (i, wi) in w.iter().enumerate() {
                                sum += row(i) * wi[s];
                            }
                            *o += (sum / scale).floor();
                        }
                    }
                    w.push(out);
                }
            }
            DEPENDENCY => {
                // W_j = sum_{i<j} t_ji W_i + C_j + o_j (J-12).
                let m = self.array(t_index, DEPENDENCY, mo * mo.saturating_sub(1) / 2)?;
                for j in 0..mo {
                    let mut out: Vec<f64> = c[j].iter().map(|&v| v + offset(j)).collect();
                    if let Some(m) = m {
                        for (i, wi) in w.iter().enumerate() {
                            let t = m[j * (j - 1) / 2 + i];
                            if t != 0.0 {
                                for (o, &v) in out.iter_mut().zip(wi) {
                                    *o += t * v;
                                }
                            }
                        }
                    }
                    w.push(out);
                }
            }
            _ => {
                // Wavelet along the component axis (J.3.2).
                let kernel = coll.t & 0xFF;
                let levels = (coll.t >> 16) & 0x3F;
                if kernel > 1 {
                    return Err(Error::unsupported(
                        CODEC,
                        "arbitrary transformation kernels (ATK) in multiple component transforms",
                    ));
                }
                if levels > 32 {
                    return Err(invalid("too many component wavelet levels"));
                }
                w = vec![vec![0.0; n]; mo];
                if kernel == 1 {
                    let mut line = vec![0i32; ni];
                    for s in 0..n {
                        for (l, cj) in line.iter_mut().zip(c) {
                            *l = cj[s] as i32;
                        }
                        let r = inverse_1d(&line, coll.origin, levels);
                        for (j, v) in r.into_iter().enumerate() {
                            w[j][s] = v as f64 + offset(j);
                        }
                    }
                } else {
                    let mut line = vec![0f32; ni];
                    for s in 0..n {
                        for (l, cj) in line.iter_mut().zip(c) {
                            *l = cj[s] as f32;
                        }
                        let r = inverse_1d(&line, coll.origin, levels);
                        for (j, v) in r.into_iter().enumerate() {
                            w[j][s] = v as f64 + offset(j);
                        }
                    }
                }
            }
        }
        Ok(w)
    }
}

fn ceil_div_pow2(v: i64, k: u32) -> i64 {
    (v + (1i64 << k) - 1).div_euclid(1i64 << k)
}

/// Sub-band extents along the component axis at decomposition `level`
/// (T.800 B-15 with tcx0 = origin, tcx1 = origin + n): (low, high).
fn band_ranges(origin: u32, n: usize, level: u32) -> ((i64, i64), (i64, i64)) {
    let (x0, x1) = (origin as i64, origin as i64 + n as i64);
    let low = (ceil_div_pow2(x0, level), ceil_div_pow2(x1, level));
    let shift = if level == 0 { 0 } else { 1i64 << (level - 1) };
    let high = (
        ceil_div_pow2(x0 - shift, level),
        ceil_div_pow2(x1 - shift, level),
    );
    (low, high)
}

/// Inverse 1D multi-level DWT of a line stored as LL_NL, HL_NL, ..., HL_1.
fn inverse_1d<T: Sample>(line: &[T], origin: u32, levels: u32) -> Vec<T> {
    let n = line.len();
    let ((l0, l1), _) = band_ranges(origin, n, levels);
    let mut pos = (l1 - l0) as usize;
    let mut cur: Vec<T> = line[..pos].to_vec();
    let mut cur_x0 = l0;
    for lev in (1..=levels).rev() {
        let (_, (h0, h1)) = band_ranges(origin, n, lev);
        let hb = &line[pos..pos + (h1 - h0) as usize];
        pos += (h1 - h0) as usize;
        let ((u0, u1), _) = band_ranges(origin, n, lev - 1);
        let mut res: Vec<T> = (u0..u1)
            .map(|u| {
                let b = u.div_euclid(2);
                if u.rem_euclid(2) == 0 {
                    cur[(b - cur_x0) as usize]
                } else {
                    hb[(b - h0) as usize]
                }
            })
            .collect();
        if !res.is_empty() {
            T::synth(&mut res, u0.rem_euclid(2) == 1);
        }
        cur = res;
        cur_x0 = u0;
    }
    cur
}

/// Forward 1D multi-level DWT; output ordered LL_NL, HL_NL, ..., HL_1.
#[cfg(feature = "encode")]
pub(crate) fn forward_1d<T: Analyse>(line: &[T], origin: u32, levels: u32) -> Vec<T> {
    let n = line.len();
    let mut cur = line.to_vec();
    let mut highs: Vec<Vec<T>> = Vec::with_capacity(levels as usize);
    for lev in 1..=levels {
        let ((u0, _), _) = band_ranges(origin, n, lev - 1);
        if !cur.is_empty() {
            T::analyse(&mut cur, u0.rem_euclid(2) == 1);
        }
        let (mut lo, mut hi) = (Vec::new(), Vec::new());
        for (k, &v) in cur.iter().enumerate() {
            if (u0 + k as i64).rem_euclid(2) == 0 {
                lo.push(v);
            } else {
                hi.push(v);
            }
        }
        highs.push(hi);
        cur = lo;
    }
    for h in highs.into_iter().rev() {
        cur.extend(h);
    }
    cur
}

/// Forward wavelet transform along the component axis (T.801 J.3.2.2) of
/// the components listed in `order` (the W indices of one collection).
/// Returns the transformed components C_0..C_n in sub-band order.
#[cfg(feature = "encode")]
pub(crate) fn forward_components<T: Analyse>(
    planes: &[Vec<T>],
    order: &[usize],
    levels: u32,
) -> Vec<Vec<T>> {
    let n = planes.first().map_or(0, Vec::len);
    let mut out: Vec<Vec<T>> = (0..order.len()).map(|_| Vec::with_capacity(n)).collect();
    let mut line = Vec::with_capacity(order.len());
    #[allow(clippy::needless_range_loop)] // indexes every plane at `s`
    for s in 0..n {
        line.clear();
        line.extend(order.iter().map(|&c| planes[c][s]));
        for (o, v) in out.iter_mut().zip(forward_1d(&line, 0, levels)) {
            o.push(v);
        }
    }
    out
}

/// Squared L2 norm of the component-axis synthesis basis function of every
/// transformed component: how strongly an error in C_k reaches the output.
#[cfg(feature = "encode")]
pub(crate) fn synthesis_weights(n: usize, levels: u32) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let mut line = vec![0f32; n];
            line[k] = 1.0;
            inverse_1d(&line, 0, levels)
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum()
        })
        .collect()
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;

    #[test]
    fn wavelet_1d_round_trip() {
        for n in 1..40usize {
            for origin in [0u32, 1, 2, 3, 7] {
                for levels in 0..6 {
                    let line: Vec<i32> = (0..n as i32).map(|i| (i * 37 % 23) - 11).collect();
                    let f = forward_1d(&line, origin, levels);
                    assert_eq!(f.len(), n);
                    assert_eq!(
                        inverse_1d(&f, origin, levels),
                        line,
                        "n{n} o{origin} L{levels}"
                    );
                    let fl: Vec<f32> = line.iter().map(|&v| v as f32).collect();
                    let r = inverse_1d(&forward_1d(&fl, origin, levels), origin, levels);
                    for (a, b) in r.iter().zip(&fl) {
                        assert!((a - b).abs() < 1e-3, "n{n} o{origin} L{levels}");
                    }
                }
            }
        }
    }
}
