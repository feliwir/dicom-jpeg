//! Codestream syntax (T.800 Annex A): main / tile-part headers and tile data.

use super::CODEC;
use crate::error::{Error, Result};

fn truncated() -> Error {
    Error::Truncated { codec: CODEC }
}

fn invalid(msg: impl Into<std::borrow::Cow<'static, str>>) -> Error {
    Error::invalid(CODEC, msg)
}

/// Per-component SIZ parameters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ComponentSiz {
    pub(crate) precision: u8,
    pub(crate) signed: bool,
    pub(crate) dx: u32,
    pub(crate) dy: u32,
}

/// Image and tile size (SIZ).
#[derive(Debug, Clone)]
pub(crate) struct Siz {
    /// Capabilities (Rsiz); kept for diagnostics.
    #[allow(dead_code)]
    pub(crate) rsiz: u16,
    pub(crate) x1: u32,
    pub(crate) y1: u32,
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) tw: u32,
    pub(crate) th: u32,
    pub(crate) tx0: u32,
    pub(crate) ty0: u32,
    pub(crate) comps: Vec<ComponentSiz>,
}

impl Siz {
    pub(crate) fn tiles_x(&self) -> u32 {
        (self.x1 - self.tx0).div_ceil(self.tw)
    }
    pub(crate) fn tiles_y(&self) -> u32 {
        (self.y1 - self.ty0).div_ceil(self.th)
    }
}

/// Coding style parameters of one component (SPcod / SPcoc).
#[derive(Debug, Clone)]
pub(crate) struct ComponentCoding {
    pub(crate) levels: u8,
    pub(crate) xcb: u8,
    pub(crate) ycb: u8,
    pub(crate) cblk_style: u8,
    pub(crate) reversible: bool,
    /// (PPx, PPy) per resolution level 0..=levels.
    pub(crate) precincts: Vec<(u8, u8)>,
}

/// Scod / SGcod part of a COD.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GlobalCoding {
    pub(crate) sop: bool,
    pub(crate) eph: bool,
    pub(crate) progression: u8,
    pub(crate) layers: u16,
    pub(crate) mct: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct Cod {
    pub(crate) global: GlobalCoding,
    pub(crate) comp: ComponentCoding,
}

/// Quantization of one component (SQcd / SPqcd).
#[derive(Debug, Clone)]
pub(crate) struct Quantization {
    pub(crate) style: u8,
    pub(crate) guard_bits: u8,
    /// (exponent, mantissa) per subband in codestream order.
    pub(crate) steps: Vec<(u8, u16)>,
}

/// Progression order change (POC entry).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Poc {
    pub(crate) rs: u8,
    pub(crate) cs: u16,
    pub(crate) lye: u16,
    pub(crate) re: u8,
    pub(crate) ce: u16,
    pub(crate) order: u8,
}

/// Markers that may appear in both main and tile headers.
#[derive(Debug, Clone, Default)]
pub(crate) struct HeaderParams {
    pub(crate) cod: Option<Cod>,
    pub(crate) coc: Vec<Option<ComponentCoding>>,
    pub(crate) qcd: Option<Quantization>,
    pub(crate) qcc: Vec<Option<Quantization>>,
    pub(crate) rgn: Vec<Option<u8>>,
    pub(crate) poc: Vec<Poc>,
    /// Part 2 MCT segments: (Imct, Zmct, array data).
    pub(crate) mct: Vec<(u16, u16, Vec<u8>)>,
    /// Part 2 MCC segments: (Imcc, Zmcc, Qmcc if Zmcc = 0, collection data).
    pub(crate) mcc: Vec<(u8, u16, Option<u16>, Vec<u8>)>,
    /// Part 2 MCO: MCC indices of the inverse transform stages.
    pub(crate) mco: Option<Vec<u8>>,
}

impl HeaderParams {
    fn new(nc: usize) -> Self {
        HeaderParams {
            coc: vec![None; nc],
            qcc: vec![None; nc],
            rgn: vec![None; nc],
            ..Default::default()
        }
    }
}

/// Collected data of one tile.
#[derive(Debug, Clone, Default)]
pub(crate) struct TileData {
    pub(crate) params: Option<HeaderParams>,
    /// Concatenated tile-part bodies.
    pub(crate) data: Vec<u8>,
    /// Packed packet headers (PPM / PPT), if used.
    pub(crate) packed_headers: Option<Vec<u8>>,
    pub(crate) seen: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Codestream {
    pub(crate) siz: Siz,
    pub(crate) main: HeaderParams,
    pub(crate) tiles: Vec<TileData>,
    /// HTJ2K (Part 15) capabilities present.
    pub(crate) high_throughput: bool,
    /// Part 2 CBD: (precision, signed) of every reconstructed image component.
    pub(crate) cbd: Option<Vec<(u8, bool)>>,
}

/// Resolved parameters for one tile.
#[derive(Debug, Clone)]
pub(crate) struct TileParams {
    pub(crate) global: GlobalCoding,
    pub(crate) comps: Vec<ComponentCoding>,
    pub(crate) quant: Vec<Quantization>,
    pub(crate) roi_shift: Vec<u8>,
    pub(crate) poc: Vec<Poc>,
}

impl Codestream {
    /// Precision, signedness and sub-sampling of the output components:
    /// the CBD marker when Part 2 multiple component transforms are used,
    /// SIZ otherwise.
    pub(crate) fn output_comps(&self) -> Vec<ComponentSiz> {
        match &self.cbd {
            Some(cbd) => cbd
                .iter()
                .enumerate()
                .map(|(i, &(precision, signed))| {
                    let s = self.siz.comps[i.min(self.siz.comps.len() - 1)];
                    ComponentSiz {
                        precision,
                        signed,
                        dx: s.dx,
                        dy: s.dy,
                    }
                })
                .collect(),
            None => self.siz.comps.clone(),
        }
    }

    /// Resolve COD/COC/QCD/QCC/RGN/POC priorities for a tile (A.6).
    pub(crate) fn tile_params(&self, t: usize) -> Result<TileParams> {
        let nc = self.siz.comps.len();
        let main = &self.main;
        let tile = self.tiles[t].params.as_ref();
        let cod = tile
            .and_then(|p| p.cod.as_ref())
            .or(main.cod.as_ref())
            .ok_or_else(|| invalid("missing COD marker"))?;
        let mut comps = Vec::with_capacity(nc);
        let mut quant = Vec::with_capacity(nc);
        let mut roi_shift = Vec::with_capacity(nc);
        for c in 0..nc {
            let cc = tile
                .and_then(|p| p.coc[c].clone())
                .or_else(|| tile.and_then(|p| p.cod.as_ref()).map(|d| d.comp.clone()))
                .or_else(|| main.coc[c].clone())
                .unwrap_or_else(|| {
                    main.cod
                        .as_ref()
                        .map(|d| d.comp.clone())
                        .unwrap_or(cod.comp.clone())
                });
            let q = tile
                .and_then(|p| p.qcc[c].clone())
                .or_else(|| tile.and_then(|p| p.qcd.clone()))
                .or_else(|| main.qcc[c].clone())
                .or_else(|| main.qcd.clone())
                .ok_or_else(|| invalid("missing QCD marker"))?;
            let r = tile.and_then(|p| p.rgn[c]).or(main.rgn[c]).unwrap_or(0);
            comps.push(cc);
            quant.push(q);
            roi_shift.push(r);
        }
        let poc = match tile {
            Some(p) if !p.poc.is_empty() => p.poc.clone(),
            _ => main.poc.clone(),
        };
        Ok(TileParams {
            global: cod.global,
            comps,
            quant,
            roi_shift,
            poc,
        })
    }
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> Result<u8> {
        let v = *self.d.get(self.p).ok_or_else(truncated)?;
        self.p += 1;
        Ok(v)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(((self.u8()? as u16) << 8) | self.u8()? as u16)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(((self.u16()? as u32) << 16) | self.u16()? as u32)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.d.get(self.p..self.p + n).ok_or_else(truncated)?;
        self.p += n;
        Ok(s)
    }
    fn remaining(&self) -> usize {
        self.d.len().saturating_sub(self.p)
    }
}

fn parse_siz(seg: &[u8]) -> Result<Siz> {
    let mut r = Reader { d: seg, p: 0 };
    let rsiz = r.u16()?;
    let (x1, y1, x0, y0) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
    let (tw, th, tx0, ty0) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
    let nc = r.u16()? as usize;
    if nc == 0 || nc > 16384 {
        return Err(invalid("invalid component count"));
    }
    let mut comps = Vec::with_capacity(nc);
    for _ in 0..nc {
        let s = r.u8()?;
        let (dx, dy) = (r.u8()? as u32, r.u8()? as u32);
        let precision = (s & 0x7F) + 1;
        if precision > 38 {
            return Err(invalid("component precision above 38 bits"));
        }
        if precision > 31 {
            return Err(Error::unsupported(
                CODEC,
                "component precision above 31 bits",
            ));
        }
        if dx == 0 || dy == 0 {
            return Err(invalid("zero component subsampling"));
        }
        comps.push(ComponentSiz {
            precision: (s & 0x7F) + 1,
            signed: s & 0x80 != 0,
            dx,
            dy,
        });
    }
    if x1 <= x0 || y1 <= y0 || tw == 0 || th == 0 || tx0 > x0 || ty0 > y0 {
        return Err(invalid("invalid image / tile geometry"));
    }
    if tx0.checked_add(tw).is_none_or(|v| v <= x0) || ty0.checked_add(th).is_none_or(|v| v <= y0) {
        return Err(invalid("first tile does not intersect the image"));
    }
    Ok(Siz {
        rsiz,
        x1,
        y1,
        x0,
        y0,
        tw,
        th,
        tx0,
        ty0,
        comps,
    })
}

fn parse_spcod(r: &mut Reader<'_>, precincts_defined: bool) -> Result<ComponentCoding> {
    let levels = r.u8()?;
    let xcb = (r.u8()? & 0x0F) + 2;
    let ycb = (r.u8()? & 0x0F) + 2;
    let cblk_style = r.u8()?;
    let transform = r.u8()?;
    if levels > 32 {
        return Err(invalid("too many decomposition levels"));
    }
    if xcb > 10 || ycb > 10 || xcb + ycb > 12 {
        return Err(invalid("invalid code-block size"));
    }
    if transform > 1 {
        return Err(Error::unsupported(
            CODEC,
            "arbitrary wavelet transforms (Part 2)",
        ));
    }
    let mut precincts = Vec::with_capacity(levels as usize + 1);
    for _ in 0..=levels {
        if precincts_defined {
            let b = r.u8()?;
            precincts.push((b & 0x0F, b >> 4));
        } else {
            precincts.push((15, 15));
        }
    }
    Ok(ComponentCoding {
        levels,
        xcb,
        ycb,
        cblk_style,
        reversible: transform == 1,
        precincts,
    })
}

fn parse_quant(r: &mut Reader<'_>) -> Result<Quantization> {
    let sq = r.u8()?;
    let style = sq & 0x1F;
    let guard_bits = sq >> 5;
    let mut steps = Vec::new();
    match style {
        0 => {
            while r.remaining() > 0 {
                steps.push((r.u8()? >> 3, 0));
            }
        }
        1 | 2 => {
            while r.remaining() >= 2 {
                let v = r.u16()?;
                steps.push(((v >> 11) as u8, v & 0x7FF));
            }
        }
        _ => return Err(invalid("invalid quantization style")),
    }
    if steps.is_empty() {
        return Err(invalid("quantization without step sizes"));
    }
    Ok(Quantization {
        style,
        guard_bits,
        steps,
    })
}

fn comp_index(r: &mut Reader<'_>, nc: usize) -> Result<usize> {
    let c = if nc < 257 {
        r.u8()? as usize
    } else {
        r.u16()? as usize
    };
    if c >= nc {
        return Err(invalid("component index out of range"));
    }
    Ok(c)
}

/// Parse a marker segment that may appear in main or tile headers.
fn parse_param_marker(m: u16, seg: &[u8], nc: usize, p: &mut HeaderParams) -> Result<bool> {
    let mut r = Reader { d: seg, p: 0 };
    match m {
        0xFF52 => {
            let scod = r.u8()?;
            let progression = r.u8()?;
            let layers = r.u16()?;
            let mct = r.u8()?;
            if progression > 4 {
                return Err(invalid("invalid progression order"));
            }
            if layers == 0 {
                return Err(invalid("zero quality layers"));
            }
            let comp = parse_spcod(&mut r, scod & 1 != 0)?;
            p.cod = Some(Cod {
                global: GlobalCoding {
                    sop: scod & 2 != 0,
                    eph: scod & 4 != 0,
                    progression,
                    layers,
                    mct,
                },
                comp,
            });
        }
        0xFF53 => {
            let c = comp_index(&mut r, nc)?;
            let scoc = r.u8()?;
            p.coc[c] = Some(parse_spcod(&mut r, scoc & 1 != 0)?);
        }
        0xFF5C => p.qcd = Some(parse_quant(&mut r)?),
        0xFF5D => {
            let c = comp_index(&mut r, nc)?;
            p.qcc[c] = Some(parse_quant(&mut r)?);
        }
        0xFF5E => {
            let c = comp_index(&mut r, nc)?;
            let style = r.u8()?;
            if style != 0 {
                return Err(invalid("invalid ROI style"));
            }
            let shift = r.u8()?;
            if shift > 30 {
                return Err(Error::unsupported(CODEC, "ROI shift above 30"));
            }
            p.rgn[c] = Some(shift);
        }
        0xFF5F => {
            let wide = nc >= 257;
            while r.remaining() > 0 {
                let rs = r.u8()?;
                let cs = if wide { r.u16()? } else { r.u8()? as u16 };
                let lye = r.u16()?;
                let re = r.u8()?;
                let ce = if wide { r.u16()? } else { r.u8()? as u16 };
                let order = r.u8()?;
                if order > 4 {
                    return Err(invalid("invalid POC progression order"));
                }
                p.poc.push(Poc {
                    rs,
                    cs,
                    lye,
                    re,
                    ce: if ce == 0 { 256 } else { ce },
                    order,
                });
            }
        }
        0xFF74 => {
            let z = r.u16()?;
            let imct = r.u16()?;
            if z == 0 {
                let _ymct = r.u16()?;
            }
            p.mct.push((imct, z, seg[r.p..].to_vec()));
        }
        0xFF75 => {
            let z = r.u16()?;
            let imcc = r.u8()?;
            let q = if z == 0 {
                let _ymcc = r.u16()?;
                Some(r.u16()?)
            } else {
                None
            };
            p.mcc.push((imcc, z, q, seg[r.p..].to_vec()));
        }
        0xFF77 => {
            let n = r.u8()? as usize;
            p.mco = Some(r.bytes(n)?.to_vec());
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_cbd(seg: &[u8]) -> Result<Vec<(u8, bool)>> {
    let mut r = Reader { d: seg, p: 0 };
    let n = r.u16()?;
    let count = (n & 0x7FFF) as usize;
    if count == 0 || count > 16384 {
        return Err(invalid("invalid CBD component count"));
    }
    let depth = |b: u8| -> Result<(u8, bool)> {
        let precision = (b & 0x7F) + 1;
        if precision > 31 {
            return Err(Error::unsupported(
                CODEC,
                "component precision above 31 bits",
            ));
        }
        Ok((precision, b & 0x80 != 0))
    };
    if n & 0x8000 != 0 {
        let d = depth(r.u8()?)?;
        Ok(vec![d; count])
    } else {
        (0..count).map(|_| depth(r.u8()?)).collect()
    }
}

/// Parse a raw J2K codestream.
pub(crate) fn parse(data: &[u8]) -> Result<Codestream> {
    if !data.starts_with(&[0xFF, 0x4F]) {
        return Err(invalid("missing SOC marker"));
    }
    let mut r = Reader { d: data, p: 2 };
    let mut siz: Option<Siz> = None;
    let mut main = HeaderParams::default();
    let mut ppm: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut high_throughput = false;
    let mut cbd = None;

    // Main header.
    loop {
        let m = r.u16()?;
        if m == 0xFF90 {
            r.p -= 2;
            break;
        }
        if m == 0xFFD9 {
            return Err(invalid("no tile-parts"));
        }
        if m < 0xFF00 {
            return Err(invalid(format!("expected marker, found {m:04X}")));
        }
        if (0xFF30..=0xFF3F).contains(&m) {
            continue; // reserved markers without a segment (A.1.3)
        }
        let len = r.u16()? as usize;
        if len < 2 {
            return Err(invalid("marker segment length < 2"));
        }
        let seg = r.bytes(len - 2)?;
        match m {
            0xFF51 => {
                let s = parse_siz(seg)?;
                main = HeaderParams::new(s.comps.len());
                siz = Some(s);
            }
            0xFF50 => high_throughput = true,
            0xFF78 => cbd = Some(parse_cbd(seg)?),
            0xFF60 => {
                if let Some((&z, rest)) = seg.split_first() {
                    ppm.push((z, rest.to_vec()));
                }
            }
            _ => {
                let nc = siz
                    .as_ref()
                    .ok_or_else(|| invalid("SIZ must be the first marker"))?
                    .comps
                    .len();
                parse_param_marker(m, seg, nc, &mut main)?;
            }
        }
    }
    let siz = siz.ok_or_else(|| invalid("missing SIZ marker"))?;
    let nc = siz.comps.len();
    let ntiles = (siz.tiles_x() as u64 * siz.tiles_y() as u64) as usize;
    if ntiles > 65535 {
        return Err(invalid("too many tiles"));
    }
    let mut tiles = vec![TileData::default(); ntiles];
    // PPM: concatenated in Zppm order, then split as Nppm-prefixed chunks per tile-part.
    ppm.sort_by_key(|(z, _)| *z);
    let ppm_stream: Vec<u8> = ppm.into_iter().flat_map(|(_, d)| d).collect();
    let mut ppm_pos = 0usize;
    let use_ppm = !ppm_stream.is_empty();

    // Tile-parts.
    while r.p + 2 <= data.len() {
        let m = r.u16()?;
        if m == 0xFFD9 {
            break;
        }
        if m != 0xFF90 {
            // Tolerate garbage between tile-parts: resynchronise on the next SOT.
            match data[r.p..].windows(2).position(|w| w == [0xFF, 0x90]) {
                Some(off) => {
                    r.p += off;
                    continue;
                }
                None => break,
            }
        }
        let sot_start = r.p - 2;
        let lsot = r.u16()?;
        if lsot != 10 {
            return Err(invalid("invalid SOT length"));
        }
        let isot = r.u16()? as usize;
        let psot = r.u32()? as usize;
        let _tpsot = r.u8()?;
        let _tnsot = r.u8()?;
        if isot >= ntiles {
            return Err(invalid("tile index out of range"));
        }
        let end = if psot == 0 {
            data.len()
        } else {
            (sot_start + psot).min(data.len())
        };
        let first = !tiles[isot].seen;
        tiles[isot].seen = true;
        if first {
            tiles[isot].params = Some(HeaderParams::new(nc));
        }
        // Tile-part header.
        let mut ppt: Vec<(u8, Vec<u8>)> = Vec::new();
        loop {
            if r.p + 2 > end {
                return Err(truncated());
            }
            let m = r.u16()?;
            if m == 0xFF93 {
                break;
            }
            if (0xFF30..=0xFF3F).contains(&m) {
                continue;
            }
            let len = r.u16()? as usize;
            if len < 2 {
                return Err(invalid("marker segment length < 2"));
            }
            let seg = r.bytes(len - 2)?;
            match m {
                0xFF61 => {
                    if let Some((&z, rest)) = seg.split_first() {
                        ppt.push((z, rest.to_vec()));
                    }
                }
                _ => {
                    let params = tiles[isot].params.as_mut().expect("initialised");
                    if !parse_param_marker(m, seg, nc, params)? {
                        // COM, PLT, TLM etc. are ignored.
                    }
                }
            }
        }
        let body = &data[r.p.min(end)..end];
        let tile = &mut tiles[isot];
        tile.data.extend_from_slice(body);
        if use_ppm {
            if ppm_pos + 4 <= ppm_stream.len() {
                let n = u32::from_be_bytes(
                    ppm_stream[ppm_pos..ppm_pos + 4]
                        .try_into()
                        .expect("4 bytes"),
                ) as usize;
                let chunk_end = (ppm_pos + 4 + n).min(ppm_stream.len());
                tile.packed_headers
                    .get_or_insert_with(Vec::new)
                    .extend_from_slice(&ppm_stream[ppm_pos + 4..chunk_end]);
                ppm_pos = chunk_end;
            }
        } else if !ppt.is_empty() {
            ppt.sort_by_key(|(z, _)| *z);
            let h = tile.packed_headers.get_or_insert_with(Vec::new);
            for (_, d) in ppt {
                h.extend_from_slice(&d);
            }
        }
        r.p = end;
    }
    Ok(Codestream {
        siz,
        main,
        tiles,
        high_throughput,
        cbd,
    })
}
