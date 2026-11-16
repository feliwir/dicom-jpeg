//! Tile structure (B.5-B.9) and packet decoding (B.10-B.12).

use super::CODEC;
use super::codestream::{ComponentCoding, ComponentSiz, Poc, TileParams};
use super::t1::{Orient, STYLE_BYPASS, STYLE_HT, STYLE_HT_MIXED, STYLE_TERMALL, Segment};
use super::tagtree::TagTree;
use dicom_jpeg_common::{Error, Result};

#[inline]
pub(crate) fn ceil_div(a: u64, b: u64) -> u64 {
    a.div_ceil(b)
}

#[derive(Debug)]
pub(crate) struct CodeBlock {
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) x1: u32,
    pub(crate) y1: u32,
    pub(crate) included: bool,
    pub(crate) zero_bitplanes: u32,
    pub(crate) lblock: u32,
    pub(crate) segments: Vec<Segment>,
    /// HT only: parsing state (see `HtState`).
    pub(crate) ht_state: HtState,
}

/// Packet-header state of an HT code-block (T.814 Annex B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum HtState {
    /// No HT cleanup pass seen yet: all passes so far are placeholders.
    #[default]
    Placeholders,
    /// The first HT cleanup pass has been received.
    Coded,
    /// Mixed mode: this code-block uses the Part 1 block coder.
    Part1,
}

#[derive(Debug)]
pub(crate) struct PrecinctBand {
    /// Code-block grid of this precinct within the band.
    pub(crate) cw: usize,
    pub(crate) ch: usize,
    pub(crate) blocks: Vec<CodeBlock>,
    incl: TagTree,
    zbp: TagTree,
}

#[derive(Debug)]
pub(crate) struct Band {
    pub(crate) orient: Orient,
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) x1: u32,
    pub(crate) y1: u32,
    /// Index into the quantization step list.
    pub(crate) step_index: usize,
}

#[derive(Debug)]
pub(crate) struct Resolution {
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) x1: u32,
    pub(crate) y1: u32,
    pub(crate) ppx: u8,
    pub(crate) ppy: u8,
    /// Number of precincts horizontally / vertically.
    pub(crate) pw: u32,
    pub(crate) ph: u32,
    pub(crate) bands: Vec<Band>,
    /// precincts[p][band]
    pub(crate) precincts: Vec<Vec<PrecinctBand>>,
}

#[derive(Debug)]
pub(crate) struct TileComponent {
    pub(crate) dx: u32,
    pub(crate) dy: u32,
    pub(crate) coding: ComponentCoding,
    pub(crate) resolutions: Vec<Resolution>,
}

fn band_bounds(tc: (u32, u32, u32, u32), nb: u32, xob: u32, yob: u32) -> (u32, u32, u32, u32) {
    let s = 1u64 << nb;
    let off_x = (xob as u64) << nb.saturating_sub(1);
    let off_y = (yob as u64) << nb.saturating_sub(1);
    let f = |v: u32, off: u64| -> u32 {
        let v = v as u64;
        if v >= off {
            ceil_div(v - off, s) as u32
        } else {
            0
        }
    };
    (
        f(tc.0, off_x),
        f(tc.1, off_y),
        f(tc.2, off_x),
        f(tc.3, off_y),
    )
}

impl TileComponent {
    pub(crate) fn new(
        tile: (u32, u32, u32, u32),
        siz: &ComponentSiz,
        coding: &ComponentCoding,
        max_blocks: &mut u64,
    ) -> Result<Self> {
        let (dx, dy) = (siz.dx as u64, siz.dy as u64);
        let x0 = ceil_div(tile.0 as u64, dx) as u32;
        let y0 = ceil_div(tile.1 as u64, dy) as u32;
        let x1 = ceil_div(tile.2 as u64, dx) as u32;
        let y1 = ceil_div(tile.3 as u64, dy) as u32;
        let nl = coding.levels as u32;
        let mut resolutions = Vec::with_capacity(nl as usize + 1);
        for r in 0..=nl {
            let s = 1u64 << (nl - r);
            let rx0 = ceil_div(x0 as u64, s) as u32;
            let ry0 = ceil_div(y0 as u64, s) as u32;
            let rx1 = ceil_div(x1 as u64, s) as u32;
            let ry1 = ceil_div(y1 as u64, s) as u32;
            let (ppx, ppy) = coding.precincts[r as usize];
            if r > 0 && (ppx == 0 || ppy == 0) {
                return Err(Error::invalid(CODEC, "precinct size 1 at a resolution > 0"));
            }
            let (pw, ph) = if rx1 > rx0 && ry1 > ry0 {
                (
                    (ceil_div(rx1 as u64, 1 << ppx) - (rx0 as u64 >> ppx)) as u32,
                    (ceil_div(ry1 as u64, 1 << ppy) - (ry0 as u64 >> ppy)) as u32,
                )
            } else {
                (0, 0)
            };
            let bands: Vec<Band> = if r == 0 {
                let b = band_bounds((x0, y0, x1, y1), nl, 0, 0);
                vec![Band {
                    orient: Orient::LL,
                    x0: b.0,
                    y0: b.1,
                    x1: b.2,
                    y1: b.3,
                    step_index: 0,
                }]
            } else {
                let nb = nl - r + 1;
                [(Orient::HL, 1, 0), (Orient::LH, 0, 1), (Orient::HH, 1, 1)]
                    .iter()
                    .enumerate()
                    .map(|(i, &(orient, xob, yob))| {
                        let b = band_bounds((x0, y0, x1, y1), nb, xob, yob);
                        Band {
                            orient,
                            x0: b.0,
                            y0: b.1,
                            x1: b.2,
                            y1: b.3,
                            step_index: 3 * (r as usize - 1) + 1 + i,
                        }
                    })
                    .collect()
            };
            // Code-block size within this resolution (B.7).
            let (xcb, ycb) = if r == 0 {
                (coding.xcb.min(ppx), coding.ycb.min(ppy))
            } else {
                (coding.xcb.min(ppx - 1), coding.ycb.min(ppy - 1))
            };
            let (bppx, bppy) = if r == 0 {
                (ppx, ppy)
            } else {
                (ppx - 1, ppy - 1)
            };
            let mut precincts = Vec::with_capacity((pw * ph) as usize);
            for j in 0..ph {
                for i in 0..pw {
                    let mut per_band = Vec::with_capacity(bands.len());
                    for b in &bands {
                        // Precinct region in band coordinates.
                        let px0 = ((rx0 as u64 >> ppx) + i as u64) << bppx;
                        let py0 = ((ry0 as u64 >> ppy) + j as u64) << bppy;
                        let px1 = px0 + (1u64 << bppx);
                        let py1 = py0 + (1u64 << bppy);
                        let (bx0, by0) = (px0.max(b.x0 as u64), py0.max(b.y0 as u64));
                        let (bx1, by1) = (px1.min(b.x1 as u64), py1.min(b.y1 as u64));
                        let (cw, ch, blocks) = if bx1 > bx0 && by1 > by0 {
                            let cx0 = bx0 >> xcb;
                            let cy0 = by0 >> ycb;
                            let cx1 = ceil_div(bx1, 1 << xcb);
                            let cy1 = ceil_div(by1, 1 << ycb);
                            let (cw, ch) = ((cx1 - cx0) as usize, (cy1 - cy0) as usize);
                            *max_blocks = max_blocks.saturating_sub((cw * ch) as u64);
                            if *max_blocks == 0 {
                                return Err(Error::LimitExceeded {
                                    what: "code-blocks",
                                    requested: u64::MAX,
                                    limit: 0,
                                });
                            }
                            let mut blocks = Vec::with_capacity(cw * ch);
                            for cy in cy0..cy1 {
                                for cx in cx0..cx1 {
                                    blocks.push(CodeBlock {
                                        x0: ((cx << xcb).max(bx0)) as u32,
                                        y0: ((cy << ycb).max(by0)) as u32,
                                        x1: (((cx + 1) << xcb).min(bx1)) as u32,
                                        y1: (((cy + 1) << ycb).min(by1)) as u32,
                                        included: false,
                                        zero_bitplanes: 0,
                                        lblock: 3,
                                        segments: Vec::new(),
                                        ht_state: HtState::Placeholders,
                                    });
                                }
                            }
                            (cw, ch, blocks)
                        } else {
                            (0, 0, Vec::new())
                        };
                        per_band.push(PrecinctBand {
                            cw,
                            ch,
                            blocks,
                            incl: TagTree::new(cw.max(1), ch.max(1)),
                            zbp: TagTree::new(cw.max(1), ch.max(1)),
                        });
                    }
                    precincts.push(per_band);
                }
            }
            resolutions.push(Resolution {
                x0: rx0,
                y0: ry0,
                x1: rx1,
                y1: ry1,
                ppx,
                ppy,
                pw,
                ph,
                bands,
                precincts,
            });
        }
        Ok(TileComponent {
            dx: siz.dx,
            dy: siz.dy,
            coding: coding.clone(),
            resolutions,
        })
    }
}

/// A packet identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PacketId {
    pub(crate) layer: u16,
    pub(crate) res: u8,
    pub(crate) comp: u16,
    pub(crate) precinct: u32,
}

/// Packet order for a tile (B.12), honouring POC changes; packets are
/// emitted once even if several progressions cover them.
pub(crate) fn packet_order(
    tile: (u32, u32, u32, u32),
    comps: &[TileComponent],
    params: &TileParams,
    max_packets: u64,
) -> Result<Vec<PacketId>> {
    let layers = params.global.layers;
    let nc = comps.len();
    let max_res = comps.iter().map(|c| c.resolutions.len()).max().unwrap_or(0);
    let progressions: Vec<Poc> = if params.poc.is_empty() {
        vec![Poc {
            rs: 0,
            cs: 0,
            lye: layers,
            re: max_res as u8,
            ce: nc as u16,
            order: params.global.progression,
        }]
    } else {
        params.poc.clone()
    };
    // Offsets of each (component, resolution) precinct range in the "done" bitmap.
    let mut offsets = vec![vec![0usize; max_res]; nc];
    let mut total = 0usize;
    for (c, tc) in comps.iter().enumerate() {
        for (r, res) in tc.resolutions.iter().enumerate() {
            offsets[c][r] = total;
            total += (res.pw as usize) * (res.ph as usize);
        }
    }
    if total as u64 * layers as u64 > max_packets {
        return Err(Error::LimitExceeded {
            what: "packets",
            requested: total as u64 * layers as u64,
            limit: max_packets,
        });
    }
    let mut done = vec![false; total * layers as usize];
    let mut out = Vec::new();
    let mut emit = |l: u16, r: usize, c: usize, p: u32, out: &mut Vec<PacketId>| {
        let tc = &comps[c];
        if r >= tc.resolutions.len() || l >= layers {
            return;
        }
        let res = &tc.resolutions[r];
        if p >= res.pw * res.ph {
            return;
        }
        let k = (offsets[c][r] + p as usize) * layers as usize + l as usize;
        if !done[k] {
            done[k] = true;
            out.push(PacketId {
                layer: l,
                res: r as u8,
                comp: c as u16,
                precinct: p,
            });
        }
    };
    for prog in progressions {
        let lye = prog.lye.min(layers);
        let (rs, re) = (prog.rs as usize, (prog.re as usize).min(max_res));
        let (cs, ce) = (prog.cs as usize, (prog.ce as usize).min(nc));
        if rs >= re || cs >= ce {
            continue;
        }
        let nprec = |c: usize, r: usize| -> u32 {
            comps[c].resolutions.get(r).map_or(0, |res| res.pw * res.ph)
        };
        match prog.order {
            0 => {
                for l in 0..lye {
                    for r in rs..re {
                        for c in cs..ce {
                            for p in 0..nprec(c, r) {
                                emit(l, r, c, p, &mut out);
                            }
                        }
                    }
                }
            }
            1 => {
                for r in rs..re {
                    for l in 0..lye {
                        for c in cs..ce {
                            for p in 0..nprec(c, r) {
                                emit(l, r, c, p, &mut out);
                            }
                        }
                    }
                }
            }
            order => {
                // Position-driven orders: iterate the reference grid.
                let mut xstep = u64::MAX;
                let mut ystep = u64::MAX;
                for tc in &comps[cs..ce] {
                    let nl = tc.resolutions.len() as u32 - 1;
                    for (r, res) in tc.resolutions.iter().enumerate() {
                        let lv = nl - r as u32;
                        xstep = xstep.min((tc.dx as u64) << (res.ppx as u32 + lv));
                        ystep = ystep.min((tc.dy as u64) << (res.ppy as u32 + lv));
                    }
                }
                let iterations = ceil_div((tile.2 - tile.0) as u64, xstep.max(1))
                    * ceil_div((tile.3 - tile.1) as u64, ystep.max(1))
                    * (ce - cs) as u64
                    * max_res as u64;
                if iterations > 1 << 28 {
                    return Err(Error::LimitExceeded {
                        what: "precinct iterations",
                        requested: iterations,
                        limit: 1 << 28,
                    });
                }
                let precinct_at = |c: usize, r: usize, x: u64, y: u64| -> Option<u32> {
                    let tc = &comps[c];
                    let res = tc.resolutions.get(r)?;
                    let nl = tc.resolutions.len() as u32 - 1;
                    let lv = nl - r as u32;
                    let (dx, dy) = (tc.dx as u64, tc.dy as u64);
                    if res.x0 == res.x1 || res.y0 == res.y1 {
                        return None;
                    }
                    let rpx = res.ppx as u32 + lv;
                    let rpy = res.ppy as u32 + lv;
                    let ok_y = y.is_multiple_of(dy << rpy)
                        || (y == tile.1 as u64
                            && !((res.y0 as u64) << lv).is_multiple_of(1u64 << rpy));
                    let ok_x = x.is_multiple_of(dx << rpx)
                        || (x == tile.0 as u64
                            && !((res.x0 as u64) << lv).is_multiple_of(1u64 << rpx));
                    if !ok_x || !ok_y {
                        return None;
                    }
                    let prci = (ceil_div(x, dx << lv) >> res.ppx) - ((res.x0 as u64) >> res.ppx);
                    let prcj = (ceil_div(y, dy << lv) >> res.ppy) - ((res.y0 as u64) >> res.ppy);
                    Some((prci + prcj * res.pw as u64) as u32)
                };
                let next = |v: u64, step: u64| v + step - (v % step);
                let (tx0, ty0, tx1, ty1) =
                    (tile.0 as u64, tile.1 as u64, tile.2 as u64, tile.3 as u64);
                match order {
                    2 => {
                        for r in rs..re {
                            let mut y = ty0;
                            while y < ty1 {
                                let mut x = tx0;
                                while x < tx1 {
                                    for c in cs..ce {
                                        if let Some(p) = precinct_at(c, r, x, y) {
                                            for l in 0..lye {
                                                emit(l, r, c, p, &mut out);
                                            }
                                        }
                                    }
                                    x = next(x, xstep);
                                }
                                y = next(y, ystep);
                            }
                        }
                    }
                    3 => {
                        let mut y = ty0;
                        while y < ty1 {
                            let mut x = tx0;
                            while x < tx1 {
                                for c in cs..ce {
                                    for r in rs..re {
                                        if let Some(p) = precinct_at(c, r, x, y) {
                                            for l in 0..lye {
                                                emit(l, r, c, p, &mut out);
                                            }
                                        }
                                    }
                                }
                                x = next(x, xstep);
                            }
                            y = next(y, ystep);
                        }
                    }
                    _ => {
                        for c in cs..ce {
                            let mut y = ty0;
                            while y < ty1 {
                                let mut x = tx0;
                                while x < tx1 {
                                    for r in rs..re {
                                        if let Some(p) = precinct_at(c, r, x, y) {
                                            for l in 0..lye {
                                                emit(l, r, c, p, &mut out);
                                            }
                                        }
                                    }
                                    x = next(x, xstep);
                                }
                                y = next(y, ystep);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Bit reader for packet headers (B.10.1, with bit stuffing after 0xFF).
pub(crate) struct HeaderReader<'a> {
    data: &'a [u8],
    pub(crate) pos: usize,
    buf: u32,
    ct: u32,
}

impl<'a> HeaderReader<'a> {
    pub(crate) fn new(data: &'a [u8], pos: usize) -> Self {
        HeaderReader {
            data,
            pos,
            buf: 0,
            ct: 0,
        }
    }

    pub(crate) fn bit(&mut self) -> Option<bool> {
        if self.ct == 0 {
            let stuffed = self.buf == 0xFF;
            self.buf = *self.data.get(self.pos)? as u32;
            self.pos += 1;
            self.ct = if stuffed { 7 } else { 8 };
        }
        self.ct -= 1;
        Some((self.buf >> self.ct) & 1 == 1)
    }

    pub(crate) fn bits(&mut self, n: u32) -> Option<u32> {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit()? as u32;
        }
        Some(v)
    }

    /// Byte-align at the end of a packet header.
    pub(crate) fn align(&mut self) {
        if self.buf == 0xFF && self.ct == 0 {
            // The byte after 0xFF carries a stuffed bit; skip it.
            self.pos += 1;
        } else if self.buf == 0xFF && self.ct > 0 && self.ct < 8 {
            // Partially consumed 0xFF: the next byte is part of this header.
            self.pos += 1;
        }
        self.ct = 0;
        self.buf = 0;
    }
}

fn floor_log2(v: u32) -> u32 {
    31 - v.leading_zeros()
}

/// Maximum number of passes of segment `index` of a code-block (B.10.7 / D.4).
fn max_passes(style: u8, segments: &[Segment]) -> u32 {
    if style & STYLE_TERMALL != 0 {
        1
    } else if style & STYLE_BYPASS != 0 {
        match segments.last() {
            None => 10,
            Some(prev) if prev.max_passes == 1 || prev.max_passes == 10 => 2,
            Some(_) => 1,
        }
    } else {
        u32::MAX
    }
}

/// New data for one code-block in a packet: (band, block, [(segment << 8 | passes, length)]).
type Contribution = (usize, usize, Vec<(u32, u32)>);

#[allow(clippy::too_many_arguments)]
/// Decode one packet: header from `headers` (tile data or packed headers),
/// body from `body`. Returns `false` if the data ran out.
pub(crate) fn decode_packet(
    tc: &mut TileComponent,
    id: PacketId,
    headers: &mut HeaderReader<'_>,
    body: &[u8],
    body_pos: &mut usize,
    packed: bool,
    sop: bool,
    eph: bool,
    band_mb: &dyn Fn(usize) -> u32,
) -> Result<bool> {
    let style = tc.coding.cblk_style;
    let res = &mut tc.resolutions[id.res as usize];
    let layer = id.layer as i32;
    // SOP marker segment (in the body stream).
    if sop && body.get(*body_pos..*body_pos + 2) == Some(&[0xFF, 0x91]) {
        *body_pos += 6;
        if !packed {
            headers.pos = *body_pos;
        }
    }
    if !packed {
        headers.pos = *body_pos;
    }
    let Some(present) = headers.bit() else {
        return Ok(false);
    };
    // (block index per band, list of (segment index, passes, length)).
    let mut contributions: Vec<Contribution> = Vec::new();
    if present {
        let precinct = &mut res.precincts[id.precinct as usize];
        for (bi, pb) in precinct.iter_mut().enumerate() {
            let step_index = res.bands[bi].step_index;
            for cy in 0..pb.ch {
                for cx in 0..pb.cw {
                    let k = cy * pb.cw + cx;
                    let mut bit = || headers.bit();
                    let included = if pb.blocks[k].included {
                        match bit() {
                            Some(b) => b,
                            None => return Ok(false),
                        }
                    } else {
                        match pb.incl.decode(cx, cy, layer + 1, &mut bit) {
                            Some(b) => b,
                            None => return Ok(false),
                        }
                    };
                    if !included {
                        continue;
                    }
                    if !pb.blocks[k].included {
                        let Some(z) = pb.zbp.decode_value(cx, cy, &mut bit) else {
                            return Ok(false);
                        };
                        let mb = band_mb(step_index);
                        if z as u32 > mb + 31 {
                            return Err(Error::invalid(CODEC, "invalid number of zero bit-planes"));
                        }
                        pb.blocks[k].zero_bitplanes = z as u32;
                        pb.blocks[k].included = true;
                    }
                    // Number of coding passes (Table B.4).
                    let passes = (|| -> Option<u32> {
                        if !headers.bit()? {
                            return Some(1);
                        }
                        if !headers.bit()? {
                            return Some(2);
                        }
                        let v = headers.bits(2)?;
                        if v < 3 {
                            return Some(3 + v);
                        }
                        let v = headers.bits(5)?;
                        if v < 31 {
                            return Some(6 + v);
                        }
                        Some(37 + headers.bits(7)?)
                    })();
                    let Some(passes) = passes else {
                        return Ok(false);
                    };
                    // Lblock increment (comma code).
                    let cb = &mut pb.blocks[k];
                    loop {
                        match headers.bit() {
                            Some(true) => cb.lblock += 1,
                            Some(false) => break,
                            None => return Ok(false),
                        }
                    }
                    if cb.lblock > 32 {
                        return Err(Error::invalid(CODEC, "Lblock too large"));
                    }
                    let mut parts = Vec::new();
                    if style & STYLE_HT != 0 && cb.ht_state != HtState::Part1 {
                        let mixed = style & STYLE_HT_MIXED == STYLE_HT_MIXED;
                        let mut remaining = passes;
                        if cb.ht_state == HtState::Placeholders {
                            // All passes so far sit in one open segment. A cleanup
                            // pass can only be at a global index divisible by 3.
                            if cb.segments.is_empty() {
                                cb.segments.push(Segment::default());
                            }
                            let index: u32 = cb.segments[0].passes;
                            let refinement = (index + passes - 1) % 3;
                            let to_cleanup = passes as i64 - refinement as i64;
                            let (n, len, cleanup) = if to_cleanup < 1 {
                                let Some(len) = headers.bits(cb.lblock + floor_log2(passes)) else {
                                    return Ok(false);
                                };
                                (passes, len, false)
                            } else {
                                let n = to_cleanup as u32;
                                let bits = cb.lblock + floor_log2(n);
                                let Some(mut len) = headers.bits(bits) else {
                                    return Ok(false);
                                };
                                let msb_clear = len >> (bits - 1) == 0;
                                let found =
                                    len != 0 && (!mixed || (cb.lblock > 3 && len > 1 && msb_clear));
                                if found {
                                    if len < 2 {
                                        return Err(Error::invalid(
                                            CODEC,
                                            "HT cleanup segment shorter than 2 bytes",
                                        ));
                                    }
                                    (n, len, true)
                                } else {
                                    // The length field spans every pass of this contribution.
                                    let extra = floor_log2(passes) - floor_log2(n);
                                    let Some(more) = headers.bits(extra) else {
                                        return Ok(false);
                                    };
                                    len = (len << extra) | more;
                                    (passes, len, false)
                                }
                            };
                            if !cleanup && len != 0 {
                                if !mixed {
                                    return Err(Error::invalid(
                                        CODEC,
                                        "HT placeholder passes carry data",
                                    ));
                                }
                                // Mixed mode: a Part 1 code-block.
                                cb.ht_state = HtState::Part1;
                                cb.segments[0].max_passes = 109;
                            }
                            parts.push((0, n, len));
                            remaining -= n;
                            if cleanup {
                                cb.ht_state = HtState::Coded;
                                cb.segments[0].cup_pass = Some(index + n - 1);
                            }
                        }
                        if cb.ht_state == HtState::Coded {
                            let mut total: u32 = cb.segments.iter().map(|s| s.passes).sum::<u32>()
                                + parts.iter().map(|&(_, n, _)| n).sum::<u32>();
                            while remaining > 0 {
                                let phase = total % 3;
                                let (seg, n) = if phase == 0 {
                                    cb.segments.push(Segment {
                                        cup_pass: Some(total),
                                        ..Segment::default()
                                    });
                                    (cb.segments.len() - 1, 1)
                                } else {
                                    let open = cb.segments.last().is_some_and(|s| {
                                        s.cup_pass.is_none() && s.max_passes == 2 && phase == 2
                                    });
                                    if !open {
                                        cb.segments.push(Segment {
                                            max_passes: 2,
                                            ..Segment::default()
                                        });
                                    }
                                    (cb.segments.len() - 1, remaining.min(3 - phase))
                                };
                                let Some(len) = headers.bits(cb.lblock + floor_log2(n)) else {
                                    return Ok(false);
                                };
                                if phase == 0 && len == 1 {
                                    return Err(Error::invalid(
                                        CODEC,
                                        "HT cleanup segment of 1 byte",
                                    ));
                                }
                                parts.push((seg as u32, n, len));
                                total += n;
                                remaining -= n;
                            }
                        }
                    } else {
                        // Split the new passes into segments and read their lengths.
                        let mut remaining = passes;
                        let mut seg_index = cb.segments.len();
                        let mut cur_passes = 0;
                        let mut cur_max = 0;
                        if let Some(last) = cb.segments.last()
                            && last.passes < last.max_passes
                        {
                            seg_index -= 1;
                            cur_passes = last.passes;
                            cur_max = last.max_passes;
                        }
                        let mut virtual_segments: Vec<Segment> = Vec::new();
                        while remaining > 0 {
                            if seg_index == cb.segments.len() + virtual_segments.len() {
                                let mut all: Vec<Segment> = cb.segments.clone();
                                all.extend(virtual_segments.iter().cloned());
                                let m = max_passes(style, &all);
                                virtual_segments.push(Segment {
                                    passes: 0,
                                    max_passes: m,
                                    data: Vec::new(),
                                    cup_pass: None,
                                });
                                cur_passes = 0;
                                cur_max = m;
                            }
                            let n = remaining.min(cur_max - cur_passes);
                            let bits = cb.lblock + floor_log2(n);
                            let Some(len) = headers.bits(bits) else {
                                return Ok(false);
                            };
                            parts.push((seg_index as u32, n, len));
                            remaining -= n;
                            seg_index += 1;
                            cur_passes = 0;
                            cur_max = 0;
                        }
                        cb.segments.extend(virtual_segments);
                    }
                    contributions.push((
                        bi,
                        k,
                        parts
                            .into_iter()
                            .map(|(s, n, l)| (s << 8 | n.min(255), l))
                            .collect(),
                    ));
                    // Fix up: store passes via the contribution list below.
                }
            }
        }
    }
    headers.align();
    if eph && headers.data.get(headers.pos..headers.pos + 2) == Some(&[0xFF, 0x92]) {
        headers.pos += 2;
    }
    if !packed {
        *body_pos = headers.pos;
    }
    // Packet body.
    let precinct = &mut res.precincts[id.precinct as usize];
    for (bi, k, parts) in contributions {
        let cb = &mut precinct[bi].blocks[k];
        for (sp, len) in parts {
            let (seg, n) = ((sp >> 8) as usize, sp & 0xFF);
            let len = len as usize;
            let start = (*body_pos).min(body.len());
            let end = (start + len).min(body.len());
            let s = &mut cb.segments[seg];
            s.data.extend_from_slice(&body[start..end]);
            s.passes += n;
            *body_pos += len;
            if end - start < len {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
