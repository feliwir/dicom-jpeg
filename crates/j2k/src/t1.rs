//! Tier-1 decoding: EBCOT bit-plane coding of code-blocks (T.800 Annex D).
//!
//! Decoded values are returned at twice their true scale with mid-point
//! reconstruction: a coefficient that became significant at bit-plane `p`
//! starts at `3 << p` (1.5 * 2^p, doubled), and every refinement moves it by
//! `±1 << q`. For a fully decoded reversible block, `value >> 1` is exact.

use super::CODEC;
use super::mq::{CTX_RL, CTX_UNI, Context, MqDecoder, NUM_CONTEXTS, RawDecoder, initial_contexts};
use dicom_jpeg_common::{Error, Result};

/// Sub-band orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Orient {
    LL = 0,
    HL = 1,
    LH = 2,
    HH = 3,
}

/// Code-block style flags (Table A.19).
pub(crate) const STYLE_BYPASS: u8 = 0x01;
pub(crate) const STYLE_RESET: u8 = 0x02;
pub(crate) const STYLE_TERMALL: u8 = 0x04;
pub(crate) const STYLE_VSC: u8 = 0x08;
pub(crate) const STYLE_SEGSYM: u8 = 0x20;
pub(crate) const STYLE_HT: u8 = 0x40;
/// HT mixed mode: code-blocks may be HT or Part 1 (bits 6 and 7 set).
pub(crate) const STYLE_HT_MIXED: u8 = 0xC0;

const SIG: u8 = 1;
const NEG: u8 = 2;
const VISIT: u8 = 4;
const REFINED: u8 = 8;

// Neighbourhood word bits: significance of the 8 neighbours, and signs of the
// 4 direct neighbours (set when they become significant).
pub(crate) const N_W: u16 = 1 << 0;
pub(crate) const N_E: u16 = 1 << 1;
pub(crate) const N_N: u16 = 1 << 2;
pub(crate) const N_S: u16 = 1 << 3;
pub(crate) const N_NW: u16 = 1 << 4;
pub(crate) const N_NE: u16 = 1 << 5;
pub(crate) const N_SW: u16 = 1 << 6;
pub(crate) const N_SE: u16 = 1 << 7;
pub(crate) const NEG_W: u16 = 1 << 8;
pub(crate) const NEG_E: u16 = 1 << 9;
pub(crate) const NEG_N: u16 = 1 << 10;
pub(crate) const NEG_S: u16 = 1 << 11;
/// Bits that refer to the row below (ignored in vertically causal mode).
const BELOW: u16 = N_S | N_SW | N_SE | NEG_S;

/// Zero-coding context (Table D.1).
pub(crate) const fn zc_context(orient: Orient, h: u32, v: u32, d: u32) -> usize {
    let (h, v) = if matches!(orient, Orient::HL) {
        (v, h)
    } else {
        (h, v)
    };
    if matches!(orient, Orient::HH) {
        let hv = h + v;
        return match (d, hv) {
            (3.., _) => 8,
            (2, 1..) => 7,
            (2, 0) => 6,
            (1, 2..) => 5,
            (1, 1) => 4,
            (1, 0) => 3,
            (0, 2..) => 2,
            (0, 1) => 1,
            _ => 0,
        };
    }
    match (h, v, d) {
        (2, _, _) => 8,
        (1, 1.., _) => 7,
        (1, 0, 1..) => 6,
        (1, 0, 0) => 5,
        (0, 2, _) => 4,
        (0, 1, _) => 3,
        (0, 0, 2..) => 2,
        (0, 0, 1) => 1,
        _ => 0,
    }
}

/// Sign-coding context and XOR bit (Table D.3) from clamped contributions.
pub(crate) const fn sc_context(h: i32, v: i32) -> (usize, u32) {
    match (h, v) {
        (1, 1) => (13, 0),
        (1, 0) => (12, 0),
        (1, -1) => (11, 0),
        (0, 1) => (10, 0),
        (0, 0) => (9, 0),
        (0, -1) => (10, 1),
        (-1, 1) => (11, 1),
        (-1, 0) => (12, 1),
        _ => (13, 1),
    }
}

/// Zero-coding contexts per orientation (`Orient as usize`), indexed by the
/// 8 neighbour significance bits of a neighbourhood word.
static ZC_LUT: [[u8; 256]; 4] = [
    zc_lut(Orient::LL),
    zc_lut(Orient::HL),
    zc_lut(Orient::LH),
    zc_lut(Orient::HH),
];

const fn zc_lut(orient: Orient) -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let w = i as u16;
        let h = (w & (N_W | N_E)).count_ones();
        let v = (w & (N_N | N_S)).count_ones();
        let d = (w & (N_NW | N_NE | N_SW | N_SE)).count_ones();
        t[i] = zc_context(orient, h, v, d) as u8;
        i += 1;
    }
    t
}

/// Sign-coding context (low 7 bits) and XOR bit (bit 7), indexed by the
/// significance (bits 0..4) and sign (bits 4..8) of the W, E, N, S
/// neighbours: [`sc_index`].
static SC_LUT: [u8; 256] = sc_lut();

const fn sc_lut() -> [u8; 256] {
    /// -1, 0 or 1: the contribution of a neighbour (significance bit `sig`,
    /// sign bit `neg` of index `i`).
    const fn contrib(i: usize, sig: usize, neg: usize) -> i32 {
        if i & sig == 0 {
            0
        } else if i & neg != 0 {
            -1
        } else {
            1
        }
    }
    const fn clamp1(x: i32) -> i32 {
        if x < -1 {
            -1
        } else if x > 1 {
            1
        } else {
            x
        }
    }
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let h = clamp1(contrib(i, 1, 16) + contrib(i, 2, 32));
        let v = clamp1(contrib(i, 4, 64) + contrib(i, 8, 128));
        let (cx, xor) = sc_context(h, v);
        t[i] = cx as u8 | (xor as u8) << 7;
        i += 1;
    }
    t
}

/// The [`SC_LUT`] index of a neighbourhood word.
#[inline]
fn sc_index(w: u16) -> usize {
    ((w & 0xF) | ((w >> 4) & 0xF0)) as usize
}

/// Buffers reused across the code-blocks decoded on a thread.
#[derive(Default)]
struct Scratch {
    flags: Vec<u8>,
    nb: Vec<u16>,
    mag: Vec<i32>,
}

thread_local! {
    static SCRATCH: std::cell::Cell<Scratch> = std::cell::Cell::new(Scratch::default());
}

enum Coder<'a> {
    Mq(MqDecoder<'a>),
    Raw(RawDecoder<'a>),
}

struct Block<'a> {
    w: usize,
    h: usize,
    stride: usize,
    /// Flags with a one-sample border.
    flags: Vec<u8>,
    /// Neighbourhood words, same layout as `flags`.
    nb: Vec<u16>,
    /// Doubled magnitudes.
    mag: Vec<i32>,
    orient: Orient,
    vsc: bool,
    ctx: [Context; NUM_CONTEXTS],
    coder: Coder<'a>,
}

impl Block<'_> {
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        (y + 1) * self.stride + x + 1
    }

    /// Whether the neighbours below must be ignored (vertically causal mode).
    #[inline]
    fn exclude_below(&self, y: usize) -> bool {
        self.vsc && y % 4 == 3
    }

    /// Neighbourhood word of (x, y), with the row below masked in
    /// vertically causal mode.
    #[inline]
    fn nbw(&self, x: usize, y: usize) -> u16 {
        let w = self.nb[self.idx(x, y)];
        if self.exclude_below(y) { w & !BELOW } else { w }
    }

    /// Zero-coding context of (x, y).
    #[inline]
    fn zc(&self, x: usize, y: usize) -> usize {
        ZC_LUT[self.orient as usize][(self.nbw(x, y) & 0xFF) as usize] as usize
    }

    #[inline]
    fn decide(&mut self, cx: usize) -> u32 {
        match &mut self.coder {
            Coder::Mq(m) => m.decode(&mut self.ctx[cx]),
            Coder::Raw(r) => r.bit(),
        }
    }

    #[inline]
    fn raw_bit(&mut self) -> u32 {
        match &mut self.coder {
            Coder::Raw(r) => r.bit(),
            Coder::Mq(_) => unreachable!("raw bit requested from MQ coder"),
        }
    }

    fn decode_sign(&mut self, x: usize, y: usize, raw: bool) -> bool {
        if raw {
            return self.raw_bit() == 1;
        }
        let e = SC_LUT[sc_index(self.nbw(x, y))];
        let bit = self.decide((e & 0x7F) as usize) ^ (e >> 7) as u32;
        bit == 1
    }

    #[inline]
    fn set_significant(&mut self, x: usize, y: usize, p: u32, negative: bool) {
        let i = self.idx(x, y);
        self.flags[i] |= SIG | if negative { NEG } else { 0 };
        self.mag[y * self.w + x] = 3 << p;
        let s = self.stride;
        let nb = &mut self.nb;
        let n = |bit: u16| if negative { bit } else { 0 };
        nb[i + 1] |= N_W | n(NEG_W);
        nb[i - 1] |= N_E | n(NEG_E);
        nb[i + s] |= N_N | n(NEG_N);
        nb[i - s] |= N_S | n(NEG_S);
        nb[i + s + 1] |= N_NW;
        nb[i + s - 1] |= N_NE;
        nb[i - s + 1] |= N_SW;
        nb[i - s - 1] |= N_SE;
    }

    fn significance_pass(&mut self, p: u32, raw: bool) {
        for s in (0..self.h).step_by(4) {
            for x in 0..self.w {
                for y in s..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    if self.flags[i] & SIG != 0 {
                        continue;
                    }
                    if self.nbw(x, y) & 0xFF == 0 {
                        continue;
                    }
                    let bit = if raw {
                        self.raw_bit()
                    } else {
                        self.decide(self.zc(x, y))
                    };
                    if bit == 1 {
                        let neg = self.decode_sign(x, y, raw);
                        self.set_significant(x, y, p, neg);
                    }
                    self.flags[i] |= VISIT;
                }
            }
        }
    }

    fn refinement_pass(&mut self, p: u32, raw: bool) {
        for s in (0..self.h).step_by(4) {
            for x in 0..self.w {
                for y in s..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    let f = self.flags[i];
                    if f & SIG == 0 || f & VISIT != 0 {
                        continue;
                    }
                    let bit = if raw {
                        self.raw_bit()
                    } else {
                        let cx = if f & REFINED != 0 {
                            16
                        } else {
                            if self.nbw(x, y) & 0xFF == 0 { 14 } else { 15 }
                        };
                        self.decide(cx)
                    };
                    let m = &mut self.mag[y * self.w + x];
                    if bit == 1 {
                        *m += 1 << p;
                    } else {
                        *m -= 1 << p;
                    }
                    self.flags[i] |= REFINED;
                }
            }
        }
    }

    fn cleanup_pass(&mut self, p: u32) {
        for s in (0..self.h).step_by(4) {
            let full = s + 4 <= self.h;
            for x in 0..self.w {
                let mut start = s;
                if full {
                    let all_clear = (s..s + 4).all(|y| {
                        let f = self.flags[self.idx(x, y)];
                        f & (SIG | VISIT) == 0 && self.nbw(x, y) & 0xFF == 0
                    });
                    if all_clear {
                        if self.decide(CTX_RL) == 0 {
                            continue;
                        }
                        let r = (self.decide(CTX_UNI) << 1) | self.decide(CTX_UNI);
                        let y = s + r as usize;
                        let neg = self.decode_sign(x, y, false);
                        self.set_significant(x, y, p, neg);
                        start = y + 1;
                    }
                }
                for y in start..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    if self.flags[i] & (SIG | VISIT) == 0 && self.decide(self.zc(x, y)) == 1 {
                        let neg = self.decode_sign(x, y, false);
                        self.set_significant(x, y, p, neg);
                    }
                    self.flags[i] &= !VISIT;
                }
                // Samples above `start` in an RL-coded column were not visited.
                for y in s..start {
                    let i = self.idx(x, y);
                    self.flags[i] &= !VISIT;
                }
            }
        }
    }
}

/// A coded segment of a code-block: number of passes and its bytes.
#[derive(Debug, Clone, Default)]
pub(crate) struct Segment {
    pub(crate) passes: u32,
    pub(crate) max_passes: u32,
    pub(crate) data: Vec<u8>,
    /// HT only: index of the cleanup pass of this (cleanup) segment, counting
    /// placeholder passes; `None` for refinement segments.
    pub(crate) cup_pass: Option<u32>,
}

/// Decode one code-block into `out` (`w * h`, row-major, doubled signed values).
///
/// `bitplanes` is the number of magnitude bit-planes (Mb - Zb, plus the ROI shift).
pub(crate) fn decode(
    w: usize,
    h: usize,
    orient: Orient,
    style: u8,
    bitplanes: u32,
    segments: &[Segment],
    out: &mut [i32],
) -> Result<()> {
    out.fill(0);
    let total: u32 = segments.iter().map(|s| s.passes).sum();
    if total == 0 || bitplanes == 0 || w == 0 || h == 0 {
        return Ok(());
    }
    if style & STYLE_HT != 0 {
        return Err(Error::unsupported(CODEC, "HTJ2K code-blocks"));
    }
    if bitplanes > 30 {
        return Err(Error::unsupported(CODEC, "more than 30 bit-planes"));
    }
    let total = total.min(3 * bitplanes - 2);
    let stride = w + 2;
    let mut scratch = SCRATCH.take();
    fn reset<T: Copy + Default>(v: &mut Vec<T>, n: usize) {
        v.clear();
        v.resize(n, T::default());
    }
    reset(&mut scratch.flags, stride * (h + 2));
    reset(&mut scratch.nb, stride * (h + 2));
    reset(&mut scratch.mag, w * h);
    let mut b = Block {
        w,
        h,
        stride,
        flags: std::mem::take(&mut scratch.flags),
        nb: std::mem::take(&mut scratch.nb),
        mag: std::mem::take(&mut scratch.mag),
        orient,
        vsc: style & STYLE_VSC != 0,
        ctx: initial_contexts(),
        coder: Coder::Raw(RawDecoder::new(&[])),
    };
    let bypass = style & STYLE_BYPASS != 0;
    let mut seg = 0usize;
    let mut left_in_seg = 0u32;
    for k in 0..total {
        // Pass type: 0 = significance, 1 = refinement, 2 = cleanup.
        let (ptype, plane) = if k == 0 {
            (2, bitplanes - 1)
        } else {
            (((k - 1) % 3) as u8, bitplanes - 1 - k.div_ceil(3))
        };
        let raw = bypass && k >= 10 && ptype != 2;
        while left_in_seg == 0 {
            let Some(s) = segments.get(seg) else {
                return Ok(());
            };
            seg += 1;
            left_in_seg = s.passes;
            if left_in_seg > 0 {
                b.coder = if raw {
                    Coder::Raw(RawDecoder::new(&s.data))
                } else {
                    Coder::Mq(MqDecoder::new(&s.data))
                };
            }
        }
        left_in_seg -= 1;
        if style & STYLE_RESET != 0 && k > 0 {
            b.ctx = initial_contexts();
        }
        match ptype {
            0 => b.significance_pass(plane, raw),
            1 => b.refinement_pass(plane, raw),
            _ => {
                b.cleanup_pass(plane);
                if style & STYLE_SEGSYM != 0 {
                    let mut sym = 0;
                    for _ in 0..4 {
                        sym = (sym << 1) | b.decide(CTX_UNI);
                    }
                    if sym != 0b1010 {
                        // Corrupt data: stop decoding this block (keep what we have).
                        break;
                    }
                }
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let f = b.flags[(y + 1) * stride + x + 1];
            let m = b.mag[y * w + x];
            out[y * w + x] = if f & NEG != 0 { -m } else { m };
        }
    }
    SCRATCH.set(Scratch {
        flags: b.flags,
        nb: b.nb,
        mag: b.mag,
    });
    Ok(())
}
