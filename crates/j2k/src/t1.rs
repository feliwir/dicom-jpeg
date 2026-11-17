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
    st: Vec<u32>,
    mag: Vec<i32>,
}

thread_local! {
    static SCRATCH: std::cell::Cell<Scratch> = std::cell::Cell::new(Scratch::default());
}

// Per-sample state word: the neighbourhood bits (`N_*`, `NEG_*`, bits
// 0..12) and the sample's own flags.
const SIG: u32 = 1 << 12;
const NEG: u32 = 1 << 13;
const VISIT: u32 = 1 << 14;
const REFINED: u32 = 1 << 15;

/// A source of decisions: the MQ decoder, or raw (bypass) bits, which
/// ignore the context.
trait Bits {
    fn decide(&mut self, cx: &mut Context) -> u32;
}

impl Bits for MqDecoder<'_> {
    #[inline(always)]
    fn decide(&mut self, cx: &mut Context) -> u32 {
        self.decode(cx)
    }
}

impl Bits for RawDecoder<'_> {
    #[inline(always)]
    fn decide(&mut self, _: &mut Context) -> u32 {
        self.bit()
    }
}

enum Coder<'a> {
    Mq(MqDecoder<'a>),
    Raw(RawDecoder<'a>),
}

/// The decoding state of a code-block.
struct Block {
    w: usize,
    h: usize,
    stride: usize,
    /// State words with a one-sample border.
    st: Vec<u32>,
    /// Doubled magnitudes.
    mag: Vec<i32>,
    zc: &'static [u8; 256],
    ctx: [Context; NUM_CONTEXTS],
}

impl Block {
    #[inline(always)]
    fn idx(&self, x: usize, y: usize) -> usize {
        (y + 1) * self.stride + x + 1
    }

    /// The state word at `i` (row `y`), with the row below masked in
    /// vertically causal mode.
    #[inline(always)]
    fn word<const VSC: bool>(&self, i: usize, y: usize) -> u32 {
        let w = self.st[i];
        if VSC && y % 4 == 3 {
            w & !(BELOW as u32)
        } else {
            w
        }
    }

    #[inline(always)]
    fn decide<B: Bits>(&mut self, bits: &mut B, cx: usize) -> u32 {
        bits.decide(&mut self.ctx[cx])
    }

    /// The sign of a sample becoming significant: a raw bit in raw passes,
    /// otherwise sign-coded from the neighbourhood word `w`.
    #[inline(always)]
    fn sign<B: Bits, const RAW: bool>(&mut self, bits: &mut B, w: u32) -> bool {
        if RAW {
            return self.decide(bits, 0) == 1;
        }
        let e = SC_LUT[sc_index(w as u16)];
        (self.decide(bits, (e & 0x7F) as usize) ^ (e >> 7) as u32) == 1
    }

    #[inline(always)]
    fn set_significant(&mut self, i: usize, x: usize, y: usize, p: u32, negative: bool) {
        self.st[i] |= SIG | if negative { NEG } else { 0 };
        self.mag[y * self.w + x] = 3 << p;
        let s = self.stride;
        let n = |bit: u16| if negative { bit as u32 } else { 0 };
        let st = &mut self.st;
        st[i + 1] |= N_W as u32 | n(NEG_W);
        st[i - 1] |= N_E as u32 | n(NEG_E);
        st[i + s] |= N_N as u32 | n(NEG_N);
        st[i - s] |= N_S as u32 | n(NEG_S);
        st[i + s + 1] |= N_NW as u32;
        st[i + s - 1] |= N_NE as u32;
        st[i - s + 1] |= N_SW as u32;
        st[i - s - 1] |= N_SE as u32;
    }

    fn significance_pass<B: Bits, const VSC: bool, const RAW: bool>(
        &mut self,
        bits: &mut B,
        p: u32,
    ) {
        for s in (0..self.h).step_by(4) {
            for x in 0..self.w {
                for y in s..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    let w = self.word::<VSC>(i, y);
                    if w & SIG != 0 || w & 0xFF == 0 {
                        continue;
                    }
                    let cx = self.zc[(w & 0xFF) as usize] as usize;
                    if self.decide(bits, cx) == 1 {
                        let neg = self.sign::<B, RAW>(bits, w);
                        self.set_significant(i, x, y, p, neg);
                    }
                    self.st[i] |= VISIT;
                }
            }
        }
    }

    fn refinement_pass<B: Bits, const VSC: bool>(&mut self, bits: &mut B, p: u32) {
        for s in (0..self.h).step_by(4) {
            for x in 0..self.w {
                for y in s..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    let w = self.word::<VSC>(i, y);
                    if w & SIG == 0 || w & VISIT != 0 {
                        continue;
                    }
                    let cx = if w & REFINED != 0 {
                        16
                    } else if w & 0xFF == 0 {
                        14
                    } else {
                        15
                    };
                    let bit = self.decide(bits, cx);
                    let m = &mut self.mag[y * self.w + x];
                    if bit == 1 {
                        *m += 1 << p;
                    } else {
                        *m -= 1 << p;
                    }
                    self.st[i] |= REFINED;
                }
            }
        }
    }

    fn cleanup_pass<B: Bits, const VSC: bool>(&mut self, bits: &mut B, p: u32) {
        for s in (0..self.h).step_by(4) {
            let full = s + 4 <= self.h;
            for x in 0..self.w {
                let mut start = s;
                if full {
                    let all_clear = (s..s + 4)
                        .all(|y| self.word::<VSC>(self.idx(x, y), y) & (SIG | VISIT | 0xFF) == 0);
                    if all_clear {
                        if self.decide(bits, CTX_RL) == 0 {
                            continue;
                        }
                        let r = (self.decide(bits, CTX_UNI) << 1) | self.decide(bits, CTX_UNI);
                        let y = s + r as usize;
                        let i = self.idx(x, y);
                        let neg = self.sign::<B, false>(bits, self.word::<VSC>(i, y));
                        self.set_significant(i, x, y, p, neg);
                        start = y + 1;
                    }
                }
                for y in start..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    let w = self.word::<VSC>(i, y);
                    if w & (SIG | VISIT) == 0
                        && self.decide(bits, self.zc[(w & 0xFF) as usize] as usize) == 1
                    {
                        let neg = self.sign::<B, false>(bits, w);
                        self.set_significant(i, x, y, p, neg);
                    }
                    self.st[i] &= !VISIT;
                }
                // Samples above `start` in an RL-coded column were not visited.
                for y in s..start {
                    let i = self.idx(x, y);
                    self.st[i] &= !VISIT;
                }
            }
        }
    }

    /// Run pass `ptype` (0 significance, 1 refinement, 2 cleanup) on
    /// bit-plane `p` with the coder's bits; `raw`: a raw (bypass) pass.
    fn pass<B: Bits>(&mut self, bits: &mut B, ptype: u8, p: u32, raw: bool, vsc: bool) {
        match (ptype, vsc, raw) {
            (0, false, false) => self.significance_pass::<B, false, false>(bits, p),
            (0, false, true) => self.significance_pass::<B, false, true>(bits, p),
            (0, true, false) => self.significance_pass::<B, true, false>(bits, p),
            (0, true, true) => self.significance_pass::<B, true, true>(bits, p),
            (1, false, _) => self.refinement_pass::<B, false>(bits, p),
            (1, true, _) => self.refinement_pass::<B, true>(bits, p),
            (_, false, _) => self.cleanup_pass::<B, false>(bits, p),
            (_, true, _) => self.cleanup_pass::<B, true>(bits, p),
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
    reset(&mut scratch.st, stride * (h + 2));
    reset(&mut scratch.mag, w * h);
    let mut b = Block {
        w,
        h,
        stride,
        st: std::mem::take(&mut scratch.st),
        mag: std::mem::take(&mut scratch.mag),
        zc: &ZC_LUT[orient as usize],
        ctx: initial_contexts(),
    };
    let vsc = style & STYLE_VSC != 0;
    let mut coder = Coder::Raw(RawDecoder::new(&[]));
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
                coder = if raw {
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
        match &mut coder {
            Coder::Mq(m) => b.pass(m, ptype, plane, raw, vsc),
            Coder::Raw(r) => b.pass(r, ptype, plane, raw, vsc),
        }
        if ptype == 2 && style & STYLE_SEGSYM != 0 {
            let mut sym = 0;
            for _ in 0..4 {
                let bit = match &mut coder {
                    Coder::Mq(m) => b.decide(m, CTX_UNI),
                    Coder::Raw(r) => b.decide(r, CTX_UNI),
                };
                sym = (sym << 1) | bit;
            }
            if sym != 0b1010 {
                // Corrupt data: stop decoding this block (keep what we have).
                break;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let f = b.st[(y + 1) * stride + x + 1];
            let m = b.mag[y * w + x];
            out[y * w + x] = if f & NEG != 0 { -m } else { m };
        }
    }
    SCRATCH.set(Scratch {
        st: b.st,
        mag: b.mag,
    });
    Ok(())
}
