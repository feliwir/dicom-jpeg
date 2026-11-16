//! High-Throughput JPEG 2000 block coding (ITU-T T.814 | ISO/IEC 15444-15).
//!
//! The decoder follows the structure of the reference software OpenJPH by
//! Aous Naman (BSD-2-Clause, see `tables.rs`): a first sweep decodes the
//! VLC / MEL streams of each pair of quads (significance pattern and
//! exponent bounds), a second sweep reads magnitudes and signs from the
//! MagSgn stream, then the optional SigProp and MagRef passes refine one more
//! bit-plane. Internally samples use OpenJPH's 32-bit sign / magnitude layout;
//! the result is converted to the doubled mid-point values of `t1::decode`.

#[cfg(feature = "encode")]
pub(crate) mod encode;
pub(crate) mod tables;

use super::CODEC;
use dicom_jpeg_common::{Error, Result};
use std::sync::OnceLock;

/// Decoding lookup tables built from the CxtVLC tables.
pub(crate) struct Lut {
    /// `vlc[c_q << 7 | 7 VLC bits]` = e_k << 12 | e_1 << 8 | rho << 4 | u_off << 3 | length.
    pub(crate) vlc0: [u16; 1024],
    pub(crate) vlc1: [u16; 1024],
    /// U-VLC prefix decoding for the first row (with MEL-dependent modes).
    pub(crate) uvlc0: [u16; 320],
    pub(crate) uvlc1: [u16; 256],
}

pub(crate) fn lut() -> &'static Lut {
    static LUT: OnceLock<Lut> = OnceLock::new();
    LUT.get_or_init(build_lut)
}

fn build_lut() -> Lut {
    let vlc = |table: &[[u8; 7]]| {
        let mut out = [0u16; 1024];
        for (i, slot) in out.iter_mut().enumerate() {
            let cwd = (i & 0x7F) as u32;
            let c_q = (i >> 7) as u8;
            for r in table {
                if r[0] == c_q && r[5] as u32 == cwd & ((1 << r[6]) - 1) {
                    *slot = ((r[1] as u16) << 4)
                        | ((r[2] as u16) << 3)
                        | ((r[3] as u16) << 12)
                        | ((r[4] as u16) << 8)
                        | r[6] as u16;
                }
            }
        }
        out
    };
    // Prefix codeword decoding: prefix length | suffix length << 2 | u prefix value << 5.
    const DEC: [u8; 8] = [
        3 | (5 << 2) | (5 << 5),
        1 | (1 << 5),
        2 | (2 << 5),
        1 | (1 << 5),
        3 | (1 << 2) | (3 << 5),
        1 | (1 << 5),
        2 | (2 << 5),
        1 | (1 << 5),
    ];
    let pack = |prefix: u32, suffix: u32, u0_suffix: u32, u0: u32, u1: u32| -> u16 {
        (prefix | (suffix << 3) | (u0_suffix << 7) | (u0 << 10) | (u1 << 13)) as u16
    };
    let mut uvlc0 = [0u16; 320];
    for (i, slot) in uvlc0.iter_mut().enumerate() {
        let mode = i >> 6;
        let mut v = i & 0x3F;
        *slot = match mode {
            0 => 0,
            1 | 2 => {
                let d = DEC[v & 7] as u32;
                let suffix = (d >> 2) & 7;
                if mode == 1 {
                    pack(d & 3, suffix, suffix, d >> 5, 0)
                } else {
                    pack(d & 3, suffix, 0, 0, d >> 5)
                }
            }
            3 => {
                let d0 = DEC[v & 7] as u32;
                v >>= d0 & 3;
                let d1 = DEC[v & 7] as u32;
                if d0 & 3 == 3 {
                    let s = (d0 >> 2) & 7;
                    pack((d0 & 3) + 1, s, s, d0 >> 5, (v & 1) as u32 + 1)
                } else {
                    let s0 = (d0 >> 2) & 7;
                    pack(
                        (d0 & 3) + (d1 & 3),
                        s0 + ((d1 >> 2) & 7),
                        s0,
                        d0 >> 5,
                        d1 >> 5,
                    )
                }
            }
            _ => {
                let d0 = DEC[v & 7] as u32;
                v >>= d0 & 3;
                let d1 = DEC[v & 7] as u32;
                let s0 = (d0 >> 2) & 7;
                pack(
                    (d0 & 3) + (d1 & 3),
                    s0 + ((d1 >> 2) & 7),
                    s0,
                    (d0 >> 5) + 2,
                    (d1 >> 5) + 2,
                )
            }
        };
    }
    let mut uvlc1 = [0u16; 256];
    for (i, slot) in uvlc1.iter_mut().enumerate() {
        let mode = i >> 6;
        let mut v = i & 0x3F;
        *slot = match mode {
            0 => 0,
            1 | 2 => {
                let d = DEC[v & 7] as u32;
                let suffix = (d >> 2) & 7;
                if mode == 1 {
                    pack(d & 3, suffix, suffix, d >> 5, 0)
                } else {
                    pack(d & 3, suffix, 0, 0, d >> 5)
                }
            }
            _ => {
                let d0 = DEC[v & 7] as u32;
                v >>= d0 & 3;
                let d1 = DEC[v & 7] as u32;
                let s0 = (d0 >> 2) & 7;
                pack(
                    (d0 & 3) + (d1 & 3),
                    s0 + ((d1 >> 2) & 7),
                    s0,
                    d0 >> 5,
                    d1 >> 5,
                )
            }
        };
    }
    Lut {
        vlc0: vlc(&tables::TABLE0),
        vlc1: vlc(&tables::TABLE1),
        uvlc0,
        uvlc1,
    }
}

// ---------------------------------------------------------------------------
// Bit readers

/// MEL decoder (forward, MSB first, adaptive run-length code).
struct Mel<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
    byte: u32,
    bits: u32,
    prev_ff: bool,
    k: usize,
    zeros: u32,
    one_after: bool,
}

const MEL_E: [u32; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];

impl<'a> Mel<'a> {
    fn new(data: &'a [u8], lcup: usize, scup: usize) -> Self {
        Mel {
            data,
            pos: lcup - scup,
            end: lcup - 1,
            byte: 0,
            bits: 0,
            prev_ff: false,
            k: 0,
            zeros: 0,
            one_after: false,
        }
    }

    fn bit(&mut self) -> u32 {
        if self.bits == 0 {
            let mut b = if self.pos < self.end {
                self.data[self.pos] as u32
            } else {
                0xFF
            };
            if self.pos + 1 == self.end {
                b |= 0x0F; // the last MEL byte shares its low nibble with Scup
            }
            self.pos += 1;
            self.bits = if self.prev_ff { 7 } else { 8 };
            self.prev_ff = b == 0xFF;
            self.byte = b;
        }
        self.bits -= 1;
        (self.byte >> self.bits) & 1
    }

    fn event(&mut self) -> bool {
        loop {
            if self.zeros > 0 {
                self.zeros -= 1;
                return false;
            }
            if self.one_after {
                self.one_after = false;
                return true;
            }
            let e = MEL_E[self.k];
            if self.bit() == 1 {
                self.zeros = 1 << e;
                self.k = (self.k + 1).min(12);
            } else {
                let mut r = 0;
                for _ in 0..e {
                    r = (r << 1) | self.bit();
                }
                self.zeros = r;
                self.one_after = true;
                self.k = self.k.saturating_sub(1);
            }
        }
    }
}

/// Backward reader (VLC and MagRef), LSB first, with unstuffing of a 0x7F
/// byte that follows (in reading order) a byte above 0x8F.
struct Rev<'a> {
    data: &'a [u8],
    /// Index of the next byte to read (moving down), `None` when exhausted.
    pos: Option<usize>,
    left: usize,
    acc: u64,
    bits: u32,
    unstuff: bool,
}

impl<'a> Rev<'a> {
    fn vlc(data: &'a [u8], lcup: usize, scup: usize) -> Self {
        let d = data[lcup - 2] as u64;
        let tmp = d >> 4;
        let mut r = Rev {
            data,
            pos: lcup.checked_sub(3),
            left: scup - 2,
            acc: tmp,
            bits: 4 - ((tmp & 7) == 7) as u32,
            unstuff: d > 0x8F, // i.e. (d | 0xF) > 0x8F
        };
        r.fill();
        r
    }

    fn mrp(data: &'a [u8], start: usize, len: usize) -> Self {
        let mut r = Rev {
            data,
            pos: (start + len).checked_sub(1),
            left: len,
            acc: 0,
            bits: 0,
            unstuff: true,
        };
        r.fill();
        r
    }

    fn fill(&mut self) {
        while self.bits <= 56 {
            let d = match (self.left, self.pos) {
                (0, _) | (_, None) => 0,
                (_, Some(p)) => {
                    self.left -= 1;
                    self.pos = p.checked_sub(1);
                    self.data[p] as u64
                }
            };
            let width = 8 - (self.unstuff && (d & 0x7F) == 0x7F) as u32;
            self.acc |= d << self.bits;
            self.bits += width;
            self.unstuff = d > 0x8F;
        }
    }

    #[inline]
    fn peek(&mut self) -> u32 {
        if self.bits < 32 {
            self.fill();
        }
        self.acc as u32
    }

    #[inline]
    fn advance(&mut self, n: u32) {
        self.acc >>= n;
        self.bits -= n;
    }
}

/// Forward reader (MagSgn, SigProp), LSB first; after 0xFF the next byte
/// carries 7 bits. Past the end, `fill` bytes are supplied.
struct Fwd<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
    fill_byte: u64,
    acc: u64,
    bits: u32,
    unstuff: bool,
}

impl<'a> Fwd<'a> {
    fn new(data: &'a [u8], start: usize, len: usize, fill_byte: u8) -> Self {
        Fwd {
            data,
            pos: start,
            end: start + len,
            fill_byte: fill_byte as u64,
            acc: 0,
            bits: 0,
            unstuff: false,
        }
    }

    fn fill(&mut self) {
        while self.bits <= 56 {
            let d = if self.pos < self.end {
                self.pos += 1;
                self.data[self.pos - 1] as u64
            } else {
                self.fill_byte
            };
            self.acc |= d << self.bits;
            self.bits += 8 - self.unstuff as u32;
            self.unstuff = d == 0xFF;
        }
    }

    #[inline]
    fn peek(&mut self) -> u32 {
        if self.bits < 32 {
            self.fill();
        }
        self.acc as u32
    }

    #[inline]
    fn advance(&mut self, n: u32) {
        self.acc >>= n;
        self.bits -= n;
    }
}

#[inline]
fn le16x2(a: &[u16], i: usize) -> u32 {
    a[i] as u32 | ((a[i + 1] as u32) << 16)
}

fn malformed(msg: &'static str) -> Error {
    Error::invalid(CODEC, msg)
}

/// Decode an HT code-block.
///
/// * `cleanup` / `refinement`: the HT cleanup segment and the optional
///   SigProp / MagRef segment;
/// * `passes`: number of HT passes (1..=3);
/// * `missing_msbs`: zero bit-planes signalled for the block (incl. placeholders);
/// * `bitplanes`: Mb - missing_msbs (+ ROI shift), i.e. the cleanup pass
///   reconstructs down to bit-plane `bitplanes - 1`;
/// * `out`: doubled mid-point values (same convention as `t1::decode`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode(
    w: usize,
    h: usize,
    cleanup: &[u8],
    refinement: &[u8],
    passes: u32,
    missing_msbs: u32,
    bitplanes: u32,
    stripe_causal: bool,
    out: &mut [i32],
) -> Result<()> {
    out.fill(0);
    if passes == 0 || w == 0 || h == 0 {
        return Ok(());
    }
    let mut passes = passes.min(3);
    if passes > 1 && refinement.is_empty() {
        passes = 1;
    }
    if missing_msbs > 29 || bitplanes == 0 {
        return Err(Error::unsupported(
            CODEC,
            "HT code-block needs more than 30 bit-planes",
        ));
    }
    if missing_msbs == 29 || bitplanes == 1 {
        passes = 1;
    }
    let p = 30 - missing_msbs;
    let lcup = cleanup.len();
    if lcup < 2 {
        return Err(malformed("HT cleanup segment shorter than 2 bytes"));
    }
    let scup = ((cleanup[lcup - 1] as usize) << 4) + (cleanup[lcup - 2] & 0xF) as usize;
    if scup < 2 || scup > lcup || scup > 4079 {
        return Err(malformed("invalid HT Scup"));
    }
    // Data buffer: cleanup followed by refinement (MagRef reads backwards from its end).
    let mut data = Vec::with_capacity(lcup + refinement.len());
    data.extend_from_slice(cleanup);
    data.extend_from_slice(refinement);
    let lut = lut();
    let mmsbp2 = missing_msbs + 2;

    let qw = w.div_ceil(2); // quads per row
    let sstr = ((w + 2) + 7) & !7;
    let qrows = h.div_ceil(2) + 1;
    let mut scratch = vec![0u16; sstr * (qrows + 1)];
    let stride = w;
    let rows = h + 1; // room for the odd last row of quads
    let mut dec = vec![0u32; stride * rows + 4];

    // First sweep: VLC + MEL.
    {
        let mut mel = Mel::new(&data, lcup, scup);
        let mut vlc = Rev::vlc(&data, lcup, scup);
        let mut c_q: usize = 0;
        let mut x = 0usize;
        let mut sp = 0usize;
        while x < w {
            let v = vlc.peek();
            let mut t0 = lut.vlc0[c_q + (v & 0x7F) as usize];
            if c_q == 0 && !mel.event() {
                t0 = 0;
            }
            scratch[sp] = t0;
            x += 2;
            c_q = (((t0 & 0x10) << 3) | ((t0 & 0xE0) << 2)) as usize;
            vlc.advance((t0 & 7) as u32);
            let v = vlc.peek();
            let mut t1 = lut.vlc0[c_q + (v & 0x7F) as usize];
            if c_q == 0 && x < w && !mel.event() {
                t1 = 0;
            }
            if x >= w {
                t1 = 0;
            }
            scratch[sp + 2] = t1;
            x += 2;
            c_q = (((t1 & 0x10) << 3) | ((t1 & 0xE0) << 2)) as usize;
            vlc.advance((t1 & 7) as u32);
            let mut mode = (((t0 & 8) << 3) | ((t1 & 8) << 4)) as usize;
            if mode == 0xC0 && mel.event() {
                mode += 0x40;
            }
            let v = vlc.peek();
            let mut e = lut.uvlc0[mode + (v & 0x3F) as usize] as u32;
            vlc.advance(e & 7);
            e >>= 3;
            let len = e & 0xF;
            let v = vlc.peek();
            let tmp = v & ((1u32 << len) - 1);
            vlc.advance(len);
            e >>= 4;
            let len0 = e & 7;
            e >>= 3;
            scratch[sp + 1] = (1 + (e & 7) + (tmp & !(0xFFu32 << len0))) as u16;
            scratch[sp + 3] = (1 + (e >> 3) + (tmp >> len0)) as u16;
            sp += 4;
        }
        scratch[sp] = 0;
        scratch[sp + 1] = 0;
        let mut y = 2;
        while y < h {
            let mut c_q: usize = 0;
            let row = (y >> 1) * sstr;
            let up = row - sstr;
            let mut sp = row;
            let mut x = 0usize;
            while x < w {
                let su = sp - row + up;
                c_q |= ((scratch[su] & 0xA0) << 2) as usize;
                c_q |= ((scratch[su + 2] & 0x20) << 4) as usize;
                let v = vlc.peek();
                let mut t0 = lut.vlc1[c_q + (v & 0x7F) as usize];
                if c_q == 0 && !mel.event() {
                    t0 = 0;
                }
                scratch[sp] = t0;
                x += 2;
                c_q = (((t0 & 0x40) << 2) | ((t0 & 0x80) << 1)) as usize;
                c_q |= (scratch[su] & 0x80) as usize;
                c_q |= ((scratch[su + 2] & 0xA0) << 2) as usize;
                c_q |= ((scratch[su + 4] & 0x20) << 4) as usize;
                vlc.advance((t0 & 7) as u32);
                let v = vlc.peek();
                let mut t1 = lut.vlc1[c_q + (v & 0x7F) as usize];
                if c_q == 0 && x < w && !mel.event() {
                    t1 = 0;
                }
                if x >= w {
                    t1 = 0;
                }
                scratch[sp + 2] = t1;
                x += 2;
                c_q = (((t1 & 0x40) << 2) | ((t1 & 0x80) << 1)) as usize;
                c_q |= (scratch[su + 2] & 0x80) as usize;
                vlc.advance((t1 & 7) as u32);
                let mode = (((t0 & 8) << 3) | ((t1 & 8) << 4)) as usize;
                let v = vlc.peek();
                let mut e = lut.uvlc1[mode + (v & 0x3F) as usize] as u32;
                vlc.advance(e & 7);
                e >>= 3;
                let len = e & 0xF;
                let v = vlc.peek();
                let tmp = v & ((1u32 << len) - 1);
                vlc.advance(len);
                e >>= 4;
                let len0 = e & 7;
                e >>= 3;
                scratch[sp + 1] = ((e & 7) + (tmp & !(0xFFu32 << len0))) as u16;
                scratch[sp + 3] = ((e >> 3) + (tmp >> len0)) as u16;
                sp += 4;
            }
            scratch[sp] = 0;
            scratch[sp + 1] = 0;
            y += 2;
        }
    }

    // Second sweep: MagSgn.
    {
        let mut ms = Fwd::new(&data, 0, lcup - scup, 0xFF);
        let mut vn = vec![0u32; qw + 4];
        let sample = |ms: &mut Fwd<'_>, inf: u32, u_q: u32, bit: u32| -> Result<(u32, u32)> {
            if inf & (1 << (4 + bit)) == 0 {
                return Ok((0, 0));
            }
            let msv = ms.peek();
            let m_n = u_q
                .checked_sub((inf >> (12 + bit)) & 1)
                .ok_or_else(|| malformed("HT exponent underflow"))?;
            if m_n > 31 {
                return Err(malformed("HT magnitude too large"));
            }
            ms.advance(m_n);
            let mut v_n = if m_n == 0 {
                0
            } else {
                msv & ((1u32 << m_n) - 1)
            };
            v_n |= ((inf >> (8 + bit)) & 1) << m_n;
            v_n |= 1;
            let mag = ((v_n as u64 + 2) << (p - 1)) as u32 & 0x7FFF_FFFF;
            Ok(((msv << 31) | mag, v_n))
        };
        let mut y = 0;
        while y < h {
            let row = (y >> 1) * sstr;
            let mut sp = row;
            let mut vp = 0usize;
            let mut prev_vn = 0u32;
            let mut x = 0usize;
            while x < w {
                let inf = scratch[sp] as u32;
                let u_q = scratch[sp + 1] as u32;
                let big_u = if y == 0 {
                    u_q
                } else {
                    let gamma = inf & 0xF0;
                    let gamma = gamma & gamma.wrapping_sub(0x10);
                    let emax = vn[vp] | vn[vp + 1];
                    let emax = 31 - (emax | 2).leading_zeros();
                    u_q + if gamma != 0 { emax } else { 1 }
                };
                if big_u > mmsbp2 {
                    return Err(malformed("HT exponent bound exceeds missing MSBs + 2"));
                }
                let (v0, _) = sample(&mut ms, inf, big_u, 0)?;
                dec[y * stride + x] = v0;
                let (v1, n1) = sample(&mut ms, inf, big_u, 1)?;
                dec[(y + 1) * stride + x] = v1;
                vn[vp] = prev_vn | n1;
                prev_vn = 0;
                x += 1;
                if x >= w {
                    vp += 1;
                    break;
                }
                let (v2, _) = sample(&mut ms, inf, big_u, 2)?;
                dec[y * stride + x] = v2;
                let (v3, n3) = sample(&mut ms, inf, big_u, 3)?;
                dec[(y + 1) * stride + x] = v3;
                prev_vn = n3;
                x += 1;
                sp += 2;
                vp += 1;
            }
            vn[vp] = prev_vn;
            // Clear the stale entry after the row (row widths can differ by one quad).
            if vp + 1 < vn.len() {
                vn[vp + 1] = 0;
            }
            y += 2;
        }
    }

    // SigProp and MagRef.
    if passes > 1 {
        let mstr = ((w.div_ceil(4) + 2) + 7) & !7;
        let srows = h.div_ceil(4) + 2;
        let mut sigma = vec![0u16; mstr * srows];
        let mut y = 0;
        while y < h {
            let sp_row = (y >> 1) * sstr;
            let dp_row = (y >> 2) * mstr;
            let mut x = 0;
            let mut sp = sp_row;
            let mut dp = dp_row;
            while x < w {
                let s = |i: usize| scratch.get(i).copied().unwrap_or(0) as u32;
                let mut t0 = ((s(sp) & 0x30) >> 4) | ((s(sp) & 0xC0) >> 2);
                t0 |= ((s(sp + 2) & 0x30) << 4) | ((s(sp + 2) & 0xC0) << 6);
                let mut t1 = ((s(sp + sstr) & 0x30) >> 2) | (s(sp + sstr) & 0xC0);
                t1 |= ((s(sp + 2 + sstr) & 0x30) << 6) | ((s(sp + 2 + sstr) & 0xC0) << 8);
                t0 |= t1;
                sigma[dp] = t0 as u16;
                x += 4;
                sp += 4;
                dp += 1;
            }
            y += 4;
        }
        let mut prev_row_sig = vec![0u16; mstr + 8];
        let mut sigprop = Fwd::new(&data, lcup, refinement.len(), 0);
        let mut y = 0;
        while y < h {
            let mut pattern: u32 = match h - y {
                1 => 0x1111,
                2 => 0x3333,
                3 => 0x7777,
                _ => 0xFFFF,
            };
            let mut prev = 0u32;
            let cur_row = (y >> 2) * mstr;
            let mut x = 0;
            let mut k = 0;
            while x < w {
                let s = (x as i64 + 4 - w as i64).max(0) as u32;
                pattern >>= s * 4;
                let ps = le16x2(&prev_row_sig, k);
                let ns = le16x2(&sigma, cur_row + mstr + k);
                let mut u = (ps & 0x8888_8888) >> 3;
                if !stripe_causal {
                    u |= (ns & 0x1111_1111) << 3;
                }
                let cs = le16x2(&sigma, cur_row + k);
                let mut mbr = cs;
                mbr |= (cs & 0x7777_7777) << 1;
                mbr |= (cs & 0xEEEE_EEEE) >> 1;
                mbr |= u;
                let t = mbr;
                mbr |= t << 4;
                mbr |= t >> 4;
                mbr |= prev >> 12;
                mbr &= pattern;
                mbr &= !cs;
                let mut new_sig = mbr;
                if new_sig != 0 {
                    let mut cwd = sigprop.peek();
                    let mut cnt = 0;
                    let mut col_mask = 0xFu32;
                    let inv_sig = !cs & pattern;
                    let mut i = 0;
                    while i < 16 {
                        if col_mask & new_sig != 0 {
                            let mut sample_mask = 0x1111u32 & col_mask;
                            for spread in [0x33u32, 0x76, 0xEC, 0xC8] {
                                if new_sig & sample_mask != 0 {
                                    new_sig &= !sample_mask;
                                    if cwd & 1 != 0 {
                                        new_sig |= (spread << i) & inv_sig;
                                    }
                                    cwd >>= 1;
                                    cnt += 1;
                                }
                                sample_mask <<= 1;
                            }
                        }
                        i += 4;
                        col_mask <<= 4;
                    }
                    if new_sig != 0 {
                        let val = 3u32 << (p - 2);
                        let mut col_mask = 0xFu32;
                        for c in 0..4usize {
                            if col_mask & new_sig != 0 {
                                let mut sample_mask = 0x1111u32 & col_mask;
                                for r in 0..4usize {
                                    if new_sig & sample_mask != 0 {
                                        let (yy, xx) = (y + r, x + c);
                                        if yy < h && xx < w {
                                            dec[yy * stride + xx] = (cwd << 31) | val;
                                        }
                                        cwd >>= 1;
                                        cnt += 1;
                                    }
                                    sample_mask <<= 1;
                                }
                            }
                            col_mask <<= 4;
                        }
                    }
                    sigprop.advance(cnt);
                }
                new_sig |= cs;
                // Significance for the stripe below (MagRef keeps using `sigma`).
                prev_row_sig[k] = (new_sig & 0xFFFF) as u16;
                let t = new_sig;
                let mut ns2 = new_sig;
                ns2 |= (t & 0x7777) << 1;
                ns2 |= (t & 0xEEEE) >> 1;
                prev = (ns2 | u) & 0xF000;
                x += 4;
                k += 1;
            }
            y += 4;
        }
        if passes > 2 {
            // MagRef refines the samples that were significant after the cleanup pass.
            let mut magref = Rev::mrp(&data, lcup, refinement.len());
            let half = 1u32 << (p - 2);
            let mut y = 0;
            while y < h {
                let row = (y >> 2) * mstr;
                let mut i = 0;
                let mut k = 0;
                while i < w {
                    let mut cwd = magref.peek();
                    let sig = le16x2(&sigma, row + k);
                    let mut col_mask = 0xFu32;
                    if sig != 0 {
                        for j in 0..8usize {
                            if sig & col_mask != 0 {
                                let mut sample_mask = 0x1111_1111u32 & col_mask;
                                for r in 0..4usize {
                                    if sig & sample_mask != 0 {
                                        let (yy, xx) = (y + r, i + j);
                                        let sym = cwd & 1;
                                        let flip = ((1 - sym) << (p - 1)) | half;
                                        if yy < h && xx < w {
                                            dec[yy * stride + xx] ^= flip;
                                        }
                                        cwd >>= 1;
                                    }
                                    sample_mask <<= 1;
                                }
                            }
                            col_mask <<= 4;
                        }
                    }
                    magref.advance(sig.count_ones());
                    i += 8;
                    k += 2;
                }
                y += 4;
            }
        }
    }

    // Convert to doubled mid-point values at bit-plane `bitplanes - 1`.
    let q = bitplanes - 1;
    for yy in 0..h {
        for xx in 0..w {
            let v = dec[yy * stride + xx];
            let m = v & 0x7FFF_FFFF;
            let doubled: u64 = if passes == 1 {
                ((m >> (p - 1)) as u64) << q
            } else {
                ((m >> (p - 2)) as u64) << (q - 1)
            };
            let d = doubled.min(i32::MAX as u64) as i32;
            out[yy * w + xx] = if v >> 31 == 1 { -d } else { d };
        }
    }
    Ok(())
}
