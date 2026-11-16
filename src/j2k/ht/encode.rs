//! HT cleanup-pass encoder (ITU-T T.814 | ISO/IEC 15444-15, clause 7).
//!
//! Every code-block is coded with a single HT cleanup pass that carries all
//! magnitude bits, which is what lossless and single-layer lossy encoders
//! need. The structure (MEL / VLC / MagSgn writers and their termination)
//! follows OpenJPH's `ojph_block_encoder.cpp` (BSD-2-Clause, see
//! `tables.rs`).

use super::tables::{TABLE0, TABLE1};
use crate::j2k::t1_enc::EncodedBlock;
use std::sync::OnceLock;

/// Encoder VLC tables indexed by `context << 8 | rho << 4 | emb`; each entry
/// is `codeword << 8 | length << 4 | e_k` (0 = not representable).
struct VlcTables {
    t0: [u16; 2048],
    t1: [u16; 2048],
}

fn build(src: &[[u8; 7]]) -> [u16; 2048] {
    let mut out = [0u16; 2048];
    for (i, entry) in out.iter_mut().enumerate() {
        let (cq, rho, emb) = ((i >> 8) as u8, ((i >> 4) & 0xF) as u8, (i & 0xF) as u8);
        if emb & rho != emb || (rho == 0 && cq == 0) {
            continue;
        }
        let rows = src.iter().filter(|r| r[0] == cq && r[1] == rho);
        let best = if emb != 0 {
            // u_off = 1: the entry whose e_k covers the most samples.
            let mut best: Option<&[u8; 7]> = None;
            for r in rows.filter(|r| r[2] == 1 && emb & r[3] == r[4]) {
                if best.is_none_or(|b| r[3].count_ones() >= b[3].count_ones()) {
                    best = Some(r);
                }
            }
            best
        } else {
            rows.into_iter().find(|r| r[2] == 0)
        };
        if let Some(r) = best {
            *entry = ((r[5] as u16) << 8) | ((r[6] as u16) << 4) | r[3] as u16;
        }
    }
    out
}

fn vlc_tables() -> &'static VlcTables {
    static T: OnceLock<VlcTables> = OnceLock::new();
    T.get_or_init(|| VlcTables {
        t0: build(&TABLE0),
        t1: build(&TABLE1),
    })
}

/// U-VLC code of `u` (0..=32): (prefix, prefix length, suffix, suffix length).
fn uvlc(u: u32) -> (u32, u32, u32, u32) {
    match u {
        0 => (0, 0, 0, 0),
        1 => (1, 1, 0, 0),
        2 => (2, 2, 0, 0),
        3 | 4 => (4, 3, u - 3, 1),
        _ => (0, 3, u - 5, 5),
    }
}

const MEL_EXP: [u32; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];

struct Mel {
    buf: Vec<u8>,
    remaining: u32,
    tmp: u32,
    run: u32,
    k: usize,
}

impl Mel {
    fn emit(&mut self, bit: u32) {
        self.tmp = (self.tmp << 1) | bit;
        self.remaining -= 1;
        if self.remaining == 0 {
            self.buf.push(self.tmp as u8);
            self.remaining = if self.tmp == 0xFF { 7 } else { 8 };
            self.tmp = 0;
        }
    }

    fn encode(&mut self, bit: bool) {
        if !bit {
            self.run += 1;
            if self.run >= 1 << MEL_EXP[self.k] {
                self.emit(1);
                self.run = 0;
                self.k = (self.k + 1).min(12);
            }
        } else {
            self.emit(0);
            let mut t = MEL_EXP[self.k];
            while t > 0 {
                t -= 1;
                self.emit((self.run >> t) & 1);
            }
            self.run = 0;
            self.k = self.k.saturating_sub(1);
        }
    }
}

/// VLC bytes are produced in reverse order (the decoder reads them backwards).
struct Vlc {
    buf: Vec<u8>,
    used: u32,
    tmp: u32,
    last_gt_8f: bool,
}

impl Vlc {
    fn encode(&mut self, mut cwd: u32, mut len: u32) {
        while len > 0 {
            let mut avail = 8 - self.last_gt_8f as u32 - self.used;
            let t = avail.min(len);
            self.tmp |= (cwd & ((1 << t) - 1)) << self.used;
            self.used += t;
            avail -= t;
            len -= t;
            cwd >>= t;
            if avail == 0 {
                if self.last_gt_8f && self.tmp != 0x7F {
                    // The stuffing bit is not needed: one more bit available.
                    self.last_gt_8f = false;
                    continue;
                }
                self.buf.push(self.tmp as u8);
                self.last_gt_8f = self.tmp > 0x8F;
                self.tmp = 0;
                self.used = 0;
            }
        }
    }
}

struct MagSgn {
    buf: Vec<u8>,
    max_bits: u32,
    used: u32,
    tmp: u32,
}

impl MagSgn {
    fn encode(&mut self, mut cwd: u64, mut len: u32) {
        while len > 0 {
            let t = (self.max_bits - self.used).min(len);
            self.tmp |= ((cwd & ((1u64 << t) - 1)) as u32) << self.used;
            self.used += t;
            cwd >>= t;
            len -= t;
            if self.used >= self.max_bits {
                self.buf.push(self.tmp as u8);
                self.max_bits = if self.tmp == 0xFF { 7 } else { 8 };
                self.tmp = 0;
                self.used = 0;
            }
        }
    }

    fn terminate(&mut self) {
        if self.used > 0 {
            let t = self.max_bits - self.used;
            self.tmp |= (0xFF & ((1 << t) - 1)) << self.used;
            if self.tmp != 0xFF {
                self.buf.push(self.tmp as u8);
            }
        } else if self.max_bits == 7 {
            self.buf.pop();
        }
    }
}

fn terminate_mel_vlc(mel: &mut Mel, vlc: &mut Vlc) {
    if mel.run > 0 {
        mel.emit(1);
    }
    mel.tmp <<= mel.remaining;
    let mel_mask = (0xFFu32 << mel.remaining) & 0xFF;
    let vlc_mask = 0xFFu32 >> (8 - vlc.used);
    if mel_mask | vlc_mask == 0 {
        return;
    }
    let fuse = mel.tmp | vlc.tmp;
    if (((fuse ^ mel.tmp) & mel_mask) | ((fuse ^ vlc.tmp) & vlc_mask)) == 0
        && fuse != 0xFF
        && vlc.buf.len() > 1
    {
        mel.buf.push(fuse as u8);
    } else {
        mel.buf.push(mel.tmp as u8);
        vlc.buf.push(vlc.tmp as u8);
    }
}

/// One 2x2 quad: significance pattern, exponents and MagSgn values
/// (`2 (mu - 1) + sign`) of samples (x,y), (x,y+1), (x+1,y), (x+1,y+1).
#[derive(Default)]
struct Quad {
    rho: u32,
    e: [u32; 4],
    s: [u64; 4],
    emax: u32,
}

impl Quad {
    fn eps(&self, u: u32) -> u32 {
        if u == 0 {
            return 0;
        }
        (0..4)
            .filter(|&i| self.e[i] == self.emax)
            .fold(0, |a, i| a | (1 << i))
    }

    fn emit_magsgn(&self, ms: &mut MagSgn, big_u: u32, tuple: u16) {
        for i in 0..4 {
            if self.rho & (1 << i) != 0 {
                let m = big_u - ((tuple as u32 >> i) & 1);
                ms.encode(self.s[i] & ((1u64 << m) - 1), m);
            }
        }
    }
}

/// Encode quantized coefficients (`w * h`, row-major, |v| < 2^30) as one HT
/// cleanup segment. Returns `passes == 0` when every coefficient is zero.
/// `None` when the block is too large for the HT segment limits.
pub(crate) fn encode(w: usize, h: usize, coefs: &[i32]) -> Option<EncodedBlock> {
    let max = coefs.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
    if max == 0 {
        return Some(EncodedBlock::default());
    }
    if w > 1024 || w * h > 4096 || max >= 1 << 30 {
        return None;
    }
    let bitplanes = 32 - max.leading_zeros();
    let tables = vlc_tables();
    let quad = |x: usize, y: usize| -> Quad {
        let mut q = Quad::default();
        for (i, (dx, dy)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
            let (sx, sy) = (x + dx, y + dy);
            if sx >= w || sy >= h {
                continue;
            }
            let v = coefs[sy * w + sx];
            let mu = v.unsigned_abs() as u64;
            if mu == 0 {
                continue;
            }
            q.rho |= 1 << i;
            q.e[i] = 64 - (2 * mu - 1).leading_zeros();
            q.emax = q.emax.max(q.e[i]);
            q.s[i] = 2 * (mu - 1) + (v < 0) as u64;
        }
        q
    };

    let mut mel = Mel {
        buf: Vec::new(),
        remaining: 8,
        tmp: 0,
        run: 0,
        k: 0,
    };
    let mut vlc = Vlc {
        buf: vec![0xFF],
        used: 4,
        tmp: 0xF,
        last_gt_8f: true,
    };
    let mut ms = MagSgn {
        buf: Vec::with_capacity(w * h),
        max_bits: 8,
        used: 0,
        tmp: 0,
    };

    let qw = w.div_ceil(2);
    // Per quad column: largest exponent / significance of the bottom row
    // samples that touch it (feeds the next quad row's kappa and context).
    let mut lep = vec![0u32; qw + 2];
    let mut lcx = vec![0u32; qw + 2];

    // First quad row (table 0, pairs share MEL / U-VLC coding).
    let mut c_q0 = 0u32;
    let mut li = 0usize;
    for x in (0..w).step_by(4) {
        let q0 = quad(x, 0);
        let big_u0 = q0.emax.max(1);
        let u_q0 = big_u0 - 1;
        let eps0 = q0.eps(u_q0);
        lep[li] = lep[li].max(q0.e[1]);
        li += 1;
        lep[li] = q0.e[3];
        lcx[li - 1] |= (q0.rho & 2) >> 1;
        lcx[li] = (q0.rho & 8) >> 3;
        let tuple0 = tables.t0[((c_q0 << 8) | (q0.rho << 4) | eps0) as usize];
        vlc.encode(tuple0 as u32 >> 8, (tuple0 as u32 >> 4) & 7);
        if c_q0 == 0 {
            mel.encode(q0.rho != 0);
        }
        q0.emit_magsgn(&mut ms, big_u0, tuple0);

        let mut u_q1 = 0;
        let mut rho1 = 0;
        if x + 2 < w {
            let q1 = quad(x + 2, 0);
            rho1 = q1.rho;
            let c_q1 = (q0.rho >> 1) | (q0.rho & 1);
            let big_u1 = q1.emax.max(1);
            u_q1 = big_u1 - 1;
            let eps1 = q1.eps(u_q1);
            lep[li] = lep[li].max(q1.e[1]);
            li += 1;
            lep[li] = q1.e[3];
            lcx[li - 1] |= (q1.rho & 2) >> 1;
            lcx[li] = (q1.rho & 8) >> 3;
            let tuple1 = tables.t0[((c_q1 << 8) | (q1.rho << 4) | eps1) as usize];
            vlc.encode(tuple1 as u32 >> 8, (tuple1 as u32 >> 4) & 7);
            if c_q1 == 0 {
                mel.encode(q1.rho != 0);
            }
            q1.emit_magsgn(&mut ms, big_u1, tuple1);
        }
        if u_q0 > 0 && u_q1 > 0 {
            mel.encode(u_q0.min(u_q1) > 2);
        }
        if u_q0 > 2 && u_q1 > 2 {
            let (a, b) = (uvlc(u_q0 - 2), uvlc(u_q1 - 2));
            vlc.encode(a.0, a.1);
            vlc.encode(b.0, b.1);
            vlc.encode(a.2, a.3);
            vlc.encode(b.2, b.3);
        } else if u_q0 > 2 && u_q1 > 0 {
            let a = uvlc(u_q0);
            vlc.encode(a.0, a.1);
            vlc.encode(u_q1 - 1, 1);
            vlc.encode(a.2, a.3);
        } else {
            let (a, b) = (uvlc(u_q0), uvlc(u_q1));
            vlc.encode(a.0, a.1);
            vlc.encode(b.0, b.1);
            vlc.encode(a.2, a.3);
            vlc.encode(b.2, b.3);
        }
        c_q0 = (rho1 >> 1) | (rho1 & 1);
    }

    // Remaining quad rows (table 1).
    for y in (2..h).step_by(2) {
        li = 0;
        let mut max_e = lep[0].max(lep[1]) as i32 - 1;
        lep[0] = 0;
        let mut c_q0 = lcx[0] + (lcx[1] << 2);
        lcx[0] = 0;
        for x in (0..w).step_by(4) {
            let q0 = quad(x, y);
            let kappa0 = if q0.rho & q0.rho.wrapping_sub(1) != 0 {
                max_e.max(1) as u32
            } else {
                1
            };
            let big_u0 = q0.emax.max(kappa0);
            let u_q0 = big_u0 - kappa0;
            let eps0 = q0.eps(u_q0);
            lep[li] = lep[li].max(q0.e[1]);
            li += 1;
            max_e = lep[li].max(lep[li + 1]) as i32 - 1;
            lep[li] = q0.e[3];
            lcx[li - 1] |= (q0.rho & 2) >> 1;
            let mut c_q1 = lcx[li] + (lcx[li + 1] << 2);
            lcx[li] = (q0.rho & 8) >> 3;
            let tuple0 = tables.t1[((c_q0 << 8) | (q0.rho << 4) | eps0) as usize];
            vlc.encode(tuple0 as u32 >> 8, (tuple0 as u32 >> 4) & 7);
            if c_q0 == 0 {
                mel.encode(q0.rho != 0);
            }
            q0.emit_magsgn(&mut ms, big_u0, tuple0);

            let mut u_q1 = 0;
            let mut rho1 = 0;
            if x + 2 < w {
                let q1 = quad(x + 2, y);
                rho1 = q1.rho;
                let kappa1 = if q1.rho & q1.rho.wrapping_sub(1) != 0 {
                    max_e.max(1) as u32
                } else {
                    1
                };
                c_q1 |= ((q0.rho & 4) >> 1) | ((q0.rho & 8) >> 2);
                let big_u1 = q1.emax.max(kappa1);
                u_q1 = big_u1 - kappa1;
                let eps1 = q1.eps(u_q1);
                lep[li] = lep[li].max(q1.e[1]);
                li += 1;
                max_e = lep[li].max(lep[li + 1]) as i32 - 1;
                lep[li] = q1.e[3];
                lcx[li - 1] |= (q1.rho & 2) >> 1;
                c_q0 = lcx[li] + (lcx[li + 1] << 2);
                lcx[li] = (q1.rho & 8) >> 3;
                let tuple1 = tables.t1[((c_q1 << 8) | (q1.rho << 4) | eps1) as usize];
                vlc.encode(tuple1 as u32 >> 8, (tuple1 as u32 >> 4) & 7);
                if c_q1 == 0 {
                    mel.encode(q1.rho != 0);
                }
                q1.emit_magsgn(&mut ms, big_u1, tuple1);
            }
            let (a, b) = (uvlc(u_q0), uvlc(u_q1));
            vlc.encode(a.0, a.1);
            vlc.encode(b.0, b.1);
            vlc.encode(a.2, a.3);
            vlc.encode(b.2, b.3);
            c_q0 |= ((rho1 & 4) >> 1) | ((rho1 & 8) >> 2);
        }
    }

    terminate_mel_vlc(&mut mel, &mut vlc);
    ms.terminate();
    let scup = mel.buf.len() + vlc.buf.len();
    if !(2..=4079).contains(&scup) {
        return None;
    }
    let mut data = ms.buf;
    data.extend_from_slice(&mel.buf);
    data.extend(vlc.buf.iter().rev());
    let n = data.len();
    data[n - 1] = (scup >> 4) as u8;
    data[n - 2] = (data[n - 2] & 0xF0) | (scup & 0xF) as u8;
    Some(EncodedBlock {
        data,
        passes: 1,
        bitplanes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        *seed >> 8
    }

    fn round_trip(w: usize, h: usize, coefs: &[i32]) {
        let enc = encode(w, h, coefs).expect("encodable");
        let mut out = vec![0; w * h];
        if enc.passes > 0 {
            // Single cleanup at bit-plane 0: missing MSBs = Kmax - 1.
            super::super::decode(w, h, &enc.data, &[], 1, 29, 1, false, &mut out)
                .unwrap_or_else(|e| panic!("{w}x{h}: {e}"));
        }
        let back: Vec<i32> = out
            .iter()
            .map(|&v| if v < 0 { -((-v) >> 1) } else { v >> 1 })
            .collect();
        assert_eq!(back, coefs, "{w}x{h}");
    }

    #[test]
    fn random_blocks_round_trip() {
        let mut seed = 7u32;
        let sizes = [
            (1, 1),
            (1, 7),
            (7, 1),
            (2, 2),
            (3, 5),
            (5, 3),
            (8, 8),
            (17, 9),
            (33, 31),
            (64, 64),
            (4, 64),
            (64, 4),
            (1024, 4),
        ];
        for &(w, h) in &sizes {
            for density in [5u32, 30, 60, 100] {
                for amp in [1u32, 3, 40, 3000, 1 << 20, 1 << 29] {
                    let coefs: Vec<i32> = (0..w * h)
                        .map(|_| {
                            if lcg(&mut seed) % 100 >= density {
                                return 0;
                            }
                            let m = (lcg(&mut seed) % amp.max(1)) as i32 + 1;
                            if lcg(&mut seed) & 1 == 1 { -m } else { m }
                        })
                        .collect();
                    round_trip(w, h, &coefs);
                }
            }
        }
    }

    #[test]
    fn extreme_values() {
        let max = (1i32 << 30) - 1;
        round_trip(4, 4, &[max; 16]);
        round_trip(4, 4, &[-max; 16]);
        let mut v = vec![0; 64 * 64];
        v[0] = 1;
        round_trip(64, 64, &v);
        v[64 * 64 - 1] = -max;
        round_trip(64, 64, &v);
        round_trip(3, 3, &[0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
}
