//! Tier-1 encoding (T.800 Annex D): the inverse of `t1::decode` for the
//! default code-block style (one terminated MQ segment, no bypass).

use super::mq::{CTX_RL, CTX_UNI, Context, MqEncoder, NUM_CONTEXTS, initial_contexts};
use super::t1::{
    N_E, N_N, N_NE, N_NW, N_S, N_SE, N_SW, N_W, NEG_E, NEG_N, NEG_S, NEG_W, Orient, sc_context,
    zc_context,
};

const SIG: u8 = 1;
const VISIT: u8 = 4;
const REFINED: u8 = 8;

/// A coded code-block.
#[derive(Debug, Clone, Default)]
pub(crate) struct EncodedBlock {
    pub(crate) data: Vec<u8>,
    /// Number of coding passes (0 = all coefficients zero).
    pub(crate) passes: u32,
    /// Number of magnitude bit-planes actually coded.
    pub(crate) bitplanes: u32,
}

struct Block<'a> {
    w: usize,
    h: usize,
    stride: usize,
    flags: Vec<u8>,
    nb: Vec<u16>,
    mag: &'a [u32],
    neg: &'a [bool],
    orient: Orient,
    ctx: [Context; NUM_CONTEXTS],
    mq: MqEncoder,
}

impl Block<'_> {
    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        (y + 1) * self.stride + x + 1
    }

    #[inline]
    fn counts(&self, i: usize) -> (u32, u32, u32) {
        let w = self.nb[i];
        (
            (w & (N_W | N_E)).count_ones(),
            (w & (N_N | N_S)).count_ones(),
            (w & (N_NW | N_NE | N_SW | N_SE)).count_ones(),
        )
    }

    fn encode_sign(&mut self, i: usize, negative: bool) {
        let w = self.nb[i];
        let contrib = |sig: u16, neg: u16| -> i32 {
            if w & sig == 0 {
                0
            } else if w & neg != 0 {
                -1
            } else {
                1
            }
        };
        let h = (contrib(N_W, NEG_W) + contrib(N_E, NEG_E)).clamp(-1, 1);
        let v = (contrib(N_N, NEG_N) + contrib(N_S, NEG_S)).clamp(-1, 1);
        let (cx, xor) = sc_context(h, v);
        self.mq.encode(&mut self.ctx[cx], negative as u32 ^ xor);
    }

    fn set_significant(&mut self, i: usize, negative: bool) {
        self.flags[i] |= SIG;
        let s = self.stride;
        let n = |bit: u16| if negative { bit } else { 0 };
        let nb = &mut self.nb;
        nb[i + 1] |= N_W | n(NEG_W);
        nb[i - 1] |= N_E | n(NEG_E);
        nb[i + s] |= N_N | n(NEG_N);
        nb[i - s] |= N_S | n(NEG_S);
        nb[i + s + 1] |= N_NW;
        nb[i + s - 1] |= N_NE;
        nb[i - s + 1] |= N_SW;
        nb[i - s - 1] |= N_SE;
    }

    fn significance_pass(&mut self, p: u32) {
        for s in (0..self.h).step_by(4) {
            for x in 0..self.w {
                for y in s..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    if self.flags[i] & SIG != 0 || self.nb[i] & 0xFF == 0 {
                        continue;
                    }
                    let k = y * self.w + x;
                    let bit = (self.mag[k] >> p) & 1;
                    let (h, v, d) = self.counts(i);
                    self.mq
                        .encode(&mut self.ctx[zc_context(self.orient, h, v, d)], bit);
                    if bit == 1 {
                        self.encode_sign(i, self.neg[k]);
                        self.set_significant(i, self.neg[k]);
                    }
                    self.flags[i] |= VISIT;
                }
            }
        }
    }

    fn refinement_pass(&mut self, p: u32) {
        for s in (0..self.h).step_by(4) {
            for x in 0..self.w {
                for y in s..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    let f = self.flags[i];
                    if f & SIG == 0 || f & VISIT != 0 {
                        continue;
                    }
                    let cx = if f & REFINED != 0 {
                        16
                    } else if self.nb[i] & 0xFF == 0 {
                        14
                    } else {
                        15
                    };
                    let bit = (self.mag[y * self.w + x] >> p) & 1;
                    self.mq.encode(&mut self.ctx[cx], bit);
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
                if full
                    && (s..s + 4).all(|y| {
                        let i = self.idx(x, y);
                        self.flags[i] & (SIG | VISIT) == 0 && self.nb[i] & 0xFF == 0
                    })
                {
                    let first = (s..s + 4).position(|y| (self.mag[y * self.w + x] >> p) & 1 == 1);
                    match first {
                        None => {
                            self.mq.encode(&mut self.ctx[CTX_RL], 0);
                            continue;
                        }
                        Some(r) => {
                            self.mq.encode(&mut self.ctx[CTX_RL], 1);
                            self.mq.encode(&mut self.ctx[CTX_UNI], (r as u32 >> 1) & 1);
                            self.mq.encode(&mut self.ctx[CTX_UNI], r as u32 & 1);
                            let y = s + r;
                            let i = self.idx(x, y);
                            let neg = self.neg[y * self.w + x];
                            self.encode_sign(i, neg);
                            self.set_significant(i, neg);
                            start = y + 1;
                        }
                    }
                }
                for y in start..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    if self.flags[i] & (SIG | VISIT) == 0 {
                        let k = y * self.w + x;
                        let bit = (self.mag[k] >> p) & 1;
                        let (h, v, d) = self.counts(i);
                        self.mq
                            .encode(&mut self.ctx[zc_context(self.orient, h, v, d)], bit);
                        if bit == 1 {
                            self.encode_sign(i, self.neg[k]);
                            self.set_significant(i, self.neg[k]);
                        }
                    }
                    self.flags[i] &= !VISIT;
                }
                for y in s..start {
                    let i = self.idx(x, y);
                    self.flags[i] &= !VISIT;
                }
            }
        }
    }
}

/// Encode quantized coefficients (`w * h`, row-major) of one code-block.
pub(crate) fn encode(w: usize, h: usize, orient: Orient, coefs: &[i32]) -> EncodedBlock {
    let mag: Vec<u32> = coefs.iter().map(|v| v.unsigned_abs()).collect();
    let neg: Vec<bool> = coefs.iter().map(|&v| v < 0).collect();
    let max = mag.iter().copied().max().unwrap_or(0);
    if max == 0 {
        return EncodedBlock::default();
    }
    let bitplanes = 32 - max.leading_zeros();
    let stride = w + 2;
    let mut b = Block {
        w,
        h,
        stride,
        flags: vec![0; stride * (h + 2)],
        nb: vec![0; stride * (h + 2)],
        mag: &mag,
        neg: &neg,
        orient,
        ctx: initial_contexts(),
        mq: MqEncoder::new(),
    };
    let passes = 3 * bitplanes - 2;
    for k in 0..passes {
        if k == 0 {
            b.cleanup_pass(bitplanes - 1);
            continue;
        }
        let plane = bitplanes - 1 - k.div_ceil(3);
        match (k - 1) % 3 {
            0 => b.significance_pass(plane),
            1 => b.refinement_pass(plane),
            _ => b.cleanup_pass(plane),
        }
    }
    EncodedBlock {
        data: b.mq.finish(),
        passes,
        bitplanes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j2k::t1::{self, Segment};

    #[test]
    fn round_trip_through_decoder() {
        let mut seed = 5u32;
        for &(w, h) in &[(1usize, 1usize), (3, 5), (8, 8), (17, 9), (64, 64), (32, 7)] {
            for orient in [Orient::LL, Orient::HL, Orient::LH, Orient::HH] {
                let coefs: Vec<i32> = (0..w * h)
                    .map(|_| {
                        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                        let r = (seed >> 16) % 100;
                        if r < 50 {
                            0
                        } else {
                            (r as i32 - 75) * ((seed >> 8) % 50) as i32
                        }
                    })
                    .collect();
                let enc = encode(w, h, orient, &coefs);
                let mut out = vec![0; w * h];
                let seg = Segment {
                    passes: enc.passes,
                    max_passes: u32::MAX,
                    data: enc.data.clone(),
                };
                t1::decode(w, h, orient, 0, enc.bitplanes, &[seg], &mut out).unwrap();
                // Decoder returns doubled mid-point values; >> 1 gives the integer.
                let back: Vec<i32> = out
                    .iter()
                    .map(|&v| if v < 0 { -((-v) >> 1) } else { v >> 1 })
                    .collect();
                assert_eq!(back, coefs, "{w}x{h} {orient:?}");
            }
        }
    }
}
