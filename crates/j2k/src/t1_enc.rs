//! Tier-1 encoding (T.800 Annex D): the inverse of `t1::decode` for the
//! default code-block style (one terminated MQ segment, no bypass).

use super::mq::{CTX_RL, CTX_UNI, Context, MqEncoder, NUM_CONTEXTS, initial_contexts};
use super::t1::{
    N_E, N_N, N_NE, N_NW, N_S, N_SE, N_SW, N_W, NEG_E, NEG_N, NEG_S, NEG_W, Orient, REFINED,
    SC_LUT, SIG, VISIT, ZC_LUT, sc_index,
};

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
    /// State words (flags and neighbourhood bits, as in `t1`) with a
    /// one-sample border.
    st: Vec<u32>,
    mag: &'a [u32],
    neg: &'a [bool],
    zc: &'static [u8; 256],
    ctx: [Context; NUM_CONTEXTS],
    mq: MqEncoder,
}

impl Block<'_> {
    #[inline(always)]
    fn idx(&self, x: usize, y: usize) -> usize {
        (y + 1) * self.stride + x + 1
    }

    #[inline(always)]
    fn encode_sign(&mut self, w: u32, negative: bool) {
        let e = SC_LUT[sc_index(w as u16)];
        let xor = (e >> 7) as u32;
        self.mq
            .encode(&mut self.ctx[(e & 0x7F) as usize], negative as u32 ^ xor);
    }

    #[inline(always)]
    fn set_significant(&mut self, i: usize, negative: bool) {
        self.st[i] |= SIG;
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

    /// Zero-code the bit of sample `k` (state `w` at `i`); sign-code and
    /// mark it if it becomes significant.
    #[inline(always)]
    fn zero_code(&mut self, i: usize, k: usize, w: u32, p: u32) {
        let bit = (self.mag[k] >> p) & 1;
        let cx = self.zc[(w & 0xFF) as usize] as usize;
        self.mq.encode(&mut self.ctx[cx], bit);
        if bit == 1 {
            let neg = self.neg[k];
            self.encode_sign(w, neg);
            self.set_significant(i, neg);
        }
    }

    fn significance_pass(&mut self, p: u32) {
        for s in (0..self.h).step_by(4) {
            for x in 0..self.w {
                for y in s..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    let w = self.st[i];
                    if w & SIG != 0 || w & 0xFF == 0 {
                        continue;
                    }
                    self.zero_code(i, y * self.w + x, w, p);
                    self.st[i] |= VISIT;
                }
            }
        }
    }

    fn refinement_pass(&mut self, p: u32) {
        for s in (0..self.h).step_by(4) {
            for x in 0..self.w {
                for y in s..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    let w = self.st[i];
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
                    let bit = (self.mag[y * self.w + x] >> p) & 1;
                    self.mq.encode(&mut self.ctx[cx], bit);
                    self.st[i] |= REFINED;
                }
            }
        }
    }

    fn cleanup_pass(&mut self, p: u32) {
        for s in (0..self.h).step_by(4) {
            let full = s + 4 <= self.h;
            for x in 0..self.w {
                let mut start = s;
                if full && (s..s + 4).all(|y| self.st[self.idx(x, y)] & (SIG | VISIT | 0xFF) == 0) {
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
                            self.encode_sign(self.st[i], neg);
                            self.set_significant(i, neg);
                            start = y + 1;
                        }
                    }
                }
                for y in start..(s + 4).min(self.h) {
                    let i = self.idx(x, y);
                    let w = self.st[i];
                    if w & (SIG | VISIT) == 0 {
                        self.zero_code(i, y * self.w + x, w, p);
                    }
                    self.st[i] &= !VISIT;
                }
                for y in s..start {
                    let i = self.idx(x, y);
                    self.st[i] &= !VISIT;
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
        st: vec![0; stride * (h + 2)],
        mag: &mag,
        neg: &neg,
        zc: &ZC_LUT[orient as usize],
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
    use crate::t1::{self, Segment};

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
                    cup_pass: None,
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
