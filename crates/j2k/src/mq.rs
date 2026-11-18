//! MQ arithmetic decoder (T.800 Annex C) and the raw (bypass) bit reader (D.6).

/// (Qe, NMPS, NLPS, SWITCH) for the 47 probability states (Table C.2).
pub(crate) const STATES: [(u32, u8, u8, bool); 47] = [
    (0x5601, 1, 1, true),
    (0x3401, 2, 6, false),
    (0x1801, 3, 9, false),
    (0x0AC1, 4, 12, false),
    (0x0521, 5, 29, false),
    (0x0221, 38, 33, false),
    (0x5601, 7, 6, true),
    (0x5401, 8, 14, false),
    (0x4801, 9, 14, false),
    (0x3801, 10, 14, false),
    (0x3001, 11, 17, false),
    (0x2401, 12, 18, false),
    (0x1C01, 13, 20, false),
    (0x1601, 29, 21, false),
    (0x5601, 15, 14, true),
    (0x5401, 16, 14, false),
    (0x5101, 17, 15, false),
    (0x4801, 18, 16, false),
    (0x3801, 19, 17, false),
    (0x3401, 20, 18, false),
    (0x3001, 21, 19, false),
    (0x2801, 22, 19, false),
    (0x2401, 23, 20, false),
    (0x2201, 24, 21, false),
    (0x1C01, 25, 22, false),
    (0x1801, 26, 23, false),
    (0x1601, 27, 24, false),
    (0x1401, 28, 25, false),
    (0x1201, 29, 26, false),
    (0x1101, 30, 27, false),
    (0x0AC1, 31, 28, false),
    (0x09C1, 32, 29, false),
    (0x08A1, 33, 30, false),
    (0x0521, 34, 31, false),
    (0x0441, 35, 32, false),
    (0x02A1, 36, 33, false),
    (0x0221, 37, 34, false),
    (0x0141, 38, 35, false),
    (0x0111, 39, 36, false),
    (0x0085, 40, 37, false),
    (0x0049, 41, 38, false),
    (0x0025, 42, 39, false),
    (0x0015, 43, 40, false),
    (0x0009, 44, 41, false),
    (0x0005, 45, 42, false),
    (0x0001, 45, 43, false),
    (0x5601, 46, 46, false),
];

/// Number of coding contexts used by EBCOT.
pub(crate) const NUM_CONTEXTS: usize = 19;
/// Run-length context.
pub(crate) const CTX_RL: usize = 17;
/// Uniform context.
pub(crate) const CTX_UNI: usize = 18;

/// A context: probability state and most probable symbol, as one index
/// `2 * state + mps` into [`TRANSITIONS`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Context {
    idx: u8,
}

impl Context {
    const fn new(state: u8, mps: u8) -> Self {
        Context {
            idx: 2 * state + mps,
        }
    }

    /// The most probable symbol.
    #[inline(always)]
    fn mps(self) -> u32 {
        (self.idx & 1) as u32
    }
}

/// Per context index (`2 * state + mps`): Qe, and the next index after an
/// MPS and after an LPS (with the MPS switched where Table C.2 says so).
const TRANSITIONS: [(u32, u8, u8); 94] = {
    let mut t = [(0, 0, 0); 94];
    let mut i = 0;
    while i < 94 {
        let (qe, nmps, nlps, switch) = STATES[i / 2];
        let mps = (i & 1) as u8;
        t[i] = (qe, 2 * nmps + mps, 2 * nlps + (mps ^ switch as u8));
        i += 1;
    }
    t
};

/// Initial context states (Table D.7).
pub(crate) fn initial_contexts() -> [Context; NUM_CONTEXTS] {
    let mut c = [Context::new(0, 0); NUM_CONTEXTS];
    c[0] = Context::new(4, 0);
    c[CTX_RL] = Context::new(3, 0);
    c[CTX_UNI] = Context::new(46, 0);
    c
}

/// Byte at `i`; past the end of the segment the decoder sees 0xFF 0xFF
/// (a marker), as if the segment were terminated.
#[inline]
fn byte(data: &[u8], i: usize) -> u32 {
    data.get(i).copied().unwrap_or(0xFF) as u32
}

#[derive(Debug)]
pub(crate) struct MqDecoder<'a> {
    data: &'a [u8],
    bp: usize,
    a: u32,
    c: u32,
    ct: u32,
}

impl<'a> MqDecoder<'a> {
    /// INITDEC.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut d = MqDecoder {
            data,
            bp: 0,
            a: 0x8000,
            c: byte(data, 0) << 16,
            ct: 0,
        };
        d.bytein();
        d.c <<= 7;
        d.ct = d.ct.saturating_sub(7);
        d.a = 0x8000;
        d
    }

    #[inline]
    fn bytein(&mut self) {
        if byte(self.data, self.bp) == 0xFF {
            if byte(self.data, self.bp + 1) > 0x8F {
                self.c += 0xFF00;
                self.ct = 8;
            } else {
                self.bp += 1;
                self.c += byte(self.data, self.bp) << 9;
                self.ct = 7;
            }
        } else {
            self.bp += 1;
            self.c += byte(self.data, self.bp) << 8;
            self.ct = 8;
        }
    }

    /// RENORMD: shift A left until its bit 15 is set (A > 0), reading a
    /// byte whenever the bit counter runs out, as the bit-by-bit loop of
    /// the standard does, but shifting as many bits at once as possible.
    #[inline]
    fn renorm(&mut self) {
        let mut n = self.a.leading_zeros() - 16;
        loop {
            if self.ct == 0 {
                self.bytein();
            }
            let s = n.min(self.ct);
            self.a <<= s;
            self.c <<= s;
            self.ct -= s;
            n -= s;
            if n == 0 {
                break;
            }
        }
    }

    /// DECODE a binary decision in context `cx` (C.3.2, Figures C.15 to
    /// C.17).
    #[inline]
    pub(crate) fn decode(&mut self, cx: &mut Context) -> u32 {
        let (qe, nmps, nlps) = TRANSITIONS[cx.idx as usize];
        let mps = cx.mps();
        self.a -= qe;
        let d;
        if (self.c >> 16) < qe {
            // LPS exchange
            if self.a < qe {
                d = mps;
                cx.idx = nmps;
            } else {
                d = 1 - mps;
                cx.idx = nlps;
            }
            self.a = qe;
            self.renorm();
        } else {
            self.c -= qe << 16;
            if self.a & 0x8000 == 0 {
                // MPS exchange
                if self.a < qe {
                    d = 1 - mps;
                    cx.idx = nlps;
                } else {
                    d = mps;
                    cx.idx = nmps;
                }
                self.renorm();
            } else {
                d = mps;
            }
        }
        d
    }
}

/// Raw bit reader for bypassed coding passes (D.6, with bit stuffing).
#[derive(Debug)]
pub(crate) struct RawDecoder<'a> {
    data: &'a [u8],
    bp: usize,
    c: u32,
    ct: u32,
}

impl<'a> RawDecoder<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        RawDecoder {
            data,
            bp: 0,
            c: 0,
            ct: 0,
        }
    }

    #[inline]
    pub(crate) fn bit(&mut self) -> u32 {
        if self.ct == 0 {
            if self.c == 0xFF {
                let b = byte(self.data, self.bp);
                if b > 0x8F {
                    self.c = 0xFF;
                    self.ct = 8;
                } else {
                    self.c = b;
                    self.bp += 1;
                    self.ct = 7;
                }
            } else {
                self.c = byte(self.data, self.bp);
                self.bp += 1;
                self.ct = 8;
            }
        }
        self.ct -= 1;
        (self.c >> self.ct) & 1
    }
}

/// MQ arithmetic encoder (C.2), producing one terminated segment.
#[cfg(feature = "encode")]
#[derive(Debug)]
pub(crate) struct MqEncoder {
    /// Output with a leading dummy byte (index 0) that absorbs the BP = -1 state.
    out: Vec<u8>,
    bp: usize,
    a: u32,
    c: u32,
    ct: u32,
}

#[cfg(feature = "encode")]
impl MqEncoder {
    pub(crate) fn new() -> Self {
        MqEncoder {
            out: vec![0],
            bp: 0,
            a: 0x8000,
            c: 0,
            ct: 12,
        }
    }

    #[inline]
    fn byteout(&mut self) {
        if self.out[self.bp] == 0xFF {
            self.bp += 1;
            self.put((self.c >> 20) as u8);
            self.c &= 0xF_FFFF;
            self.ct = 7;
        } else if self.c < 0x800_0000 {
            self.bp += 1;
            self.put((self.c >> 19) as u8);
            self.c &= 0x7_FFFF;
            self.ct = 8;
        } else {
            self.out[self.bp] = self.out[self.bp].wrapping_add(1);
            if self.out[self.bp] == 0xFF {
                self.c &= 0x7FF_FFFF;
                self.bp += 1;
                self.put((self.c >> 20) as u8);
                self.c &= 0xF_FFFF;
                self.ct = 7;
            } else {
                self.bp += 1;
                self.put((self.c >> 19) as u8);
                self.c &= 0x7_FFFF;
                self.ct = 8;
            }
        }
    }

    #[inline]
    fn put(&mut self, b: u8) {
        if self.bp == self.out.len() {
            self.out.push(b);
        } else {
            self.out[self.bp] = b;
        }
    }

    #[inline]
    fn renorm(&mut self) {
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                self.byteout();
            }
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }

    /// ENCODE decision `d` in context `cx`.
    #[inline]
    pub(crate) fn encode(&mut self, cx: &mut Context, d: u32) {
        let (qe, nmps, nlps) = TRANSITIONS[cx.idx as usize];
        if d == cx.mps() {
            self.a -= qe;
            if self.a & 0x8000 == 0 {
                if self.a < qe {
                    self.a = qe;
                } else {
                    self.c += qe;
                }
                cx.idx = nmps;
                self.renorm();
            } else {
                self.c += qe;
            }
        } else {
            self.a -= qe;
            if self.a < qe {
                self.c += qe;
            } else {
                self.a = qe;
            }
            cx.idx = nlps;
            self.renorm();
        }
    }

    /// FLUSH and return the segment bytes.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        // SETBITS
        let temp = self.c + self.a;
        self.c |= 0xFFFF;
        if self.c >= temp {
            self.c -= 0x8000;
        }
        self.c <<= self.ct;
        self.byteout();
        self.c <<= self.ct;
        self.byteout();
        // A segment must not end with 0xFF.
        let mut end = self.bp;
        if self.out[end] != 0xFF {
            end += 1;
        }
        let mut out = self.out;
        out.truncate(end.max(1));
        out.remove(0);
        out
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;

    /// The decoder as the flowcharts of T.800 C.3 draw it (state and MPS
    /// apart, bit-by-bit renormalisation): the reference for `decode`.
    struct Reference<'a> {
        d: MqDecoder<'a>,
    }

    impl Reference<'_> {
        fn renorm(&mut self) {
            let d = &mut self.d;
            loop {
                if d.ct == 0 {
                    d.bytein();
                }
                d.a <<= 1;
                d.c <<= 1;
                d.ct -= 1;
                if d.a & 0x8000 != 0 {
                    break;
                }
            }
        }

        fn decode(&mut self, state: &mut u8, mps: &mut u8) -> u32 {
            let (qe, nmps, nlps, switch) = STATES[*state as usize];
            self.d.a -= qe;
            let d;
            if (self.d.c >> 16) < qe {
                if self.d.a < qe {
                    d = *mps as u32;
                    *state = nmps;
                } else {
                    d = 1 - *mps as u32;
                    if switch {
                        *mps = 1 - *mps;
                    }
                    *state = nlps;
                }
                self.d.a = qe;
                self.renorm();
            } else {
                self.d.c -= qe << 16;
                if self.d.a & 0x8000 == 0 {
                    if self.d.a < qe {
                        d = 1 - *mps as u32;
                        if switch {
                            *mps = 1 - *mps;
                        }
                        *state = nlps;
                    } else {
                        d = *mps as u32;
                        *state = nmps;
                    }
                    self.renorm();
                } else {
                    d = *mps as u32;
                }
            }
            d
        }
    }

    #[test]
    fn decoder_matches_reference_on_any_data() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for len in [0usize, 1, 2, 5, 64, 1000] {
            // Random bytes (with 0xFF runs, as markers and stuffing).
            let data: Vec<u8> = (0..len)
                .map(|_| if rnd() % 7 == 0 { 0xFF } else { rnd() as u8 })
                .collect();
            let mut fast = MqDecoder::new(&data);
            let mut slow = Reference {
                d: MqDecoder::new(&data),
            };
            let mut ctx = initial_contexts();
            let mut states: Vec<(u8, u8)> = ctx.iter().map(|c| (c.idx / 2, c.idx & 1)).collect();
            for i in 0..20_000 {
                let k = (rnd() % NUM_CONTEXTS as u32) as usize;
                let (st, m) = &mut states[k];
                let want = slow.decode(st, m);
                assert_eq!(fast.decode(&mut ctx[k]), want, "length {len}, decision {i}");
                assert_eq!(ctx[k], Context::new(*st, *m), "length {len}, decision {i}");
                assert_eq!(
                    (fast.a, fast.c, fast.ct, fast.bp),
                    (slow.d.a, slow.d.c, slow.d.ct, slow.d.bp)
                );
            }
        }
    }
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;

    #[test]
    fn mq_round_trip() {
        let mut seed = 99u32;
        let decisions: Vec<(usize, u32)> = (0..20000)
            .map(|i| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let cx = (seed >> 16) as usize % NUM_CONTEXTS;
                // Skewed probabilities per context.
                let bit = ((seed >> 8) % 100 < (cx as u32 * 5 + 3)) as u32 ^ (i % 7 == 0) as u32;
                (cx, bit)
            })
            .collect();
        let mut enc = MqEncoder::new();
        let mut ctx = initial_contexts();
        for &(cx, d) in &decisions {
            enc.encode(&mut ctx[cx], d);
        }
        let data = enc.finish();
        let mut dec = MqDecoder::new(&data);
        let mut ctx = initial_contexts();
        for (i, &(cx, d)) in decisions.iter().enumerate() {
            assert_eq!(dec.decode(&mut ctx[cx]), d, "decision {i}");
        }
    }
}
