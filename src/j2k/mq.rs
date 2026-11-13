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

/// A context: probability state index and most probable symbol.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Context {
    pub(crate) state: u8,
    pub(crate) mps: u8,
}

/// Initial context states (Table D.7).
pub(crate) fn initial_contexts() -> [Context; NUM_CONTEXTS] {
    let mut c = [Context::default(); NUM_CONTEXTS];
    c[0].state = 4;
    c[CTX_RL].state = 3;
    c[CTX_UNI].state = 46;
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

    #[inline]
    fn renorm(&mut self) {
        loop {
            if self.ct == 0 {
                self.bytein();
            }
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.a & 0x8000 != 0 {
                break;
            }
        }
    }

    /// DECODE a binary decision in context `cx`.
    #[inline]
    pub(crate) fn decode(&mut self, cx: &mut Context) -> u32 {
        let (qe, nmps, nlps, switch) = STATES[cx.state as usize];
        self.a -= qe;
        let d;
        if (self.c >> 16) < qe {
            // LPS exchange
            if self.a < qe {
                d = cx.mps as u32;
                cx.state = nmps;
            } else {
                d = 1 - cx.mps as u32;
                if switch {
                    cx.mps = 1 - cx.mps;
                }
                cx.state = nlps;
            }
            self.a = qe;
            self.renorm();
        } else {
            self.c -= qe << 16;
            if self.a & 0x8000 == 0 {
                // MPS exchange
                if self.a < qe {
                    d = 1 - cx.mps as u32;
                    if switch {
                        cx.mps = 1 - cx.mps;
                    }
                    cx.state = nlps;
                } else {
                    d = cx.mps as u32;
                    cx.state = nmps;
                }
                self.renorm();
            } else {
                d = cx.mps as u32;
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
