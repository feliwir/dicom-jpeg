//! JPEG-LS context modelling and coding procedures shared by the decoder and
//! encoder (ITU-T T.87 Annex A).

use super::CODEC;
use dicom_jpeg_common::{Error, Result};

/// Run-length order table J (A.7.1.2).
pub(crate) const J: [u32; 32] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13,
    14, 15,
];

/// Coding parameters of a scan (C.2.4.1 / A.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Params {
    pub(crate) maxval: i32,
    pub(crate) t1: i32,
    pub(crate) t2: i32,
    pub(crate) t3: i32,
    pub(crate) reset: i32,
    pub(crate) near: i32,
    pub(crate) range: i32,
    pub(crate) qbpp: u32,
    pub(crate) limit: u32,
}

fn ceil_log2(v: i32) -> u32 {
    let mut n = 0;
    while (1i64 << n) < v as i64 {
        n += 1;
    }
    n
}

/// `CLAMP(i, j, MAXVAL)` of C.2.4.1.1.
fn clamp_param(i: i32, j: i32, maxval: i32) -> i32 {
    if i > maxval || i < j { j } else { i }
}

/// Default thresholds (C.2.4.1.1).
pub(crate) fn default_thresholds(maxval: i32, near: i32) -> (i32, i32, i32) {
    const T1: i32 = 3;
    const T2: i32 = 7;
    const T3: i32 = 21;
    if maxval >= 128 {
        let factor = (maxval.min(4095) + 128) / 256;
        let t1 = clamp_param(factor * (T1 - 2) + 2 + 3 * near, near + 1, maxval);
        let t2 = clamp_param(factor * (T2 - 3) + 3 + 5 * near, t1, maxval);
        let t3 = clamp_param(factor * (T3 - 4) + 4 + 7 * near, t2, maxval);
        (t1, t2, t3)
    } else {
        let factor = 256 / (maxval + 1);
        let t1 = clamp_param((T1 / factor + 3 * near).max(2), near + 1, maxval);
        let t2 = clamp_param((T2 / factor + 5 * near).max(3), t1, maxval);
        let t3 = clamp_param((T3 / factor + 7 * near).max(4), t2, maxval);
        (t1, t2, t3)
    }
}

/// Preset parameters as given in an LSE marker (0 = default).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Preset {
    pub(crate) maxval: i32,
    pub(crate) t1: i32,
    pub(crate) t2: i32,
    pub(crate) t3: i32,
    pub(crate) reset: i32,
}

impl Params {
    pub(crate) fn new(precision: u8, near: i32, preset: Preset) -> Result<Self> {
        let maxval = if preset.maxval != 0 {
            preset.maxval
        } else {
            ((1u32 << precision) - 1) as i32
        };
        if !(1..=65535).contains(&maxval) {
            return Err(Error::invalid(CODEC, "invalid MAXVAL"));
        }
        if near < 0 || near > (maxval / 2).min(255) {
            return Err(Error::invalid(CODEC, format!("invalid NEAR {near}")));
        }
        let (d1, d2, d3) = default_thresholds(maxval, near);
        let t1 = if preset.t1 != 0 { preset.t1 } else { d1 };
        let t2 = if preset.t2 != 0 { preset.t2 } else { d2 };
        let t3 = if preset.t3 != 0 { preset.t3 } else { d3 };
        if t1 < near + 1 || t1 > maxval || t2 < t1 || t2 > maxval || t3 < t2 || t3 > maxval {
            return Err(Error::invalid(CODEC, "invalid preset thresholds"));
        }
        let reset = if preset.reset != 0 { preset.reset } else { 64 };
        if reset < 3 || reset > maxval.max(255) {
            return Err(Error::invalid(CODEC, "invalid RESET value"));
        }
        let range = (maxval + 2 * near) / (2 * near + 1) + 1;
        let qbpp = ceil_log2(range);
        let bpp = ceil_log2(maxval + 1).max(2);
        let limit = 2 * (bpp + bpp.max(8));
        Ok(Params {
            maxval,
            t1,
            t2,
            t3,
            reset,
            near,
            range,
            qbpp,
            limit,
        })
    }

    /// Local gradient quantization (A.3.3).
    #[inline]
    pub(crate) fn quantize_gradient(&self, d: i32) -> i32 {
        if d <= -self.t3 {
            -4
        } else if d <= -self.t2 {
            -3
        } else if d <= -self.t1 {
            -2
        } else if d < -self.near {
            -1
        } else if d <= self.near {
            0
        } else if d < self.t1 {
            1
        } else if d < self.t2 {
            2
        } else if d < self.t3 {
            3
        } else {
            4
        }
    }

    /// Prediction error quantization for near-lossless coding (A.4.4), with
    /// `near` given (a constant 0 in lossless code paths).
    #[cfg(feature = "encode")]
    #[inline(always)]
    pub(crate) fn quantize_error_near(&self, e: i32, near: i32) -> i32 {
        if near == 0 {
            e
        } else if e > 0 {
            (e + near) / (2 * near + 1)
        } else {
            -((near - e) / (2 * near + 1))
        }
    }

    /// Modulo reduction of the error (A.4.5).
    #[cfg(feature = "encode")]
    #[inline]
    pub(crate) fn modulo_range(&self, mut e: i32) -> i32 {
        if e < 0 {
            e += self.range;
        }
        if e >= (self.range + 1) / 2 {
            e -= self.range;
        }
        e
    }

    #[inline]
    pub(crate) fn clamp(&self, v: i32) -> i32 {
        v.clamp(0, self.maxval)
    }

    /// Reconstructed sample from prediction and (quantized) error, with
    /// `near` given (a constant 0 in lossless code paths).
    #[inline(always)]
    pub(crate) fn reconstruct_near(&self, px: i32, e: i32, near: i32) -> i32 {
        let step = 2 * near + 1;
        let mut v = px + e * step;
        if v < -near {
            v += self.range * step;
        } else if v > self.maxval + near {
            v -= self.range * step;
        }
        self.clamp(v)
    }
}

/// The quantized value of every local gradient `-MAXVAL..=MAXVAL` (the
/// range of differences of reconstructed samples), from
/// [`Params::quantize_gradient`].
#[derive(Debug, Clone)]
pub(crate) struct Quantizer {
    lut: Vec<i8>,
    offset: i32,
}

impl Quantizer {
    pub(crate) fn new(p: &Params) -> Self {
        Quantizer {
            lut: (-p.maxval..=p.maxval)
                .map(|d| p.quantize_gradient(d) as i8)
                .collect(),
            offset: p.maxval,
        }
    }

    #[inline(always)]
    pub(crate) fn q(&self, d: i32) -> i32 {
        self.lut[(d + self.offset) as usize] as i32
    }
}

/// The smallest `k` with `n << k >= a` (A.5.1), at most 24. Starts the
/// search at the bit-length bound, below which every `k` fails.
#[inline(always)]
fn golomb_k(n: i32, a: i32) -> Result<u32> {
    let bits = |v: i32| 32 - (v.max(0) as u32).leading_zeros();
    let mut k = bits(a).saturating_sub(bits(n) + 1);
    if k > 24 {
        return Err(Error::invalid(CODEC, "Golomb parameter out of range"));
    }
    while (n << k) < a {
        k += 1;
        if k > 24 {
            return Err(Error::invalid(CODEC, "Golomb parameter out of range"));
        }
    }
    Ok(k)
}

/// Context index (0..=364) and sign from the three quantized gradients (A.3.4).
#[inline]
pub(crate) fn context(q1: i32, q2: i32, q3: i32) -> (usize, i32) {
    let q = (q1 * 9 + q2) * 9 + q3;
    if q < 0 {
        ((-q) as usize, -1)
    } else {
        (q as usize, 1)
    }
}

/// Median edge detector (A.4.1).
#[inline]
pub(crate) fn predict(ra: i32, rb: i32, rc: i32) -> i32 {
    if rc >= ra.max(rb) {
        ra.min(rb)
    } else if rc <= ra.min(rb) {
        ra.max(rb)
    } else {
        ra + rb - rc
    }
}

/// Regular-mode context statistics.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RegularContext {
    pub(crate) a: i32,
    pub(crate) b: i32,
    pub(crate) c: i32,
    pub(crate) n: i32,
}

impl RegularContext {
    #[inline]
    pub(crate) fn golomb_k(&self) -> Result<u32> {
        golomb_k(self.n, self.a)
    }

    /// The special error mapping of A.5.2 applies (k = 0, NEAR = 0, 2B <= -N).
    #[inline]
    pub(crate) fn special_mapping(&self, k: u32, near: i32) -> bool {
        k == 0 && near == 0 && 2 * self.b <= -self.n
    }

    /// Context update and bias correction (A.6).
    #[inline]
    pub(crate) fn update(&mut self, e: i32, near: i32, reset: i32) {
        self.b += e * (2 * near + 1);
        self.a += e.abs();
        if self.n == reset {
            self.a >>= 1;
            self.b >>= 1;
            self.n >>= 1;
        }
        self.n += 1;
        if self.b <= -self.n {
            self.b += self.n;
            if self.c > -128 {
                self.c -= 1;
            }
            if self.b <= -self.n {
                self.b = -self.n + 1;
            }
        } else if self.b > 0 {
            self.b -= self.n;
            if self.c < 127 {
                self.c += 1;
            }
            if self.b > 0 {
                self.b = 0;
            }
        }
    }
}

/// Run-interruption context statistics (A.7.2).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RunContext {
    pub(crate) a: i32,
    pub(crate) n: i32,
    pub(crate) nn: i32,
    pub(crate) ritype: i32,
}

impl RunContext {
    #[inline]
    pub(crate) fn golomb_k(&self) -> Result<u32> {
        golomb_k(self.n, self.a + (self.n >> 1) * self.ritype)
    }

    #[cfg(feature = "encode")]
    #[inline]
    pub(crate) fn map(&self, e: i32, k: u32) -> bool {
        (k == 0 && e > 0 && 2 * self.nn < self.n)
            || (e < 0 && 2 * self.nn >= self.n)
            || (e < 0 && k != 0)
    }

    /// Error value from `EMErrval + RItype` (inverse of the mapping).
    #[inline]
    pub(crate) fn unmap(&self, temp: i32, k: u32) -> i32 {
        let map = temp & 1 == 1;
        let abs = (temp + map as i32) / 2;
        if (k != 0 || 2 * self.nn >= self.n) == map {
            -abs
        } else {
            abs
        }
    }

    #[inline]
    pub(crate) fn update(&mut self, e: i32, em: i32, reset: i32) {
        if e < 0 {
            self.nn += 1;
        }
        self.a += (em + 1 - self.ritype) >> 1;
        if self.n == reset {
            self.a >>= 1;
            self.n >>= 1;
            self.nn >>= 1;
        }
        self.n += 1;
    }
}

/// All adaptive state of a scan.
#[derive(Debug, Clone)]
pub(crate) struct State {
    pub(crate) regular: Vec<RegularContext>,
    pub(crate) run: [RunContext; 2],
    pub(crate) run_index: usize,
}

impl State {
    pub(crate) fn new(p: &Params) -> Self {
        let a = ((p.range + 32) / 64).max(2);
        State {
            regular: vec![
                RegularContext {
                    a,
                    b: 0,
                    c: 0,
                    n: 1
                };
                365
            ],
            run: [
                RunContext {
                    a,
                    n: 1,
                    nn: 0,
                    ritype: 0,
                },
                RunContext {
                    a,
                    n: 1,
                    nn: 0,
                    ritype: 1,
                },
            ],
            run_index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golomb_k_matches_the_loop() {
        let reference = |n: i32, a: i32| -> Option<u32> {
            let mut k = 0;
            while (n << k) < a {
                k += 1;
                if k > 24 {
                    return None;
                }
            }
            Some(k)
        };
        for n in 1..300 {
            for a in (-5..70_000)
                .step_by(7)
                .chain([1 << 20, (1 << 24) * 3, 1 << 30])
            {
                assert_eq!(golomb_k(n, a).ok(), reference(n, a), "n {n} a {a}");
            }
        }
    }

    #[test]
    fn quantizer_matches_the_thresholds() {
        for (precision, near) in [(8u8, 0), (8, 3), (12, 0), (16, 2), (2, 0)] {
            let p = Params::new(precision, near, Preset::default()).unwrap();
            let q = Quantizer::new(&p);
            for d in -p.maxval..=p.maxval {
                assert_eq!(q.q(d), p.quantize_gradient(d));
            }
        }
    }
}
