//! Inverse discrete wavelet transforms (T.800 Annex F): reversible 5/3 and
//! irreversible 9/7 lifting with symmetric extension, for arbitrary
//! (odd / even) sample origins.

/// A rectangular array of coefficients with its origin in the coordinate
/// system of its resolution level or sub-band.
#[derive(Debug, Clone)]
pub(crate) struct Plane<T> {
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) data: Vec<T>,
}

impl<T: Copy + Default> Plane<T> {
    pub(crate) fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Self {
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        Plane {
            x0,
            y0,
            w,
            h,
            data: vec![T::default(); w * h],
        }
    }
}

#[inline]
fn mirror(k: isize, n: isize) -> usize {
    let mut k = k;
    if k < 0 {
        k = -k;
    }
    if k >= n {
        k = 2 * (n - 1) - k;
    }
    k.clamp(0, n - 1) as usize
}

/// One-dimensional 5/3 synthesis of `x` (interleaved, first sample at an
/// absolute coordinate with parity `odd_start`).
fn synth53(x: &mut [i32], odd_start: bool) {
    let n = x.len() as isize;
    if n == 1 {
        if odd_start {
            x[0] /= 2;
        }
        return;
    }
    let first_even = if odd_start { 1 } else { 0 };
    let mut k = first_even;
    while k < n {
        let a = x[mirror(k - 1, n)];
        let b = x[mirror(k + 1, n)];
        x[k as usize] -= (a + b + 2) >> 2;
        k += 2;
    }
    let mut k = 1 - first_even;
    while k < n {
        let a = x[mirror(k - 1, n)];
        let b = x[mirror(k + 1, n)];
        x[k as usize] += (a + b) >> 1;
        k += 2;
    }
}

const ALPHA: f32 = -1.586_134_3;
const BETA: f32 = -0.052_980_118;
const GAMMA: f32 = 0.882_911_1;
const DELTA: f32 = 0.443_506_87;
const K: f32 = 1.230_174_1;

/// One-dimensional 9/7 synthesis.
fn synth97(x: &mut [f32], odd_start: bool) {
    let n = x.len() as isize;
    if n == 1 {
        if odd_start {
            x[0] *= 0.5;
        }
        return;
    }
    let even0 = if odd_start { 1 } else { 0 };
    let odd0 = 1 - even0;
    let mut k = even0;
    while k < n {
        x[k as usize] *= K;
        k += 2;
    }
    let mut k = odd0;
    while k < n {
        x[k as usize] *= 1.0 / K;
        k += 2;
    }
    for (start, coef) in [(even0, DELTA), (odd0, GAMMA), (even0, BETA), (odd0, ALPHA)] {
        let mut k = start;
        while k < n {
            let s = x[mirror(k - 1, n)] + x[mirror(k + 1, n)];
            x[k as usize] -= coef * s;
            k += 2;
        }
    }
}

/// Values that can go through a synthesis filter.
pub(crate) trait Sample: Copy + Default {
    fn synth(line: &mut [Self], odd_start: bool);
}

impl Sample for i32 {
    fn synth(line: &mut [Self], odd_start: bool) {
        synth53(line, odd_start)
    }
}

impl Sample for f32 {
    fn synth(line: &mut [Self], odd_start: bool) {
        synth97(line, odd_start)
    }
}

/// One 2D synthesis level (F.3.2): interleave LL (`ll`, the previous
/// resolution) with HL, LH, HH into the resolution `[u0,u1) x [v0,v1)`,
/// then filter rows (HOR_SR) and columns (VER_SR).
pub(crate) fn synth_level<T: Sample>(
    ll: &Plane<T>,
    hl: &Plane<T>,
    lh: &Plane<T>,
    hh: &Plane<T>,
    (u0, v0, u1, v1): (u32, u32, u32, u32),
) -> Plane<T> {
    let mut out: Plane<T> = Plane::new(u0, v0, u1, v1);
    let (w, h) = (out.w, out.h);
    if w == 0 || h == 0 {
        return out;
    }
    let get = |p: &Plane<T>, bx: u32, by: u32| -> T {
        let (x, y) = (
            bx.wrapping_sub(p.x0) as usize,
            by.wrapping_sub(p.y0) as usize,
        );
        if x < p.w && y < p.h {
            p.data[y * p.w + x]
        } else {
            T::default()
        }
    };
    for v in v0..v1 {
        for u in u0..u1 {
            let val = match (u % 2, v % 2) {
                (0, 0) => get(ll, u / 2, v / 2),
                (1, 0) => get(hl, u / 2, v / 2),
                (0, 1) => get(lh, u / 2, v / 2),
                _ => get(hh, u / 2, v / 2),
            };
            out.data[(v - v0) as usize * w + (u - u0) as usize] = val;
        }
    }
    for row in out.data.chunks_exact_mut(w) {
        T::synth(row, u0 % 2 == 1);
    }
    let mut col = vec![T::default(); h];
    for x in 0..w {
        for (y, c) in col.iter_mut().enumerate() {
            *c = out.data[y * w + x];
        }
        T::synth(&mut col, v0 % 2 == 1);
        for (y, &c) in col.iter().enumerate() {
            out.data[y * w + x] = c;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward 5/3 (F.4.8.1) for testing the inverse.
    fn analyse53(x: &mut [i32], odd_start: bool) {
        let n = x.len() as isize;
        if n == 1 {
            if odd_start {
                x[0] *= 2;
            }
            return;
        }
        let first_even = if odd_start { 1 } else { 0 };
        let mut k = 1 - first_even;
        while k < n {
            let a = x[mirror(k - 1, n)];
            let b = x[mirror(k + 1, n)];
            x[k as usize] -= (a + b) >> 1;
            k += 2;
        }
        let mut k = first_even;
        while k < n {
            let a = x[mirror(k - 1, n)];
            let b = x[mirror(k + 1, n)];
            x[k as usize] += (a + b + 2) >> 2;
            k += 2;
        }
    }

    #[test]
    fn reversible_53_roundtrip_all_lengths_and_parities() {
        for n in 1..20 {
            for odd in [false, true] {
                let orig: Vec<i32> = (0..n).map(|i| (i * 37 % 23) - 11).collect();
                let mut x = orig.clone();
                analyse53(&mut x, odd);
                synth53(&mut x, odd);
                assert_eq!(x, orig, "n={n} odd={odd}");
            }
        }
    }
}
