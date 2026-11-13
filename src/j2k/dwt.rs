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

/// One-dimensional 5/3 analysis (F.4.8.1).
#[cfg(feature = "encode")]
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

/// One-dimensional 9/7 analysis (F.4.8.2), inverse of [`synth97`].
#[cfg(feature = "encode")]
fn analyse97(x: &mut [f32], odd_start: bool) {
    let n = x.len() as isize;
    if n == 1 {
        if odd_start {
            x[0] *= 2.0;
        }
        return;
    }
    let even0 = if odd_start { 1 } else { 0 };
    let odd0 = 1 - even0;
    for (start, coef) in [(odd0, ALPHA), (even0, BETA), (odd0, GAMMA), (even0, DELTA)] {
        let mut k = start;
        while k < n {
            let s = x[mirror(k - 1, n)] + x[mirror(k + 1, n)];
            x[k as usize] += coef * s;
            k += 2;
        }
    }
    let mut k = even0;
    while k < n {
        x[k as usize] *= 1.0 / K;
        k += 2;
    }
    let mut k = odd0;
    while k < n {
        x[k as usize] *= K;
        k += 2;
    }
}

/// Values that can go through an analysis filter.
#[cfg(feature = "encode")]
pub(crate) trait Analyse: Sample {
    fn analyse(line: &mut [Self], odd_start: bool);
}

#[cfg(feature = "encode")]
impl Analyse for i32 {
    fn analyse(line: &mut [Self], odd_start: bool) {
        analyse53(line, odd_start)
    }
}

#[cfg(feature = "encode")]
impl Analyse for f32 {
    fn analyse(line: &mut [Self], odd_start: bool) {
        analyse97(line, odd_start)
    }
}

/// One 2D analysis level (F.4.2): filter columns (VER_SD) then rows
/// (HOR_SD) of `res`, and split into (LL, HL, LH, HH) by absolute parity.
#[cfg(feature = "encode")]
pub(crate) fn analyse_level<T: Analyse>(res: &Plane<T>) -> [Plane<T>; 4] {
    let (u0, v0) = (res.x0, res.y0);
    let (u1, v1) = (u0 + res.w as u32, v0 + res.h as u32);
    let mut data = res.data.clone();
    let (w, h) = (res.w, res.h);
    if w > 0 && h > 0 {
        let mut col = vec![T::default(); h];
        for x in 0..w {
            for (y, c) in col.iter_mut().enumerate() {
                *c = data[y * w + x];
            }
            T::analyse(&mut col, v0 % 2 == 1);
            for (y, &c) in col.iter().enumerate() {
                data[y * w + x] = c;
            }
        }
        for row in data.chunks_exact_mut(w) {
            T::analyse(row, u0 % 2 == 1);
        }
    }
    let lo = |a: u32| a.div_ceil(2);
    let hi = |a: u32| a / 2;
    let mut bands = [
        Plane::new(lo(u0), lo(v0), lo(u1), lo(v1)),
        Plane::new(hi(u0), lo(v0), hi(u1), lo(v1)),
        Plane::new(lo(u0), hi(v0), lo(u1), hi(v1)),
        Plane::new(hi(u0), hi(v0), hi(u1), hi(v1)),
    ];
    for v in v0..v1 {
        for u in u0..u1 {
            let b = ((u % 2) + 2 * (v % 2)) as usize;
            let p = &mut bands[b];
            let (bx, by) = ((u / 2 - p.x0) as usize, (v / 2 - p.y0) as usize);
            p.data[by * p.w + bx] = data[(v - v0) as usize * w + (u - u0) as usize];
        }
    }
    bands
}

#[cfg(all(test, feature = "encode"))]
mod tests {
    use super::*;

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

#[cfg(all(test, feature = "encode"))]
mod level_tests {
    use super::*;

    #[test]
    fn two_d_round_trip_with_odd_origins() {
        for (x0, y0, w, h) in [
            (0u32, 0u32, 7usize, 5usize),
            (3, 1, 6, 9),
            (1, 1, 1, 4),
            (5, 2, 2, 1),
        ] {
            let mut p: Plane<i32> = Plane::new(x0, y0, x0 + w as u32, y0 + h as u32);
            for (i, v) in p.data.iter_mut().enumerate() {
                *v = (i as i32 * 37) % 101 - 50;
            }
            let [ll, hl, lh, hh] = analyse_level(&p);
            let back = synth_level(&ll, &hl, &lh, &hh, (x0, y0, x0 + w as u32, y0 + h as u32));
            assert_eq!(back.data, p.data, "{x0},{y0} {w}x{h}");

            let pf = Plane {
                x0,
                y0,
                w,
                h,
                data: p.data.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            };
            let [ll, hl, lh, hh] = analyse_level(&pf);
            let back = synth_level(&ll, &hl, &lh, &hh, (x0, y0, x0 + w as u32, y0 + h as u32));
            for (a, b) in back.data.iter().zip(&pf.data) {
                assert!((a - b).abs() < 1e-3, "9/7 {a} vs {b}");
            }
        }
    }
}
