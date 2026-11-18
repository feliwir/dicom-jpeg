//! Inverse discrete wavelet transforms (T.800 Annex F): reversible 5/3 and
//! irreversible 9/7 lifting with symmetric extension, for arbitrary
//! (odd / even) sample origins.
//!
//! The 2-D synthesis lifts each row in its de-interleaved form (low-pass and
//! high-pass halves), then the columns a whole row at a time, so both
//! passes are element-wise operations on contiguous slices: the kernels in
//! `scalar.rs` and the SIMD files. Every step is computed exactly as in the
//! 1-D reference ([`synth53`], [`synth97`]), rows before columns, so the
//! result is bit-identical to it.

use dicom_jpeg_common::dispatch;
use dicom_jpeg_common::simd::{self, Level};

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse41;

/// The lifting kernels, for the benchmarks.
#[cfg(feature = "kernels")]
pub(crate) mod kernels {
    use super::*;

    /// 5/3 synthesis update: `x -= (a + b + 2) >> 2`.
    pub fn update53(level: Level, cur: &mut [i32], a: &[i32], b: &[i32]) {
        dispatch!(level => update53(cur, a, b))
    }

    /// 5/3 synthesis predict: `x += (a + b) >> 1`.
    pub fn predict53(level: Level, cur: &mut [i32], a: &[i32], b: &[i32]) {
        dispatch!(level => predict53(cur, a, b))
    }

    /// 5/3 analysis predict: `x -= (a + b) >> 1`.
    pub fn apredict53(level: Level, cur: &mut [i32], a: &[i32], b: &[i32]) {
        dispatch!(level => apredict53(cur, a, b))
    }

    /// 5/3 analysis update: `x += (a + b + 2) >> 2`.
    pub fn aupdate53(level: Level, cur: &mut [i32], a: &[i32], b: &[i32]) {
        dispatch!(level => aupdate53(cur, a, b))
    }

    /// 9/7 lifting step: `x -= k * (a + b)`.
    pub fn lift97(level: Level, cur: &mut [f32], a: &[f32], b: &[f32], k: f32) {
        dispatch!(level => lift97(cur, a, b, k))
    }

    /// 9/7 scaling: `x *= k`.
    pub fn scale97(level: Level, cur: &mut [f32], k: f32) {
        dispatch!(level => scale97(cur, k))
    }
}

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

/// One step of a synthesis lifting program, on the low-pass (even
/// coordinate) or high-pass (odd coordinate) samples.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Step {
    /// Multiply by a constant.
    Scale { low: bool, k: f32 },
    /// Combine with the two neighbours of the other kind.
    Lift { low: bool, op: Op },
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Op {
    /// 5/3: x -= (a + b + 2) >> 2.
    Update53,
    /// 5/3: x += (a + b) >> 1.
    Predict53,
    /// 9/7: x -= k * (a + b) (analysis uses -k for x += k * (a + b), which
    /// rounds identically).
    Lift97(f32),
    /// 5/3 analysis: x -= (a + b) >> 1.
    #[cfg(feature = "encode")]
    APredict53,
    /// 5/3 analysis: x += (a + b + 2) >> 2.
    #[cfg(feature = "encode")]
    AUpdate53,
}

/// Values that can go through a synthesis filter.
pub(crate) trait Sample: Copy + Default {
    /// 1-D synthesis of an interleaved line (the reference).
    fn synth(line: &mut [Self], odd_start: bool);

    /// The lifting steps of a line of at least two samples.
    const PROGRAM: &'static [Step];

    /// A line of one sample at an odd coordinate.
    fn single_odd(x: &mut [Self]);

    /// `op` on `cur` with neighbours `a` and `b` (equal lengths).
    fn lift(level: Level, op: Op, cur: &mut [Self], a: &[Self], b: &[Self]);

    fn scale(level: Level, cur: &mut [Self], k: f32);
}

impl Sample for i32 {
    fn synth(line: &mut [Self], odd_start: bool) {
        synth53(line, odd_start)
    }

    const PROGRAM: &'static [Step] = &[
        Step::Lift {
            low: true,
            op: Op::Update53,
        },
        Step::Lift {
            low: false,
            op: Op::Predict53,
        },
    ];

    fn single_odd(x: &mut [Self]) {
        for v in x {
            *v /= 2;
        }
    }

    fn lift(level: Level, op: Op, cur: &mut [Self], a: &[Self], b: &[Self]) {
        match op {
            Op::Update53 => dispatch!(level => update53(cur, a, b)),
            Op::Predict53 => dispatch!(level => predict53(cur, a, b)),
            #[cfg(feature = "encode")]
            Op::APredict53 => dispatch!(level => apredict53(cur, a, b)),
            #[cfg(feature = "encode")]
            Op::AUpdate53 => dispatch!(level => aupdate53(cur, a, b)),
            Op::Lift97(_) => unreachable!("9/7 step on integers"),
        }
    }

    fn scale(_: Level, _: &mut [Self], _: f32) {
        unreachable!("the 5/3 program does not scale")
    }
}

impl Sample for f32 {
    fn synth(line: &mut [Self], odd_start: bool) {
        synth97(line, odd_start)
    }

    const PROGRAM: &'static [Step] = &[
        Step::Scale { low: true, k: K },
        Step::Scale {
            low: false,
            k: 1.0 / K,
        },
        Step::Lift {
            low: true,
            op: Op::Lift97(DELTA),
        },
        Step::Lift {
            low: false,
            op: Op::Lift97(GAMMA),
        },
        Step::Lift {
            low: true,
            op: Op::Lift97(BETA),
        },
        Step::Lift {
            low: false,
            op: Op::Lift97(ALPHA),
        },
    ];

    fn single_odd(x: &mut [Self]) {
        for v in x {
            *v *= 0.5;
        }
    }

    fn lift(level: Level, op: Op, cur: &mut [Self], a: &[Self], b: &[Self]) {
        match op {
            Op::Lift97(k) => dispatch!(level => lift97(cur, a, b, k)),
            _ => unreachable!("5/3 step on floats"),
        }
    }

    fn scale(level: Level, cur: &mut [Self], k: f32) {
        dispatch!(level => scale97(cur, k));
    }
}

/// Lift `cur` against `other` (the samples of the other kind of the same
/// line): sample `i` of `cur` has the neighbours `other[i + off - 1]` and
/// `other[i + off]`. Symmetric extension makes a missing neighbour equal to
/// the other one (the line has at least two samples).
fn lift_seq<T: Sample>(level: Level, op: Op, cur: &mut [T], other: &[T], off: usize) {
    let (n, m) = (cur.len() as isize, other.len() as isize);
    let off = off as isize;
    // Interior: both neighbours exist.
    let i0 = (1 - off).clamp(0, n);
    let i1 = (m - off).clamp(i0, n);
    if i1 > i0 {
        let (i0u, i1u) = (i0 as usize, i1 as usize);
        let a = &other[(i0 + off - 1) as usize..(i1 + off - 1) as usize];
        let b = &other[(i0 + off) as usize..(i1 + off) as usize];
        T::lift(level, op, &mut cur[i0u..i1u], a, b);
    }
    for i in (0..i0).chain(i1..n) {
        let (ia, ib) = (i + off - 1, i + off);
        let get = |k: isize| (0..m).contains(&k).then(|| other[k as usize]);
        let (a, b) = match (get(ia), get(ib)) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => (a, a),
            (None, Some(b)) => (b, b),
            (None, None) => unreachable!("a line of two samples has a neighbour"),
        };
        let i = i as usize;
        T::lift(Level::Scalar, op, &mut cur[i..i + 1], &[a], &[b]);
    }
}

/// A lifting program: its steps, and the operation on a line of one sample
/// at an odd coordinate.
struct Program<T> {
    steps: &'static [Step],
    single_odd: fn(&mut [T]),
}

impl<T: Sample> Program<T> {
    fn synthesis() -> Self {
        Program {
            steps: T::PROGRAM,
            single_odd: T::single_odd,
        }
    }
}

/// Run `prog` on a line given as its low-pass (`lo`, even coordinates)
/// and high-pass (`hi`) samples; `odd_start`: the line starts at an odd
/// coordinate.
fn lift_lh<T: Sample>(
    level: Level,
    prog: &Program<T>,
    lo: &mut [T],
    hi: &mut [T],
    odd_start: bool,
) {
    match lo.len() + hi.len() {
        0 => {}
        1 => {
            if odd_start {
                (prog.single_odd)(hi);
            }
        }
        _ => {
            // Low sample i neighbours high i - 1 + p and i + p; high
            // sample j neighbours low j - p and j + 1 - p.
            let p = odd_start as usize;
            for step in prog.steps {
                match *step {
                    Step::Scale { low: true, k } => T::scale(level, lo, k),
                    Step::Scale { low: false, k } => T::scale(level, hi, k),
                    Step::Lift { low: true, op } => lift_seq(level, op, lo, hi, p),
                    Step::Lift { low: false, op } => lift_seq(level, op, hi, lo, 1 - p),
                }
            }
        }
    }
}

/// The vertical pass: `prog` on every column of the `h` rows of width `w`
/// in `data`, a whole row per operation.
fn lift_columns<T: Sample>(
    level: Level,
    prog: &Program<T>,
    data: &mut [T],
    w: usize,
    h: usize,
    odd_start: bool,
) {
    let p = odd_start as usize;
    // Row k holds low-pass samples if its coordinate is even.
    let is_low = |k: usize| (k + p).is_multiple_of(2);
    if h == 1 {
        if odd_start {
            (prog.single_odd)(data);
        }
        return;
    }
    for step in prog.steps {
        match *step {
            Step::Scale { low, k } => {
                for (r, row) in data.chunks_exact_mut(w).enumerate() {
                    if is_low(r) == low {
                        T::scale(level, row, k);
                    }
                }
            }
            Step::Lift { low, op } => {
                for k in (0..h).filter(|&k| is_low(k) == low) {
                    // Symmetric extension: a missing neighbour row is the
                    // other one.
                    let up = if k > 0 { k - 1 } else { k + 1 };
                    let down = if k + 1 < h { k + 1 } else { k - 1 };
                    let (before, rest) = data.split_at_mut(k * w);
                    let (cur, after) = rest.split_at_mut(w);
                    let row = |r: usize| -> &[T] {
                        if r < k {
                            &before[r * w..(r + 1) * w]
                        } else {
                            &after[(r - k - 1) * w..(r - k) * w]
                        }
                    };
                    T::lift(level, op, cur, row(up), row(down));
                }
            }
        }
    }
}

/// Copy the samples `x0..x0 + dst.len()` of row `y` of band `p` (absolute
/// band coordinates) into `dst`; samples outside the band are zero.
fn band_row<T: Copy + Default>(p: &Plane<T>, y: u32, x0: u32, dst: &mut [T]) {
    dst.fill(T::default());
    let row = y.wrapping_sub(p.y0) as usize;
    if row >= p.h {
        return;
    }
    let src = &p.data[row * p.w..(row + 1) * p.w];
    for (k, d) in dst.iter_mut().enumerate() {
        let x = (x0 + k as u32).wrapping_sub(p.x0) as usize;
        if let Some(&v) = src.get(x) {
            *d = v;
        }
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
    let level = simd::level();
    let prog = Program::<T>::synthesis();
    let mut out: Plane<T> = Plane::new(u0, v0, u1, v1);
    let (w, h) = (out.w, out.h);
    if w == 0 || h == 0 {
        return out;
    }
    let odd_u = u0 % 2 == 1;
    // Low-pass samples sit at even coordinates.
    let (lo_x0, hi_x0) = (u0.div_ceil(2), u0 / 2);
    let mut lo = vec![T::default(); (u1.div_ceil(2) - lo_x0) as usize];
    let mut hi = vec![T::default(); (u1 / 2 - hi_x0) as usize];
    for (v, row) in (v0..v1).zip(out.data.chunks_exact_mut(w)) {
        let (lb, hb) = if v % 2 == 0 { (ll, hl) } else { (lh, hh) };
        band_row(lb, v / 2, lo_x0, &mut lo);
        band_row(hb, v / 2, hi_x0, &mut hi);
        lift_lh(level, &prog, &mut lo, &mut hi, odd_u);
        let (first, second) = if odd_u { (&hi, &lo) } else { (&lo, &hi) };
        let pairs = row.as_chunks_mut::<2>().0.iter_mut();
        for (pair, (&a, &b)) in pairs.zip(first.iter().zip(second.iter())) {
            pair[0] = a;
            pair[1] = b;
        }
        if w % 2 == 1 {
            row[w - 1] = first[w / 2];
        }
    }
    lift_columns(level, &prog, &mut out.data, w, h, v0 % 2 == 1);
    out
}

/// The previous implementation of [`synth_level`]: interleave, then the
/// 1-D reference filters along rows and columns.
#[cfg(test)]
fn synth_level_reference<T: Sample>(
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
    /// 1-D analysis of an interleaved line (the reference).
    fn analyse(line: &mut [Self], odd_start: bool);

    /// The analysis lifting steps of a line of at least two samples.
    const ANALYSIS: &'static [Step];

    /// Analysis of a line of one sample at an odd coordinate.
    fn single_odd_analysis(x: &mut [Self]);
}

#[cfg(feature = "encode")]
impl Analyse for i32 {
    fn analyse(line: &mut [Self], odd_start: bool) {
        analyse53(line, odd_start)
    }

    const ANALYSIS: &'static [Step] = &[
        Step::Lift {
            low: false,
            op: Op::APredict53,
        },
        Step::Lift {
            low: true,
            op: Op::AUpdate53,
        },
    ];

    fn single_odd_analysis(x: &mut [Self]) {
        for v in x {
            *v = v.wrapping_mul(2);
        }
    }
}

#[cfg(feature = "encode")]
impl Analyse for f32 {
    fn analyse(line: &mut [Self], odd_start: bool) {
        analyse97(line, odd_start)
    }

    const ANALYSIS: &'static [Step] = &[
        Step::Lift {
            low: false,
            op: Op::Lift97(-ALPHA),
        },
        Step::Lift {
            low: true,
            op: Op::Lift97(-BETA),
        },
        Step::Lift {
            low: false,
            op: Op::Lift97(-GAMMA),
        },
        Step::Lift {
            low: true,
            op: Op::Lift97(-DELTA),
        },
        Step::Scale {
            low: true,
            k: 1.0 / K,
        },
        Step::Scale { low: false, k: K },
    ];

    fn single_odd_analysis(x: &mut [Self]) {
        for v in x {
            *v *= 2.0;
        }
    }
}

/// One 2D analysis level (F.4.2): filter columns (VER_SD) then rows
/// (HOR_SD) of `res`, and split into (LL, HL, LH, HH) by absolute parity.
#[cfg(feature = "encode")]
pub(crate) fn analyse_level<T: Analyse>(res: &Plane<T>) -> [Plane<T>; 4] {
    let level = simd::level();
    let prog = Program {
        steps: T::ANALYSIS,
        single_odd: T::single_odd_analysis,
    };
    let (u0, v0) = (res.x0, res.y0);
    let (u1, v1) = (u0 + res.w as u32, v0 + res.h as u32);
    let lo = |a: u32| a.div_ceil(2);
    let hi = |a: u32| a / 2;
    let mut bands = [
        Plane::new(lo(u0), lo(v0), lo(u1), lo(v1)),
        Plane::new(hi(u0), lo(v0), hi(u1), lo(v1)),
        Plane::new(lo(u0), hi(v0), lo(u1), hi(v1)),
        Plane::new(hi(u0), hi(v0), hi(u1), hi(v1)),
    ];
    let (w, h) = (res.w, res.h);
    if w == 0 || h == 0 {
        return bands;
    }
    let mut data = res.data.clone();
    lift_columns(level, &prog, &mut data, w, h, v0 % 2 == 1);
    let odd_u = u0 % 2 == 1;
    let (nl, nh) = ((lo(u1) - lo(u0)) as usize, (hi(u1) - hi(u0)) as usize);
    let (mut l, mut hrow) = (vec![T::default(); nl], vec![T::default(); nh]);
    let [ll, hl, lh, hh] = &mut bands;
    for (v, row) in (v0..v1).zip(data.chunks_exact(w)) {
        // De-interleave: even coordinates are low-pass.
        let (first, second) = if odd_u {
            (&mut hrow, &mut l)
        } else {
            (&mut l, &mut hrow)
        };
        for (k, &x) in row.iter().enumerate() {
            if k % 2 == 0 {
                first[k / 2] = x;
            } else {
                second[k / 2] = x;
            }
        }
        lift_lh(level, &prog, &mut l, &mut hrow, odd_u);
        let (lb, hb) = if v % 2 == 0 {
            (&mut *ll, &mut *hl)
        } else {
            (&mut *lh, &mut *hh)
        };
        let y = (v / 2 - lb.y0) as usize;
        lb.data[y * lb.w..(y + 1) * lb.w].copy_from_slice(&l);
        let y = (v / 2 - hb.y0) as usize;
        hb.data[y * hb.w..(y + 1) * hb.w].copy_from_slice(&hrow);
    }
    bands
}

/// The previous implementation of [`analyse_level`] (1-D reference filters
/// per column and row).
#[cfg(all(test, feature = "encode"))]
fn analyse_level_reference<T: Analyse>(res: &Plane<T>) -> [Plane<T>; 4] {
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

#[cfg(test)]
mod synth_tests {
    use super::*;

    /// Bands of the right geometry for the resolution `[u0,u1) x [v0,v1)`.
    fn bands<T: Copy + Default>(
        (u0, v0, u1, v1): (u32, u32, u32, u32),
        mut value: impl FnMut() -> T,
    ) -> [Plane<T>; 4] {
        let lo = |a: u32| a.div_ceil(2);
        let hi = |a: u32| a / 2;
        let mut b = [
            Plane::new(lo(u0), lo(v0), lo(u1), lo(v1)),
            Plane::new(hi(u0), lo(v0), hi(u1), lo(v1)),
            Plane::new(lo(u0), hi(v0), lo(u1), hi(v1)),
            Plane::new(hi(u0), hi(v0), hi(u1), hi(v1)),
        ];
        for p in &mut b {
            for v in &mut p.data {
                *v = value();
            }
        }
        b
    }

    #[test]
    fn matches_the_reference_at_every_level() {
        let mut seed = 1u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut regions = Vec::new();
        for (u0, v0) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1), (5, 2), (6, 7)] {
            for (w, h) in [
                (1u32, 1u32),
                (1, 5),
                (5, 1),
                (2, 2),
                (3, 3),
                (17, 9),
                (40, 33),
                (64, 1),
                (1, 64),
            ] {
                regions.push((u0, v0, u0 + w, v0 + h));
            }
        }
        for r in regions {
            let [ll, hl, lh, hh] = bands(r, || (rnd() % 4001) as i32 - 2000);
            let want = synth_level_reference(&ll, &hl, &lh, &hh, r);
            let [fll, fhl, flh, fhh] = bands(r, || (rnd() % 100_000) as f32 / 37.0 - 1300.0);
            let fwant = synth_level_reference(&fll, &fhl, &flh, &fhh, r);
            for level in Level::supported() {
                simd::with_level(level, || {
                    let got = synth_level(&ll, &hl, &lh, &hh, r);
                    assert_eq!(got.data, want.data, "5/3 {level:?} {r:?}");
                    let got = synth_level(&fll, &fhl, &flh, &fhh, r);
                    let bits = |d: &[f32]| d.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&got.data), bits(&fwant.data), "9/7 {level:?} {r:?}");
                });
            }
        }
    }
}

#[cfg(all(test, feature = "encode"))]
mod analysis_tests {
    use super::*;

    #[test]
    fn matches_the_reference_at_every_level() {
        let mut seed = 7u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for (x0, y0) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1), (5, 2), (6, 7)] {
            for (w, h) in [
                (1usize, 1usize),
                (1, 5),
                (5, 1),
                (2, 2),
                (3, 3),
                (17, 9),
                (40, 33),
                (64, 1),
                (1, 64),
            ] {
                let mut p: Plane<i32> = Plane::new(x0, y0, x0 + w as u32, y0 + h as u32);
                for v in &mut p.data {
                    *v = (rnd() % 65536) as i32 - 32768;
                }
                let pf = Plane {
                    x0,
                    y0,
                    w,
                    h,
                    data: p
                        .data
                        .iter()
                        .map(|&v| v as f32 * 0.37)
                        .collect::<Vec<f32>>(),
                };
                let want = analyse_level_reference(&p);
                let fwant = analyse_level_reference(&pf);
                for level in Level::supported() {
                    simd::with_level(level, || {
                        let got = analyse_level(&p);
                        for (g, w) in got.iter().zip(&want) {
                            assert_eq!(g.data, w.data, "5/3 {level:?} {x0},{y0} {w}x{h}", w = p.w);
                        }
                        let got = analyse_level(&pf);
                        for (g, w) in got.iter().zip(&fwant) {
                            let bits =
                                |d: &[f32]| d.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                            assert_eq!(bits(&g.data), bits(&w.data), "9/7 {level:?} {x0},{y0}");
                        }
                    });
                }
            }
        }
    }
}
