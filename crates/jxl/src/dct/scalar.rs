//! The portable DCT-II / DCT-III: Lee's recursion over all columns of a
//! block at once (`w` lanes), so every step is a loop over whole rows; rows
//! are transformed as the columns of the transpose. The SIMD inverse 2-D
//! transforms repeat these steps per lane, in the same order.

/// Rows `i` and `j > i` of `x` (`w` wide), mutably.
#[inline]
fn two_rows(x: &mut [f32], w: usize, i: usize, j: usize) -> (&mut [f32], &mut [f32]) {
    let (lo, hi) = x.split_at_mut(j * w);
    (&mut lo[i * w..][..w], &mut hi[..w])
}

/// Column-wise y_i = sum_k x_k cos((2i + 1) k pi / 2n) of the n x w block
/// `x`, in place; `tmp` is scratch of the same size.
pub(super) fn dct3(mul: &[Vec<f32>], x: &mut [f32], tmp: &mut [f32], n: usize, w: usize) {
    match n {
        2 => dct3_n::<2>(mul, x, tmp, w),
        4 => dct3_n::<4>(mul, x, tmp, w),
        8 => dct3_n::<8>(mul, x, tmp, w),
        16 => dct3_n::<16>(mul, x, tmp, w),
        32 => dct3_n::<32>(mul, x, tmp, w),
        64 => dct3_n::<64>(mul, x, tmp, w),
        128 => dct3_n::<128>(mul, x, tmp, w),
        256 => dct3_n::<256>(mul, x, tmp, w),
        _ => {}
    }
}

/// Column-wise z_k = sum_i x_i cos((2i + 1) k pi / 2n) (the transpose of
/// `dct3`).
fn dct2(mul: &[Vec<f32>], x: &mut [f32], tmp: &mut [f32], n: usize, w: usize) {
    match n {
        2 => dct2_n::<2>(mul, x, tmp, w),
        4 => dct2_n::<4>(mul, x, tmp, w),
        8 => dct2_n::<8>(mul, x, tmp, w),
        16 => dct2_n::<16>(mul, x, tmp, w),
        32 => dct2_n::<32>(mul, x, tmp, w),
        64 => dct2_n::<64>(mul, x, tmp, w),
        128 => dct2_n::<128>(mul, x, tmp, w),
        256 => dct2_n::<256>(mul, x, tmp, w),
        _ => {}
    }
}

/// `dct3` for size `N`; the recursion halves go through `dct3`, which
/// dispatches to the specialisation of size `N / 2`.
#[inline(always)]
fn dct3_n<const N: usize>(mul: &[Vec<f32>], x: &mut [f32], tmp: &mut [f32], w: usize) {
    if N == 2 {
        let (r0, r1) = x[..2 * w].split_at_mut(w);
        for (a, b) in r0.iter_mut().zip(r1) {
            let (p, q) = (*a, *b * std::f32::consts::FRAC_1_SQRT_2);
            *a = p + q;
            *b = p - q;
        }
        return;
    }
    let h = N / 2;
    let (even, odd) = tmp[..N * w].split_at_mut(h * w);
    for k in 0..h {
        even[k * w..][..w].copy_from_slice(&x[2 * k * w..][..w]);
        let o = &mut odd[k * w..][..w];
        let cur = &x[(2 * k + 1) * w..][..w];
        if k == 0 {
            o.copy_from_slice(cur);
        } else {
            let prev = &x[(2 * k - 1) * w..][..w];
            for ((o, &c), &p) in o.iter_mut().zip(cur).zip(prev) {
                *o = c + p;
            }
        }
    }
    let (s0, s1) = x[..N * w].split_at_mut(h * w);
    dct3(mul, even, s0, h, w);
    dct3(mul, odd, s1, h, w);
    let m = &mul[N.trailing_zeros() as usize];
    for i in 0..h {
        let mi = m[i];
        let (e, o) = (&even[i * w..][..w], &odd[i * w..][..w]);
        let (xi, xj) = two_rows(x, w, i, N - 1 - i);
        for (((xi, xj), &e), &o) in xi.iter_mut().zip(xj).zip(e).zip(o) {
            let o = o * mi;
            *xi = e + o;
            *xj = e - o;
        }
    }
}

/// `dct2` for size `N`.
#[inline(always)]
fn dct2_n<const N: usize>(mul: &[Vec<f32>], x: &mut [f32], tmp: &mut [f32], w: usize) {
    if N == 2 {
        let (r0, r1) = x[..2 * w].split_at_mut(w);
        for (a, b) in r0.iter_mut().zip(r1) {
            let (p, q) = (*a, *b);
            *a = p + q;
            *b = (p - q) * std::f32::consts::FRAC_1_SQRT_2;
        }
        return;
    }
    let h = N / 2;
    let m = &mul[N.trailing_zeros() as usize];
    let (even, odd) = tmp[..N * w].split_at_mut(h * w);
    for i in 0..h {
        let mi = m[i];
        let (xi, xj) = two_rows(x, w, i, N - 1 - i);
        let (e, o) = (&mut even[i * w..][..w], &mut odd[i * w..][..w]);
        for (((e, o), &a), &b) in e.iter_mut().zip(o).zip(&*xi).zip(&*xj) {
            *e = a + b;
            *o = (a - b) * mi;
        }
    }
    let (s0, s1) = x[..N * w].split_at_mut(h * w);
    dct2(mul, even, s0, h, w);
    dct2(mul, odd, s1, h, w);
    for k in 0..h {
        x[2 * k * w..][..w].copy_from_slice(&even[k * w..][..w]);
        let dst = &mut x[(2 * k + 1) * w..][..w];
        if k + 1 < h {
            let (cur, next) = (&odd[k * w..][..w], &odd[(k + 1) * w..][..w]);
            for ((d, &c), &nx) in dst.iter_mut().zip(cur).zip(next) {
                *d = c + nx;
            }
        } else {
            dst.copy_from_slice(&odd[k * w..][..w]);
        }
    }
}

/// The JPEG XL transform along the columns of the n x w block `x`.
pub(super) fn transform_cols(
    mul: &[Vec<f32>],
    x: &mut [f32],
    tmp: &mut [f32],
    n: usize,
    w: usize,
    inverse: bool,
) {
    if n <= 1 {
        return;
    }
    let sqrt2 = std::f32::consts::SQRT_2;
    let x = &mut x[..n * w];
    if inverse {
        // x_i = c_0 + sum_k sqrt(2) c_k cos(...)
        for c in &mut x[w..] {
            *c *= sqrt2;
        }
        dct3(mul, x, tmp, n, w);
    } else {
        dct2(mul, x, tmp, n, w);
        let inv = 1.0 / n as f32;
        for c in &mut x[..w] {
            *c *= inv;
        }
        for c in &mut x[w..] {
            *c *= sqrt2 * inv;
        }
    }
}

/// `dst` (w x h) = transpose of `src` (h x w).
pub(super) fn transpose(src: &[f32], dst: &mut [f32], w: usize, h: usize) {
    for (y, row) in src[..w * h].chunks_exact(w).enumerate() {
        for (x, &v) in row.iter().enumerate() {
            dst[x * h + y] = v;
        }
    }
}

/// The 2-D transform of the h x w block `a` (rows first, as the columns of
/// the transpose, then the columns); `b` and `tmp` are scratch of the same
/// size.
pub(super) fn transform_2d(
    mul: &[Vec<f32>],
    a: &mut [f32],
    b: &mut [f32],
    tmp: &mut [f32],
    w: usize,
    h: usize,
    inverse: bool,
) {
    transpose(a, b, w, h);
    transform_cols(mul, b, tmp, w, h, inverse);
    transpose(b, a, h, w);
    transform_cols(mul, a, tmp, h, w, inverse);
}

/// The inverse 2-D transform (the SIMD kernels' signature).
pub(super) fn idct_2d(
    mul: &[Vec<f32>],
    a: &mut [f32],
    b: &mut [f32],
    tmp: &mut [f32],
    w: usize,
    h: usize,
) {
    transform_2d(mul, a, b, tmp, w, h, true);
}
