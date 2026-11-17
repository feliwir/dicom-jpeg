//! Portable YCbCr -> RGB.

use super::*;

pub(super) fn ycbcr_to_rgb(
    y: &[u16],
    cb: &[u16],
    cr: &[u16],
    precision: u8,
    r: &mut [u16],
    g: &mut [u16],
    b: &mut [u16],
) {
    let max = (1i32 << precision) - 1;
    let center = 1i32 << (precision - 1);
    let src = y.iter().zip(cb).zip(cr);
    for (((&yy, &cb), &cr), ((r, g), b)) in src.zip(r.iter_mut().zip(g.iter_mut()).zip(b)) {
        let yy = yy as i32;
        let cbv = cb as i32 - center;
        let crv = cr as i32 - center;
        *r = (yy + ((FR * crv + ONE_HALF) >> SCALEBITS)).clamp(0, max) as u16;
        *g = (yy + ((-FGB * cbv + ONE_HALF - FGR * crv) >> SCALEBITS)).clamp(0, max) as u16;
        *b = (yy + ((FB * cbv + ONE_HALF) >> SCALEBITS)).clamp(0, max) as u16;
    }
}
