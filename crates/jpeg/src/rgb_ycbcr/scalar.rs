//! Portable RGB -> YCbCr.

use super::*;

pub(super) fn rgb_to_ycbcr(precision: u8, a: &mut [u16], b: &mut [u16], c: &mut [u16]) {
    let offset = (1i32 << (precision - 1)) << SCALEBITS;
    for ((y, cb), cr) in a.iter_mut().zip(b.iter_mut()).zip(c.iter_mut()) {
        let (r, g, bl) = (*y as i32, *cb as i32, *cr as i32);
        *y = ((YR * r + YG * g + YB * bl + HALF) >> SCALEBITS) as u16;
        *cb = ((-CBR * r - CBG * g + CBB * bl + offset + HALF - 1) >> SCALEBITS) as u16;
        *cr = ((CRR * r - CRG * g - CRB * bl + offset + HALF - 1) >> SCALEBITS) as u16;
    }
}
