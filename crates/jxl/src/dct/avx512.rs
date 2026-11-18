//! AVX-512: the AVX2 inverse 2-D DCT. A VarDCT block row is often only 8
//! samples wide, so 16-lane vectors would leave half of the lanes idle.

pub(super) use super::avx2::idct_2d;
