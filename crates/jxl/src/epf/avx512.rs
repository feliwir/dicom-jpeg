//! AVX-512: the AVX2 EPF. Its unit is a group of 8 pixels (one block
//! column, one sigma), which fills exactly one AVX2 vector.

pub(super) use super::avx2::{epf_row0, epf_row1, epf_row2};
