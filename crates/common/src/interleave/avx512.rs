//! AVX-512 interleaving: uses the SSE4.1 code (see `avx2.rs`).

pub(super) use super::sse41::{
    deinterleave3_u8, deinterleave3_u16, interleave3_u8, interleave3_u16,
};
