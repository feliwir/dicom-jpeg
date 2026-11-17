//! AVX2 interleaving. The work is byte shuffles, and `pshufb` only shuffles
//! within 128-bit lanes, so wider registers would need extra cross-lane
//! moves: this level uses the SSE4.1 code.

pub(super) use super::sse41::{
    deinterleave3_u8, deinterleave3_u16, interleave3_u8, interleave3_u16,
};
