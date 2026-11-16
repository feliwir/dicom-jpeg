#![no_main]

use dicom_jpeg::{DecodeOptions, Limits, Codec, decode};
use libfuzzer_sys::fuzz_target;

// Decoding arbitrary input must never panic, hang, or allocate without bound.
fuzz_target!(|data: &[u8]| {
    let opts = DecodeOptions::new().with_limits(Limits {
        max_samples: 1 << 24,
        max_alloc_bytes: 1 << 28,
    });
    let _ = decode(Codec::JpegXl, data, &opts);
});
