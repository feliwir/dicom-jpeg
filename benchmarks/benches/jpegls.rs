//! JPEG-LS (T.87): dicom-jpeg vs CharLS.

use charls::{CharLS, FrameInfo, InterleaveMode};
use criterion::{Criterion, criterion_group, criterion_main};
use dicom_jpeg::jpegls::{EncoderOptions, encode_with};
use dicom_jpeg::{Codec, Image, Samples};
use dicom_jpeg_benchmarks::{
    DecodeFn, EncodeFn, decode_group, encode_group, ours_decode, references, wg04,
};

/// CharLS; its Rust wrapper cannot reuse a decoder, so each call creates one.
fn charls_decode<'a>() -> DecodeFn<'a> {
    Box::new(|d| {
        CharLS::default()
            .decode(d)
            .map(|v| v.len())
            .map_err(|e| e.to_string())
    })
}

fn decode(c: &mut Criterion) {
    for (set, mode) in [("JLSL", "lossless"), ("JLSN", "near-lossless")] {
        decode_group(
            c,
            &format!("jpegls-decode/{set} {mode}"),
            &wg04(set),
            vec![
                ("dicom-jpeg", ours_decode(Codec::JpegLs)),
                ("CharLS", charls_decode()),
            ],
        );
    }
}

fn ours_encode<'a>(near: u16) -> EncodeFn<'a> {
    let mut o = EncoderOptions::default();
    o.near = near;
    Box::new(move |img| {
        encode_with(img, &o)
            .map(|e| e.data)
            .map_err(|e| e.to_string())
    })
}

/// CharLS with sample interleaving, as dicom-jpeg's default. Its Rust
/// wrapper copies the input once per call.
fn charls_encode<'a>(near: i32) -> EncodeFn<'a> {
    Box::new(move |img: &Image| {
        let info = &img.info;
        let bytes: Vec<u8> = match &img.samples {
            Samples::U8(s) => s.clone(),
            Samples::U16(s) => s.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        };
        let mut c = CharLS::default();
        if info.components > 1 {
            c.set_interleave_mode(InterleaveMode::Sample)
                .map_err(|e| e.to_string())?;
        }
        let frame = FrameInfo {
            width: info.width,
            height: info.height,
            bits_per_sample: info.bits_per_sample as i32,
            component_count: info.components as i32,
        };
        c.encode(frame, near, &bytes).map_err(|e| e.to_string())
    })
}

fn encode(c: &mut Criterion) {
    let refs = references();
    for (near, name) in [(0, "lossless"), (2, "near 2")] {
        encode_group(
            c,
            &format!("jpegls-encode/{name}"),
            &refs,
            vec![
                ("dicom-jpeg", ours_encode(near as u16)),
                ("CharLS", charls_encode(near)),
            ],
        );
    }
}

criterion_group! {
    name = benches;
    config = dicom_jpeg_benchmarks::criterion();
    targets = decode, encode
}
criterion_main!(benches);
