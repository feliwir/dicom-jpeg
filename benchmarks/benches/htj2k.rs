//! HTJ2K (Part 15) decoding: dicom-jpeg vs OpenJPEG (C), openjp2 and
//! hayro-jpeg2000. None of the Rust-wrapped references encodes HTJ2K (OpenJPH,
//! Kakadu and Grok have no Rust bindings), so the decode inputs are the WG04
//! reference images encoded by dicom-jpeg plus the pydicom HTJ2K files, and
//! the encoder is benchmarked on its own.

use criterion::{Criterion, criterion_group, criterion_main};
use dicom_jpeg::j2k::{EncoderOptions, encode_with};
use dicom_jpeg::{Codec, Image};
use dicom_jpeg_benchmarks::j2k::{hayro_decode, openjp2_decode, openjpeg_decode};
use dicom_jpeg_benchmarks::{
    DecodeFn, EncodeFn, Input, decode_group, dicom_frames, encode_group, ours_decode, pixels,
    references, root,
};

fn opts(reversible: bool) -> EncoderOptions {
    let mut o = EncoderOptions::default();
    o.high_throughput = true;
    o.reversible = reversible;
    o
}

fn impls<'a>() -> Vec<(&'static str, DecodeFn<'a>)> {
    vec![
        ("dicom-jpeg", ours_decode(Codec::HtJ2k)),
        ("OpenJPEG", openjpeg_decode()),
        ("openjp2", openjp2_decode()),
        ("hayro-jpeg2000", hayro_decode()),
    ]
}

fn encoded(refs: &[(String, Image)], reversible: bool) -> Vec<Input> {
    refs.iter()
        .filter_map(|(name, img)| {
            let data = encode_with(img, &opts(reversible)).ok()?.data;
            Some(Input {
                name: name.clone(),
                data,
                pixels: pixels(img),
            })
        })
        .collect()
}

fn decode(c: &mut Criterion) {
    let refs = references();
    decode_group(
        c,
        "htj2k-decode/REF lossless",
        &encoded(&refs, true),
        impls(),
    );
    decode_group(c, "htj2k-decode/REF lossy", &encoded(&refs, false), impls());
    let files: Vec<_> = ["HTJ2KLossless_08_RGB.dcm", "HTJ2K_08_RGB.dcm"]
        .iter()
        .map(|f| root().join("testdata/dicom/pydicom").join(f))
        .collect();
    let inputs: Vec<Input> = dicom_frames(&files).into_iter().map(|(_, i)| i).collect();
    decode_group(c, "htj2k-decode/pydicom RGB", &inputs, impls());
}

fn encode(c: &mut Criterion) {
    let refs = references();
    for (reversible, name) in [(true, "lossless"), (false, "lossy")] {
        let o = opts(reversible);
        let ours: EncodeFn = Box::new(move |img| {
            encode_with(img, &o)
                .map(|e| e.data)
                .map_err(|e| e.to_string())
        });
        encode_group(
            c,
            &format!("htj2k-encode/{name}"),
            &refs,
            vec![("dicom-jpeg", ours)],
        );
    }
}

criterion_group! {
    name = benches;
    config = dicom_jpeg_benchmarks::criterion();
    targets = decode, encode
}
criterion_main!(benches);
