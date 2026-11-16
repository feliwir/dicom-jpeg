//! JPEG 2000 Part 1: dicom-jpeg vs OpenJPEG (C), openjp2 (Rust port of
//! OpenJPEG) and hayro-jpeg2000.

use std::collections::HashMap;
use std::rc::Rc;

use criterion::{Criterion, criterion_group, criterion_main};
use dicom_jpeg::j2k::{EncoderOptions, encode_with};
use dicom_jpeg::{Codec, Image};
use dicom_jpeg_benchmarks::j2k::{
    hayro_decode, openjp2_decode, openjp2_encode, openjpeg_decode, openjpeg_encode,
};
use dicom_jpeg_benchmarks::openjpeg::{self, Mode};
use dicom_jpeg_benchmarks::{
    EncodeFn, decode_group, encode_group, ours_decode, pixels, references, wg04,
};

fn decode(c: &mut Criterion) {
    for (set, mode) in [("J2KR", "lossless"), ("J2KI", "lossy")] {
        decode_group(
            c,
            &format!("j2k-decode/{set} {mode}"),
            &wg04(set),
            vec![
                ("dicom-jpeg", ours_decode(Codec::Jpeg2000)),
                ("OpenJPEG", openjpeg_decode()),
                ("openjp2", openjp2_decode()),
                ("hayro-jpeg2000", hayro_decode()),
            ],
        );
    }
}

fn ours<'a>(reversible: bool) -> EncodeFn<'a> {
    let mut o = EncoderOptions::default();
    o.reversible = reversible;
    Box::new(move |img| {
        encode_with(img, &o)
            .map(|e| e.data)
            .map_err(|e| e.to_string())
    })
}

fn encode(c: &mut Criterion) {
    let refs = references();
    encode_group(
        c,
        "j2k-encode/lossless",
        &refs,
        vec![
            ("dicom-jpeg", ours(true)),
            ("OpenJPEG", openjpeg_encode(Mode::Lossless)),
            ("openjp2", openjp2_encode(None)),
        ],
    );

    // Lossy: the references get, per image, the compression ratio that
    // dicom-jpeg reaches at its default quality, so all spend the same bytes.
    let mut ours_lossy = ours(false);
    let key = |img: &Image| img as *const Image as usize;
    let ratio: Rc<HashMap<usize, f32>> = Rc::new(
        refs.iter()
            .filter_map(|(_, img)| {
                // OpenJPEG's rate is relative to the precision, not the
                // stored sample size.
                let len = ours_lossy(img).ok()?.len();
                let i = &img.info;
                let bits = pixels(img) * i.components as u64 * i.bits_per_sample as u64;
                Some((key(img), bits as f32 / 8.0 / len as f32))
            })
            .collect(),
    );
    let r = ratio.clone();
    let opj: EncodeFn = Box::new(move |img| {
        openjpeg::encode(
            img,
            Mode::Lossy {
                ratio: r[&key(img)],
            },
        )
    });
    let r = ratio.clone();
    let ojp2: EncodeFn = Box::new(move |img| openjp2_encode(Some(r[&key(img)]))(img));
    encode_group(
        c,
        "j2k-encode/lossy (rate of dicom-jpeg q90)",
        &refs,
        vec![
            ("dicom-jpeg", ours_lossy),
            ("OpenJPEG", opj),
            ("openjp2", ojp2),
        ],
    );
}

criterion_group! {
    name = benches;
    config = dicom_jpeg_benchmarks::criterion();
    targets = decode, encode
}
criterion_main!(benches);
