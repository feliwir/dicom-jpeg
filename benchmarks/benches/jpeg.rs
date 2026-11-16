//! JPEG (T.81): dicom-jpeg vs libjpeg-turbo 3, zune-jpeg and jpeg-decoder.

use criterion::{Criterion, criterion_group, criterion_main};
use dicom_jpeg::Codec;
use dicom_jpeg::jpeg::{EncoderOptions, Process, encode_with};
use dicom_jpeg_benchmarks::turbojpeg::{Mode, TurboJpeg};
use dicom_jpeg_benchmarks::{
    DecodeFn, EncodeFn, Input, decode_group, encode_group, ours_decode, pixels, references, wg04,
};
use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_jpeg::JpegDecoder;

fn turbojpeg<'a>() -> DecodeFn<'a> {
    let mut tj = TurboJpeg::decompressor();
    Box::new(move |d| tj.decompress(d))
}

fn zune<'a>() -> DecodeFn<'a> {
    Box::new(|d| {
        let mut dec = JpegDecoder::new(ZCursor::new(d));
        dec.decode_headers().map_err(|e| format!("{e:?}"))?;
        if dec.input_colorspace() == Some(ColorSpace::Luma) {
            let opts = dec.options().jpeg_set_out_colorspace(ColorSpace::Luma);
            dec.set_options(opts);
        }
        dec.decode().map(|v| v.len()).map_err(|e| format!("{e:?}"))
    })
}

fn jpeg_decoder<'a>() -> DecodeFn<'a> {
    Box::new(|d| {
        jpeg_decoder::Decoder::new(d)
            .decode()
            .map(|v| v.len())
            .map_err(|e| e.to_string())
    })
}

fn decode(c: &mut Criterion) {
    let all = || -> Vec<(&str, DecodeFn)> {
        vec![
            ("dicom-jpeg", ours_decode(Codec::Jpeg)),
            ("libjpeg-turbo", turbojpeg()),
            ("zune-jpeg", zune()),
            ("jpeg-decoder", jpeg_decoder()),
        ]
    };
    decode_group(c, "jpeg-decode/JPLL lossless", &wg04("JPLL"), all());
    // All WG04 JPLY files are 12-bit.
    decode_group(c, "jpeg-decode/JPLY 12-bit", &wg04("JPLY"), all());
    // 8-bit DCT: the 8-bit reference images as libjpeg-turbo encodes them.
    let mut tj = TurboJpeg::compressor();
    let inputs: Vec<Input> = references()
        .into_iter()
        .filter(|(_, i)| i.info.bits_per_sample == 8)
        .filter_map(|(name, img)| {
            Some(Input {
                name,
                data: tj.compress(&img, Mode::Dct { quality: 90 }).ok()?,
                pixels: pixels(&img),
            })
        })
        .collect();
    decode_group(
        c,
        "jpeg-decode/REF 8-bit q90 (libjpeg-turbo)",
        &inputs,
        all(),
    );
}

fn ours_encode<'a>(process: Process) -> EncodeFn<'a> {
    let mut o = EncoderOptions::default();
    o.process = process;
    o.quality = 90;
    Box::new(move |img| {
        encode_with(img, &o)
            .map(|e| e.data)
            .map_err(|e| e.to_string())
    })
}

fn turbojpeg_encode<'a>(mode: Mode) -> EncodeFn<'a> {
    let mut tj = TurboJpeg::compressor();
    Box::new(move |img| tj.compress(img, mode))
}

fn encode(c: &mut Criterion) {
    let refs = references();
    let bits = |lo, hi| -> Vec<_> {
        refs.iter()
            .filter(|(_, i)| (lo..=hi).contains(&i.info.bits_per_sample))
            .cloned()
            .collect()
    };
    let q90 = Mode::Dct { quality: 90 };
    encode_group(
        c,
        "jpeg-encode/baseline q90 8-bit",
        &bits(8, 8),
        vec![
            ("dicom-jpeg", ours_encode(Process::Baseline)),
            ("libjpeg-turbo", turbojpeg_encode(q90)),
        ],
    );
    encode_group(
        c,
        "jpeg-encode/extended q90 12-bit",
        &bits(9, 12),
        vec![
            ("dicom-jpeg", ours_encode(Process::Extended)),
            ("libjpeg-turbo", turbojpeg_encode(q90)),
        ],
    );
    encode_group(
        c,
        "jpeg-encode/lossless SV1",
        &refs,
        vec![
            ("dicom-jpeg", ours_encode(Process::Lossless)),
            ("libjpeg-turbo", turbojpeg_encode(Mode::LosslessSv1)),
        ],
    );
}

criterion_group! {
    name = benches;
    config = dicom_jpeg_benchmarks::criterion();
    targets = decode, encode
}
criterion_main!(benches);
