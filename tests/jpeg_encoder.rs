//! T.81 encoder: parameter sweeps on synthetic images, decoded both by this
//! crate and by the independent `jpeg-decoder` crate.

use dicom_jpeg::jpeg::{ChromaSubsampling, EncoderOptions, Process};
use dicom_jpeg::{
    Codec, ColorOutput, ColorSpace, DecodeOptions, Encoded, Image, ImageInfo, Result, Samples,
    decode,
};

/// Encode with `opts` using the given coding process.
fn encode_with(process: Process, image: &Image, opts: &EncoderOptions) -> Result<Encoded> {
    let mut o = opts.clone();
    o.process = process;
    dicom_jpeg::jpeg::encode_with(image, &o)
}

fn synthetic(w: u32, h: u32, nc: u16, bits: u8, seed: u32) -> Image {
    let max = (1u32 << bits) - 1;
    let mut s = seed;
    let mut v = Vec::with_capacity((w * h) as usize * nc as usize);
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc as u32 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                // Smooth gradient plus some noise and a hard edge.
                let base = (x * 7 + y * 3 + c * 50) % (max + 1);
                let edge = if x > w / 2 { max / 3 } else { 0 };
                let noise = (s >> 24) % 9;
                v.push(((base + edge + noise) % (max + 1)) as u16);
            }
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        components: nc,
        bits_per_sample: bits,
        signed: false,
        color_space: match nc {
            1 => ColorSpace::Gray,
            3 => ColorSpace::Rgb,
            _ => ColorSpace::Unknown,
        },
    };
    let samples = if bits <= 8 {
        Samples::U8(v.into_iter().map(|x| x as u8).collect())
    } else {
        Samples::U16(v)
    };
    Image::new(info, samples).unwrap()
}

fn values(img: &Image) -> Vec<i32> {
    match &img.samples {
        Samples::U8(v) => v.iter().map(|&x| x as i32).collect(),
        Samples::U16(v) => v.iter().map(|&x| x as i32).collect(),
    }
}

fn decode_rgb(data: &[u8]) -> Image {
    decode(
        Codec::Jpeg,
        data,
        &DecodeOptions::new().with_color_output(ColorOutput::Rgb),
    )
    .unwrap()
}

#[test]
fn lossless_is_exact_for_all_predictors_bits_and_sizes() {
    let sizes = [(1, 1), (2, 3), (7, 5), (17, 9), (64, 33)];
    for &(w, h) in &sizes {
        for bits in [2u8, 7, 8, 10, 12, 15, 16] {
            for nc in [1u16, 3] {
                let img = synthetic(w, h, nc, bits, w * 31 + h + bits as u32);
                for predictor in 1..=7 {
                    for restart in [0u16, 5] {
                        let mut opts = EncoderOptions::default();
                        opts.predictor = predictor;
                        opts.restart_interval = restart;
                        let enc = encode_with(Process::Lossless, &img, &opts).unwrap();
                        assert!(!enc.lossy);
                        let dec = decode_rgb(&enc.data);
                        assert_eq!(
                            values(&dec),
                            values(&img),
                            "{w}x{h} {bits} bit x{nc}, predictor {predictor}, restart {restart}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn lossless_point_transform_drops_low_bits() {
    let img = synthetic(20, 10, 1, 12, 3);
    let mut opts = EncoderOptions::default();
    opts.point_transform = 3;
    let enc = encode_with(Process::Lossless, &img, &opts).unwrap();
    assert!(enc.lossy);
    let dec = decode_rgb(&enc.data);
    let expected: Vec<i32> = values(&img).iter().map(|v| v & !7).collect();
    assert_eq!(values(&dec), expected);
}

#[test]
fn lossless_output_decodes_with_jpeg_decoder() {
    for (bits, nc) in [(8u8, 1u16), (8, 3), (12, 1), (16, 1)] {
        let img = synthetic(33, 17, nc, bits, 99);
        let enc = encode_with(Process::Lossless, &img, &EncoderOptions::default()).unwrap();
        let mut d = jpeg_decoder::Decoder::new(&enc.data[..]);
        let out = d
            .decode()
            .unwrap_or_else(|e| panic!("{bits} bit x{nc}: {e}"));
        let oracle: Vec<i32> = if bits <= 8 {
            out.iter().map(|&v| v as i32).collect()
        } else {
            // jpeg-decoder returns native-endian 16-bit samples
            out.as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_ne_bytes([c[0], c[1]]) as i32)
                .collect()
        };
        assert_eq!(oracle, values(&img), "{bits} bit x{nc}");
    }
}

fn psnr(a: &[i32], b: &[i32], bits: u8) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    let peak = ((1u32 << bits) - 1) as f64;
    10.0 * (peak * peak / mse.max(1e-12)).log10()
}

#[test]
fn lossy_quality_sweep_and_oracle() {
    let subsamplings = [
        ChromaSubsampling::S444,
        ChromaSubsampling::S422,
        ChromaSubsampling::S420,
    ];
    for &(w, h) in &[(1u32, 1u32), (9, 7), (16, 16), (63, 41)] {
        for nc in [1u16, 3] {
            let img = synthetic(w, h, nc, 8, w + h);
            for &sub in &subsamplings {
                for restart in [0u16, 3] {
                    let mut last_size = usize::MAX;
                    for quality in [100u8, 90, 75, 50, 10] {
                        let mut opts = EncoderOptions::default();
                        opts.quality = quality;
                        opts.subsampling = sub;
                        opts.restart_interval = restart;
                        let enc = encode_with(Process::Baseline, &img, &opts).unwrap();
                        let ours = values(&decode_rgb(&enc.data));
                        let oracle: Vec<i32> = jpeg_decoder::Decoder::new(&enc.data[..])
                            .decode()
                            .unwrap()
                            .into_iter()
                            .map(|v| v as i32)
                            .collect();
                        assert_eq!(ours.len(), oracle.len());
                        let max = ours
                            .iter()
                            .zip(&oracle)
                            .map(|(a, b)| (a - b).abs())
                            .max()
                            .unwrap();
                        assert!(
                            max <= 8,
                            "{w}x{h}x{nc} {sub:?} q{quality}: ours vs jpeg-decoder max {max}"
                        );
                        let p = psnr(&ours, &values(&img), 8);
                        if quality >= 90 && w >= 16 && sub == ChromaSubsampling::S444 {
                            assert!(p > 30.0, "{w}x{h}x{nc} q{quality}: PSNR {p}");
                        }
                        // Lower quality should not produce larger files (allow small slack).
                        assert!(
                            enc.data.len() <= last_size.saturating_add(64),
                            "size grew at q{quality}"
                        );
                        last_size = enc.data.len();
                    }
                }
            }
        }
    }
}

#[test]
fn twelve_bit_extended_round_trip() {
    for nc in [1u16, 3] {
        let img = synthetic(40, 24, nc, 12, 5);
        let mut opts = EncoderOptions::default();
        opts.quality = 95;
        let enc = encode_with(Process::Extended, &img, &opts).unwrap();
        let dec = decode_rgb(&enc.data);
        assert_eq!(dec.info.bits_per_sample, 12);
        let p = psnr(&values(&dec), &values(&img), 12);
        assert!(p > 40.0, "x{nc}: PSNR {p}");
    }
}

#[test]
fn invalid_parameters_are_rejected() {
    let img12 = synthetic(8, 8, 1, 12, 1);
    assert!(encode_with(Process::Baseline, &img12, &EncoderOptions::default()).is_err());
    let img16 = synthetic(8, 8, 1, 16, 1);
    assert!(encode_with(Process::Extended, &img16, &EncoderOptions::default()).is_err());
    let mut opts = EncoderOptions::default();
    opts.predictor = 0;
    assert!(encode_with(Process::Lossless, &img16, &opts).is_err());
}
