//! JPEG-LS encoder parameter sweeps on synthetic images.

use dicom_jpeg::jpegls::{EncoderOptions, encode_with};
use dicom_jpeg::{Codec, ColorSpace, DecodeOptions, Image, ImageInfo, Samples, decode};

fn synthetic(w: u32, h: u32, nc: u16, bits: u8, seed: u32) -> Image {
    let max = (1u32 << bits) - 1;
    let mut s = seed;
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc as u32 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                // Flat areas (run mode), gradients, noise and edges.
                let val = if (x / 8 + y / 8) % 3 == 0 {
                    max / 2
                } else {
                    (x * 5 + y * 11 + c * 40 + (s >> 26)) % (max + 1)
                };
                v.push(val as u16);
            }
        }
    }
    let info = ImageInfo {
        width: w,
        height: h,
        components: nc,
        bits_per_sample: bits,
        signed: false,
        color_space: if nc == 3 {
            ColorSpace::Rgb
        } else {
            ColorSpace::Gray
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

#[test]
fn sweep_bits_near_interleave_restart_sizes() {
    for &(w, h) in &[(1u32, 1u32), (2, 3), (13, 7), (40, 33)] {
        for bits in [2u8, 5, 8, 10, 12, 16] {
            for nc in [1u16, 3, 4] {
                let img = synthetic(w, h, nc, bits, w * 7 + h + bits as u32);
                let max_near = (((1u32 << bits) - 1) / 2).min(255) as u16;
                for near in [0u16, 1, 3, 10].into_iter().filter(|&n| n <= max_near) {
                    for interleave in 0..=2u8 {
                        for restart in [0u16, 2] {
                            let mut opts = EncoderOptions::default();
                            opts.near = near;
                            opts.interleave = interleave;
                            opts.restart_interval = restart;
                            let enc = encode_with(&img, &opts).unwrap();
                            assert_eq!(enc.lossy, near > 0);
                            let dec = decode(Codec::JpegLs, &enc.data, &DecodeOptions::new())
                                .unwrap_or_else(|e| panic!("{w}x{h} {bits}b x{nc} near {near} ilv {interleave} rst {restart}: {e}"));
                            let max = values(&dec)
                                .iter()
                                .zip(values(&img))
                                .map(|(a, b)| (a - b).abs())
                                .max()
                                .unwrap();
                            assert!(
                                max <= near as i32,
                                "{w}x{h} {bits}b x{nc} near {near} ilv {interleave} rst {restart}: max |err| {max}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_near_is_rejected() {
    let img = synthetic(8, 8, 1, 2, 1);
    let mut opts = EncoderOptions::default();
    opts.near = 2; // MAXVAL 3 allows NEAR <= 1
    assert!(encode_with(&img, &opts).is_err());
}
