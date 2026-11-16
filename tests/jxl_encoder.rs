//! Lossless JPEG XL encoder: round trips through our decoder and
//! cross-decoding with jxl-oxide.

use dicom_jpeg::{
    ColorSpace, DecodeOptions, EncodeOptions, Image, ImageInfo, Samples, TransferSyntax,
};

fn synthetic(w: u32, h: u32, nc: u16, bits: u8, seed: u32) -> Image {
    let max = (1u32 << bits) - 1;
    let mut s = seed;
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            for c in 0..nc as u32 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let smooth = (x * 3 + y * 5 + c * 17) * (max / 255).max(1);
                let edge = if (x + y) % 23 < 4 { max / 2 } else { 0 };
                v.push(((smooth + edge + (s >> 28)) % (max + 1)) as u16);
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

fn values(img: &Image) -> Vec<u32> {
    match &img.samples {
        Samples::U8(v) => v.iter().map(|&x| x as u32).collect(),
        Samples::U16(v) => v.iter().map(|&x| x as u32).collect(),
    }
}

/// Decode with jxl-oxide and scale its [0, 1] output back to integers.
fn oxide_decode(data: &[u8], bits: u8) -> Vec<u32> {
    let image = jxl_oxide::JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .expect("jxl-oxide parses the codestream");
    let render = image.render_frame(0).expect("jxl-oxide renders");
    let fb = render.image_all_channels();
    let max = ((1u32 << bits) - 1) as f32;
    fb.buf().iter().map(|&v| (v * max).round() as u32).collect()
}

#[test]
fn lossless_sweep() {
    let opts = EncodeOptions::new();
    for &(w, h) in &[
        (1u32, 1u32),
        (7, 3),
        (64, 64),
        (255, 257),
        (300, 200),
        (513, 20),
    ] {
        for (nc, bits) in [
            (1u16, 1u8),
            (1, 8),
            (1, 12),
            (1, 16),
            (3, 8),
            (3, 10),
            (3, 16),
        ] {
            let img = synthetic(w, h, nc, bits, w * 31 + h);
            let enc = dicom_jpeg::jxl::encode(TransferSyntax::JpegXlLossless, &img, &opts)
                .unwrap_or_else(|e| panic!("{w}x{h} x{nc} {bits}b: {e}"));
            assert!(!enc.lossy);
            let dec = dicom_jpeg::jxl::decode(&enc.data, &DecodeOptions::default())
                .unwrap_or_else(|e| panic!("{w}x{h} x{nc} {bits}b: decode: {e}"));
            assert_eq!(
                (dec.info.width, dec.info.height, dec.info.components),
                (w, h, nc)
            );
            assert!(
                values(&dec) == values(&img),
                "{w}x{h} x{nc} {bits}b: round trip"
            );
            let oxide = oxide_decode(&enc.data, bits);
            assert!(
                oxide == values(&img),
                "{w}x{h} x{nc} {bits}b: jxl-oxide differs"
            );
        }
    }
}
