//! ITU-T T.81 decoder conformance: ITU-T T.83 compliance streams,
//! libjpeg-turbo reference decodes, W3C samples (vs. the jpeg-decoder crate)
//! and robustness against truncated / corrupted input.

mod common;

use dicom_jpeg::{
    ColorOutput, ColorSpace, DecodeOptions, Error, Image, Samples, TransferSyntax, decode,
    read_info,
};

fn read(rel: &str) -> Vec<u8> {
    std::fs::read(common::testdata(rel)).unwrap()
}

fn decode_jpeg(data: &[u8], rgb: bool) -> Image {
    let opts = DecodeOptions::new().with_color_output(if rgb {
        ColorOutput::Rgb
    } else {
        ColorOutput::Native
    });
    decode(TransferSyntax::JpegExtended, data, &opts).expect("decode")
}

fn u8_samples(img: &Image) -> &[u8] {
    match &img.samples {
        Samples::U8(v) => v,
        Samples::U16(_) => panic!("expected 8-bit samples"),
    }
}

/// Parse a binary PNM (P5/P6, 8 bit).
fn read_pnm(data: &[u8]) -> (usize, usize, usize, Vec<u8>) {
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while data[i].is_ascii_whitespace() {
            i += 1;
        }
        if data[i] == b'#' {
            while data[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        while !data[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(std::str::from_utf8(&data[start..i]).unwrap().to_string());
    }
    let channels = if fields[0] == "P6" { 3 } else { 1 };
    let (w, h): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    assert_eq!(fields[3], "255");
    (w, h, channels, data[i + 1..].to_vec())
}

/// PPM as written by libjpeg-turbo's `example decompress`
/// (12-bit samples are written as native-endian 16-bit words).
fn example_ppm(img: &Image) -> Vec<u8> {
    let max = (1u32 << img.info.bits_per_sample) - 1;
    let mut out = format!("P6\n{} {}\n{max}\n", img.info.width, img.info.height).into_bytes();
    match &img.samples {
        Samples::U8(v) => out.extend_from_slice(v),
        Samples::U16(v) => v
            .iter()
            .for_each(|s| out.extend_from_slice(&s.to_le_bytes())),
    }
    out
}

/// Digest from libjpeg-turbo's CMakeLists.txt (commit 3ef3367),
/// `MD5_PPM_EXAMPLE_DECOMPRESS` of the 8-bit test build.
///
/// The 12-bit digest (`49aac011...` for monkey12.jpg) is not used: libjpeg-turbo
/// 3.x 12-bit output differs from IJG libjpeg 6b, which DCMTK uses and which
/// this decoder reproduces (see `monkey12_matches_dcmtk`).
#[test]
fn libjpeg_turbo_example_decompress_is_bit_exact() {
    let img = decode_jpeg(&read("conformance/t81/libjpeg-turbo/testorig.jpg"), true);
    assert_eq!(
        common::md5_hex(&example_ppm(&img)),
        "dea1d7bbc37e39adf628342c86096641"
    );
}

/// The source image of testorig.jpg is testorig.ppm: sanity-check the quality.
#[test]
fn libjpeg_turbo_testorig_close_to_source() {
    let img = decode_jpeg(&read("conformance/t81/libjpeg-turbo/testorig.jpg"), true);
    let (w, h, c, source) = read_pnm(&read("conformance/t81/libjpeg-turbo/testorig.ppm"));
    assert_eq!(
        (img.info.width as usize, img.info.height as usize, c),
        (w, h, 3)
    );
    let a: Vec<i32> = u8_samples(&img).iter().map(|&v| v as i32).collect();
    let b: Vec<i32> = source.iter().map(|&v| v as i32).collect();
    let p = common::psnr(&a, &b, 8);
    assert!(p > 30.0, "PSNR {p}");
}

#[test]
fn libjpeg_turbo_testimgint_decodes() {
    let img = decode_jpeg(&read("conformance/t81/libjpeg-turbo/testimgint.jpg"), true);
    assert_eq!((img.info.width, img.info.height), (227, 149));
}

#[test]
fn progressive_is_reported_unsupported() {
    let mut data = read("conformance/t81/libjpeg-turbo/testorig.jpg");
    let sof = data.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
    data[sof + 1] = 0xC2;
    let err = decode(TransferSyntax::JpegExtended, &data, &DecodeOptions::new()).unwrap_err();
    assert!(matches!(err, Error::Unsupported { .. }), "{err}");
}

/// All T.83 compliance streams for the DICOM processes decode with the
/// geometry announced in their frame header.
#[test]
fn t83_streams_decode() {
    let b1 = read("conformance/t81/t83/p1/B1.JPG");
    for rel in [
        "p1/A1.JPG",
        "p1/B2.JPG",
        "p2/A1.JPG",
        "p2/B2.JPG",
        "p2/C1.JPG",
        "p2/C2.JPG",
        "p4/A1.JPG",
        "p4/B2.JPG",
        "p4/C1.JPG",
        "p4/C2.JPG",
        "p4/E1.JPG",
        "p4/E2.JPG",
        "p14/O1.JPG",
        "p14/O2.JPG",
    ] {
        let mut data = read(&format!("conformance/t81/t83/{rel}"));
        if rel.ends_with("B2.JPG") {
            // B2 is an abbreviated image stream; its tables are in B1.
            let mut joined = b1[..b1.len() - 2].to_vec();
            joined.extend_from_slice(&data[2..]);
            data = joined;
        }
        let info =
            read_info(TransferSyntax::JpegExtended, &data).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let img = decode_jpeg(&data, false);
        assert_eq!(img.info, info, "{rel}");
        eprintln!(
            "{rel}: {}x{} x{} @ {} bit",
            info.width, info.height, info.components, info.bits_per_sample
        );
    }
}

/// O2 (16-bit lossless, one scan per component): the fourth component
/// (sampling 1x4 of 3x4) equals the source image D16.SRC.
#[test]
fn t83_o2_matches_source() {
    let img = decode_jpeg(&read("conformance/t81/t83/p14/O2.JPG"), false);
    let src = read("conformance/t81/t83/p14/D16.SRC");
    let Samples::U16(samples) = &img.samples else {
        panic!("expected 16-bit samples")
    };
    let (w, h) = (img.info.width as usize, img.info.height as usize);
    assert_eq!((w, h, img.info.components), (255, 257, 4));
    // Component 3 is replicated 3x horizontally by the upsampler.
    let plane: Vec<u16> = (0..h)
        .flat_map(|y| (0..85).map(move |x| (y, x)))
        .map(|(y, x)| samples[(y * w + x * 3) * 4 + 3])
        .collect();
    let le: Vec<u16> = src
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let be: Vec<u16> = src
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();
    assert!(
        plane == le || plane == be,
        "component differs from D16.SRC (first values {:?} vs {:?})",
        &plane[..4],
        &be[..4]
    );
}

/// W3C samples: compare with the independent `jpeg-decoder` crate.
#[test]
fn w3_samples_match_oracle() {
    for name in ["jpeg400jfif", "jpeg420exif", "jpeg422jfif", "jpeg444"] {
        let data = read(&format!("conformance/t81/w3/{name}.jpg"));
        let img = decode_jpeg(&data, true);
        let mut oracle = jpeg_decoder::Decoder::new(&data[..]);
        let expected = oracle.decode().unwrap();
        let ours = u8_samples(&img);
        assert_eq!(ours.len(), expected.len(), "{name}: size");
        let max = ours
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .max()
            .unwrap();
        let mean = ours
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| (a as i32 - b as i32).abs() as f64)
            .sum::<f64>()
            / ours.len() as f64;
        eprintln!("{name}: max |diff| vs jpeg-decoder = {max}, mean = {mean:.4}");
        // Different (but conforming) IDCT / upsampling implementations.
        assert!(max <= 6 && mean < 0.5, "{name}: max {max}, mean {mean}");
        let expected_color = if img.info.components == 3 {
            ColorSpace::Rgb
        } else {
            ColorSpace::Gray
        };
        assert_eq!(img.info.color_space, expected_color);
    }
}

/// Truncated or corrupted streams must produce an image or an error, never a panic.
#[test]
fn corrupted_streams_do_not_panic() {
    let files = [
        "conformance/t81/libjpeg-turbo/testorig.jpg",
        "conformance/t81/libjpeg-turbo/monkey12.jpg",
        "conformance/t81/t83/p14/O1.JPG",
        "conformance/t81/t83/p4/E1.JPG",
        "conformance/t81/w3/jpeg420exif.jpg",
    ];
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for rel in files {
        let data = read(rel);
        let step = (data.len() / 97).max(1);
        for cut in (0..data.len()).step_by(step) {
            let _ = decode(
                TransferSyntax::JpegExtended,
                &data[..cut],
                &DecodeOptions::new(),
            );
        }
        for _ in 0..200 {
            let mut d = data.clone();
            for _ in 0..1 + rnd() % 8 {
                let i = (rnd() as usize) % d.len();
                d[i] = rnd() as u8;
            }
            let _ = decode(TransferSyntax::JpegExtended, &d, &DecodeOptions::new());
        }
    }
}

/// 12-bit reference: monkey12.jpg wrapped into DICOM with DCMTK `img2dcm` and
/// decoded with DCMTK `dcmdjpeg` (IJG 6b 12-bit), with and without
/// YCbCr -> RGB conversion.
#[test]
fn monkey12_matches_dcmtk() {
    let data = read("conformance/t81/libjpeg-turbo/monkey12.jpg");
    for (rgb, reference) in [
        (false, "dicom/dcmtk/monkey12_dcmd_ybr.dcm"),
        (true, "dicom/dcmtk/monkey12_dcmd_rgb.dcm"),
    ] {
        let img = decode_jpeg(&data, rgb);
        let Samples::U16(ours) = &img.samples else {
            panic!("12 bit")
        };
        let ref_obj = common::open(reference);
        let expected = common::samples(&common::native_frame(&ref_obj, 0), 16, 12, false);
        let diffs: Vec<(usize, i32, i32)> = ours
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (a, b))| **a as i32 != **b)
            .map(|(i, (a, b))| (i, *a as i32, *b))
            .collect();
        assert!(
            diffs.is_empty(),
            "rgb={rgb}: {} differences, first {:?}",
            diffs.len(),
            &diffs[..diffs.len().min(8)]
        );
    }
}
