//! JPEG-LS (T.87) conformance: the ISO/IEC 14495-1 test set, CharLS
//! regression streams, and bit-exact re-encoding of the conformance streams.

mod common;

use dicom_jpeg::{Codec, DecodeOptions, Error, Image, Samples, decode};

fn read(rel: &str) -> Vec<u8> {
    std::fs::read(common::testdata(rel)).unwrap()
}

/// Binary PGM / PPM (8 or 16 bit) -> (width, height, components, samples).
fn read_pnm(data: &[u8]) -> (usize, usize, usize, Vec<i32>) {
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
        let s = i;
        while !data[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(std::str::from_utf8(&data[s..i]).unwrap().to_string());
    }
    let nc = if fields[0] == "P6" { 3 } else { 1 };
    let (w, h, max): (usize, usize, u32) = (
        fields[1].parse().unwrap(),
        fields[2].parse().unwrap(),
        fields[3].parse().unwrap(),
    );
    let px = &data[i + 1..];
    let samples = if max > 255 {
        px.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_be_bytes([c[0], c[1]]) as i32)
            .collect()
    } else {
        px.iter().map(|&b| b as i32).collect()
    };
    (w, h, nc, samples)
}

fn values(img: &Image) -> Vec<i32> {
    match &img.samples {
        Samples::U8(v) => v.iter().map(|&x| x as i32).collect(),
        Samples::U16(v) => v.iter().map(|&x| x as i32).collect(),
    }
}

fn decode_jls(data: &[u8]) -> Result<Image, Error> {
    decode(Codec::JpegLs, data, &DecodeOptions::new())
}

fn check(stream: &str, reference: &str, max_err: i32) {
    let img = decode_jls(&read(stream)).unwrap_or_else(|e| panic!("{stream}: {e}"));
    let (w, h, nc, expected) = read_pnm(&read(reference));
    assert_eq!(
        (
            img.info.width as usize,
            img.info.height as usize,
            img.info.components as usize
        ),
        (w, h, nc),
        "{stream}: geometry"
    );
    let got = values(&img);
    let max = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .max()
        .unwrap();
    assert!(max <= max_err, "{stream}: max |err| {max} > {max_err}");
}

const C: &str = "conformance/jpegls";

#[test]
fn iso_14495_lossless() {
    for s in ["t8c0e0", "t8c1e0", "t8c2e0"] {
        check(&format!("{C}/{s}.jls"), &format!("{C}/test8.ppm"), 0);
    }
    // Non-default parameters (T1=T2=T3=9, RESET=31), coded from test8bs2.
    check(&format!("{C}/t8nde0.jls"), &format!("{C}/test8bs2.pgm"), 0);
    check(&format!("{C}/t16e0.jls"), &format!("{C}/test16.pgm"), 0);
}

#[test]
fn iso_14495_near_lossless() {
    for s in ["t8c0e3", "t8c1e3", "t8c2e3"] {
        check(&format!("{C}/{s}.jls"), &format!("{C}/test8.ppm"), 3);
    }
    check(&format!("{C}/t8nde3.jls"), &format!("{C}/test8bs2.pgm"), 3);
    check(&format!("{C}/t16e3.jls"), &format!("{C}/test16.pgm"), 3);
    // t16e3.pgm is the reference decode of t16e3.jls.
    check(&format!("{C}/t16e3.jls"), &format!("{C}/t16e3.pgm"), 0);
}

#[test]
fn iso_14495_subsampled_is_unsupported() {
    for s in ["t8sse0", "t8sse3"] {
        let err = decode_jls(&read(&format!("{C}/{s}.jls"))).unwrap_err();
        assert!(matches!(err, Error::Unsupported { .. }), "{s}: {err}");
    }
}

#[test]
fn charls_streams() {
    let cases = [
        ("test8_ilv_none_rm_7.jls", "test8.ppm"),
        ("test8_ilv_line_rm_7.jls", "test8.ppm"),
        ("test8_ilv_sample_rm_7.jls", "test8.ppm"),
        ("test8_ilv_sample_rm_300.jls", "test8.ppm"),
        ("test16_rm_5.jls", "test16.pgm"),
        ("banny.jls", "banny.ppm"),
        ("banny-hp1.jls", "banny.ppm"),
        ("banny-hp2.jls", "banny.ppm"),
        ("banny-hp3.jls", "banny.ppm"),
        (
            "tulips-gray-8bit-512-512-hp-encoder.jls",
            "tulips-gray-8bit-512-512.pgm",
        ),
    ];
    for (stream, reference) in cases {
        let r = if reference.starts_with("test") {
            format!("{C}/{reference}")
        } else {
            format!("{C}/charls/{reference}")
        };
        check(&format!("{C}/charls/{stream}"), &r, 0);
    }
}

/// Fuzzed / malformed CharLS regression streams must not panic.
#[test]
fn charls_malformed_streams_do_not_panic() {
    for s in [
        "fuzzy_input_golomb_16.jls",
        "fuzzy-input-no-valid-bits-at-the-end.jls",
        "fuzzy-input-bad-run-mode-golomb-code.jls",
        "ff_in_entropy_data.jls",
        "no_start_byte_after_encoded_scan.jls",
        "land10-10bit-rgb-hp3-invalid.jls",
        "8bit-monochrome-2x2.jls",
    ] {
        let r = decode_jls(&read(&format!("{C}/charls/{s}")));
        eprintln!(
            "{s}: {:?}",
            r.as_ref().map(|i| i.info).map_err(|e| e.to_string())
        );
    }
}

#[test]
fn truncated_and_corrupted_streams_do_not_panic() {
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for s in ["t8c0e0", "t8c1e3", "t8c2e0", "t16e3", "t8nde3"] {
        let data = read(&format!("{C}/{s}.jls"));
        for cut in (0..data.len()).step_by((data.len() / 61).max(1)) {
            let _ = decode_jls(&data[..cut]);
        }
        for _ in 0..100 {
            let mut d = data.clone();
            for _ in 0..1 + rnd() % 6 {
                let i = (rnd() as usize) % d.len();
                d[i] = rnd() as u8;
            }
            let _ = decode_jls(&d);
        }
    }
}

/// The entropy-coded segment(s) of a JPEG-LS stream (SOS payload through the
/// following marker), for comparing encoders independent of header choices.
#[cfg(feature = "encode")]
fn scan_data(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut scans = Vec::new();
    let mut i = 2;
    while i + 4 <= stream.len() {
        if stream[i] != 0xFF {
            i += 1;
            continue;
        }
        let m = stream[i + 1];
        if m == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([stream[i + 2], stream[i + 3]]) as usize;
        if m != 0xDA {
            i += 2 + len;
            continue;
        }
        let start = i + 2 + len;
        let mut j = start;
        while !(stream[j] == 0xFF && stream[j + 1] & 0x80 != 0) {
            j += 1;
        }
        scans.push(stream[start..j].to_vec());
        i = j;
    }
    scans
}

/// JPEG-LS coding is deterministic: encoding the conformance source images
/// with the conformance parameters must reproduce the reference scan data.
#[cfg(feature = "encode")]
#[test]
fn encoder_reproduces_conformance_streams_bit_exactly() {
    use dicom_jpeg::jpegls::{EncoderOptions, encode_with};
    use dicom_jpeg::{ColorSpace, ImageInfo};
    let image = |pnm: &str| {
        let (w, h, nc, s) = read_pnm(&read(&format!("{C}/{pnm}")));
        let bits = if pnm == "test16.pgm" { 12 } else { 8 };
        let info = ImageInfo {
            width: w as u32,
            height: h as u32,
            components: nc as u16,
            bits_per_sample: bits,
            signed: false,
            color_space: if nc == 3 {
                ColorSpace::Rgb
            } else {
                ColorSpace::Gray
            },
        };
        let samples = if bits <= 8 {
            Samples::U8(s.iter().map(|&v| v as u8).collect())
        } else {
            Samples::U16(s.iter().map(|&v| v as u16).collect())
        };
        Image::new(info, samples).unwrap()
    };
    let test8 = image("test8.ppm");
    let test16 = image("test16.pgm");
    for (stream, img, ilv, near) in [
        ("t8c0e0", &test8, 0u8, 0u16),
        ("t8c1e0", &test8, 1, 0),
        ("t8c2e0", &test8, 2, 0),
        ("t8c0e3", &test8, 0, 3),
        ("t8c1e3", &test8, 1, 3),
        ("t8c2e3", &test8, 2, 3),
        ("t16e0", &test16, 0, 0),
        ("t16e3", &test16, 0, 3),
    ] {
        let mut opts = EncoderOptions::default();
        opts.interleave = ilv;
        opts.near = near;
        let ours = encode_with(img, &opts).unwrap();
        let reference = read(&format!("{C}/{stream}.jls"));
        let (a, b) = (scan_data(&ours.data), scan_data(&reference));
        assert_eq!(a.len(), b.len(), "{stream}: scan count");
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            if x != y {
                let first = x
                    .iter()
                    .zip(y)
                    .position(|(p, q)| p != q)
                    .unwrap_or(x.len().min(y.len()));
                panic!(
                    "{stream} scan {k}: differs at byte {first} (ours {} bytes, reference {} bytes)",
                    x.len(),
                    y.len()
                );
            }
        }
        // And it decodes back within NEAR.
        let back = decode_jls(&ours.data).unwrap();
        let max = values(&back)
            .iter()
            .zip(values(img))
            .map(|(a, b)| (a - b).abs())
            .max()
            .unwrap();
        assert!(max <= near as i32, "{stream}: round trip max |err| {max}");
    }
}

/// pydicom-data: the same RGB image in the three interleave modes decodes
/// identically, and the near-lossless version stays within NEAR.
#[test]
fn dicom_interleave_modes_agree() {
    let frames = |rel: &str| {
        let obj = common::open(rel);
        common::decode_frames(&obj).unwrap_or_else(|e| panic!("{rel}: {e}"))
    };
    let ilv0 = frames("dicom/pydicom/JLSL_RGB_ILV0.dcm");
    assert_eq!(ilv0, frames("dicom/pydicom/JLSL_RGB_ILV1.dcm"));
    assert_eq!(ilv0, frames("dicom/pydicom/JLSL_RGB_ILV2.dcm"));
    let near_obj = common::open("dicom/pydicom/JLSN_RGB_ILV0.dcm");
    let near_frames = common::decode_frames(&near_obj).unwrap();
    let data =
        dicom_jpeg::dicom_rs::frame_codestream(&near_obj, dicom_jpeg::Codec::JpegLs, 0).unwrap();
    let near = common::jls_near(&data) as i32;
    let max = ilv0[0]
        .iter()
        .zip(&near_frames[0])
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .max()
        .unwrap();
    assert!(near > 0 && max <= near, "max |err| {max}, NEAR {near}");
}
