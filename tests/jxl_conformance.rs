//! JPEG XL decoder conformance (libjxl/conformance test cases, ISO/IEC 18181-3).

use std::path::{Path, PathBuf};

use dicom_jpeg::{DecodeOptions, Samples};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/conformance/jxl/testcases")
}

/// Decode a PNG reference into (width, height, channels, bit depth, samples).
fn read_png(path: &Path) -> (u32, u32, usize, u8, Vec<u16>) {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    let channels = info.color_type.samples();
    let depth = info.bit_depth as u8;
    let data = if depth == 16 {
        buf[..info.buffer_size()]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_be_bytes(*c))
            .collect()
    } else {
        buf[..info.buffer_size()]
            .iter()
            .map(|&b| b as u16)
            .collect()
    };
    (info.width, info.height, channels, depth, data)
}

/// `peak_error` of the first frame in `test.json`.
fn peak_tolerance(name: &str) -> f64 {
    let json = std::fs::read_to_string(root().join(name).join("test.json")).unwrap();
    let i = json.find("\"peak_error\"").unwrap();
    let rest = &json[i + 13..];
    let rest = rest.trim_start_matches([':', ' ']);
    let end = rest.find([',', '\n', '}']).unwrap();
    rest[..end].trim().parse().unwrap()
}

/// Compare a decoded image with `ref.png` in normalised sample values;
/// returns the peak error relative to full scale.
fn compare(name: &str, img: &dicom_jpeg::Image) -> f64 {
    let (w, h, ch, depth, reference) = read_png(&root().join(name).join("ref.png"));
    assert_eq!((w, h), (img.info.width, img.info.height), "{name}: size");
    let got: Vec<u16> = match &img.samples {
        Samples::U8(v) => v.iter().map(|&x| x as u16).collect(),
        Samples::U16(v) => v.clone(),
    };
    let nc = img.info.components as usize;
    let our_max = ((1u32 << img.info.bits_per_sample) - 1) as f64;
    let ref_max = ((1u32 << depth) - 1) as f64;
    // Grey references may be stored as RGB.
    let ref_c = |i: usize, c: usize| reference[i * ch + if nc == 1 { 0 } else { c.min(ch - 1) }];
    let mut peak = 0f64;
    for i in 0..(w * h) as usize {
        for c in 0..nc {
            let a = got[i * nc + c] as f64 / our_max;
            let b = ref_c(i, c) as f64 / ref_max;
            peak = peak.max((a - b).abs());
        }
    }
    peak
}

/// Test cases the decoder must reproduce (the others must be reported as
/// unsupported, never decoded wrongly).
const REQUIRED: &[&str] = &[
    "alpha_nonpremultiplied",
    "alpha_triangles",
    "delta_palette",
    "lz77_flower",
];

#[test]
fn conformance() {
    let mut names: Vec<_> = std::fs::read_dir(root())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("input.jxl").exists())
        .map(|e| e.file_name().into_string().unwrap())
        .collect();
    names.sort();
    let mut passed = Vec::new();
    let mut failures = Vec::new();
    for name in &names {
        let data = std::fs::read(root().join(name).join("input.jxl")).unwrap();
        match dicom_jpeg::jxl::decode(&data, &DecodeOptions::default()) {
            Ok(img) => {
                if !root().join(name).join("ref.png").exists() {
                    continue;
                }
                let peak = compare(name, &img);
                let ref_depth = read_png(&root().join(name).join("ref.png")).3;
                // The reference is quantised to its own bit depth.
                let tol = peak_tolerance(name)
                    + 0.5 / ((1u32 << ref_depth) - 1) as f64
                    + 0.5 / ((1u32 << img.info.bits_per_sample) - 1) as f64;
                eprintln!("{name}: peak {peak:.6} (tolerance {tol:.6})");
                if peak <= tol {
                    passed.push(name.as_str());
                } else {
                    failures.push(format!("{name}: peak {peak} > {tol}"));
                }
            }
            Err(dicom_jpeg::Error::Unsupported { .. }) => {}
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    for r in REQUIRED {
        if !passed.contains(r) {
            failures.push(format!("{r}: required case did not pass"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore = "diagnostic listing"]
fn probe() {
    let mut names: Vec<_> = std::fs::read_dir(root())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("input.jxl").exists())
        .map(|e| e.file_name().into_string().unwrap())
        .collect();
    names.sort();
    for name in names {
        let data = std::fs::read(root().join(&name).join("input.jxl")).unwrap();
        let info = dicom_jpeg::jxl::read_info(&data);
        let res = dicom_jpeg::jxl::decode(&data, &DecodeOptions::default());
        let verdict = match &res {
            Ok(img) => {
                let png = root().join(&name).join("ref.png");
                if png.exists() {
                    let (w, h, ch, depth, reference) = read_png(&png);
                    let got: Vec<u16> = match &img.samples {
                        Samples::U8(v) => v.iter().map(|&x| x as u16).collect(),
                        Samples::U16(v) => v.clone(),
                    };
                    let nc = img.info.components as usize;
                    if (w, h) != (img.info.width, img.info.height) {
                        format!("size mismatch {w}x{h}")
                    } else {
                        let mut peak = 0i32;
                        for i in 0..(w * h) as usize {
                            for c in 0..nc.min(ch) {
                                let a = got[i * nc + c] as i32;
                                let b = reference[i * ch + c] as i32;
                                peak = peak.max((a - b).abs());
                            }
                        }
                        format!("decoded, peak {peak} vs ref (png {ch}ch {depth}bit)")
                    }
                } else {
                    "decoded (no ref.png)".to_string()
                }
            }
            Err(e) => format!("{e}"),
        };
        eprintln!(
            "{name:32} {:?} -> {verdict}",
            info.map(|i| (i.width, i.height, i.components, i.bits_per_sample))
        );
    }
}

#[test]
#[ignore = "diagnostic"]
fn locate_mismatch() {
    let name = std::env::var("JXL_CASE").unwrap_or("grayscale_public_university".into());
    let data = std::fs::read(root().join(&name).join("input.jxl")).unwrap();
    let img = dicom_jpeg::jxl::decode(&data, &DecodeOptions::default()).unwrap();
    let (w, h, ch, _, reference) = read_png(&root().join(&name).join("ref.png"));
    let got: Vec<u16> = match &img.samples {
        Samples::U8(v) => v.iter().map(|&x| x as u16).collect(),
        Samples::U16(v) => v.clone(),
    };
    let nc = img.info.components as usize;
    let mut bad = 0;
    let mut rows = std::collections::BTreeMap::new();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * w as usize + x;
            if got[i * nc] != reference[i * ch] {
                if bad < 5 {
                    eprintln!("({x},{y}): got {} ref {}", got[i * nc], reference[i * ch]);
                }
                bad += 1;
                *rows.entry(y / 256).or_insert(0) += 1;
            }
        }
    }
    eprintln!("{bad} mismatches; per 256-row band: {rows:?}");
}
