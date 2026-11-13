//! JPEG 2000 Part 1 decoder conformance (ITU-T T.803 | ISO/IEC 15444-4):
//! class 1 decoding of the profile 0 and profile 1 codestreams, compared per
//! component with the reference images using the standard's peak-error and
//! MSE limits (Tables C.6 / C.7; as transcribed in OpenJPEG's test suite).

mod common;

use dicom_jpeg::j2k::{DecoderOptions, decode_components};

const ROOT: &str = "conformance/j2k/iso15444-4";

/// Read a PGX file: "PG <ML|LM> [+|-]<depth> <width> <height>\n" + samples.
fn read_pgx(data: &[u8]) -> (u32, u32, Vec<i32>) {
    let nl = data.iter().position(|&b| b == b'\n').unwrap();
    let header = std::str::from_utf8(&data[..nl]).unwrap();
    let mut it = header.split_whitespace();
    assert_eq!(it.next(), Some("PG"));
    let endian = it.next().unwrap();
    let mut depth_s = it.next().unwrap().to_string();
    let signed = depth_s.starts_with('-');
    if depth_s.starts_with('+') || depth_s.starts_with('-') {
        depth_s.remove(0);
    }
    // Some writers put the sign separately: "PG ML + 12 ...".
    let (depth, w, h): (u32, u32, u32) = if depth_s.is_empty() {
        (
            it.next().unwrap().parse().unwrap(),
            it.next().unwrap().parse().unwrap(),
            it.next().unwrap().parse().unwrap(),
        )
    } else {
        (
            depth_s.parse().unwrap(),
            it.next().unwrap().parse().unwrap(),
            it.next().unwrap().parse().unwrap(),
        )
    };
    let signed = signed || header.contains(" - ");
    let px = &data[nl + 1..];
    let bytes = if depth <= 8 {
        1
    } else if depth <= 16 {
        2
    } else {
        4
    };
    let big = endian == "ML";
    let values = px
        .chunks_exact(bytes)
        .take((w * h) as usize)
        .map(|c| {
            let raw: u32 = match (bytes, big) {
                (1, _) => c[0] as u32,
                (2, true) => u16::from_be_bytes([c[0], c[1]]) as u32,
                (2, false) => u16::from_le_bytes([c[0], c[1]]) as u32,
                (_, true) => u32::from_be_bytes([c[0], c[1], c[2], c[3]]),
                (_, false) => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            };
            if signed {
                let shift = 32 - 8 * bytes as u32;
                ((raw << shift) as i32) >> shift
            } else {
                raw as i32
            }
        })
        .collect();
    (w, h, values)
}

struct Case {
    stream: &'static str,
    reference: &'static str,
    reduce: u8,
    peak: &'static [i64],
    mse: &'static [f64],
}

/// Table C.6 (class 1, profile 0) and C.7 (class 1, profile 1).
const CASES: &[Case] = &[
    Case {
        stream: "codestreams_profile0/p0_01.j2k",
        reference: "reference_class1_profile0/c1p0_01",
        reduce: 0,
        peak: &[0],
        mse: &[0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_02.j2k",
        reference: "reference_class1_profile0/c1p0_02",
        reduce: 0,
        peak: &[0],
        mse: &[0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_03.j2k",
        reference: "reference_class1_profile0/c1p0_03",
        reduce: 0,
        peak: &[0],
        mse: &[0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_04.j2k",
        reference: "reference_class1_profile0/c1p0_04",
        reduce: 0,
        peak: &[5, 4, 6],
        mse: &[0.776, 0.626, 1.070],
    },
    Case {
        stream: "codestreams_profile0/p0_05.j2k",
        reference: "reference_class1_profile0/c1p0_05",
        reduce: 0,
        peak: &[2, 2, 2, 0],
        mse: &[0.302, 0.307, 0.269, 0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_06.j2k",
        reference: "reference_class1_profile0/c1p0_06",
        reduce: 0,
        peak: &[635, 403, 378, 0],
        mse: &[11287.0, 6124.0, 3968.0, 0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_07.j2k",
        reference: "reference_class1_profile0/c1p0_07",
        reduce: 0,
        peak: &[0, 0, 0],
        mse: &[0.0, 0.0, 0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_08.j2k",
        reference: "reference_class1_profile0/c1p0_08",
        reduce: 1,
        peak: &[0, 0, 0],
        mse: &[0.0, 0.0, 0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_09.j2k",
        reference: "reference_class1_profile0/c1p0_09",
        reduce: 0,
        peak: &[0],
        mse: &[0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_10.j2k",
        reference: "reference_class1_profile0/c1p0_10",
        reduce: 0,
        peak: &[0, 0, 0],
        mse: &[0.0, 0.0, 0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_11.j2k",
        reference: "reference_class1_profile0/c1p0_11",
        reduce: 0,
        peak: &[0],
        mse: &[0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_12.j2k",
        reference: "reference_class1_profile0/c1p0_12",
        reduce: 0,
        peak: &[0],
        mse: &[0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_13.j2k",
        reference: "reference_class1_profile0/c1p0_13",
        reduce: 0,
        peak: &[0, 0, 0, 0],
        mse: &[0.0, 0.0, 0.0, 0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_14.j2k",
        reference: "reference_class1_profile0/c1p0_14",
        reduce: 0,
        peak: &[0, 0, 0],
        mse: &[0.0, 0.0, 0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_15.j2k",
        reference: "reference_class1_profile0/c1p0_15",
        reduce: 0,
        peak: &[0],
        mse: &[0.0],
    },
    Case {
        stream: "codestreams_profile0/p0_16.j2k",
        reference: "reference_class1_profile0/c1p0_16",
        reduce: 0,
        peak: &[0],
        mse: &[0.0],
    },
    Case {
        stream: "codestreams_profile1/p1_01.j2k",
        reference: "reference_class1_profile1/c1p1_01",
        reduce: 0,
        peak: &[0],
        mse: &[0.0],
    },
    Case {
        stream: "codestreams_profile1/p1_02.j2k",
        reference: "reference_class1_profile1/c1p1_02",
        reduce: 0,
        peak: &[5, 4, 6],
        mse: &[0.765, 0.616, 1.051],
    },
    Case {
        stream: "codestreams_profile1/p1_03.j2k",
        reference: "reference_class1_profile1/c1p1_03",
        reduce: 0,
        peak: &[2, 2, 1, 0],
        mse: &[0.3, 0.210, 0.200, 0.0],
    },
    Case {
        stream: "codestreams_profile1/p1_04.j2k",
        reference: "reference_class1_profile1/c1p1_04",
        reduce: 0,
        peak: &[624],
        mse: &[3080.0],
    },
    Case {
        stream: "codestreams_profile1/p1_05.j2k",
        reference: "reference_class1_profile1/c1p1_05",
        reduce: 0,
        peak: &[40, 40, 40],
        mse: &[8.458, 9.816, 10.154],
    },
    Case {
        stream: "codestreams_profile1/p1_06.j2k",
        reference: "reference_class1_profile1/c1p1_06",
        reduce: 0,
        peak: &[2, 2, 2],
        mse: &[0.6, 0.6, 0.6],
    },
    Case {
        stream: "codestreams_profile1/p1_07.j2k",
        reference: "reference_class1_profile1/c1p1_07",
        reduce: 0,
        peak: &[0, 0],
        mse: &[0.0, 0.0],
    },
];

fn run_case(case: &Case) -> Result<String, String> {
    let data = std::fs::read(common::testdata(&format!("{ROOT}/{}", case.stream))).unwrap();
    let mut opts = DecoderOptions::default();
    opts.reduce = case.reduce;
    let comps = decode_components(&data, &opts).map_err(|e| format!("decode failed: {e}"))?;
    let mut report = String::new();
    let mut ok = true;
    for (k, (&peak_limit, &mse_limit)) in case.peak.iter().zip(case.mse).enumerate() {
        let path = common::testdata(&format!("{ROOT}/{}-{k}.pgx", case.reference));
        let (w, h, reference) = read_pgx(&std::fs::read(path).unwrap());
        let c = comps
            .get(k)
            .ok_or_else(|| format!("component {k} missing"))?;
        if (c.width, c.height) != (w, h) {
            return Err(format!(
                "component {k}: {}x{} vs reference {w}x{h}",
                c.width, c.height
            ));
        }
        let peak = c
            .data
            .iter()
            .zip(&reference)
            .map(|(a, b)| (*a as i64 - *b as i64).abs())
            .max()
            .unwrap_or(0);
        let mse = c
            .data
            .iter()
            .zip(&reference)
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum::<f64>()
            / reference.len() as f64;
        let pass = peak <= peak_limit && mse <= mse_limit + 1e-9;
        ok &= pass;
        report.push_str(&format!(
            "  c{k}: peak {peak} (≤{peak_limit}), mse {mse:.3} (≤{mse_limit}) {}\n",
            if pass { "ok" } else { "FAIL" }
        ));
    }
    if ok { Ok(report) } else { Err(report) }
}

#[test]
fn class1_profile0_and_profile1() {
    let mut failures = Vec::new();
    for case in CASES {
        match run_case(case) {
            Ok(r) => eprintln!("{}: ok\n{r}", case.stream),
            Err(r) => {
                eprintln!("{}: FAILED\n{r}", case.stream);
                failures.push(case.stream);
            }
        }
    }
    assert!(failures.is_empty(), "failing codestreams: {failures:?}");
}

/// Truncated or corrupted codestreams must produce an image or an error,
/// never a panic. Set `DICOM_JPEG_FUZZ_ITERS` for a deeper run.
#[test]
fn corrupted_codestreams_do_not_panic() {
    let mut seed: u64 = 0x853C_49E6_748F_EA9B;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let files = [
        "codestreams_profile0/p0_01.j2k",
        "codestreams_profile0/p0_03.j2k",
        "codestreams_profile0/p0_05.j2k",
        "codestreams_profile0/p0_09.j2k",
        "codestreams_profile0/p0_12.j2k",
        "codestreams_profile0/p0_14.j2k",
        "codestreams_profile1/p1_03.j2k",
        "codestreams_profile1/p1_05.j2k",
        "codestreams_profile1/p1_07.j2k",
    ];
    let opts = DecoderOptions::default();
    for f in files {
        let data = std::fs::read(common::testdata(&format!("{ROOT}/{f}"))).unwrap();
        for cut in (0..data.len()).step_by((data.len() / 25).max(1)) {
            let _ = decode_components(&data[..cut], &opts);
        }
        let iters: usize = std::env::var("DICOM_JPEG_FUZZ_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(25);
        for _ in 0..iters {
            let mut d = data.clone();
            for _ in 0..1 + rnd() % 6 {
                let i = (rnd() as usize) % d.len();
                d[i] = rnd() as u8;
            }
            let _ = decode_components(&d, &opts);
        }
    }
}

/// Every JP2 / JPH test file and codestream of the ISO set decodes or
/// reports an error, without panicking.
#[test]
fn all_iso_files_decode_or_fail_cleanly() {
    let mut results = Vec::new();
    for dir in [
        "testfiles_jp2",
        "testfiles_jph",
        "codestreams_extended",
        "codestreams_hifi",
    ] {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join(ROOT)
            .join(dir);
        let mut entries: Vec<_> = std::fs::read_dir(&base)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .collect();
        entries.sort();
        for p in entries {
            let Some(ext) = p.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            if !matches!(ext, "jp2" | "jph" | "j2k" | "j2c" | "jhc" | "jpx") {
                continue;
            }
            let data = std::fs::read(&p).unwrap();
            let r = decode_components(&data, &DecoderOptions::default());
            results.push(format!(
                "{}: {}",
                p.file_name().unwrap().to_string_lossy(),
                match r {
                    Ok(c) => format!("ok, {} components, {}x{}", c.len(), c[0].width, c[0].height),
                    Err(e) => format!("error: {e}"),
                }
            ));
        }
    }
    eprintln!("{}", results.join("\n"));
}
