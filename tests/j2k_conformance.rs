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
        "htj2k_bsets_profile0/p0_02_bset/ds0_ht_02_b11.j2k",
        "htj2k_bsets_profile0/p0_06_bset/ds0_hm_06_b11.j2k",
        "htj2k_bsets_profile1/p1_03_bset/ds1_ht_03_b11.j2k",
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

/// Per-component (peak, MSE) limits of a HTJ2K bset, keyed by (B, mixed).
/// Two layouts occur: one "PEAK, MSE, PSNR" row per component, or a row of
/// per-component peaks followed by a row of MSEs (last column = max / mean).
/// Per (B, mixed): per-component (peak, MSE).
type BsetStats = Vec<((u32, bool), Vec<(i64, f64)>)>;

fn bset_stats(text: &str) -> BsetStats {
    let mut out: BsetStats = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let flush = |out: &mut BsetStats, rows: &mut Vec<Vec<f64>>| {
        if let Some(last) = out.last_mut() {
            if rows.iter().all(|r| r.len() == 3) {
                last.1 = rows.iter().map(|r| (r[0] as i64, r[1])).collect();
            } else if rows.len() >= 2 {
                let n = rows[0].len() - 1;
                last.1 = (0..n).map(|k| (rows[0][k] as i64, rows[1][k])).collect();
            }
        }
        rows.clear();
    };
    for line in text.lines() {
        let t = line.trim();
        if let Some(key) = t
            .strip_prefix("----B=")
            .and_then(|r| r.strip_suffix("----"))
        {
            flush(&mut out, &mut rows);
            let mixed = key.ends_with("HTMIX");
            let b: u32 = key.trim_end_matches("HTMIX").trim().parse().unwrap();
            out.push(((b, mixed), Vec::new()));
            continue;
        }
        let fields: Vec<f64> = t
            .split(',')
            .filter_map(|f| f.trim().parse::<f64>().ok())
            .collect();
        if !fields.is_empty() && fields.len() == t.split(',').count() {
            rows.push(fields);
        }
    }
    flush(&mut out, &mut rows);
    out
}

/// HTJ2K conformance (15444-4 ed. 4). Every HT codestream was transcoded from
/// a Part 1 conformance codestream; its decode is compared per component with
/// the decode of that Part 1 codestream (which itself passes class 1), within
/// the peak / MSE of the transcoding published in the set's stats file.
#[test]
fn htj2k_bsets() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(ROOT);
    let mut failures = Vec::new();
    let mut count = 0;
    for profile in [0, 1] {
        let dir = base.join(format!("htj2k_bsets_profile{profile}"));
        let mut sets: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        sets.sort();
        for set in sets {
            let name = set.file_name().unwrap().to_string_lossy().to_string(); // p0_04_bset
            let num = &name[3..5];
            let stats =
                bset_stats(&std::fs::read_to_string(set.join("stats.txt")).unwrap_or_default());
            let case = CASES
                .iter()
                .find(|c| c.stream.ends_with(&format!("p{profile}_{num}.j2k")))
                .expect("matching Part 1 case");
            let mut opts = DecoderOptions::default();
            opts.reduce = case.reduce;
            let original =
                std::fs::read(common::testdata(&format!("{ROOT}/{}", case.stream))).unwrap();
            let part1 = decode_components(&original, &opts).unwrap();
            let mut files: Vec<_> = std::fs::read_dir(&set)
                .unwrap()
                .flatten()
                .map(|e| e.path())
                .collect();
            files.sort();
            for f in files
                .iter()
                .filter(|p| p.extension().is_some_and(|e| e == "j2k"))
            {
                let fname = f.file_name().unwrap().to_string_lossy().to_string();
                let b: u32 = fname
                    .rsplit("_b")
                    .next()
                    .unwrap()
                    .trim_end_matches(".j2k")
                    .parse()
                    .unwrap();
                let mixed = fname.contains("_hm_");
                let limits = stats
                    .iter()
                    .find(|(k, _)| *k == (b, mixed))
                    .map(|(_, l)| l.clone())
                    .unwrap_or_default();
                count += 1;
                let comps = match decode_components(&std::fs::read(f).unwrap(), &opts) {
                    Ok(c) => c,
                    Err(e) => {
                        failures.push(format!("{fname}: decode failed: {e}"));
                        continue;
                    }
                };
                let mut line = format!("{fname}:");
                let mut ok = comps.len() == part1.len();
                for (k, (c, r)) in comps.iter().zip(&part1).enumerate() {
                    if (c.width, c.height) != (r.width, r.height) {
                        ok = false;
                        line.push_str(&format!(" c{k} size mismatch"));
                        continue;
                    }
                    let peak = c
                        .data
                        .iter()
                        .zip(&r.data)
                        .map(|(a, b)| (*a as i64 - *b as i64).abs())
                        .max()
                        .unwrap_or(0);
                    let mse = c
                        .data
                        .iter()
                        .zip(&r.data)
                        .map(|(a, b)| ((*a - *b) as f64).powi(2))
                        .sum::<f64>()
                        / r.data.len() as f64;
                    let (lp, lm) = limits.get(k).copied().unwrap_or((0, 0.0));
                    // Kakadu's figures come from its own decoder; reconstruction
                    // points of heavily quantized data differ slightly.
                    let pass = peak <= lp + 1 + lp / 50 && mse <= lm * 1.05 + 0.01;
                    ok &= pass;
                    line.push_str(&format!(
                        " c{k} peak {peak}/{lp} mse {mse:.3}/{lm:.3}{}",
                        if pass { "" } else { " FAIL" }
                    ));
                }
                eprintln!("{line}");
                if !ok {
                    failures.push(line);
                }
            }
        }
    }
    assert!(count >= 40, "only {count} HT codestreams found");
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
}

/// Part 2 multiple component transform codestreams (15444-4 "extended"):
/// mct1 (9/7 wavelet along 60 components), mct2 (wavelet stage on three
/// collections followed by 60 array decorrelation collections) and mct3o
/// (2 -> 3 components irreversible decorrelation with offsets). The
/// reference images carry no tolerances. They were produced by truncating
/// the reconstructed values while we round, so about half the samples
/// differ by exactly 1: allow a peak error of 1 and an MSE of 0.5.
#[test]
#[cfg(feature = "jpeg2000-part2")]
fn part2_multiple_component_transforms() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(ROOT);
    for (name, ncomp) in [("mct1", 60usize), ("mct2", 180), ("mct3o", 3)] {
        let data = std::fs::read(
            base.join("codestreams_extended")
                .join(format!("{name}.j2k")),
        )
        .unwrap();
        let comps = decode_components(&data, &DecoderOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(comps.len(), ncomp, "{name}");
        // The electronic insert does not number the reference files in
        // component order. The components form 60 equal units (single
        // components in mct1, the colour triples of mct2) and the units are
        // numbered in a punctuation-ignoring name sort ("0", "10".."19",
        // "1", "20", ...). Established by matching every decoded component
        // against all reference files.
        let files: Vec<usize> = if ncomp % 60 == 0 {
            let unit = ncomp / 60;
            let mut units: Vec<usize> = (0..60).collect();
            units.sort_by_key(|k| format!("{k}pgx"));
            (0..ncomp)
                .map(|i| units[i / unit] * unit + i % unit)
                .collect()
        } else {
            (0..ncomp).collect()
        };
        let (mut worst_peak, mut worst_mse) = (0i64, 0f64);
        for (i, c) in comps.iter().enumerate() {
            let pgx = std::fs::read(
                base.join("reference_extended")
                    .join(format!("{name}-{}.pgx", files[i])),
            )
            .unwrap();
            let (w, h, reference) = read_pgx(&pgx);
            assert_eq!((c.width, c.height), (w, h), "{name} component {i}");
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
            worst_peak = worst_peak.max(peak);
            worst_mse = worst_mse.max(mse);
        }
        eprintln!("{name}: worst peak {worst_peak}, worst MSE {worst_mse:.4}");
        assert!(
            worst_peak <= 1 && worst_mse <= 0.5,
            "{name}: peak {worst_peak}, MSE {worst_mse}"
        );
    }
}
