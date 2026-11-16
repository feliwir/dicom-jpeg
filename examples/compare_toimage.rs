//! Compare this crate's decoders with dicom-rs's own (via `dicom-toimage`).
//!
//! ```sh
//! cargo run --release --example compare_toimage                    # all of testdata/dicom
//! cargo run --release --example compare_toimage -- testdata/dicom/wg04/JPLY
//! cargo run --release --example compare_toimage -- --toimage /path/to/dicom-toimage
//! ```
//!
//! For every encapsulated JPEG-family DICOM file, each frame (at most 8,
//! evenly spaced) is decoded by the dicom-rs adapter of this crate and by
//! `dicom-toimage --no-modality-lut` (stored values, no windowing). Where a
//! reference exists (WG04 `REF/*_UNC` originals, pydicom originals, DCMTK
//! decodes of DCMTK-encoded files), both are also compared against it. Results go to `decoded/compare/report.md` (plus difference
//! images for frames where the two decoders disagree).

#[path = "../tests/common/pnm.rs"]
mod pnm;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use dicom_dictionary_std::tags;
use dicom_encoding::adapters::PixelDataReader;
use dicom_jpeg::dicom_rs::PixelAdapter;
use dicom_object::{DefaultDicomObject, OpenFileOptions};

#[derive(Debug, Clone, Copy)]
struct Attrs {
    rows: usize,
    cols: usize,
    spp: usize,
    bits_allocated: u32,
    bits_stored: u32,
    signed: bool,
    mono1: bool,
    frames: u32,
}

fn attrs(obj: &DefaultDicomObject) -> Option<Attrs> {
    let int = |tag| -> Option<u32> { obj.get(tag)?.to_int::<u32>().ok() };
    let pi = obj
        .get(tags::PHOTOMETRIC_INTERPRETATION)
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_string()))
        .unwrap_or_default();
    Some(Attrs {
        rows: int(tags::ROWS)? as usize,
        cols: int(tags::COLUMNS)? as usize,
        spp: int(tags::SAMPLES_PER_PIXEL).unwrap_or(1) as usize,
        bits_allocated: int(tags::BITS_ALLOCATED)?,
        bits_stored: int(tags::BITS_STORED)?,
        signed: int(tags::PIXEL_REPRESENTATION).unwrap_or(0) == 1,
        mono1: pi == "MONOCHROME1",
        frames: int(tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1),
    })
}

/// Little-endian stored samples -> integers (masked, sign-extended).
fn to_values(bytes: &[u8], a: &Attrs) -> Vec<i32> {
    let bits = a.bits_stored.min(16);
    let mask = if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let conv = |v: u32| {
        let v = v & mask;
        if a.signed && v & (1 << (bits - 1)) != 0 {
            (v | !mask) as i32
        } else {
            v as i32
        }
    };
    if a.bits_allocated == 8 {
        bytes.iter().map(|&b| conv(b as u32)).collect()
    } else {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| conv(u16::from_le_bytes([c[0], c[1]]) as u32))
            .collect()
    }
}

fn ours(obj: &DefaultDicomObject, frame: u32, a: &Attrs) -> Result<Vec<i32>, String> {
    let adapter = PixelAdapter::from_uid(obj.meta().transfer_syntax()).ok_or("not JPEG family")?;
    let mut out = Vec::new();
    adapter
        .decode_frame(obj, frame, &mut out)
        .map_err(|e| e.to_string())?;
    Ok(to_values(&out, a))
}

/// Parse a PAM (P7) file written by dicom-toimage.
fn read_pam(data: &[u8]) -> Result<(usize, Vec<u32>), String> {
    let end = data
        .windows(7)
        .position(|w| w == b"ENDHDR\n")
        .ok_or("not a PAM file")?;
    let header = std::str::from_utf8(&data[..end]).map_err(|e| e.to_string())?;
    let mut maxval = 255u32;
    let mut depth = 1usize;
    for line in header.lines() {
        let mut it = line.split_whitespace();
        match (it.next(), it.next()) {
            (Some("MAXVAL"), Some(v)) => maxval = v.parse().map_err(|_| "bad MAXVAL")?,
            (Some("DEPTH"), Some(v)) => depth = v.parse().map_err(|_| "bad DEPTH")?,
            _ => {}
        }
    }
    let px = &data[end + 7..];
    let values = if maxval > 255 {
        px.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_be_bytes([c[0], c[1]]) as u32)
            .collect()
    } else {
        px.iter().map(|&b| b as u32).collect()
    };
    let _ = depth;
    Ok((maxval as usize, values))
}

fn theirs(bin: &Path, file: &Path, frame: u32, a: &Attrs, tmp: &Path) -> Result<Vec<i32>, String> {
    let _ = std::fs::remove_file(tmp);
    let depth = if a.bits_allocated == 8 {
        "--8bit"
    } else {
        "--16bit"
    };
    let out = Command::new(bin)
        .args(["--no-modality-lut", depth, "-F", &frame.to_string(), "-o"])
        .arg(tmp)
        .arg(file)
        .output()
        .map_err(|e| format!("cannot run {}: {e}", bin.display()))?;
    if !out.status.success() || !tmp.exists() {
        // dicom-toimage logs to stdout with ANSI colours; keep the innermost
        // cause ("N: message" lines list the most recent error first).
        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&out.stderr));
        let plain = strip_ansi(&text);
        let cause = plain
            .lines()
            .map(str::trim)
            .rfind(|l| {
                l.split_once(':')
                    .is_some_and(|(n, _)| n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty())
            })
            .map(|l| l.split_once(':').unwrap().1.trim().to_string())
            .or_else(|| {
                plain
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .map(|l| l.trim().to_string())
            })
            .unwrap_or_else(|| "failed".into());
        return Err(cause);
    }
    let (maxval, raw) = read_pam(&std::fs::read(tmp).map_err(|e| e.to_string())?)?;
    // Undo what dicom-pixeldata does without LUTs: MONOCHROME1 inversion and
    // the offset of signed samples to the unsigned range.
    Ok(raw
        .into_iter()
        .map(|t| {
            let t = if a.mono1 && a.spp == 1 {
                maxval as u32 - t
            } else {
                t
            };
            if a.signed {
                t as i32 - (maxval as i32 + 1) / 2
            } else {
                t as i32
            }
        })
        .collect())
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[derive(Debug, Default, Clone, Copy)]
struct Diff {
    max: u32,
    mismatches: usize,
    psnr: f64,
}

fn diff(a: &[i32], b: &[i32], bits: u32) -> Option<Diff> {
    if a.len() != b.len() {
        return None;
    }
    let mut d = Diff::default();
    let mut se = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let e = (x - y).unsigned_abs();
        if e > 0 {
            d.mismatches += 1;
            d.max = d.max.max(e);
            se += (e as f64) * (e as f64);
        }
    }
    let mse = se / a.len().max(1) as f64;
    let peak = ((1u64 << bits.min(16)) - 1) as f64;
    d.psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (peak * peak / mse).log10()
    };
    Some(d)
}

fn fmt_diff(d: Option<Diff>, n: usize) -> String {
    match d {
        None => "size mismatch".into(),
        Some(d) if d.max == 0 => "identical".into(),
        Some(d) => format!(
            "max {} · {:.2}% differ · {:.1} dB",
            d.max,
            100.0 * d.mismatches as f64 / n.max(1) as f64,
            d.psnr
        ),
    }
}

/// Reference image for a test file: the WG04 original for
/// `.../wg04/<SET>/<NAME>_<SET>`, or a known reference for pydicom / DCMTK
/// files (originals, or DCMTK's own decodes of DCMTK-encoded JPEGs).
fn reference(file: &Path) -> Option<PathBuf> {
    let set_dir = file.parent()?;
    let dicom = set_dir.parent()?;
    let name = file.file_name()?.to_str()?;
    let r = if dicom.file_name()? == "wg04" {
        let stem = name.split('_').next()?;
        dicom.join("REF").join(format!("{stem}_UNC"))
    } else {
        const KNOWN: &[(&str, &str)] = &[
            ("693_J2KI.dcm", "693_UNCI.dcm"),
            ("693_J2KR.dcm", "693_UNCR.dcm"),
            ("emri_small_jpeg_2k_lossless.dcm", "emri_small.dcm"),
            ("emri_small_jpeg_ls_lossless.dcm", "emri_small.dcm"),
            ("MR_small_jp2klossless.dcm", "MR_small.dcm"),
            ("MR_small_jpeg_ls_lossless.dcm", "MR_small.dcm"),
            ("JPEG2000.dcm", "JPEG2000_UNC.dcm"),
            ("SC_rgb_jpeg_gdcm.dcm", "SC_rgb.dcm"),
            ("SC_rgb_jpeg_dcmtk.dcm", "SC_rgb.dcm"),
            ("SC_rgb_jpeg_lossy_gdcm.dcm", "SC_rgb.dcm"),
            ("SC_rgb_small_odd_jpeg.dcm", "SC_rgb_small_odd.dcm"),
            ("SC_rgb_dcmtk_+eb+cr.dcm", "SC_rgb_dcmtk_ebcr_dcmd.dcm"),
            ("SC_rgb_dcmtk_+eb+cy+n1.dcm", "SC_rgb_dcmtk_ebcyn1_dcmd.dcm"),
            ("SC_rgb_dcmtk_+eb+cy+n2.dcm", "SC_rgb_dcmtk_ebcyn2_dcmd.dcm"),
            ("SC_rgb_dcmtk_+eb+cy+np.dcm", "SC_rgb_dcmtk_ebcynp_dcmd.dcm"),
            ("SC_rgb_dcmtk_+eb+cy+s2.dcm", "SC_rgb_dcmtk_ebcys2_dcmd.dcm"),
            ("SC_rgb_dcmtk_+eb+cy+s4.dcm", "SC_rgb_dcmtk_ebcys4_dcmd.dcm"),
            ("SC_rgb_gdcm_KY.dcm", "SC_rgb_gdcm2k_uncompressed.dcm"),
            ("monkey12_jply.dcm", "monkey12_dcmd_rgb.dcm"),
        ];
        let (_, r) = KNOWN.iter().find(|(c, _)| *c == name)?;
        set_dir.join(r)
    };
    r.exists().then_some(r)
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .collect();
        entries.sort();
        for e in entries {
            collect(&e, out);
        }
    } else if !matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("txt" | "rst" | "md")
    ) {
        out.push(path.to_path_buf());
    }
}

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut bin = PathBuf::from("dicom-toimage");
    let mut inputs = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--toimage" => bin = PathBuf::from(args.next().expect("--toimage needs a path")),
            _ => inputs.push(PathBuf::from(a)),
        }
    }
    if inputs.is_empty() {
        inputs.push(root.join("testdata/dicom"));
    }
    let mut files = Vec::new();
    for i in &inputs {
        collect(i, &mut files);
    }
    let out_dir = root.join("decoded/compare");
    std::fs::create_dir_all(&out_dir).expect("create output folder");
    let tmp = out_dir.join("toimage_tmp.pam");

    let mut report = String::new();
    let _ = writeln!(report, "# dicom-jpeg vs dicom-rs (dicom-toimage)\n");
    let _ = writeln!(
        report,
        "| File | Transfer syntax | Frames | dicom-jpeg vs dicom-rs | dicom-jpeg vs reference | dicom-rs vs reference |\n|---|---|---|---|---|---|"
    );
    let mut counts = [0usize; 4]; // identical, differ, dicom-rs failed, ours failed
    for file in &files {
        let Ok(obj) = OpenFileOptions::new().open_file(file) else {
            continue;
        };
        let Some(adapter) = PixelAdapter::from_uid(obj.meta().transfer_syntax()) else {
            continue;
        };
        let Some(a) = attrs(&obj) else { continue };
        let rel = file
            .strip_prefix(root)
            .unwrap_or(file)
            .display()
            .to_string();
        let reference = reference(file).and_then(|r| OpenFileOptions::new().open_file(r).ok());
        let step = (a.frames as usize).div_ceil(8).max(1);
        let frames: Vec<u32> = (0..a.frames).step_by(step).collect();

        let (mut vs_ref_ours, mut vs_ref_theirs) = (Vec::new(), Vec::new());
        let (mut ours_err, mut theirs_err) = (None, None);
        let mut worst: Option<Diff> = Some(Diff {
            psnr: f64::INFINITY,
            ..Diff::default()
        });
        let mut worst_ref = [Some(Diff {
            psnr: f64::INFINITY,
            ..Diff::default()
        }); 2];
        let n = a.rows * a.cols * a.spp;
        for &f in &frames {
            let o = ours(&obj, f, &a).map_err(|e| ours_err = Some(e)).ok();
            let t = theirs(&bin, file, f, &a, &tmp)
                .map_err(|e| theirs_err = Some(e))
                .ok();
            let r = reference.as_ref().and_then(|r| {
                let ra = attrs(r)?;
                let bytes = dicom_encoding::adapters::PixelDataObject::frame_pixel_data(r, f)?;
                // Compare in the compressed object's bit depth.
                let ra = Attrs {
                    bits_stored: ra.bits_stored.min(a.bits_stored),
                    ..ra
                };
                Some(to_values(&bytes, &ra))
            });
            let merge = |w: &mut Option<Diff>, d: Option<Diff>| {
                *w = match (*w, d) {
                    (Some(w), Some(d)) => Some(Diff {
                        max: w.max.max(d.max),
                        mismatches: w.mismatches.max(d.mismatches),
                        psnr: w.psnr.min(d.psnr),
                    }),
                    _ => None,
                }
            };
            if let (Some(o), Some(t)) = (&o, &t) {
                let d = diff(o, t, a.bits_stored);
                merge(&mut worst, d);
                if d.is_some_and(|d| d.max > 0) {
                    let scaled: Vec<i32> = o
                        .iter()
                        .zip(t)
                        .map(|(x, y)| ((x - y).unsigned_abs() * 255 / d.unwrap().max) as i32)
                        .collect();
                    let name = format!("{}_f{f:03}_diff", pnm::slug(&rel));
                    let _ = pnm::write_pnm(&out_dir.join(name), a.cols, a.rows, a.spp, 8, &scaled);
                }
            }
            if let Some(r) = &r {
                // Mask both decodes the same way as the reference (bits of the compressed object).
                if let Some(o) = &o {
                    let d = diff(o, r, a.bits_stored);
                    merge(&mut worst_ref[0], d);
                    vs_ref_ours.push(d);
                }
                if let Some(t) = &t {
                    let d = diff(t, r, a.bits_stored);
                    merge(&mut worst_ref[1], d);
                    vs_ref_theirs.push(d);
                }
            }
        }
        let fail = |e: &Option<String>| e.as_ref().map(|e| format!("**failed**: {e}"));
        let col_cmp = match (&ours_err, &theirs_err) {
            (Some(_), _) => {
                counts[3] += 1;
                fail(&ours_err).unwrap()
            }
            (_, Some(_)) => {
                counts[2] += 1;
                format!("dicom-rs {}", fail(&theirs_err).unwrap())
            }
            _ => {
                if worst.is_some_and(|w| w.max == 0) {
                    counts[0] += 1;
                } else {
                    counts[1] += 1;
                }
                fmt_diff(worst, n)
            }
        };
        let col_ref = |v: &Vec<Option<Diff>>, w: Option<Diff>| {
            if v.is_empty() {
                "–".to_string()
            } else {
                fmt_diff(w, n)
            }
        };
        let _ = writeln!(
            report,
            "| `{rel}` | {} | {} | {col_cmp} | {} | {} |",
            adapter.uid().trim_start_matches("1.2.840.10008.1.2."),
            if frames.len() as u32 == a.frames {
                a.frames.to_string()
            } else {
                format!("{} of {}", frames.len(), a.frames)
            },
            col_ref(&vs_ref_ours, worst_ref[0]),
            col_ref(&vs_ref_theirs, worst_ref[1]),
        );
        eprintln!("{rel}: {col_cmp}");
    }
    let _ = std::fs::remove_file(&tmp);
    let summary = format!(
        "\n**{} files:** {} identical, {} differ, {} not decodable by dicom-rs, {} not decodable by dicom-jpeg.\n",
        counts.iter().sum::<usize>(),
        counts[0],
        counts[1],
        counts[2],
        counts[3]
    );
    report.push_str(&summary);
    let path = out_dir.join("report.md");
    std::fs::write(&path, &report).expect("write report");
    println!("{}", summary.trim());
    println!("report: {}", path.display());
}
