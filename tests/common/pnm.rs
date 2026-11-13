//! Minimal PGM / PPM writer for inspecting decoded images.
//!
//! Shared by the test helpers and `examples/decode.rs`.

#![allow(dead_code)]

use std::io::Write;
use std::path::Path;

/// Write interleaved samples as binary PGM (1 component) or PPM (3 components).
/// Other component counts are written as one PGM per component
/// (`<stem>_c<N>.pgm`). Samples wider than 8 bits are written as 16-bit
/// big-endian PNM; negative values are written as their `bits`-wide two's
/// complement pattern (use [`to_display`] first for viewable images).
/// Returns the files written.
pub fn write_pnm(
    path: &Path,
    width: usize,
    height: usize,
    components: usize,
    bits: u32,
    samples: &[i32],
) -> std::io::Result<Vec<std::path::PathBuf>> {
    let bits = bits.clamp(1, 16);
    let maxval = (1u32 << bits) - 1;
    let to_u = |v: i32| v as u32 & maxval;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let write_one = |p: &Path, channels: &[usize]| -> std::io::Result<()> {
        let magic = if channels.len() == 3 { "P6" } else { "P5" };
        let mut out = format!("{magic}\n{width} {height}\n{maxval}\n").into_bytes();
        for px in samples.chunks_exact(components) {
            for &c in channels {
                let v = to_u(px[c]);
                if maxval > 255 {
                    out.extend_from_slice(&(v as u16).to_be_bytes());
                } else {
                    out.push(v as u8);
                }
            }
        }
        std::fs::File::create(p)?.write_all(&out)
    };
    match components {
        1 => {
            let p = path.with_extension("pgm");
            write_one(&p, &[0])?;
            Ok(vec![p])
        }
        3 => {
            let p = path.with_extension("ppm");
            write_one(&p, &[0, 1, 2])?;
            Ok(vec![p])
        }
        n => (0..n)
            .map(|c| {
                let stem = path.file_stem().unwrap_or_default().to_string_lossy();
                let p = path.with_file_name(format!("{stem}_c{c}.pgm"));
                write_one(&p, &[c])?;
                Ok(p)
            })
            .collect(),
    }
}

/// Root folder for dumped images: the `DICOM_JPEG_DUMP` environment variable
/// (`1` selects the default `decoded/` folder of the crate), or `None` when unset.
pub fn dump_dir() -> Option<std::path::PathBuf> {
    let v = std::env::var_os("DICOM_JPEG_DUMP")?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    if v.is_empty() || v == "0" {
        return None;
    }
    if v == "1" {
        return Some(root.join("decoded"));
    }
    let p = std::path::PathBuf::from(v);
    Some(if p.is_absolute() { p } else { root.join(p) })
}

/// File-name friendly version of a test-data path or label.
pub fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

/// Like [`slug`], but keeps `/` so names can contain sub-folders.
pub fn slug_path(name: &str) -> String {
    name.split('/')
        .map(slug)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("/")
}

/// Value-of-interest transform used to make images viewable, following the
/// DICOM grayscale pipeline (PS3.4 C.11): Modality LUT (rescale), VOI LUT
/// (window), Presentation (MONOCHROME1 inversion).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voi {
    /// (Rescale Slope, Rescale Intercept)
    pub rescale: (f64, f64),
    /// (Window Center, Window Width) if present
    pub window: Option<(f64, f64)>,
    /// MONOCHROME1: low values are white
    pub invert: bool,
}

impl Default for Voi {
    fn default() -> Self {
        Voi {
            rescale: (1.0, 0.0),
            window: None,
            invert: false,
        }
    }
}

/// Read rescale / window / photometric attributes (first value of multi-valued ones).
pub fn voi_from_dicom(obj: &dicom_object::DefaultDicomObject) -> Voi {
    use dicom_dictionary_std::tags;
    let first = |tag| -> Option<f64> {
        let s = obj.get(tag)?.to_str().ok()?;
        s.split('\\').next()?.trim().parse().ok()
    };
    let window = match (first(tags::WINDOW_CENTER), first(tags::WINDOW_WIDTH)) {
        (Some(c), Some(w)) if w >= 1.0 => Some((c, w)),
        _ => None,
    };
    let pi = obj
        .get(tags::PHOTOMETRIC_INTERPRETATION)
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_string()))
        .unwrap_or_default();
    Voi {
        rescale: (
            first(tags::RESCALE_SLOPE).unwrap_or(1.0),
            first(tags::RESCALE_INTERCEPT).unwrap_or(0.0),
        ),
        window,
        invert: pi == "MONOCHROME1",
    }
}

/// Map samples to 8-bit display values. Grayscale images get the rescale,
/// the window (or an automatic 0.5 %..99.5 % stretch when there is none) and
/// MONOCHROME1 inversion; colour samples are scaled from `bits` to 8 bits.
pub fn to_display(samples: &[i32], components: usize, bits: u32, voi: &Voi) -> Vec<i32> {
    if components != 1 {
        let shift = bits.saturating_sub(8);
        return samples
            .iter()
            .map(|&v| (v.max(0) >> shift).min(255))
            .collect();
    }
    let (slope, intercept) = voi.rescale;
    let values: Vec<f64> = samples
        .iter()
        .map(|&v| v as f64 * slope + intercept)
        .collect();
    let (lo, hi) = match voi.window {
        // Linear VOI function (PS3.3 C.11.2.1.2.1).
        Some((c, w)) => (c - 0.5 - (w - 1.0) / 2.0, c - 0.5 + (w - 1.0) / 2.0),
        None => {
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);
            let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
            (at(0.005), at(0.995))
        }
    };
    let span = (hi - lo).max(1e-9);
    values
        .iter()
        .map(|&x| {
            let y = (((x - lo) / span) * 255.0).round().clamp(0.0, 255.0) as i32;
            if voi.invert { 255 - y } else { y }
        })
        .collect()
}
