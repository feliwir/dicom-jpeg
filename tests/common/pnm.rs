//! Minimal PGM / PPM writer for inspecting decoded images.
//!
//! Shared by the test helpers and `examples/decode.rs`.

#![allow(dead_code)]

use std::io::Write;
use std::path::Path;

/// Write interleaved samples as binary PGM (1 component) or PPM (3 components).
/// Other component counts are written as one PGM per component
/// (`<stem>_c<N>.pgm`). Samples wider than 8 bits are written as 16-bit
/// big-endian PNM. Signed samples are offset by 2^(bits-1) for display.
/// Returns the files written.
pub fn write_pnm(
    path: &Path,
    width: usize,
    height: usize,
    components: usize,
    bits: u32,
    signed: bool,
    samples: &[i32],
) -> std::io::Result<Vec<std::path::PathBuf>> {
    let bits = bits.clamp(1, 16);
    let maxval = (1u32 << bits) - 1;
    let offset = if signed { 1i64 << (bits - 1) } else { 0 };
    let to_u = |v: i32| (v as i64 + offset).clamp(0, maxval as i64) as u32;
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
