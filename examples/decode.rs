//! Decode DICOM files or raw codestreams to PGM / PPM for viewing.
//!
//! ```sh
//! cargo run --example decode -- <input>... [-o <dir>] [--raw]
//! cargo run --example decode -- testdata/dicom/wg04/JPLY/MR2_JPLY
//! cargo run --example decode -- testdata/conformance/t81/w3/*.jpg -o /tmp/out
//! ```
//!
//! DICOM files are decoded frame by frame through the dicom-rs adapter
//! (output is what a dicom-rs application would see: RGB for colour images).
//! Raw codestreams are recognised by their signature (JPEG, JPEG-LS,
//! JPEG 2000 / HTJ2K, JPEG XL). Output goes to `decoded/` (git-ignored) unless
//! `-o` is given.
//!
//! By default images are written for viewing: 8-bit, with the DICOM rescale,
//! window (or an automatic contrast stretch) and MONOCHROME1 inversion
//! applied. `--raw` writes the exact decoded sample values instead (16-bit PNM
//! above 8 bits; negative values as two's complement bit patterns).

#[path = "../tests/common/pnm.rs"]
mod pnm;

use std::path::{Path, PathBuf};

use dicom_encoding::adapters::PixelDataReader;
use dicom_jpeg::dicom_rs::PixelAdapter;
use dicom_jpeg::{Codec, ColorOutput, DecodeOptions, Samples};

type BoxResult<T> = Result<T, Box<dyn std::error::Error>>;

fn sniff(data: &[u8]) -> Option<Codec> {
    if data.starts_with(&[0xFF, 0xD8]) {
        // JPEG-LS frames use SOF55 (FF F7); otherwise treat as T.81.
        let is_ls = data.windows(2).take(4096).any(|w| w == [0xFF, 0xF7]);
        return Some(if is_ls { Codec::JpegLs } else { Codec::Jpeg });
    }
    if data.starts_with(&[0xFF, 0x4F, 0xFF, 0x51]) || data.get(4..8) == Some(b"jP  ") {
        let is_ht = data.windows(2).take(4096).any(|w| w == [0xFF, 0x50]);
        return Some(if is_ht { Codec::HtJ2k } else { Codec::Jpeg2000 });
    }
    if data.starts_with(&[0xFF, 0x0A]) || data.get(4..8) == Some(b"JXL ") {
        return Some(Codec::JpegXl);
    }
    None
}

fn to_i32(samples: &Samples) -> Vec<i32> {
    match samples {
        Samples::U8(v) => v.iter().map(|&x| x as i32).collect(),
        Samples::U16(v) => v.iter().map(|&x| x as i32).collect(),
    }
}

fn decode_dicom(input: &Path, out_dir: &Path, stem: &str, raw: bool) -> BoxResult<Vec<PathBuf>> {
    let obj = dicom_object::OpenFileOptions::new().open_file(input)?;
    let uid = obj
        .meta()
        .transfer_syntax()
        .trim_end_matches('\0')
        .to_string();
    let adapter = PixelAdapter::from_uid(&uid)
        .ok_or_else(|| format!("transfer syntax {uid} is not handled by dicom-jpeg"))?;
    let int = |tag| -> Option<u32> { obj.get(tag)?.to_int::<u32>().ok() };
    use dicom_dictionary_std::tags;
    let rows = int(tags::ROWS).ok_or("missing Rows")? as usize;
    let cols = int(tags::COLUMNS).ok_or("missing Columns")? as usize;
    let spp = int(tags::SAMPLES_PER_PIXEL).unwrap_or(1) as usize;
    let bits_allocated = int(tags::BITS_ALLOCATED).ok_or("missing BitsAllocated")?;
    let bits_stored = int(tags::BITS_STORED).unwrap_or(bits_allocated);
    let signed = int(tags::PIXEL_REPRESENTATION).unwrap_or(0) == 1;
    let frames = int(tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1);
    let voi = pnm::voi_from_dicom(&obj);
    eprintln!(
        "{}: {} ({cols}x{rows}, {spp} samples, {bits_stored} bits{}, {frames} frame(s))",
        input.display(),
        adapter.uid(),
        if signed { " signed" } else { "" }
    );
    let mut written = Vec::new();
    for f in 0..frames {
        let mut bytes = Vec::new();
        adapter.decode_frame(&obj, f, &mut bytes)?;
        let mask = if bits_stored >= 32 {
            u32::MAX
        } else {
            (1 << bits_stored) - 1
        };
        let samples: Vec<i32> = match bits_allocated {
            8 => bytes.iter().map(|&b| b as u32).collect::<Vec<_>>(),
            _ => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                .collect(),
        }
        .into_iter()
        .map(|v| {
            let v = v & mask;
            if signed && v & (1 << (bits_stored - 1)) != 0 {
                (v | !mask) as i32
            } else {
                v as i32
            }
        })
        .collect();
        let name = if frames > 1 {
            format!("{stem}_f{f:03}")
        } else {
            stem.to_string()
        };
        let (bits, samples) = if raw {
            (bits_stored, samples)
        } else {
            (8, pnm::to_display(&samples, spp, bits_stored, &voi))
        };
        written.extend(pnm::write_pnm(
            &out_dir.join(name),
            cols,
            rows,
            spp,
            bits,
            &samples,
        )?);
    }
    Ok(written)
}

fn decode_raw(
    input: &Path,
    data: &[u8],
    out_dir: &Path,
    stem: &str,
    raw: bool,
) -> BoxResult<Vec<PathBuf>> {
    let ts = sniff(data).ok_or("unknown file format (not DICOM or a known codestream)")?;
    let opts = DecodeOptions::new().with_color_output(ColorOutput::Rgb);
    let img = dicom_jpeg::decode(ts, data, &opts)?;
    let i = &img.info;
    eprintln!(
        "{}: {} codestream ({}x{}, {} components, {} bits, {:?})",
        input.display(),
        ts,
        i.width,
        i.height,
        i.components,
        i.bits_per_sample,
        i.color_space
    );
    let (nc, bits) = (i.components as usize, i.bits_per_sample as u32);
    let samples = to_i32(&img.samples);
    let (bits, samples) = if raw {
        (bits, samples)
    } else {
        (8, pnm::to_display(&samples, nc, bits, &pnm::Voi::default()))
    };
    Ok(pnm::write_pnm(
        &out_dir.join(stem),
        i.width as usize,
        i.height as usize,
        nc,
        bits,
        &samples,
    )?)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut inputs = Vec::new();
    let mut out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("decoded");
    let mut raw = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-o" | "--out" => out_dir = PathBuf::from(args.next().expect("-o needs a directory")),
            "--raw" => raw = true,
            "-h" | "--help" => {
                eprintln!("usage: decode <input>... [-o <dir>] [--raw]");
                return;
            }
            _ => inputs.push(PathBuf::from(a)),
        }
    }
    if inputs.is_empty() {
        eprintln!("usage: decode <input>... [-o <dir>] [--raw]");
        std::process::exit(2);
    }
    let mut failed = false;
    for input in &inputs {
        let stem = pnm::slug(&input.file_stem().unwrap_or_default().to_string_lossy());
        let result = std::fs::read(input).map_err(Into::into).and_then(|data| {
            let is_dicom = data.get(128..132) == Some(b"DICM");
            if is_dicom {
                decode_dicom(input, &out_dir, &stem, raw)
            } else {
                decode_raw(input, &data, &out_dir, &stem, raw)
            }
        });
        match result {
            Ok(files) => files.iter().for_each(|p| println!("{}", p.display())),
            Err(e) => {
                eprintln!("{}: {e}", input.display());
                failed = true;
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
