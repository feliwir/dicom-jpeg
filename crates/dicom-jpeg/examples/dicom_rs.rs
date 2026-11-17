//! Decode DICOM files to PNG with plain dicom-rs, which uses dicom-jpeg for
//! the JPEG family through the `inventory-registry` feature.
//!
//! ```sh
//! cargo run --release --example dicom_rs --features inventory-registry -- <file.dcm>... [-o <dir>] [--window <center> <width>]
//! cargo run --release --example dicom_rs --features inventory-registry -- testdata/dicom/wg04/J2KR/CT1_J2KR --window 40 400
//! ```
//!
//! Nothing here calls dicom-jpeg: `decode_pixel_data` looks the transfer
//! syntax up in the dicom-rs registry, where dicom-jpeg registered itself.
//! `to_dynamic_image_with_options` then converts each frame to an 8-bit image
//! as dicom-rs does: modality LUT (rescale), then the object's first VOI LUT
//! or window, or a min-max normalisation if it has none (`--window` sets
//! one). One PNG per frame goes to `decoded/` (git-ignored) unless `-o` is
//! given.

use std::error::Error;
use std::path::{Path, PathBuf};

use dicom_encoding::TransferSyntaxIndex;
use dicom_object::open_file;
use dicom_pixeldata::{ConvertOptions, PixelDecoder, VoiLutOption, WindowLevel};
use dicom_transfer_syntax_registry::TransferSyntaxRegistry;

// The only mention of dicom-jpeg: link the crate so that dicom-rs collects
// its registry entries (with `inventory-registry`, it registers itself).
use dicom_jpeg as _;

fn convert(input: &Path, out_dir: &Path, options: &ConvertOptions) -> Result<(), Box<dyn Error>> {
    let obj = open_file(input)?;
    let uid = obj.meta().transfer_syntax().trim_end_matches('\0');
    let name = TransferSyntaxRegistry.get(uid).map_or(uid, |ts| ts.name());

    let pixels = obj.decode_pixel_data()?;
    let frames = pixels.number_of_frames();
    println!(
        "{}: {name}, {} x {}, {frames} frame(s)",
        input.display(),
        pixels.columns(),
        pixels.rows()
    );
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    for frame in 0..frames {
        let path = if frames == 1 {
            out_dir.join(format!("{stem}.png"))
        } else {
            out_dir.join(format!("{stem}_{frame:04}.png"))
        };
        pixels
            .to_dynamic_image_with_options(frame, options)?
            .save(&path)?;
        println!("  -> {}", path.display());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut inputs = Vec::new();
    let mut out_dir = PathBuf::from("decoded");
    // 8 bits per sample for viewing.
    let mut options = ConvertOptions::new().force_8bit();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "-o" {
            out_dir = args.next().ok_or("-o needs a directory")?.into();
        } else if arg == "--window" {
            let mut value = || -> Result<f64, Box<dyn Error>> {
                Ok(args
                    .next()
                    .ok_or("--window needs a center and a width")?
                    .parse()?)
            };
            let (center, width) = (value()?, value()?);
            options = options.with_voi_lut(VoiLutOption::Custom(WindowLevel { center, width }));
        } else {
            inputs.push(PathBuf::from(arg));
        }
    }
    if inputs.is_empty() {
        return Err("usage: dicom_rs <file.dcm>... [-o <dir>] [--window <center> <width>]".into());
    }
    std::fs::create_dir_all(&out_dir)?;
    let mut failed = 0;
    for input in &inputs {
        if let Err(e) = convert(input, &out_dir, &options) {
            eprintln!("{}: {e}", input.display());
            failed += 1;
        }
    }
    if failed > 0 {
        return Err(format!("{failed} of {} files failed", inputs.len()).into());
    }
    Ok(())
}
