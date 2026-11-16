//! Shared data loading and Criterion helpers for the dicom-jpeg benchmarks.
//!
//! Every benchmark group compares implementations on the *same* inputs, one
//! iteration being the whole input set; throughput is reported in pixels.
//! The first implementation of a group is dicom-jpeg itself. Inputs that an
//! implementation cannot handle are dropped for all of them, unless a
//! reference fails on most inputs (a missing feature), in which case it is
//! skipped; both are reported on stderr. All implementations run
//! single-threaded.

pub mod j2k;
pub mod openjpeg;
pub mod turbojpeg;

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{Criterion, SamplingMode, Throughput};
use dicom_dictionary_std::tags;
use dicom_encoding::adapters::PixelDataObject;
use dicom_jpeg::dicom_rs::{PixelAdapter, frame_codestream};
use dicom_jpeg::{Codec, ColorSpace, Image, ImageInfo, Samples};

/// Criterion settings shared by all benchmarks: the minimum of 10 flat
/// samples, since one iteration decodes or encodes a whole test set, and a
/// short warm-up, since every implementation has already run once over all
/// inputs while the inputs were selected, and 10k bootstrap resamples
/// instead of 100k, which halves the analysis and plotting time.
pub fn criterion() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .nresamples(10_000)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_secs(1))
}

/// Largest image (pixels per frame) taken from WG04. The larger images
/// (MG1 mammography, RG radiographs, SC1, VL4/VL5 scans) would make one
/// iteration take seconds for the slower codecs; all modalities, bit depths
/// and colour types remain represented below this size.
pub const MAX_FRAME_PIXELS: u64 = 1024 * 1024;

/// Repository root.
pub fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
}

/// One codestream to decode.
#[derive(Debug, Clone)]
pub struct Input {
    /// File (and frame) the codestream came from.
    pub name: String,
    /// The codestream.
    pub data: Vec<u8>,
    /// Pixels per frame.
    pub pixels: u64,
}

fn sorted_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e} (run `git lfs pull`)", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    paths.sort();
    paths
}

/// All frames of the encapsulated DICOM files `files`, with their codec.
pub fn dicom_frames(files: &[PathBuf]) -> Vec<(Codec, Input)> {
    let mut out = Vec::new();
    for p in files {
        let obj = dicom_object::OpenFileOptions::new()
            .open_file(p)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let Some(adapter) = PixelAdapter::from_uid(obj.meta().transfer_syntax()) else {
            continue;
        };
        let codec = adapter.codec();
        let n = obj.number_of_frames().unwrap_or(1).max(1);
        let pixels = obj.rows().unwrap_or(0) as u64 * obj.cols().unwrap_or(0) as u64;
        let stem = p.file_stem().unwrap_or_default().to_string_lossy();
        for f in 0..n {
            let cs = frame_codestream(&obj, codec, f).expect("frame codestream");
            let name = if n > 1 {
                format!("{stem}#{f}")
            } else {
                stem.to_string()
            };
            out.push((
                codec,
                Input {
                    name,
                    data: cs.into_owned(),
                    pixels,
                },
            ));
        }
    }
    out
}

/// All frames of at most [`MAX_FRAME_PIXELS`] of the NEMA WG04 set `set`
/// (e.g. `"JPLL"`).
pub fn wg04(set: &str) -> Vec<Input> {
    let files = sorted_files(&root().join("testdata/dicom/wg04").join(set));
    dicom_frames(&files)
        .into_iter()
        .map(|(_, i)| i)
        .filter(|i| i.pixels <= MAX_FRAME_PIXELS)
        .collect()
}

/// The WG04 uncompressed reference images of at most [`MAX_FRAME_PIXELS`]
/// (first frame each, gray or RGB), with their file names.
pub fn references() -> Vec<(String, Image)> {
    let mut out = Vec::new();
    for p in sorted_files(&root().join("testdata/dicom/wg04/REF")) {
        let obj = dicom_object::OpenFileOptions::new()
            .open_file(&p)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let int = |tag| -> Option<u32> { obj.get(tag)?.to_int::<u32>().ok() };
        let rows = int(tags::ROWS).unwrap_or(0);
        let cols = int(tags::COLUMNS).unwrap_or(0);
        let spp = int(tags::SAMPLES_PER_PIXEL).unwrap_or(1) as u16;
        let ba = int(tags::BITS_ALLOCATED).unwrap_or(8);
        let bs = int(tags::BITS_STORED).unwrap_or(ba) as u8;
        let pi = obj
            .get(tags::PHOTOMETRIC_INTERPRETATION)
            .and_then(|e| e.to_str().ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if !matches!(pi.as_str(), "MONOCHROME1" | "MONOCHROME2" | "RGB")
            || rows as u64 * cols as u64 > MAX_FRAME_PIXELS
        {
            continue;
        }
        let bytes = obj
            .element(tags::PIXEL_DATA)
            .expect("pixel data")
            .to_bytes()
            .expect("native pixel data");
        let frame = (rows * cols * spp as u32 * (ba / 8)) as usize;
        let info = ImageInfo {
            width: cols,
            height: rows,
            components: spp,
            bits_per_sample: bs,
            signed: false,
            color_space: if spp == 3 {
                ColorSpace::Rgb
            } else {
                ColorSpace::Gray
            },
        };
        if bytes.len() >= frame
            && let Ok(img) = Image::from_le_bytes(info, &bytes[..frame], (ba / 8) as usize)
        {
            let name = p.file_stem().unwrap_or_default().to_string_lossy();
            out.push((name.trim_end_matches("_UNC").to_string(), img));
        }
    }
    out
}

/// Pixels of `img`.
pub fn pixels(img: &Image) -> u64 {
    img.info.width as u64 * img.info.height as u64
}

/// Stored size of `img` in bytes (8-bit samples as 1 byte, others as 2).
pub fn raw_bytes(img: &Image) -> usize {
    match &img.samples {
        Samples::U8(s) => s.len(),
        Samples::U16(s) => s.len() * 2,
    }
}

/// A decoder under test: codestream -> number of decoded samples.
pub type DecodeFn<'a> = Box<dyn FnMut(&[u8]) -> Result<usize, String> + 'a>;

/// An encoder under test: image -> codestream.
pub type EncodeFn<'a> = Box<dyn FnMut(&Image) -> Result<Vec<u8>, String> + 'a>;

/// Which inputs and implementations take part in a group.
#[derive(Debug)]
struct Selection {
    /// Indices of the inputs every active implementation handles.
    inputs: Vec<usize>,
    /// Per implementation: `Some(output sizes over the kept inputs)` if it
    /// takes part.
    active: Vec<Option<Vec<usize>>>,
}

/// Run every implementation once on every input. Inputs dicom-jpeg (the
/// first implementation) rejects are dropped; a reference that fails on more
/// than half of the rest lacks the feature and is skipped, otherwise the
/// inputs it fails on are dropped for everyone, so all implementations are
/// timed on identical inputs.
fn select(
    group: &str,
    names: &[&str],
    labels: &[&str],
    mut run: impl FnMut(usize, usize) -> Result<usize, String>,
) -> Selection {
    let results: Vec<Vec<Result<usize, String>>> = (0..names.len())
        .map(|k| (0..labels.len()).map(|i| run(k, i)).collect())
        .collect();
    let mut keep: Vec<bool> = results[0].iter().map(Result::is_ok).collect();
    for (i, r) in results[0].iter().enumerate() {
        if let Err(e) = r {
            eprintln!("{group}: dropping {} ({}: {e})", labels[i], names[0]);
        }
    }
    let base = keep.clone();
    let mut active = vec![true; names.len()];
    for k in 1..names.len() {
        let fails: Vec<usize> = (0..labels.len())
            .filter(|&i| base[i] && results[k][i].is_err())
            .collect();
        let total = base.iter().filter(|&&b| b).count();
        if fails.len() * 2 > total {
            let e = results[k][fails[0]].as_ref().unwrap_err();
            eprintln!(
                "{group}: skipping {} (fails on {}/{total} inputs, e.g. {}: {e})",
                names[k],
                fails.len(),
                labels[fails[0]]
            );
            active[k] = false;
        } else {
            for i in fails {
                let e = results[k][i].as_ref().unwrap_err();
                eprintln!("{group}: dropping {} ({}: {e})", labels[i], names[k]);
                keep[i] = false;
            }
        }
    }
    let inputs: Vec<usize> = (0..labels.len()).filter(|&i| keep[i]).collect();
    let active = (0..names.len())
        .map(|k| {
            active[k].then(|| {
                inputs
                    .iter()
                    .map(|&i| *results[k][i].as_ref().unwrap())
                    .collect()
            })
        })
        .collect();
    Selection { inputs, active }
}

/// Benchmark the decoders `impls` (dicom-jpeg first) on `inputs` as group
/// `group`.
pub fn decode_group(
    c: &mut Criterion,
    group: &str,
    inputs: &[Input],
    mut impls: Vec<(&str, DecodeFn)>,
) {
    let names: Vec<&str> = impls.iter().map(|i| i.0).collect();
    let labels: Vec<&str> = inputs.iter().map(|i| i.name.as_str()).collect();
    let sel = select(group, &names, &labels, |k, i| (impls[k].1)(&inputs[i].data));
    if sel.inputs.is_empty() {
        eprintln!("{group}: no inputs");
        return;
    }
    let kept: Vec<&Input> = sel.inputs.iter().map(|&i| &inputs[i]).collect();
    let px: u64 = kept.iter().map(|i| i.pixels).sum();
    eprintln!(
        "{group}: {} codestreams, {:.2} Mpixel, {:.2} MB",
        kept.len(),
        px as f64 / 1e6,
        kept.iter().map(|i| i.data.len()).sum::<usize>() as f64 / 1e6
    );
    let mut g = c.benchmark_group(group);
    g.sampling_mode(SamplingMode::Flat);
    g.throughput(Throughput::Elements(px));
    for ((name, f), active) in impls.iter_mut().zip(&sel.active) {
        if active.is_some() {
            g.bench_function(*name, |b| {
                b.iter(|| {
                    for i in &kept {
                        black_box(f(black_box(&i.data)).ok());
                    }
                })
            });
        }
    }
    g.finish();
}

/// Benchmark the encoders `impls` (dicom-jpeg first) on `images` as group
/// `group`; the compression ratio of each encoder is printed to stderr.
pub fn encode_group(
    c: &mut Criterion,
    group: &str,
    images: &[(String, Image)],
    mut impls: Vec<(&str, EncodeFn)>,
) {
    let names: Vec<&str> = impls.iter().map(|i| i.0).collect();
    let labels: Vec<&str> = images.iter().map(|i| i.0.as_str()).collect();
    let sel = select(group, &names, &labels, |k, i| {
        (impls[k].1)(&images[i].1).map(|d| d.len())
    });
    if sel.inputs.is_empty() {
        eprintln!("{group}: no inputs");
        return;
    }
    let kept: Vec<&Image> = sel.inputs.iter().map(|&i| &images[i].1).collect();
    let px: u64 = kept.iter().map(|i| pixels(i)).sum();
    let raw: usize = kept.iter().map(|i| raw_bytes(i)).sum();
    eprintln!(
        "{group}: {} images, {:.2} Mpixel, {:.2} MB raw",
        kept.len(),
        px as f64 / 1e6,
        raw as f64 / 1e6
    );
    let mut g = c.benchmark_group(group);
    g.sampling_mode(SamplingMode::Flat);
    g.throughput(Throughput::Elements(px));
    for ((name, f), active) in impls.iter_mut().zip(&sel.active) {
        let Some(sizes) = active else { continue };
        let size: usize = sizes.iter().sum();
        eprintln!(
            "{group}: {name} compression ratio {:.3}",
            raw as f64 / size as f64
        );
        g.bench_function(*name, |b| {
            b.iter(|| {
                for img in &kept {
                    black_box(f(black_box(img)).ok());
                }
            })
        });
    }
    g.finish();
}

/// dicom-jpeg's decoder for `codec`, delivering RGB for colour data so the
/// work matches decoders that always convert.
pub fn ours_decode<'a>(codec: Codec) -> DecodeFn<'a> {
    let opts = dicom_jpeg::DecodeOptions::new().with_color_output(dicom_jpeg::ColorOutput::Rgb);
    Box::new(move |d| {
        dicom_jpeg::decode(codec, d, &opts)
            .map(|img| img.samples.len())
            .map_err(|e| e.to_string())
    })
}
