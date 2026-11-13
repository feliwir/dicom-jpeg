//! Shared helpers for the DICOM integration tests.
//!
//! All test data lives in `testdata/` (Git LFS). See `testdata/SOURCES.md`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

#[cfg(feature = "encode")]
use dicom_core::ops::ApplyOp;
#[cfg(feature = "encode")]
use dicom_core::value::{PixelFragmentSequence, Value};
#[cfg(feature = "encode")]
use dicom_core::{DataElement, VR};
use dicom_dictionary_std::tags;
#[cfg(feature = "encode")]
use dicom_encoding::adapters::{EncodeOptions, PixelDataWriter};
use dicom_encoding::adapters::{PixelDataObject, PixelDataReader};
#[cfg(feature = "encode")]
use dicom_jpeg::TransferSyntax;
use dicom_jpeg::dicom_rs::PixelAdapter;
use dicom_object::{DefaultDicomObject, OpenFileOptions};

/// How a decoded image must relate to its reference.
#[derive(Debug, Clone, Copy)]
pub enum Expect {
    /// Bit-exact match (after masking to Bits Stored).
    Exact,
    /// JPEG-LS near-lossless: max |error| <= NEAR read from the codestream.
    JlsNear,
    /// Max |error| <= the given bound.
    MaxAbs(u32),
    /// PSNR (dB, peak = 2^BitsStored - 1) at least the given value.
    Psnr(f64),
    /// Only check that every frame decodes with the declared geometry.
    Decodes,
    /// Decoding must fail with an error (and must not panic).
    Fails,
}

/// Absolute path of a file below `testdata/`, verifying it is real data
/// and not a Git LFS pointer.
pub fn testdata(rel: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(rel);
    let head = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("cannot read test file {}: {e}", path.display()));
    if head.starts_with(b"version https://git-lfs.github.com/spec") {
        panic!(
            "{} is a Git LFS pointer; run `git lfs pull` to fetch the test data",
            path.display()
        );
    }
    path
}

pub fn open(rel: &str) -> DefaultDicomObject {
    OpenFileOptions::new()
        .open_file(testdata(rel))
        .unwrap_or_else(|e| panic!("cannot open DICOM file {rel}: {e}"))
}

fn uint(obj: &DefaultDicomObject, tag: dicom_core::Tag) -> Option<u32> {
    obj.get(tag)?.to_int::<u32>().ok()
}

/// Image attributes needed for comparisons.
#[derive(Debug, Clone, Copy)]
pub struct Attrs {
    pub rows: u32,
    pub cols: u32,
    pub spp: u32,
    pub bits_allocated: u32,
    pub bits_stored: u32,
    pub signed: bool,
    pub frames: u32,
}

pub fn attrs(obj: &DefaultDicomObject) -> Attrs {
    Attrs {
        rows: uint(obj, tags::ROWS).expect("Rows"),
        cols: uint(obj, tags::COLUMNS).expect("Columns"),
        spp: uint(obj, tags::SAMPLES_PER_PIXEL).unwrap_or(1),
        bits_allocated: uint(obj, tags::BITS_ALLOCATED).expect("BitsAllocated"),
        bits_stored: uint(obj, tags::BITS_STORED).expect("BitsStored"),
        signed: uint(obj, tags::PIXEL_REPRESENTATION).unwrap_or(0) == 1,
        frames: obj
            .get(tags::NUMBER_OF_FRAMES)
            .and_then(|e| e.to_int::<u32>().ok())
            .unwrap_or(1)
            .max(1),
    }
}

/// Adapter for the object's own transfer syntax.
pub fn adapter_for(obj: &DefaultDicomObject) -> PixelAdapter {
    let uid = obj.meta().transfer_syntax();
    PixelAdapter::from_uid(uid)
        .unwrap_or_else(|| panic!("not a JPEG-family transfer syntax: {uid}"))
}

/// Decode every frame through the dicom-rs adapter.
pub fn decode_frames(
    obj: &DefaultDicomObject,
) -> Result<Vec<Vec<u8>>, dicom_encoding::adapters::DecodeError> {
    let adapter = adapter_for(obj);
    let a = attrs(obj);
    (0..a.frames)
        .map(|f| {
            let mut out = Vec::new();
            adapter.decode_frame(obj, f, &mut out)?;
            Ok(out)
        })
        .collect()
}

/// Native pixel data of one frame of an uncompressed object.
pub fn native_frame(obj: &DefaultDicomObject, frame: u32) -> Vec<u8> {
    obj.frame_pixel_data(frame)
        .unwrap_or_else(|| panic!("reference frame {frame} missing"))
        .into_owned()
}

/// Interpret little-endian samples as integers, masked to `bits` and
/// sign-extended if `signed`.
pub fn samples(bytes: &[u8], bits_allocated: u32, bits: u32, signed: bool) -> Vec<i32> {
    let mask: u32 = if bits >= 32 {
        u32::MAX
    } else {
        (1 << bits) - 1
    };
    let conv = |v: u32| -> i32 {
        let v = v & mask;
        if signed && bits < 32 && v & (1 << (bits - 1)) != 0 {
            (v | !mask) as i32
        } else {
            v as i32
        }
    };
    match bits_allocated {
        8 => bytes.iter().map(|&b| conv(b as u32)).collect(),
        16 => bytes
            .chunks_exact(2)
            .map(|c| conv(u16::from_le_bytes([c[0], c[1]]) as u32))
            .collect(),
        other => panic!("unsupported Bits Allocated {other}"),
    }
}

pub fn psnr(a: &[i32], b: &[i32], bits: u32) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    let peak = ((1u64 << bits) - 1) as f64;
    10.0 * (peak * peak / mse).log10()
}

/// NEAR parameter of the first scan of a JPEG-LS codestream.
pub fn jls_near(codestream: &[u8]) -> u32 {
    let mut i = 2;
    while i + 4 <= codestream.len() {
        if codestream[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker = codestream[i + 1];
        let len = u16::from_be_bytes([codestream[i + 2], codestream[i + 3]]) as usize;
        if marker == 0xDA {
            let ns = codestream[i + 4] as usize;
            return codestream[i + 5 + 2 * ns] as u32;
        }
        i += 2 + len;
    }
    panic!("no SOS marker found in JPEG-LS codestream")
}

/// Compare decoded frames against reference frames.
pub fn compare(
    name: &str,
    decoded: &[Vec<u8>],
    reference: &[Vec<u8>],
    dec: Attrs,
    reference_attrs: Attrs,
    expect: Expect,
    near: impl Fn() -> u32,
) {
    assert_eq!(decoded.len(), reference.len(), "{name}: frame count");
    let bits = dec.bits_stored.min(reference_attrs.bits_stored);
    let signed = reference_attrs.signed;
    for (f, (d, r)) in decoded.iter().zip(reference).enumerate() {
        let d = samples(d, dec.bits_allocated, bits, signed);
        let r = samples(r, reference_attrs.bits_allocated, bits, signed);
        assert_eq!(d.len(), r.len(), "{name}: frame {f}: sample count");
        let max_abs = d
            .iter()
            .zip(&r)
            .map(|(&a, &b)| (a - b).unsigned_abs())
            .max()
            .unwrap_or(0);
        let p = psnr(&d, &r, bits);
        eprintln!("{name}: frame {f}: max |err| = {max_abs}, PSNR = {p:.2} dB");
        match expect {
            Expect::Exact => {
                if let Some(i) = d.iter().zip(&r).position(|(a, b)| a != b) {
                    panic!(
                        "{name}: frame {f}: first mismatch at sample {i}: decoded {} reference {} (max |err| {max_abs})",
                        d[i], r[i]
                    );
                }
            }
            Expect::JlsNear => {
                let n = near();
                assert!(
                    max_abs <= n,
                    "{name}: frame {f}: max |err| {max_abs} > NEAR {n}"
                );
            }
            Expect::MaxAbs(n) => {
                assert!(max_abs <= n, "{name}: frame {f}: max |err| {max_abs} > {n}");
            }
            Expect::Psnr(min) => {
                assert!(p >= min, "{name}: frame {f}: PSNR {p:.2} dB < {min} dB");
            }
            Expect::Decodes | Expect::Fails => unreachable!(),
        }
    }
}

/// Decode `rel` through the adapter and check it against `reference`.
pub fn check_decode(rel: &str, reference: Option<&str>, expect: Expect) {
    let obj = open(rel);
    let a = attrs(&obj);
    let decoded = decode_frames(&obj);
    let decoded = match (expect, decoded) {
        (Expect::Fails, Err(e)) => {
            eprintln!("{rel}: failed as expected: {e}");
            return;
        }
        (Expect::Fails, Ok(_)) => panic!("{rel}: decoding succeeded but was expected to fail"),
        (_, Err(e)) => panic!("{rel}: decoding failed: {e}"),
        (_, Ok(d)) => d,
    };
    let frame_bytes = (a.rows * a.cols * a.spp * (a.bits_allocated / 8)) as usize;
    for (f, d) in decoded.iter().enumerate() {
        assert_eq!(d.len(), frame_bytes, "{rel}: frame {f}: decoded size");
    }
    if let Expect::Decodes = expect {
        return;
    }
    let reference = reference.unwrap_or_else(|| panic!("{rel}: expectation needs a reference"));
    let ref_obj = open(reference);
    let ra = attrs(&ref_obj);
    assert_eq!(
        (a.rows, a.cols, a.spp),
        (ra.rows, ra.cols, ra.spp),
        "{rel}: geometry vs reference"
    );
    let ref_frames: Vec<Vec<u8>> = (0..ra.frames).map(|f| native_frame(&ref_obj, f)).collect();
    let codec_data = || {
        let data = dicom_jpeg::dicom_rs::frame_codestream(&obj, dicom_jpeg::Codec::JpegLs, 0)
            .expect("frame codestream");
        jls_near(&data)
    };
    compare(rel, &decoded, &ref_frames, a, ra, expect, codec_data);
}

/// Encode the native object `reference` to `ts` through the adapter, write it
/// as a complete DICOM file with dicom-rs, read it back, decode it through the
/// adapter and compare with the original pixels.
#[cfg(feature = "encode")]
pub fn check_roundtrip(reference: &str, ts: TransferSyntax, quality: Option<u8>, expect: Expect) {
    let original = open(reference);
    let oa = attrs(&original);
    let adapter = PixelAdapter::new(ts);

    let mut options = EncodeOptions::new();
    options.quality = quality;
    let mut fragments = Vec::new();
    let mut offset_table = Vec::new();
    let ops = adapter
        .encode(&original, options, &mut fragments, &mut offset_table)
        .unwrap_or_else(|e| panic!("{reference} -> {ts}: encoding failed: {e}"));
    let encoded_size: usize = fragments.iter().map(Vec::len).sum();

    let mut obj = original.clone();
    for op in ops {
        obj.apply(op).expect("apply attribute op");
    }
    obj.put(DataElement::new(
        tags::PIXEL_DATA,
        VR::OB,
        Value::PixelSequence(PixelFragmentSequence::new(offset_table, fragments)),
    ));
    obj.update_meta(|meta| meta.transfer_syntax = ts.uid().to_string());
    let mut file = Vec::new();
    obj.write_all(&mut file)
        .unwrap_or_else(|e| panic!("{reference} -> {ts}: writing DICOM failed: {e}"));
    let reread = OpenFileOptions::new()
        .from_reader(&file[..])
        .unwrap_or_else(|e| panic!("{reference} -> {ts}: re-reading DICOM failed: {e}"));
    assert_eq!(
        reread.meta().transfer_syntax().trim_end_matches('\0'),
        ts.uid()
    );

    let decoded = decode_frames(&reread)
        .unwrap_or_else(|e| panic!("{reference} -> {ts}: decoding failed: {e}"));
    let ra = attrs(&reread);
    let ref_frames: Vec<Vec<u8>> = (0..oa.frames).map(|f| native_frame(&original, f)).collect();
    let raw_size: usize = ref_frames.iter().map(Vec::len).sum();
    eprintln!(
        "{reference} -> {ts}: {raw_size} -> {encoded_size} bytes (ratio {:.2})",
        raw_size as f64 / encoded_size.max(1) as f64
    );
    let name = format!("{reference} -> {ts}");
    let near = || {
        let data = dicom_jpeg::dicom_rs::frame_codestream(&reread, dicom_jpeg::Codec::JpegLs, 0)
            .expect("frame codestream");
        jls_near(&data)
    };
    compare(&name, &decoded, &ref_frames, ra, oa, expect, near);
}

/// Generate one `#[test]` per case. The first token is the codec gate: a
/// codec that is not implemented yet produces `#[ignore]`d tests, so
/// `cargo test -- --ignored` lists the remaining work.
#[macro_export]
macro_rules! gate {
    (ready, $feature:literal, $name:ident, $body:block) => {
        #[cfg(feature = $feature)]
        #[test]
        fn $name() $body
    };
    (pending, $feature:literal, $name:ident, $body:block) => {
        #[cfg(feature = $feature)]
        #[test]
        #[ignore = "codec not implemented yet"]
        fn $name() $body
    };
}
