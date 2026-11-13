//! Shared helpers for the DICOM integration tests.
//!
//! All test data lives in `testdata/` (Git LFS). See `testdata/SOURCES.md`.

#![allow(dead_code)]

pub mod pnm;

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
    /// Display transform (rescale, window, MONOCHROME1) for dumped images.
    pub voi: pnm::Voi,
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
        voi: pnm::voi_from_dicom(obj),
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

/// Write one frame below the dump folder when `DICOM_JPEG_DUMP` is set
/// (see [`pnm::dump_dir`]). With `display`, the samples are mapped to a
/// viewable 8-bit image using the object's rescale / window / MONOCHROME1
/// attributes; otherwise they are written as-is with `bits` bits.
pub fn dump_frame(
    name: &str,
    frame: usize,
    frames: usize,
    a: Attrs,
    bits: u32,
    display: bool,
    samples: &[i32],
) {
    let Some(dir) = pnm::dump_dir() else { return };
    let mut file = pnm::slug_path(name);
    if frames > 1 {
        file.push_str(&format!("_f{frame:03}"));
    }
    let path = dir.join(file);
    let (bits, samples) = if display {
        (8, pnm::to_display(samples, a.spp as usize, bits, &a.voi))
    } else {
        (bits, samples.to_vec())
    };
    let written = pnm::write_pnm(
        &path,
        a.cols as usize,
        a.rows as usize,
        a.spp as usize,
        bits,
        &samples,
    )
    .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
    for p in written {
        eprintln!("wrote {}", p.display());
    }
}

/// Dump location of a decoded test file: `decode/<set>/<file>`.
fn decode_dump_name(rel: &str) -> String {
    let rel = rel.strip_prefix("dicom/").unwrap_or(rel);
    format!("decode/{}", rel.strip_suffix(".dcm").unwrap_or(rel))
}

/// Compare decoded frames against reference frames; `dump_as` names the
/// images written when `DICOM_JPEG_DUMP` is set.
#[allow(clippy::too_many_arguments)]
pub fn compare(
    name: &str,
    dump_as: &str,
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
        dump_frame(dump_as, f, decoded.len(), dec, bits, true, &d);
        if max_abs > 0 {
            // Difference image, normalised so the largest error is white.
            let diff: Vec<i32> = d
                .iter()
                .zip(&r)
                .map(|(&a, &b)| ((a - b).unsigned_abs() * 255 / max_abs) as i32)
                .collect();
            dump_frame(
                &format!("{dump_as}_diff"),
                f,
                decoded.len(),
                dec,
                8,
                false,
                &diff,
            );
        }
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
        for (f, d) in decoded.iter().enumerate() {
            let v = samples(d, a.bits_allocated, a.bits_stored, a.signed);
            dump_frame(
                &decode_dump_name(rel),
                f,
                decoded.len(),
                a,
                a.bits_stored,
                true,
                &v,
            );
        }
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
    compare(
        rel,
        &decode_dump_name(rel),
        &decoded,
        &ref_frames,
        a,
        ra,
        expect,
        codec_data,
    );
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
    if let Some(dir) = pnm::dump_dir() {
        // Keep the encoded DICOM file for inspection with other tools.
        let stem = Path::new(reference)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let path = dir.join("roundtrip").join(format!("{stem}__{ts:?}.dcm"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &file).unwrap();
    }
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
    let stem = Path::new(reference)
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let dump_as = format!("roundtrip/{stem}__{ts:?}");
    compare(&name, &dump_as, &decoded, &ref_frames, ra, oa, expect, near);
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

/// MD5 (RFC 1321), used to compare with reference digests published by
/// other projects (e.g. libjpeg-turbo's test suite).
pub fn md5_hex(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32)
        .collect();
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    for chunk in msg.chunks_exact(64) {
        let m: Vec<u32> = chunk
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    [a0, b0, c0, d0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn md5_known_vectors() {
    assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(
        md5_hex(b"The quick brown fox jumps over the lazy dog"),
        "9e107d9d372bb6826bd81d3542a419d6"
    );
}
