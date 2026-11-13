//! Sanity checks of the test data and of the adapter plumbing that do not
//! depend on any codec being implemented.

mod common;

use std::path::Path;

use dicom_encoding::adapters::{PixelDataObject, PixelDataReader};
use dicom_jpeg::Codec;
use dicom_jpeg::dicom_rs::frame_codestream;

fn dicom_files() -> Vec<String> {
    fn walk(dir: &Path, base: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, base, out);
            } else if !matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("txt" | "rst" | "md")
            ) {
                out.push(
                    path.strip_prefix(base)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    let mut out = Vec::new();
    walk(&base.join("dicom"), &base, &mut out);
    out.sort();
    assert!(
        out.len() > 100,
        "test data missing? found {} files",
        out.len()
    );
    out
}

/// Every test file is readable by dicom-rs.
#[test]
fn all_files_open() {
    let failures: Vec<String> = dicom_files()
        .into_iter()
        .filter_map(|rel| {
            dicom_object::OpenFileOptions::new()
                .open_file(common::testdata(&rel))
                .err()
                .map(|e| format!("{rel}: {e}"))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "unreadable files:\n{}",
        failures.join("\n")
    );
}

/// Every encapsulated JPEG-family file opens with dicom-rs and every frame
/// extracted by the adapter starts with the codestream signature.
#[test]
fn frames_extract_with_codestream_signature() {
    let mut checked = 0;
    for rel in dicom_files() {
        let obj = common::open(&rel);
        let Some(adapter) =
            dicom_jpeg::dicom_rs::PixelAdapter::from_uid(obj.meta().transfer_syntax())
        else {
            continue; // native reference image
        };
        let codec = adapter.transfer_syntax().codec();
        let frames = common::attrs(&obj).frames;
        for f in 0..frames {
            let data = frame_codestream(&obj, codec, f)
                .unwrap_or_else(|e| panic!("{rel}: frame {f}: {e}"));
            let ok = match codec {
                Codec::Jpeg | Codec::JpegLs => data.starts_with(&[0xFF, 0xD8]),
                Codec::Jpeg2000 | Codec::HtJ2k => {
                    data.starts_with(&[0xFF, 0x4F, 0xFF, 0x51]) || data[4..8] == *b"jP  "
                }
                Codec::JpegXl => data.starts_with(&[0xFF, 0x0A]) || data[4..8] == *b"JXL ",
                _ => unreachable!(),
            };
            assert!(
                ok,
                "{rel}: frame {f} does not start with a {codec} signature: {:02X?}",
                &data[..8.min(data.len())]
            );
        }
        assert!(
            frame_codestream(&obj, codec, frames).is_err(),
            "{rel}: frame index past the end must fail"
        );
        checked += 1;
    }
    assert!(checked >= 140, "only {checked} encapsulated files checked");
}

/// Reference images are native and readable through `PixelDataObject`.
#[test]
fn references_are_native() {
    for rel in dicom_files()
        .into_iter()
        .filter(|r| r.contains("/REF/") || r.contains("_UNC"))
    {
        let obj = common::open(&rel);
        assert!(obj.number_of_fragments().is_none(), "{rel} is encapsulated");
        let a = common::attrs(&obj);
        let expected = (a.rows * a.cols * a.spp * (a.bits_allocated / 8)) as usize;
        for f in 0..a.frames {
            assert_eq!(
                common::native_frame(&obj, f).len(),
                expected,
                "{rel}: frame {f}"
            );
        }
    }
}

/// Codecs that are not implemented yet must report an error, never panic.
#[test]
fn unimplemented_codecs_return_errors() {
    for rel in dicom_files() {
        let obj = common::open(&rel);
        let Some(adapter) =
            dicom_jpeg::dicom_rs::PixelAdapter::from_uid(obj.meta().transfer_syntax())
        else {
            continue;
        };
        let mut out = Vec::new();
        if let Err(e) = adapter.decode_frame(&obj, 0, &mut out) {
            let msg = e.to_string();
            assert!(!msg.is_empty(), "{rel}: empty error message");
        }
    }
}
