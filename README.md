# dicom-jpeg

Pure-Rust encoders and decoders for every active JPEG-family DICOM transfer syntax.
Each codec sits behind its own cargo feature.

| Transfer syntax UID (1.2.840.10008.1.2.4.x) | Codec | Feature | Status |
|---|---|---|---|
| .50 .51 .57 .70 | JPEG baseline / extended / lossless (T.81) | `jpeg` | decoder ✔ encoder ✔ |
| .80 .81 | JPEG-LS (T.87) | `jpeg-ls` | decoder ✔ encoder ✔ |
| .90 .91 | JPEG 2000 (15444-1) | `jpeg2000` | decoder ✔ encoder ✔ |
| .92 .93 | JPEG 2000 Part 2 multi-component | `jpeg2000-part2` | decoder ✔ encoder ✔ |
| .201 .202 .203 | HTJ2K (15444-15) | `htj2k` | decoder ✔ encoder ✔ |
| .110 .111 .112 | JPEG XL (18181) | `jpeg-xl` | decoder: modular and VarDCT (sRGB, no ICC, no patches/noise/splines/blending); encoder: lossless |

Other features:

* `encode`: enables the encoders.
* `dicom`: dicom-rs `PixelDataReader` / `PixelDataWriter` adapters (`dicom_jpeg::dicom_rs::PixelAdapter`, crate `dicom-jpeg-dicom`).
* `inventory-registry`: also register the adapters in the dicom-rs transfer syntax registry (see below).
* `parallel`: rayon parallelism.

## Using it with dicom-rs

With the `inventory-registry` feature, dicom-rs itself uses dicom-jpeg, e.g.
in `dicom_pixeldata::PixelDecoder::decode_pixel_data`:

```toml
[dependencies]
dicom = "0.10"
dicom-jpeg = { git = "https://github.com/feliwir/dicom-jpeg", features = ["inventory-registry"] }
```

If nothing else in your code refers to dicom-jpeg, add `use dicom_jpeg as _;`
so that the crate is linked and its registrations are seen.

dicom-rs keeps a built-in codec when it already decodes and encodes a
transfer syntax. In dicom-rs 0.10 that is only JPEG baseline (.50), with the
`jpeg` feature that dicom-pixeldata's default `native` feature enables; every
other JPEG-family syntax goes to dicom-jpeg. To have dicom-jpeg handle JPEG
baseline as well, use dicom-pixeldata without `native` (keeping RLE and
deflate):

```toml
[dependencies]
dicom = { version = "0.10", default-features = false, features = ["inventory-registry", "ul"] }
dicom-pixeldata = { version = "0.10", default-features = false, features = ["rle", "deflate", "rayon"] }
dicom-jpeg = { git = "https://github.com/feliwir/dicom-jpeg", features = ["inventory-registry"] }
```

Without the feature, use the adapter directly:
`dicom_jpeg::dicom_rs::PixelAdapter::from_uid(ts)` implements
`PixelDataReader` and `PixelDataWriter`.

## Crates

The repository is a virtual cargo workspace. `crates/dicom-jpeg` is the
facade: it re-exports the crates below behind the features above (the
adapter as `dicom_jpeg::dicom_rs`) and dispatches decoding by `Codec`. It
also holds the integration tests, fuzz targets and test data.

| Crate | Contents |
|---|---|
| `crates/common` (`dicom-jpeg-common`) | `Image`, options, limits, errors, `Codec`; canonical Huffman code construction and colour transforms (YCbCr / ICT, RCT) shared by the codecs |
| `crates/jpeg` (`dicom-jpeg-jpeg`) | ITU-T T.81 |
| `crates/jpegls` (`dicom-jpeg-jpegls`) | ITU-T T.87 |
| `crates/j2k` (`dicom-jpeg-j2k`) | JPEG 2000 with the `htj2k` and `part2` features |
| `crates/jxl` (`dicom-jpeg-jxl`) | JPEG XL |
| `crates/dicom` (`dicom-jpeg-dicom`) | dicom-rs `PixelDataReader` / `PixelDataWriter` adapter; maps transfer syntax UIDs (`dicom_dictionary_std::uids`) to a codec and its parameters |

The codec crates know nothing about DICOM; each takes its own
`EncoderOptions` and can be used on its own.

## Testing

Test data is stored with Git LFS. Its provenance is documented in `testdata/SOURCES.md`.

```sh
git lfs pull
cargo test --all-features             # implemented codecs
cargo test --all-features -- --ignored  # cases for codecs that are not implemented yet
```

`crates/dicom-jpeg/tests/dicom_decode.rs` decodes every NEMA WG04, pydicom and pylibjpeg file through the dicom-rs adapter and compares it with the uncompressed reference.
`crates/dicom-jpeg/tests/dicom_roundtrip.rs` encodes native references, writes and re-reads a DICOM file with dicom-rs, then decodes and compares the result.

## Benchmarks

`benchmarks/` holds Criterion benchmarks against reference implementations
that have Rust bindings: libjpeg-turbo, zune-jpeg, jpeg-decoder, CharLS,
OpenJPEG, openjp2, hayro-jpeg2000, libjxl, jxl-oxide, jxl-rs and zune-jpegxl.
They are all single-threaded and run on the WG04 test set. See
[benchmarks/README.md](benchmarks/README.md).

```sh
cargo bench -p dicom-jpeg-benchmarks
```

## License

MIT OR Apache-2.0
