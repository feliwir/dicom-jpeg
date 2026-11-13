# dicom-jpeg

Pure-Rust encoders and decoders for every active JPEG-family DICOM transfer syntax.
Each codec sits behind its own cargo feature.

| Transfer syntax UID (1.2.840.10008.1.2.4.x) | Codec | Feature | Status |
|---|---|---|---|
| .50 .51 .57 .70 | JPEG baseline / extended / lossless (T.81) | `jpeg` | decoder ✔ encoder ✔ |
| .80 .81 | JPEG-LS (T.87) | `jpeg-ls` | decoder ✔ encoder ✔ |
| .90 .91 | JPEG 2000 (15444-1) | `jpeg2000` | planned (phase 4–5) |
| .92 .93 | JPEG 2000 Part 2 multi-component | `jpeg2000-part2` | planned (phase 7) |
| .201 .202 .203 | HTJ2K (15444-15) | `htj2k` | planned (phase 6) |
| .110 .111 .112 | JPEG XL (18181) | `jpeg-xl` | planned (phase 8–9) |

Other features:

* `encode`: enables the encoders.
* `dicom`: dicom-rs `PixelDataReader` / `PixelDataWriter` adapters (`dicom_jpeg::dicom_rs::PixelAdapter`).
* `parallel`: rayon parallelism.

## Testing

Test data is stored with Git LFS. Its provenance is documented in `testdata/SOURCES.md`.

```sh
git lfs pull
cargo test --all-features             # implemented codecs
cargo test --all-features -- --ignored  # cases for codecs that are not implemented yet
```

`tests/dicom_decode.rs` decodes every NEMA WG04, pydicom and pylibjpeg file through the dicom-rs adapter and compares it with the uncompressed reference.
`tests/dicom_roundtrip.rs` encodes native references, writes and re-reads a DICOM file with dicom-rs, then decodes and compares the result.

## License

MIT OR Apache-2.0
