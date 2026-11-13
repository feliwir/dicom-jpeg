# Test data provenance

All binary files below `testdata/` are stored with Git LFS (`git lfs pull`).
Per-file SHA-256 checksums are in [`SHA256SUMS`](SHA256SUMS)
(verify with `cd testdata && sha256sum -c SHA256SUMS`).
Files were collected on 2026-11-13.

## DICOM

| Directory | Origin | Version | Licence / terms |
|---|---|---|---|
| `dicom/wg04/` | NEMA DICOM WG04 compression sample set, via [robyoung/dicom-test-files](https://github.com/robyoung/dicom-test-files) `data/WG04` | commit `6279c96a8a9d4ed2cfba89ac67ebcf669c4471f9` | NEMA public sample data |
| `dicom/pydicom/` | pydicom test files via robyoung/dicom-test-files `data/pydicom`, plus HTJ2K/JPEG-LS files from [pydicom/pydicom-data](https://github.com/pydicom/pydicom-data) `data_store/data` | dicom-test-files `6279c96`, pydicom-data `abc42b90985fb6cf385aa4af766d2c9c94a257a4` | see `dicom/pydicom/README.txt` (MIT, pydicom) |
| `dicom/pylibjpeg/` | [pydicom/pylibjpeg-data](https://github.com/pydicom/pylibjpeg-data) `ljdata/ds` | commit `8253566715800a7fc3d4d949abab102c8172bca0` | MIT, see `dicom/pylibjpeg/LICENSE.txt` |

Changes made to the originals:

* `dicom/wg04/REF/*_UNC` were stored zstd-compressed upstream (`*.zst`) and are kept decompressed here.
* WG04 `RLE/` and the uncompressed pylibjpeg sets (`LittleEndianExplicit`, `LittleEndianImplicit`) are not included.
* Byte-identical duplicates across the three sets were removed. Priority was wg04, then pydicom, then pylibjpeg.
* The File Meta Information of these files was regenerated with `dcmconv +F` (DCMTK), because dicom-rs rejects their incomplete meta group (no group length, missing SOP Instance UID). The data set, including the encapsulated pixel data, is byte-identical:
  * `pydicom/HTJ2KLossless_08_RGB.dcm`, `pydicom/HTJ2K_08_RGB.dcm`
  * `pydicom/JLSL_08_07_0_1F.dcm`, `pydicom/JLSL_16_15_1_1F.dcm`, `pydicom/JLSL_RGB_ILV{0,1,2}.dcm`, `pydicom/JLSN_RGB_ILV0.dcm`
  * `pylibjpeg/JPEG2000/MAROTECH_CT_JP2Lossy.dcm`

Reference images: for J2KI files, `pydicom/{693,MR2,RG1,RG3}_UNCI.dcm` are *decoded lossy* images, not originals. `US1_UNCI` equals the lossless original.

## Codec conformance

| Directory | Content | Origin | Version | Licence / terms |
|---|---|---|---|---|
| `conformance/t81/t83/` | ITU-T T.83 \| ISO/IEC 10918-2 compliance bitstreams (processes 1, 2, 4, 14; Huffman only) | pylibjpeg-data `ljdata/jpg/10918` | commit `8253566` | ISO/IEC JTC1/SC29/WG1 compliance data |
| `conformance/t81/libjpeg-turbo/` | `testorig.jpg`/`.ppm`, `testimgint.jpg`, `monkey12.jpg`, `monkey16.pgm`/`.png` | [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) `testimages/` | commit `3ef3367eb68aabbc25ce91bf2136c87ed4ec7218` | see `LICENSE.txt` there |
| `conformance/t81/w3/` | W3C JPEG baseline samples (400/420/422/444) | copied from the earlier `jpeg-rs` repository | — | W3C sample images |
| `conformance/jpegls/` | ISO/IEC 14495-1 conformance set (`t8*.jls`, `t16*.jls`, `test8*.p?m`, `test16.pgm`) | [team-charls/charls](https://github.com/team-charls/charls) `test/data/conformance` | commit `e682efabdf5f55bb6758b56902dd741958db0e70` | see `README.md` there |
| `conformance/jpegls/charls/` | extra CharLS regression streams (interleave modes, HP colour transforms, fuzz cases) | charls `test/data` | same | BSD-3-Clause (CharLS) |
| `conformance/j2k/iso15444-4/` | ITU-T T.803 \| ISO/IEC 15444-4 ed. 4 electronic insert: Part 1 profile 0/1 codestreams and references, HTJ2K profile 0/1 codestreams, JP2/JPH/JPX files | <https://standards.iso.org/iso-iec/15444/-4/ed-4/en/electronic_insert.zip> | ed. 4 | see `files/COPYRIGHT.txt` (free use for JPEG 2000 conformance testing) |
| `conformance/jxl/testcases/` | JPEG XL decoder conformance bitstreams and reference renderings (ISO/IEC 18181-3) | [libjxl/conformance](https://github.com/libjxl/conformance) | commit `b1d0f990b03e57bf6d137c365cd5dc8b470b9191` | see `testcases/README.md` and `LICENSE.txt` |

Notes:

* T.83 publishes no reference decodes for the lossy streams. They are checked
  with the T.83 accuracy criterion against a floating-point reference IDCT,
  and cross-checked against oracle decoders.
* The ISO/IEC 18181-3 electronic insert (ed. 2) archive was corrupt when downloaded. The
  libjxl/conformance repository holds the same test cases.
  `reference_image.npy` files are not in that repository; tests use `ref.png` / `ref.pfm`.
* No public DICOM samples exist for the JPEG XL transfer syntaxes. The tests wrap the
  conformance codestreams into DICOM objects with dicom-rs.
