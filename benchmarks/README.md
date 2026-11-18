# Benchmarks

Criterion benchmarks that compare dicom-jpeg with reference implementations
that have Rust bindings. They run single-threaded on the NEMA WG04 DICOM
test set from `testdata/`, so run `git lfs pull` first.

```sh
cargo bench -p dicom-jpeg-benchmarks                    # everything
cargo bench -p dicom-jpeg-benchmarks --bench j2k        # one codec
cargo bench -p dicom-jpeg-benchmarks --bench jpeg -- JPLL  # filter groups
```

dicom-jpeg runs at the best SIMD level the CPU supports. To compare its
levels side by side, list them in `DICOM_JPEG_BENCH_LEVELS` (`all`, or e.g.
`scalar,avx2`); each one is then benchmarked as `dicom-jpeg (level)`:

```sh
DICOM_JPEG_BENCH_LEVELS=all cargo bench -p dicom-jpeg-benchmarks --bench jpeg
```

`DICOM_JPEG_SIMD=<level>` instead lowers the level for the whole run.

The `kernels` bench times each SIMD kernel by itself at every level the
CPU supports, on 64 K samples of synthetic data: the JPEG 8- and 12-bit
IDCT, FDCT and quantisation, up/downsampling and colour conversion; the
JPEG 2000 DWT lifting steps; the JPEG XL inverse DCT, edge-preserving
filter, Gaborish and XYB conversion; and interleaving:

```sh
cargo bench -p dicom-jpeg-benchmarks --bench kernels
cargo bench -p dicom-jpeg-benchmarks --bench kernels -- idct
```

Criterion writes an HTML report to `target/criterion/report/index.html`,
with one page per group (violin plot of all implementations) and per
benchmark.

The C and C++ references are built from source by their `-sys` crates. That
needs cmake, a C++ compiler and nasm (for libjpeg-turbo's SIMD).

## Charts

`plot.py` turns the latest Criterion results into the SVG charts and tables in
[results/](results/README.md): one chart per codec, plus one for the SIMD
kernels. It needs only Python 3. Pass the benches' stderr with `--log` to
label encoders with their compression ratio:

```sh
cargo bench -p dicom-jpeg-benchmarks 2> bench.log
python3 benchmarks/plot.py --log bench.log
```

The charts adapt to light and dark mode. Regenerate them after a full run, so
that every group comes from the same build and machine.

## Method

- **What one iteration covers:** a whole input set, such as every frame of
  WG04 J2KR. Throughput is in pixels (`elem/s` = pixels/s).
- **Image size:** only WG04 images of at most 1 Mpixel are used
  (`MAX_FRAME_PIXELS`). This leaves out MG1, the RG radiographs, SC1 and
  VL4/VL5, and keeps each set at about 5 Mpixel, so the whole suite runs in
  about 3 minutes.
- **Sampling:** 10 flat samples (Criterion's minimum) in a 1 s window, after
  a 0.1 s warm-up.
- **Same inputs for everyone:**
  - Inputs dicom-jpeg rejects are dropped.
  - If a reference fails on more than half of the inputs, it is treated as
    lacking that feature and skipped.
  - Otherwise, the inputs it fails on are dropped for every implementation.
  - Each case is reported on stderr.
- **Encoders:** every encoder gets the same settings, and its compression
  ratio is printed on stderr. Lossy JPEG 2000 references are given, per
  image, the byte budget that dicom-jpeg reaches at its default quality.
- **Threads:** all implementations run single-threaded:
  - jxl-oxide, openjp2, jpeg-decoder and zune-jpegxl are built without their
    threading features;
  - OpenJPEG is set to 0 threads;
  - libjxl gets no parallel runner.
- **Colour:** colour JPEG output is converted to RGB by every decoder,
  including dicom-jpeg.

## Implementations

| Codec | Reference | Binding | Coverage |
|---|---|---|---|
| JPEG | libjpeg-turbo 3.1 | `turbojpeg` (+ raw `tj3*12/16`) | 8/12-bit DCT, lossless 2–16 bit |
| | zune-jpeg | pure Rust | 8-bit DCT |
| | jpeg-decoder | pure Rust | 8-bit DCT, lossless |
| JPEG-LS | CharLS 2.4 | `charls` | all |
| JPEG 2000 | OpenJPEG 2.5 (C) | `openjpeg-sys` | Part 1 decode and encode, HTJ2K decode |
| | openjp2 | pure-Rust port of OpenJPEG | as OpenJPEG |
| | hayro-jpeg2000 | pure Rust | Part 1 decode |
| JPEG XL | libjxl 0.12 | `jpegxl-rs` (vendored) | all |
| | jxl-oxide | pure Rust | decode |
| | jxl-rs | pure Rust (`jxl` crate) | decode |
| | zune-jpegxl | pure Rust | lossless encode (simple) |

The following are not included because they have no Rust bindings: IJG
libjpeg, OpenJPH, Grok, Kakadu. As a result, the HTJ2K decoders are fed
dicom-jpeg's own HTJ2K output plus the pydicom HTJ2K files, and the HTJ2K
encoder has nothing to compare against.

## Inputs

| Group | Inputs |
|---|---|
| `jpeg-decode/JPLL lossless` | WG04 JPLL (process 14, 8–16 bit) |
| `jpeg-decode/JPLY 12-bit` | WG04 JPLY (all 12-bit extended) |
| `jpeg-decode/REF 8-bit q90` | 8-bit WG04 REF images encoded by libjpeg-turbo (q90, 4:2:2) |
| `jpegls-decode/*` | WG04 JLSL, JLSN |
| `j2k-decode/*` | WG04 J2KR, J2KI |
| `htj2k-decode/REF *` | WG04 REF images encoded by dicom-jpeg (HT, lossless / lossy) |
| `jxl-decode/*` | WG04 REF images encoded by libjxl e7 (lossless, lossy d1) and by dicom-jpeg (lossless) |
| `*-encode/*` | WG04 REF images (gray and RGB, 8–16 bit) |
