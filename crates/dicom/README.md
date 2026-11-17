# Using dicom-jpeg with dicom-rs

`dicom-jpeg-dicom` connects the dicom-jpeg codecs to
[dicom-rs](https://github.com/Enet4/dicom-rs) 0.10. It covers every active
JPEG-family transfer syntax:

| Transfer syntaxes (1.2.840.10008.1.2.4.x) | Codec | Cargo feature |
|---|---|---|
| .50 .51 .57 .70 | JPEG baseline, extended, lossless | `jpeg` |
| .80 .81 | JPEG-LS | `jpeg-ls` |
| .90 .91 | JPEG 2000 | `jpeg2000` |
| .92 .93 | JPEG 2000 Part 2 multi-component | `jpeg2000-part2` |
| .201 .202 .203 | HTJ2K | `htj2k` |
| .110 .111 .112 | JPEG XL | `jpeg-xl` |

There are two ways to use it:

1. **Plug it into dicom-rs** (`inventory-registry` feature). dicom-rs then
   decodes, encodes and transcodes these transfer syntaxes with dicom-jpeg
   wherever it handles pixel data, for example in
   `dicom_pixeldata::PixelDecoder::decode_pixel_data` and
   `dicom_pixeldata::Transcode::transcode`. Your code doesn't change.
2. **Call the adapter yourself.** `PixelAdapter` implements dicom-rs's
   `PixelDataReader` and `PixelDataWriter` traits.

## 1. Plugging it into dicom-rs

### Cargo.toml

Use either the `dicom-jpeg` facade crate or `dicom-jpeg-dicom` on its own,
with the `inventory-registry` feature:

```toml
[dependencies]
dicom = "0.10"
dicom-jpeg = { git = "https://github.com/feliwir/dicom-jpeg", features = ["inventory-registry"] }
```

Selecting only some codecs:

```toml
dicom-jpeg-dicom = { git = "https://github.com/feliwir/dicom-jpeg", default-features = false, features = ["inventory-registry", "jpeg", "jpeg2000", "encode"] }
```

This works with the `dicom` umbrella crate or with `dicom-object` and
`dicom-pixeldata` on their own.

### Linking the crate

dicom-jpeg registers itself when the program starts, so dicom-rs only sees it
if the crate is linked into the program. If your code doesn't otherwise refer
to it, add this line anywhere in your crate:

```rust
use dicom_jpeg as _; // or: use dicom_jpeg_dicom as _;
```

### Decoding and transcoding

From here on it is plain dicom-rs:

```rust
use dicom_object::open_file;
use dicom_pixeldata::{ConvertOptions, PixelDecoder, Transcode};

use dicom_jpeg as _;

let obj = open_file("image.dcm")?;            // e.g. JPEG 2000 or JPEG-LS
let pixels = obj.decode_pixel_data()?;        // decoded by dicom-jpeg
let image = pixels.to_dynamic_image_with_options(0, &ConvertOptions::new().force_8bit())?;
image.save("image.png")?;

// Transcoding uses dicom-jpeg's encoder:
let mut obj = open_file("uncompressed.dcm")?;
let ts = dicom_transfer_syntax_registry::TransferSyntaxRegistry
    .get(dicom_dictionary_std::uids::HTJ2K_LOSSLESS)
    .unwrap();
obj.transcode(ts)?;
obj.write_to_file("htj2k.dcm")?;
```

`to_dynamic_image` requires dicom-pixeldata's `image` feature.

The complete program is
[`crates/dicom-jpeg/examples/dicom_rs.rs`](../dicom-jpeg/examples/dicom_rs.rs).
It converts DICOM files to PNG and refers to dicom-jpeg only in the
`use ... as _` line:

```sh
cargo run --release -p dicom-jpeg --example dicom_rs --features inventory-registry -- \
    testdata/dicom/wg04/J2KR/CT1_J2KR --window 40 400 -o out/
```

To display an image, dicom-rs applies the modality LUT and the object's first
VOI LUT or window. If the object has neither, it stretches the full value
range instead, which makes CT images look dark. In that case, pass a window,
as the example does with `--window <center> <width>`.

### Which syntaxes dicom-jpeg takes over

dicom-rs adds dicom-jpeg's entries after its own built-in ones. An entry
replaces a built-in one only if it does strictly more: a built-in stub that
can't decode at all, or a built-in decoder that can't encode. If both can do
the same, dicom-rs keeps its own.

With dicom-rs 0.10, every JPEG-family syntax therefore goes to dicom-jpeg
except JPEG baseline (.50). That one is only kept by dicom-rs when its `jpeg`
feature is on, which happens through dicom-pixeldata's default `native`
feature. In that case dicom-rs's own baseline codec decodes and encodes it.

To have dicom-jpeg handle baseline as well, depend on dicom-pixeldata without
`native`. `native` also brings RLE and deflate support, so enable those two
explicitly:

```toml
[dependencies]
dicom = { version = "0.10", default-features = false, features = ["inventory-registry", "ul"] }
dicom-pixeldata = { version = "0.10", default-features = false, features = ["rle", "deflate", "rayon", "image"] }
dicom-jpeg = { git = "https://github.com/feliwir/dicom-jpeg", features = ["inventory-registry"] }
```

With this setup, use `dicom_pixeldata` directly instead of
`dicom::pixeldata`. Cargo merges features across your whole dependency tree:
if any crate in it enables dicom-pixeldata's default features, `native`
comes back and baseline returns to dicom-rs.

To check what a syntax resolves to:

```rust
use dicom_encoding::{Codec, TransferSyntaxIndex};
use dicom_transfer_syntax_registry::TransferSyntaxRegistry;

let ts = TransferSyntaxRegistry.get("1.2.840.10008.1.2.4.90").unwrap();
let encodes = matches!(ts.codec(), Codec::EncapsulatedPixelData(_, Some(_)));
println!("{}: decodes {}, encodes {encodes}", ts.name(), ts.can_decode_all());
```

## 2. Calling the adapter yourself

Without the registry, get the adapter for an object's transfer syntax and use
dicom-rs's adapter traits directly:

```rust
use dicom_encoding::adapters::{EncodeOptions, PixelDataReader, PixelDataWriter};
use dicom_jpeg::dicom_rs::PixelAdapter;

let obj = dicom_object::open_file("image.dcm")?;
if let Some(adapter) = PixelAdapter::from_uid(obj.meta().transfer_syntax()) {
    let mut pixels = Vec::new();
    adapter.decode_frame(&obj, 0, &mut pixels)?; // or decode(&obj, ...) for all frames
}

// Encoding a native object (frame 0) as JPEG 2000:
let native = dicom_object::open_file("uncompressed.dcm")?;
let adapter = PixelAdapter::from_uid(dicom_dictionary_std::uids::JPEG2000).unwrap();
let mut fragment = Vec::new();
let ops = adapter.encode_frame(&native, 0, EncodeOptions::default(), &mut fragment)?;
// `fragment` is the codestream; apply `ops` (Photometric Interpretation,
// Lossy Image Compression, ...) to the object that stores it.
```

[`crates/dicom-jpeg/tests/dicom_roundtrip.rs`](../dicom-jpeg/tests/dicom_roundtrip.rs)
shows a complete encode, write, read and decode cycle.

`PixelAdapter::with_limits` sets the maximum image size and allocation size.
The defaults are 2^31 samples and 8 GiB.

## Behaviour

- **Output format:** decoded samples are little-endian at the object's Bits
  Allocated.
- **Colour:** images with three samples per pixel come out as interleaved RGB.
  YCbCr JPEG is converted, and J2K RCT/ICT and JPEG XL XYB are inverted.
- **Encoding:** the writer sets the attributes dicom-rs expects, such as
  Photometric Interpretation (e.g. YBR_RCT, YBR_ICT, YBR_FULL_422) and the
  Lossy Image Compression attributes.
- **JPEG 2000 Part 2 (.92 / .93):** all frames are coded as the components of
  one codestream (PS3.5 A.4.4). Encode with `PixelDataWriter::encode`;
  `encode_frame` only works for single-frame objects.
- **Assumptions:** dicom-rs's `PixelDataObject` doesn't expose Pixel
  Representation or Planar Configuration. Native input is therefore assumed
  to be interleaved (planar configuration 0), and decoding uses the
  signedness stored in the codestream.
- **Versions:** the traits are those of dicom-rs 0.10 (`dicom-encoding`
  0.10). With another dicom-rs version the adapter's traits won't match
  yours.
