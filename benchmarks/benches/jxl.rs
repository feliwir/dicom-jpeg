//! JPEG XL: dicom-jpeg vs libjxl (via jpegxl-rs), jxl-oxide, jxl-rs and
//! zune-jpegxl. Decode inputs are the WG04 reference images encoded by
//! libjxl (lossless and lossy) and by dicom-jpeg (lossless).

use criterion::{Criterion, criterion_group, criterion_main};
use dicom_jpeg::{Codec, Image, Samples};
use dicom_jpeg_benchmarks::{
    DecodeFn, EncodeFn, Input, decode_group, encode_group, ours_decode, pixels, references,
};
use jpegxl_rs::encode::{ColorEncoding, EncoderFrame, EncoderSpeed, ImageInfo};
use jxl::api::{Endianness, Event, JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer};

/// libjxl encoder: lossless, or lossy at butteraugli `distance`.
fn libjxl_encode<'a>(speed: EncoderSpeed, distance: Option<f32>) -> EncodeFn<'a> {
    Box::new(move |img: &Image| {
        let info = &img.info;
        let gray = info.components == 1;
        let b = jpegxl_rs::encoder_builder()
            .speed(speed)
            .color_encoding(if gray {
                ColorEncoding::SrgbLuma
            } else {
                ColorEncoding::Srgb
            });
        let mut enc = match distance {
            None => b.lossless(true).uses_original_profile(true).build(),
            Some(d) => b.quality(d).build(),
        }
        .map_err(|e| e.to_string())?;
        let ii = ImageInfo::builder()
            .width(info.width)
            .height(info.height)
            .bits_per_sample(info.bits_per_sample as u32)
            .build();
        let mut s = enc.session(&ii).map_err(|e| e.to_string())?;
        let c = info.components as u32;
        match &img.samples {
            Samples::U8(px) => {
                s.add_frame(&EncoderFrame::new(px).num_channels(c).bit_depth_from_image())
            }
            Samples::U16(px) => {
                s.add_frame(&EncoderFrame::new(px).num_channels(c).bit_depth_from_image())
            }
        }
        .map_err(|e| e.to_string())?;
        s.finish().map_err(|e| e.to_string())
    })
}

fn libjxl_decode<'a>() -> DecodeFn<'a> {
    let dec = jpegxl_rs::decoder_builder()
        .build()
        .expect("libjxl decoder");
    Box::new(move |d| {
        let (meta, _) = dec.decode(d).map_err(|e| e.to_string())?;
        Ok(meta.width as usize * meta.height as usize * meta.num_color_channels as usize)
    })
}

fn oxide_decode<'a>() -> DecodeFn<'a> {
    Box::new(|d| {
        let img = jxl_oxide::JxlImage::builder()
            .read(d)
            .map_err(|e| e.to_string())?;
        let r = img.render_frame(0).map_err(|e| e.to_string())?;
        Ok(r.image_all_channels().buf().len())
    })
}

/// jxl-rs, driving its state machine with 8/16-bit interleaved output.
fn jxl_rs_decode<'a>() -> DecodeFn<'a> {
    Box::new(|data| {
        let err = |e: jxl::error::Error| e.to_string();
        let mut dec = JxlDecoder::new(JxlDecoderOptions::default());
        let mut input = data;
        loop {
            match dec.process(&mut input, None, None).map_err(err)? {
                Event::BasicInfo => break,
                Event::NeedMoreInput { .. } => return Err("truncated".into()),
                _ => {}
            }
        }
        let info = dec.basic_info().ok_or("no basic info")?;
        let (w, h) = info.size;
        let bits = info.bit_depth.bits_per_sample();
        let mut fmt = dec.current_pixel_format().ok_or("no pixel format")?.clone();
        let df = if bits <= 8 {
            JxlDataFormat::U8 { bit_depth: 8 }
        } else {
            JxlDataFormat::U16 {
                endianness: Endianness::native(),
                bit_depth: 16,
            }
        };
        fmt.color_data_format = Some(df);
        fmt.extra_channel_format.iter_mut().for_each(|f| *f = None);
        let spp = fmt.color_type.samples_per_pixel();
        dec.set_pixel_format(fmt).map_err(err)?;
        let row = w * spp * df.bytes_per_sample();
        // u16 rows need 2-byte alignment.
        let mut store = vec![0u16; row * h / 2 + 1];
        let buf: &mut [u8] = &mut bytes_of_mut(&mut store)[..row * h];
        loop {
            let mut out = [JxlOutputBuffer::new(buf, h, row)];
            match dec.process(&mut input, Some(&mut out), None).map_err(err)? {
                Event::FrameComplete { .. } | Event::Complete => return Ok(w * h * spp),
                Event::NeedMoreInput { .. } => return Err("truncated".into()),
                _ => {}
            }
        }
    })
}

fn bytes_of_mut(v: &mut [u16]) -> &mut [u8] {
    let len = std::mem::size_of_val(v);
    // SAFETY: u16 has no padding and any byte pattern is valid; the length
    // covers exactly the same memory.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), len) }
}

fn impls<'a>() -> Vec<(&'static str, DecodeFn<'a>)> {
    vec![
        ("dicom-jpeg", ours_decode(Codec::JpegXl)),
        ("libjxl", libjxl_decode()),
        ("jxl-oxide", oxide_decode()),
        ("jxl-rs", jxl_rs_decode()),
    ]
}

fn encoded(refs: &[(String, Image)], mut enc: EncodeFn) -> Vec<Input> {
    refs.iter()
        .filter_map(|(name, img)| {
            Some(Input {
                name: name.clone(),
                data: enc(img).ok()?,
                pixels: pixels(img),
            })
        })
        .collect()
}

fn ours_encode<'a>() -> EncodeFn<'a> {
    Box::new(|img| {
        dicom_jpeg::jxl::encode_lossless(img)
            .map(|e| e.data)
            .map_err(|e| e.to_string())
    })
}

fn decode(c: &mut Criterion) {
    let refs = references();
    let sets = [
        (
            "REF lossless (libjxl e7)",
            libjxl_encode(EncoderSpeed::Squirrel, None),
        ),
        ("REF lossless (dicom-jpeg)", ours_encode()),
        (
            "REF lossy d1 (libjxl e7)",
            libjxl_encode(EncoderSpeed::Squirrel, Some(1.0)),
        ),
    ];
    for (name, enc) in sets {
        decode_group(
            c,
            &format!("jxl-decode/{name}"),
            &encoded(&refs, enc),
            impls(),
        );
    }
}

fn zune_encode<'a>() -> EncodeFn<'a> {
    use zune_core::bit_depth::BitDepth;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::EncoderOptions;
    Box::new(|img: &Image| {
        let info = &img.info;
        let (bytes, depth) = match &img.samples {
            Samples::U8(s) => (s.clone(), BitDepth::Eight),
            Samples::U16(s) => (
                s.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                BitDepth::Sixteen,
            ),
        };
        let cs = if info.components == 3 {
            ColorSpace::RGB
        } else {
            ColorSpace::Luma
        };
        let opts = EncoderOptions::new(info.width as usize, info.height as usize, cs, depth)
            .set_num_threads(0);
        let mut out = Vec::new();
        zune_jpegxl::JxlSimpleEncoder::new(&bytes, opts)
            .encode(&mut out)
            .map_err(|e| format!("{e:?}"))?;
        Ok(out)
    })
}

fn encode(c: &mut Criterion) {
    let refs = references();
    encode_group(
        c,
        "jxl-encode/lossless",
        &refs,
        vec![
            ("dicom-jpeg", ours_encode()),
            ("libjxl e1", libjxl_encode(EncoderSpeed::Lightning, None)),
            ("libjxl e3", libjxl_encode(EncoderSpeed::Falcon, None)),
            ("libjxl e7", libjxl_encode(EncoderSpeed::Squirrel, None)),
            ("zune-jpegxl", zune_encode()),
        ],
    );
}

criterion_group! {
    name = benches;
    config = dicom_jpeg_benchmarks::criterion();
    targets = decode, encode
}
criterion_main!(benches);
