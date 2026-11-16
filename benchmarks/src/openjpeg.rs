//! OpenJPEG (C library, vendored by `openjpeg-sys`) with in-memory streams.
//!
//! Decodes Part 1 and HTJ2K codestreams; encodes Part 1 (OpenJPEG has no HT
//! encoder). Threading is switched off with `opj_codec_set_threads(.., 0)`.

use std::ffi::{c_int, c_void};
use std::ptr;

use dicom_jpeg::{Image, Samples};
use openjpeg_sys as opj;

const CHUNK: usize = 1 << 20;

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

unsafe extern "C" fn read(buf: *mut c_void, n: usize, user: *mut c_void) -> usize {
    // SAFETY: `user` is the `Reader` installed by `decode`, alive for the
    // stream's lifetime; `buf` has room for `n` bytes.
    let r = unsafe { &mut *user.cast::<Reader>() };
    let left = r.data.len() - r.pos;
    if left == 0 {
        return usize::MAX; // (OPJ_SIZE_T)-1: end of stream
    }
    let k = n.min(left);
    // SAFETY: see above; source and destination do not overlap.
    unsafe { ptr::copy_nonoverlapping(r.data.as_ptr().add(r.pos), buf.cast::<u8>(), k) };
    r.pos += k;
    k
}

unsafe extern "C" fn read_skip(n: i64, user: *mut c_void) -> i64 {
    // SAFETY: as in `read`.
    let r = unsafe { &mut *user.cast::<Reader>() };
    let to = (r.pos as i64 + n).clamp(0, r.data.len() as i64);
    let moved = to - r.pos as i64;
    r.pos = to as usize;
    moved
}

unsafe extern "C" fn read_seek(n: i64, user: *mut c_void) -> c_int {
    // SAFETY: as in `read`.
    let r = unsafe { &mut *user.cast::<Reader>() };
    if n < 0 || n as usize > r.data.len() {
        return 0;
    }
    r.pos = n as usize;
    1
}

struct Writer {
    data: Vec<u8>,
    pos: usize,
}

unsafe extern "C" fn write(buf: *mut c_void, n: usize, user: *mut c_void) -> usize {
    // SAFETY: `user` is the `Writer` installed by `encode`; `buf` holds `n`
    // readable bytes.
    let w = unsafe { &mut *user.cast::<Writer>() };
    // SAFETY: see above.
    let src = unsafe { std::slice::from_raw_parts(buf.cast::<u8>(), n) };
    let end = w.pos + n;
    if end > w.data.len() {
        w.data.resize(end, 0);
    }
    w.data[w.pos..end].copy_from_slice(src);
    w.pos = end;
    n
}

unsafe extern "C" fn write_skip(n: i64, user: *mut c_void) -> i64 {
    // SAFETY: as in `write`.
    let w = unsafe { &mut *user.cast::<Writer>() };
    let to = (w.pos as i64 + n).max(0) as usize;
    if to > w.data.len() {
        w.data.resize(to, 0);
    }
    w.pos = to;
    n
}

unsafe extern "C" fn write_seek(n: i64, user: *mut c_void) -> c_int {
    // SAFETY: as in `write`.
    let w = unsafe { &mut *user.cast::<Writer>() };
    if n < 0 {
        return 0;
    }
    if n as usize > w.data.len() {
        w.data.resize(n as usize, 0);
    }
    w.pos = n as usize;
    1
}

/// Owns the OpenJPEG objects of one operation and frees them on drop.
struct Guard {
    codec: *mut opj::opj_codec_t,
    stream: *mut opj::opj_stream_t,
    image: *mut opj::opj_image_t,
}

impl Drop for Guard {
    fn drop(&mut self) {
        // SAFETY: each pointer is null or owned by this guard.
        unsafe {
            if !self.image.is_null() {
                opj::opj_image_destroy(self.image);
            }
            if !self.stream.is_null() {
                opj::opj_stream_destroy(self.stream);
            }
            if !self.codec.is_null() {
                opj::opj_destroy_codec(self.codec);
            }
        }
    }
}

/// Decode a raw J2K codestream; returns the number of decoded samples.
pub fn decode(data: &[u8]) -> Result<usize, String> {
    let mut reader = Reader { data, pos: 0 };
    let mut g = Guard {
        codec: ptr::null_mut(),
        stream: ptr::null_mut(),
        image: ptr::null_mut(),
    };
    // SAFETY: standard OpenJPEG decode sequence. `reader` outlives the
    // stream (dropped by `g` before `reader` goes out of scope), every
    // returned object is checked and owned by `g`.
    unsafe {
        g.codec = opj::opj_create_decompress(opj::OPJ_CODEC_FORMAT::OPJ_CODEC_J2K);
        if g.codec.is_null() {
            return Err("opj_create_decompress".into());
        }
        let mut params = std::mem::zeroed::<opj::opj_dparameters_t>();
        opj::opj_set_default_decoder_parameters(&mut params);
        if opj::opj_setup_decoder(g.codec, &mut params) == 0 {
            return Err("opj_setup_decoder".into());
        }
        opj::opj_codec_set_threads(g.codec, 0);
        g.stream = opj::opj_stream_create(CHUNK, 1);
        opj::opj_stream_set_read_function(g.stream, Some(read));
        opj::opj_stream_set_skip_function(g.stream, Some(read_skip));
        opj::opj_stream_set_seek_function(g.stream, Some(read_seek));
        opj::opj_stream_set_user_data(g.stream, (&raw mut reader).cast(), None);
        opj::opj_stream_set_user_data_length(g.stream, data.len() as u64);
        if opj::opj_read_header(g.stream, g.codec, &mut g.image) == 0 {
            return Err("opj_read_header".into());
        }
        if opj::opj_decode(g.codec, g.stream, g.image) == 0
            || opj::opj_end_decompress(g.codec, g.stream) == 0
        {
            return Err("opj_decode".into());
        }
        let img = &*g.image;
        let comps = std::slice::from_raw_parts(img.comps, img.numcomps as usize);
        Ok(comps.iter().map(|c| c.w as usize * c.h as usize).sum())
    }
}

/// Encoding mode for [`encode`].
#[derive(Debug, Clone, Copy)]
pub enum Mode {
    /// 5/3 wavelet, RCT for colour.
    Lossless,
    /// 9/7 wavelet, ICT for colour, one layer at `ratio`:1.
    Lossy {
        /// Target compression ratio.
        ratio: f32,
    },
}

/// Encode `image` as a raw J2K codestream with 5 decomposition levels and
/// 64x64 code-blocks (the OpenJPEG defaults, same as dicom-jpeg's).
pub fn encode(image: &Image, mode: Mode) -> Result<Vec<u8>, String> {
    let info = &image.info;
    let n = info.components as usize;
    let mut writer = Writer {
        data: Vec::new(),
        pos: 0,
    };
    let mut g = Guard {
        codec: ptr::null_mut(),
        stream: ptr::null_mut(),
        image: ptr::null_mut(),
    };
    // SAFETY: standard OpenJPEG encode sequence; component buffers are
    // allocated by `opj_image_create` with w*h samples each, `writer`
    // outlives the stream, all objects are owned by `g`.
    unsafe {
        let mut cmpt: Vec<opj::opj_image_cmptparm_t> = (0..n)
            .map(|_| {
                let mut c = std::mem::zeroed::<opj::opj_image_cmptparm_t>();
                c.dx = 1;
                c.dy = 1;
                c.w = info.width;
                c.h = info.height;
                c.prec = info.bits_per_sample as u32;
                c.sgnd = info.signed as u32;
                c
            })
            .collect();
        let cs = if n == 3 {
            opj::OPJ_COLOR_SPACE::OPJ_CLRSPC_SRGB
        } else {
            opj::OPJ_COLOR_SPACE::OPJ_CLRSPC_GRAY
        };
        g.image = opj::opj_image_create(n as u32, cmpt.as_mut_ptr(), cs);
        if g.image.is_null() {
            return Err("opj_image_create".into());
        }
        let img = &mut *g.image;
        img.x1 = info.width;
        img.y1 = info.height;
        let comps = std::slice::from_raw_parts_mut(img.comps, n);
        let px = info.width as usize * info.height as usize;
        for (k, comp) in comps.iter_mut().enumerate() {
            let dst = std::slice::from_raw_parts_mut(comp.data, px);
            match &image.samples {
                Samples::U8(s) => {
                    for (d, &v) in dst.iter_mut().zip(s.iter().skip(k).step_by(n)) {
                        *d = v as i32;
                    }
                }
                Samples::U16(s) => {
                    for (d, &v) in dst.iter_mut().zip(s.iter().skip(k).step_by(n)) {
                        *d = if info.signed {
                            v as i16 as i32
                        } else {
                            v as i32
                        };
                    }
                }
            }
        }

        let mut p = std::mem::zeroed::<opj::opj_cparameters_t>();
        opj::opj_set_default_encoder_parameters(&mut p);
        p.tcp_numlayers = 1;
        p.cp_disto_alloc = 1;
        p.tcp_mct = (n == 3) as _;
        match mode {
            Mode::Lossless => p.tcp_rates[0] = 0.0,
            Mode::Lossy { ratio } => {
                p.irreversible = 1;
                p.tcp_rates[0] = ratio;
            }
        }
        g.codec = opj::opj_create_compress(opj::OPJ_CODEC_FORMAT::OPJ_CODEC_J2K);
        if g.codec.is_null() || opj::opj_setup_encoder(g.codec, &mut p, g.image) == 0 {
            return Err("opj_setup_encoder".into());
        }
        opj::opj_codec_set_threads(g.codec, 0);
        g.stream = opj::opj_stream_create(CHUNK, 0);
        opj::opj_stream_set_write_function(g.stream, Some(write));
        opj::opj_stream_set_skip_function(g.stream, Some(write_skip));
        opj::opj_stream_set_seek_function(g.stream, Some(write_seek));
        opj::opj_stream_set_user_data(g.stream, (&raw mut writer).cast(), None);
        if opj::opj_start_compress(g.codec, g.image, g.stream) == 0
            || opj::opj_encode(g.codec, g.stream) == 0
            || opj::opj_end_compress(g.codec, g.stream) == 0
        {
            return Err("opj_encode".into());
        }
    }
    drop(g);
    Ok(writer.data)
}
