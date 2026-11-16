//! libjpeg-turbo 3 through its TurboJPEG 3 C API.
//!
//! The safe `turbojpeg` wrapper only covers 8-bit samples, so 12-bit DCT and
//! 2–16 bit lossless (process 14) go through the raw `tj3*12` / `tj3*16`
//! entry points here.

use std::ffi::{CStr, c_int, c_void};
use std::ptr;

use dicom_jpeg::{Image, Samples};
use turbojpeg::raw;

/// A TurboJPEG instance (decompressor or compressor).
#[derive(Debug)]
pub struct TurboJpeg {
    handle: raw::tjhandle,
}

impl Drop for TurboJpeg {
    fn drop(&mut self) {
        // SAFETY: `handle` came from `tj3Init` and is not used afterwards.
        unsafe { raw::tj3Destroy(self.handle) };
    }
}

/// How [`TurboJpeg::compress`] codes the image.
#[derive(Debug, Clone, Copy)]
pub enum Mode {
    /// Sequential DCT at the given quality, 4:2:2 for colour, optimised
    /// Huffman tables (as dicom-jpeg's encoder does).
    Dct {
        /// IJG quality 1..=100.
        quality: u8,
    },
    /// Lossless process 14 with predictor 1 (selection value 1).
    LosslessSv1,
}

impl TurboJpeg {
    fn init(kind: raw::TJINIT) -> Self {
        // SAFETY: plain constructor; a null handle is checked below.
        let handle = unsafe { raw::tj3Init(kind as c_int) };
        assert!(!handle.is_null(), "tj3Init failed");
        TurboJpeg { handle }
    }

    /// A decompressor.
    pub fn decompressor() -> Self {
        Self::init(raw::TJINIT_TJINIT_DECOMPRESS)
    }

    /// A compressor.
    pub fn compressor() -> Self {
        Self::init(raw::TJINIT_TJINIT_COMPRESS)
    }

    fn error(&self) -> String {
        // SAFETY: `tj3GetErrorStr` returns a NUL-terminated static or
        // per-handle buffer that stays valid until the next call.
        unsafe { CStr::from_ptr(raw::tj3GetErrorStr(self.handle)) }
            .to_string_lossy()
            .into_owned()
    }

    fn check(&self, rc: c_int) -> Result<(), String> {
        if rc == 0 { Ok(()) } else { Err(self.error()) }
    }

    fn get(&self, param: raw::TJPARAM) -> c_int {
        // SAFETY: valid handle, valid parameter id.
        unsafe { raw::tj3Get(self.handle, param as c_int) }
    }

    fn set(&self, param: raw::TJPARAM, value: c_int) -> Result<(), String> {
        // SAFETY: valid handle, valid parameter id.
        self.check(unsafe { raw::tj3Set(self.handle, param as c_int, value) })
    }

    /// Decode `jpeg` (any precision, DCT or lossless) to interleaved gray or
    /// RGB samples; returns the number of samples.
    pub fn decompress(&mut self, jpeg: &[u8]) -> Result<usize, String> {
        // SAFETY: valid handle; the slice is valid for its length.
        self.check(unsafe {
            raw::tj3DecompressHeader(self.handle, jpeg.as_ptr(), jpeg.len() as raw::size_t)
        })?;
        let w = self.get(raw::TJPARAM_TJPARAM_JPEGWIDTH);
        let h = self.get(raw::TJPARAM_TJPARAM_JPEGHEIGHT);
        let precision = self.get(raw::TJPARAM_TJPARAM_PRECISION);
        let gray = self.get(raw::TJPARAM_TJPARAM_COLORSPACE) == raw::TJCS_TJCS_GRAY as c_int;
        let (pf, channels) = if gray {
            (raw::TJPF_TJPF_GRAY, 1)
        } else {
            (raw::TJPF_TJPF_RGB, 3)
        };
        let n = w as usize * h as usize * channels;
        let pitch = 0; // packed rows
        // Each arm's destination holds exactly width * height * channels
        // samples of the type its entry point writes, which is what a packed
        // pitch with `pf` requires.
        let rc = match precision {
            ..=8 => {
                let mut out = vec![0u8; n];
                // SAFETY: `out` holds the n samples written; see above.
                unsafe {
                    raw::tj3Decompress8(
                        self.handle,
                        jpeg.as_ptr(),
                        jpeg.len() as raw::size_t,
                        out.as_mut_ptr(),
                        pitch,
                        pf,
                    )
                }
            }
            9..=12 => {
                let mut out = vec![0i16; n];
                // SAFETY: `out` holds the n samples written; see above.
                unsafe {
                    raw::tj3Decompress12(
                        self.handle,
                        jpeg.as_ptr(),
                        jpeg.len() as raw::size_t,
                        out.as_mut_ptr(),
                        pitch,
                        pf,
                    )
                }
            }
            _ => {
                let mut out = vec![0u16; n];
                // SAFETY: `out` holds the n samples written; see above.
                unsafe {
                    raw::tj3Decompress16(
                        self.handle,
                        jpeg.as_ptr(),
                        jpeg.len() as raw::size_t,
                        out.as_mut_ptr(),
                        pitch,
                        pf,
                    )
                }
            }
        };
        // A warning (e.g. JWRN_NOT_SEQUENTIAL for the Se = 0 that the WG04
        // JPLY files declare) still yields the complete image.
        // SAFETY: valid handle.
        let code = unsafe { raw::tj3GetErrorCode(self.handle) };
        if rc != 0 && code != raw::TJERR_TJERR_WARNING as c_int {
            return Err(self.error());
        }
        Ok(n)
    }

    /// Encode `image` (gray or RGB, 2–16 bits).
    pub fn compress(&mut self, image: &Image, mode: Mode) -> Result<Vec<u8>, String> {
        let info = &image.info;
        let (pf, samp) = match info.components {
            1 => (raw::TJPF_TJPF_GRAY, raw::TJSAMP_TJSAMP_GRAY),
            3 => (
                raw::TJPF_TJPF_RGB,
                match mode {
                    Mode::Dct { .. } => raw::TJSAMP_TJSAMP_422,
                    Mode::LosslessSv1 => raw::TJSAMP_TJSAMP_444,
                },
            ),
            c => return Err(format!("{c} components")),
        };
        let bits = info.bits_per_sample as c_int;
        self.set(raw::TJPARAM_TJPARAM_SUBSAMP, samp as c_int)?;
        self.set(raw::TJPARAM_TJPARAM_PRECISION, bits)?;
        match mode {
            Mode::Dct { quality } => {
                if bits != 8 && bits != 12 {
                    // libjpeg-turbo's DCT path takes exactly 8 or 12 bits.
                    self.set(
                        raw::TJPARAM_TJPARAM_PRECISION,
                        if bits < 8 { 8 } else { 12 },
                    )?;
                }
                self.set(raw::TJPARAM_TJPARAM_LOSSLESS, 0)?;
                self.set(raw::TJPARAM_TJPARAM_QUALITY, quality as c_int)?;
                self.set(raw::TJPARAM_TJPARAM_OPTIMIZE, 1)?;
            }
            Mode::LosslessSv1 => {
                self.set(raw::TJPARAM_TJPARAM_LOSSLESS, 1)?;
                self.set(raw::TJPARAM_TJPARAM_LOSSLESSPSV, 1)?;
                self.set(raw::TJPARAM_TJPARAM_LOSSLESSPT, 0)?;
            }
        }
        let precision = self.get(raw::TJPARAM_TJPARAM_PRECISION);
        let (w, h) = (info.width as c_int, info.height as c_int);
        let mut buf: *mut u8 = ptr::null_mut();
        let mut size: raw::size_t = 0;
        // In each arm the source holds width * height * channels samples of
        // the entry point's type (packed rows); `buf`/`size` receive a buffer
        // allocated by TurboJPEG, freed below.
        let rc = match (&image.samples, precision) {
            // SAFETY: see above.
            (Samples::U8(s), ..=8) => unsafe {
                raw::tj3Compress8(self.handle, s.as_ptr(), w, 0, h, pf, &mut buf, &mut size)
            },
            // SAFETY: see above.
            (Samples::U16(s), 9..=12) => unsafe {
                raw::tj3Compress12(
                    self.handle,
                    s.as_ptr().cast(),
                    w,
                    0,
                    h,
                    pf,
                    &mut buf,
                    &mut size,
                )
            },
            // SAFETY: see above.
            (Samples::U16(s), 13..) => unsafe {
                raw::tj3Compress16(self.handle, s.as_ptr(), w, 0, h, pf, &mut buf, &mut size)
            },
            _ => return Err(format!("unsupported precision {precision}")),
        };
        let out = if rc == 0 && !buf.is_null() {
            // SAFETY: on success `buf` points to `size` initialised bytes.
            Ok(unsafe { std::slice::from_raw_parts(buf, size as usize) }.to_vec())
        } else {
            Err(self.error())
        };
        // SAFETY: `buf` was allocated by TurboJPEG (or is null).
        unsafe { raw::tj3Free(buf.cast::<c_void>()) };
        out
    }
}
