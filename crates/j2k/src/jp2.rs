//! JP2 / JPH file format (T.800 Annex I, T.814): locate the codestream box.

use super::CODEC;
use dicom_jpeg_common::{Error, Result};

/// Return the contents of the first contiguous codestream box (`jp2c`).
pub(crate) fn codestream(data: &[u8]) -> Result<&[u8]> {
    let mut pos = 0usize;
    while pos + 8 <= data.len() {
        let lbox = u32::from_be_bytes(data[pos..pos + 4].try_into().expect("4 bytes")) as u64;
        let tbox = &data[pos + 4..pos + 8];
        let (header, len) = match lbox {
            0 => (8u64, (data.len() - pos) as u64),
            1 => {
                let xl = data
                    .get(pos + 8..pos + 16)
                    .ok_or(Error::Truncated { codec: CODEC })?;
                (16, u64::from_be_bytes(xl.try_into().expect("8 bytes")))
            }
            n => (8, n),
        };
        if len < header {
            return Err(Error::invalid(CODEC, "invalid JP2 box length"));
        }
        let end = (pos as u64).saturating_add(len).min(data.len() as u64) as usize;
        let start = pos + header as usize;
        if tbox == b"jp2c" {
            return Ok(&data[start.min(end)..end]);
        }
        pos = end;
    }
    Err(Error::invalid(CODEC, "JP2 file without codestream box"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fuzzer find: an XLBox length near u64::MAX overflowed `pos + len`.
    #[test]
    fn huge_extended_box_length() {
        let mut data = vec![0, 0, 0, 13];
        data.extend_from_slice(b"jP  ");
        data.extend_from_slice(&[3, 0, 0, 0, 0, 0, 0, 0, 1]);
        data.extend_from_slice(b"\x01P\x00");
        data.extend_from_slice(&[0xFF; 16]);
        assert!(codestream(&data).is_err());
        let mut xl = vec![0, 0, 0, 1];
        xl.extend_from_slice(b"jp2c");
        xl.extend_from_slice(&[0xFF; 8]);
        xl.extend_from_slice(&[0xFF, 0x4F]);
        assert_eq!(codestream(&xl).unwrap(), &[0xFF, 0x4F]);
    }
}
