//! ISO/IEC 18181: JPEG XL.

use crate::error::{Error, Result};
use crate::{Codec, DecodeOptions, ImageInfo};
#[cfg(feature = "encode")]
use crate::{EncodeOptions, Encoded, Image, TransferSyntax};

pub(crate) const CODEC: Codec = Codec::JpegXl;

/// Parse the codestream headers and describe the image.
pub fn read_info(_data: &[u8]) -> Result<ImageInfo> {
    Err(Error::unsupported(CODEC, "decoder not implemented yet"))
}

/// Decode a complete codestream.
pub fn decode(_data: &[u8], _opts: &DecodeOptions) -> Result<crate::Image> {
    Err(Error::unsupported(CODEC, "decoder not implemented yet"))
}

/// Encode an image for the given transfer syntax.
#[cfg(feature = "encode")]
pub fn encode(_ts: TransferSyntax, _image: &Image, _opts: &EncodeOptions) -> Result<Encoded> {
    Err(Error::unsupported(CODEC, "encoder not implemented yet"))
}
