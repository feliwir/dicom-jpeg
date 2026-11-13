//! ITU-T T.800 | ISO/IEC 15444-1 JPEG 2000 (with Part 2 multi-component and Part 15 HTJ2K).

use crate::error::{Error, Result};
use crate::{Codec, DecodeOptions, ImageInfo};
#[cfg(feature = "encode")]
use crate::{EncodeOptions, Encoded, Image, TransferSyntax};

pub(crate) const CODEC: Codec = Codec::Jpeg2000;

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
