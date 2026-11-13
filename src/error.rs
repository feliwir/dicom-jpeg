//! Error type shared by all codecs.

use std::borrow::Cow;
use std::fmt;

use crate::Codec;

/// Result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors produced while decoding or encoding.
///
/// Decoders never panic on malformed input; every problem is reported
/// through one of these variants.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The codestream is malformed or violates the codec specification.
    InvalidData {
        /// Codec that detected the problem.
        codec: Codec,
        /// Human readable description.
        message: Cow<'static, str>,
    },
    /// The codestream ended before decoding was complete.
    Truncated {
        /// Codec that detected the problem.
        codec: Codec,
    },
    /// The codestream (or the requested encoding) uses a feature
    /// that is valid but not supported by this implementation.
    Unsupported {
        /// Codec that detected the problem.
        codec: Codec,
        /// Description of the missing feature.
        feature: Cow<'static, str>,
    },
    /// Support for the codec was not compiled in (cargo feature disabled).
    CodecDisabled(Codec),
    /// The image would exceed the configured [`Limits`](crate::Limits).
    LimitExceeded {
        /// What exceeded the limit (e.g. "pixels", "allocation").
        what: &'static str,
        /// The requested amount.
        requested: u64,
        /// The configured limit.
        limit: u64,
    },
    /// The caller supplied inconsistent parameters or buffers.
    InvalidInput(Cow<'static, str>),
}

// Helpers are unused when every codec feature is disabled.
#[allow(dead_code)]
impl Error {
    pub(crate) fn invalid(codec: Codec, message: impl Into<Cow<'static, str>>) -> Self {
        Error::InvalidData {
            codec,
            message: message.into(),
        }
    }

    pub(crate) fn unsupported(codec: Codec, feature: impl Into<Cow<'static, str>>) -> Self {
        Error::Unsupported {
            codec,
            feature: feature.into(),
        }
    }

    pub(crate) fn input(message: impl Into<Cow<'static, str>>) -> Self {
        Error::InvalidInput(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidData { codec, message } => write!(f, "{codec}: invalid data: {message}"),
            Error::Truncated { codec } => write!(f, "{codec}: codestream is truncated"),
            Error::Unsupported { codec, feature } => write!(f, "{codec}: unsupported: {feature}"),
            Error::CodecDisabled(codec) => {
                write!(f, "{codec}: support not enabled (cargo feature is off)")
            }
            Error::LimitExceeded {
                what,
                requested,
                limit,
            } => write!(f, "limit exceeded: {what} {requested} > {limit}"),
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
