//! Transport configuration.
//!
//! One [`WsConfig`] describes the PCM format of a connection, the bounded
//! ingress ring that absorbs client audio, and the WebSocket message limits
//! applied to the socket. Everything is validated when the transport is bound,
//! so a malformed configuration fails before a peer can connect.

use parlevox_core::{OverflowPolicy, ParlevoxError};

/// Default PCM sample rate for a WebSocket session (24 kHz mono).
pub const DEFAULT_SAMPLE_RATE: u32 = 24_000;

/// Default channel count for a WebSocket session.
pub const DEFAULT_CHANNELS: u16 = 1;

/// Default ingress ring length, in seconds of audio.
pub const DEFAULT_INGRESS_SECONDS: u64 = 2;

/// Default maximum size of one incoming WebSocket message, in bytes.
///
/// The socket rejects anything larger before it reaches the PCM decoder, which
/// keeps a single peer's message from allocating without bound.
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 256 * 1024;

/// Configuration for one WebSocket transport listener.
///
/// The struct is `#[non_exhaustive]`: construct it with [`WsConfig::default`],
/// [`WsConfig::with_ingress`], or [`WsConfig::new`] so the validation in this
/// module always runs.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WsConfig {
    /// Sample rate of every PCM frame on this connection.
    pub sample_rate: u32,
    /// Interleaved channel count of every PCM frame on this connection.
    pub channels: u16,
    /// Capacity of the bounded ingress ring, in interleaved `i16` samples.
    pub ingress_capacity_samples: usize,
    /// What the ingress ring does when a message would overflow it.
    pub ingress_overflow: OverflowPolicy,
    /// Largest accepted WebSocket message, in bytes.
    ///
    /// Must be even because client audio is signed 16-bit little-endian PCM.
    pub max_message_bytes: usize,
}

impl Default for WsConfig {
    fn default() -> Self {
        WsConfig {
            sample_rate: DEFAULT_SAMPLE_RATE,
            channels: DEFAULT_CHANNELS,
            ingress_capacity_samples: default_ingress_capacity(
                DEFAULT_SAMPLE_RATE,
                DEFAULT_CHANNELS,
            ),
            ingress_overflow: OverflowPolicy::DropOldest,
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
        }
    }
}

impl WsConfig {
    /// Build a fully specified configuration.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class [`ParlevoxError`] when `sample_rate`,
    /// `channels`, `ingress_capacity_samples`, or `max_message_bytes` is zero,
    /// when the capacity cannot hold even one interleaved sample frame, or when
    /// `max_message_bytes` is odd.
    pub fn new(
        sample_rate: u32,
        channels: u16,
        ingress_capacity_samples: usize,
        ingress_overflow: OverflowPolicy,
        max_message_bytes: usize,
    ) -> Result<Self, ParlevoxError> {
        WsConfig {
            sample_rate,
            channels,
            ingress_capacity_samples,
            ingress_overflow,
            max_message_bytes,
        }
        .validated()
    }

    /// Check every field of an already-built configuration.
    ///
    /// [`WsConfig::default`] is always valid; a value built by struct
    /// construction from another module cannot be, so the transport calls this
    /// before it creates a listener.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class [`ParlevoxError`] when `sample_rate` or
    /// `channels` is zero, when `ingress_capacity_samples` cannot hold even one
    /// interleaved sample frame, or when `max_message_bytes` is smaller than
    /// two bytes or odd. Every message names the offending field.
    pub fn validate(&self) -> Result<(), ParlevoxError> {
        if self.sample_rate == 0 {
            return Err(ParlevoxError::Protocol(
                "ws config: sample_rate must be non-zero".into(),
            ));
        }
        if self.channels == 0 {
            return Err(ParlevoxError::Protocol(
                "ws config: channels must be non-zero".into(),
            ));
        }
        if self.ingress_capacity_samples == 0 {
            return Err(ParlevoxError::Protocol(
                "ws config: ingress_capacity_samples must be non-zero".into(),
            ));
        }
        if self.ingress_capacity_samples < usize::from(self.channels) {
            return Err(ParlevoxError::Protocol(
                "ws config: ingress_capacity_samples must hold at least one sample frame".into(),
            ));
        }
        if self.max_message_bytes < 2 {
            return Err(ParlevoxError::Protocol(
                "ws config: max_message_bytes must be at least two".into(),
            ));
        }
        if self.max_message_bytes % 2 != 0 {
            return Err(ParlevoxError::Protocol(
                "ws config: max_message_bytes must be even for 16-bit pcm".into(),
            ));
        }
        Ok(())
    }

    /// Consume `self` after [`WsConfig::validate`] succeeds.
    fn validated(self) -> Result<Self, ParlevoxError> {
        self.validate()?;
        Ok(self)
    }

    /// Build a configuration with the given PCM format and ingress capacity,
    /// keeping the default overflow policy and message limit.
    ///
    /// # Errors
    ///
    /// Returns the same protocol-class errors as [`WsConfig::new`].
    pub fn with_ingress(
        sample_rate: u32,
        channels: u16,
        ingress_capacity_samples: usize,
    ) -> Result<Self, ParlevoxError> {
        WsConfig::new(
            sample_rate,
            channels,
            ingress_capacity_samples,
            OverflowPolicy::DropOldest,
            DEFAULT_MAX_MESSAGE_BYTES,
        )
    }

    /// Default ingress capacity for a PCM format: [`DEFAULT_INGRESS_SECONDS`]
    /// seconds of interleaved audio, with saturating arithmetic.
    #[must_use]
    pub fn default_ingress_capacity(sample_rate: u32, channels: u16) -> usize {
        default_ingress_capacity(sample_rate, channels)
    }
}

/// `sample_rate * seconds * channels`, saturating instead of wrapping.
fn default_ingress_capacity(sample_rate: u32, channels: u16) -> usize {
    usize::try_from(sample_rate)
        .unwrap_or(usize::MAX)
        .saturating_mul(usize::try_from(DEFAULT_INGRESS_SECONDS).unwrap_or(usize::MAX))
        .saturating_mul(usize::from(channels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_two_seconds_of_24khz_mono() {
        let config = WsConfig::default();
        assert_eq!(config.sample_rate, DEFAULT_SAMPLE_RATE);
        assert_eq!(config.channels, DEFAULT_CHANNELS);
        assert_eq!(config.ingress_capacity_samples, 48_000);
        assert_eq!(config.ingress_overflow, OverflowPolicy::DropOldest);
        assert_eq!(config.max_message_bytes, DEFAULT_MAX_MESSAGE_BYTES);
    }

    #[test]
    fn zero_values_are_rejected() {
        assert!(WsConfig::with_ingress(0, 1, 16).is_err());
        assert!(WsConfig::with_ingress(24_000, 0, 16).is_err());
        assert!(WsConfig::with_ingress(24_000, 1, 0).is_err());
        assert!(WsConfig::new(24_000, 1, 16, OverflowPolicy::DropOldest, 0).is_err());
        assert!(WsConfig::new(24_000, 1, 16, OverflowPolicy::DropOldest, 3).is_err());
    }

    #[test]
    fn capacity_must_hold_one_interleaved_frame() {
        assert!(WsConfig::with_ingress(24_000, 2, 1).is_err());
        assert!(WsConfig::with_ingress(24_000, 2, 2).is_ok());
    }

    #[test]
    fn default_capacity_scales_with_rate_and_channels() {
        assert_eq!(WsConfig::default_ingress_capacity(24_000, 1), 48_000);
        assert_eq!(WsConfig::default_ingress_capacity(24_000, 2), 96_000);
        assert!(WsConfig::default_ingress_capacity(u32::MAX, u16::MAX) > 0);
    }
}
