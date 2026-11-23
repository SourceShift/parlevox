//! Connector failure taxonomy.
//!
//! Connectors classify a failure once, at the boundary where the wire is
//! understood, and keep it classified until it becomes a
//! [`ParlevoxError`] for the session runtime. The internal variant carries
//! richer detail than [`ErrorClass`]; [`ConnectorError::into_provider`] is the
//! single place that projects it onto the public taxonomy.
//!
//! # The rate-limit gap
//!
//! `parlevox-core`'s [`ErrorClass`] has no rate-limit variant, so a provider
//! `429` is reported internally as [`ConnectorError::RateLimited`] and
//! surfaces as [`ErrorClass::Transient`] with `retry_after` set. The internal
//! classification is what drives retry policy; the public class is what the
//! session acts on. Widening `ErrorClass` is a `parlevox-core` change and out
//! of scope for this crate.

use parlevox_core::{DurationMs, ErrorClass, ParlevoxError};

/// A connector-local failure with an explicit retry decision.
///
/// Marked non-exhaustive so a new wire condition can be added without breaking
/// downstream matches.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ConnectorError {
    /// The provider refused the request with a retryable rate limit.
    ///
    /// Retried only before the first body byte.
    #[error("rate limited")]
    RateLimited {
        /// Delay the provider asked the caller to observe, when it said so.
        retry_after: Option<DurationMs>,
    },
    /// A failure worth retrying before the first body byte.
    #[error("transient connector failure: {0}")]
    Transient(String),
    /// A failure retrying cannot fix: authentication, quota, missing route.
    #[error("fatal connector failure: {0}")]
    Fatal(String),
    /// The peer violated the wire contract.
    #[error("connector protocol violation: {0}")]
    Protocol(String),
    /// A bounded connector buffer could not accept the payload.
    #[error("connector capacity exhausted: {0}")]
    Capacity(String),
    /// The transport itself failed: connection, DNS, TLS, body read.
    #[error("connector transport failure: {0}")]
    Transport(String),
    /// The caller asked to stop.
    #[error("cancelled")]
    Cancelled,
}

impl ConnectorError {
    /// Classify a non-success HTTP status.
    ///
    /// Returns `None` for any `2xx`. `429` is [`ConnectorError::RateLimited`];
    /// `408`, `425` and `5xx` are transient; other `4xx` statuses are fatal
    /// because the same request will not succeed on retry; anything else is a
    /// protocol violation.
    #[must_use]
    pub fn from_status(status: u16, retry_after: Option<DurationMs>) -> Option<Self> {
        match status {
            200..=299 => None,
            429 => Some(ConnectorError::RateLimited { retry_after }),
            408 | 425 | 500..=599 => Some(ConnectorError::Transient(format!(
                "provider returned HTTP {status}"
            ))),
            400..=499 => Some(ConnectorError::Fatal(format!(
                "provider returned HTTP {status}"
            ))),
            _ => Some(ConnectorError::Protocol(format!(
                "unexpected HTTP status {status}"
            ))),
        }
    }

    /// Whether retrying before the first body byte could succeed.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ConnectorError::RateLimited { .. }
                | ConnectorError::Transient(_)
                | ConnectorError::Transport(_)
        )
    }

    /// Project this connector failure onto the public error the session sees.
    ///
    /// The provider name is supplied by the connector so the runtime never
    /// has to guess which provider produced the failure.
    #[must_use]
    pub fn into_provider(self, provider: &str) -> ParlevoxError {
        match self {
            ConnectorError::RateLimited { retry_after } => ParlevoxError::Provider {
                provider: provider.to_owned(),
                message: "provider rate limited the request".to_owned(),
                class: ErrorClass::Transient,
                retry_after,
            },
            ConnectorError::Transient(message) => ParlevoxError::Provider {
                provider: provider.to_owned(),
                message,
                class: ErrorClass::Transient,
                retry_after: None,
            },
            ConnectorError::Fatal(message) => ParlevoxError::Provider {
                provider: provider.to_owned(),
                message,
                class: ErrorClass::Fatal,
                retry_after: None,
            },
            ConnectorError::Protocol(message) => ParlevoxError::Protocol(message),
            ConnectorError::Capacity(message) => ParlevoxError::Capacity(message),
            ConnectorError::Transport(message) => ParlevoxError::Transport(message),
            ConnectorError::Cancelled => ParlevoxError::Cancelled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_and_rate_limit_are_classified() {
        assert_eq!(ConnectorError::from_status(204, None), None);
        let limited = ConnectorError::from_status(429, Some(DurationMs::from_millis(1500)))
            .expect("429 is an error");
        assert_eq!(
            limited,
            ConnectorError::RateLimited {
                retry_after: Some(DurationMs::from_millis(1500))
            }
        );
        assert!(limited.is_retryable());
    }

    #[test]
    fn server_errors_are_transient_and_client_errors_fatal() {
        assert!(ConnectorError::from_status(503, None)
            .expect("503 is an error")
            .is_retryable());
        let denied = ConnectorError::from_status(401, None).expect("401 is an error");
        assert!(!denied.is_retryable());
        assert!(matches!(denied, ConnectorError::Fatal(_)));
        assert!(!ConnectorError::from_status(302, None)
            .expect("302 is an error")
            .is_retryable());
    }

    #[test]
    fn rate_limit_maps_to_transient_with_retry_after() {
        // Core has no RateLimit variant; the retry hint survives anyway.
        let error = ConnectorError::RateLimited {
            retry_after: Some(DurationMs::from_millis(900)),
        }
        .into_provider("openai-llm");
        assert_eq!(error.class(), ErrorClass::Transient);
        match error {
            ParlevoxError::Provider { retry_after, .. } => {
                assert_eq!(retry_after, Some(DurationMs::from_millis(900)));
            }
            other => panic!("expected provider error, got {other:?}"),
        }
    }

    #[test]
    fn cancellation_stays_cancellation() {
        let error = ConnectorError::Cancelled.into_provider("x");
        assert!(matches!(error, ParlevoxError::Cancelled));
    }
}
