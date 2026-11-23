//! Error taxonomy.
//!
//! The class, not the message, decides what the session runtime does:
//! [`ErrorClass::Transient`] is retried or degraded, [`ErrorClass::Fatal`]
//! closes the session, [`ErrorClass::Protocol`] marks a plugin bug, and
//! [`ErrorClass::Cancelled`] is never surfaced to the user.

use serde::{Deserialize, Serialize};

use crate::clock::DurationMs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    /// Retry, fall back, or degrade. The session survives.
    Transient,
    /// The session cannot continue.
    Fatal,
    /// A peer violated the wire contract.
    Protocol,
    /// The caller asked for this; not an incident.
    Cancelled,
    /// A bounded buffer had no room under the current policy.
    Capacity,
}

#[derive(Debug, thiserror::Error)]
pub enum ParlevoxError {
    #[error("transport failure: {0}")]
    Transport(String),

    #[error("provider '{provider}' failed ({class:?}): {message}")]
    Provider {
        provider: String,
        message: String,
        class: ErrorClass,
        /// How long the caller should wait before retrying, when known.
        retry_after: Option<DurationMs>,
    },

    #[error("protocol violation: {0}")]
    Protocol(String),

    #[error("capacity exhausted: {0}")]
    Capacity(String),

    #[error("session closed")]
    Closed,

    #[error("cancelled")]
    Cancelled,
}

impl ParlevoxError {
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        ParlevoxError::Provider {
            provider: provider.into(),
            message: message.into(),
            class: ErrorClass::Transient,
            retry_after: None,
        }
    }

    pub fn fatal(provider: impl Into<String>, message: impl Into<String>) -> Self {
        ParlevoxError::Provider {
            provider: provider.into(),
            message: message.into(),
            class: ErrorClass::Fatal,
            retry_after: None,
        }
    }

    pub fn class(&self) -> ErrorClass {
        match self {
            ParlevoxError::Transport(_) => ErrorClass::Transient,
            ParlevoxError::Provider { class, .. } => *class,
            ParlevoxError::Protocol(_) => ErrorClass::Protocol,
            ParlevoxError::Capacity(_) => ErrorClass::Capacity,
            ParlevoxError::Closed => ErrorClass::Fatal,
            ParlevoxError::Cancelled => ErrorClass::Cancelled,
        }
    }

    /// Whether the session may continue after this error.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self.class(),
            ErrorClass::Transient | ErrorClass::Cancelled | ErrorClass::Capacity
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_is_recoverable() {
        let e = ParlevoxError::Transport("socket reset".into());
        assert_eq!(e.class(), ErrorClass::Transient);
        assert!(e.is_recoverable());
    }

    #[test]
    fn fatal_provider_error_closes_session() {
        let e = ParlevoxError::fatal("openai", "401 unauthorized");
        assert!(!e.is_recoverable());
    }

    #[test]
    fn cancellation_is_not_an_incident() {
        let e = ParlevoxError::Cancelled;
        assert_eq!(e.class(), ErrorClass::Cancelled);
        assert!(e.is_recoverable());
    }
}
