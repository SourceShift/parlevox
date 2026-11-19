//! Bounded restart policy for a crashed plugin.
//!
//! A crash is a transport failure, not a reason to spin: restarts are capped
//! and the delay grows exponentially up to a ceiling. A healthy client is
//! reused; a dead one is killed and replaced.

use std::sync::Arc;
use std::time::Duration;

use voicekit_core::VoicekitError;

use crate::client::SidecarClient;
use crate::config::SidecarConfig;

#[derive(Debug, Clone)]
pub struct RestartPolicy {
    pub max_restarts: u32,
    pub base_backoff_ms: u64,
    pub max_backoff_ms: u64,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        RestartPolicy {
            max_restarts: 3,
            base_backoff_ms: 50,
            max_backoff_ms: 1_000,
        }
    }
}

impl RestartPolicy {
    /// Delay before restart number `attempt` (1-based), capped.
    pub fn backoff_ms(&self, attempt: u32) -> u64 {
        if attempt == 0 {
            return 0;
        }
        let shift = attempt.saturating_sub(1).min(32);
        let delay = self.base_backoff_ms.saturating_mul(1u64 << shift.min(63));
        delay.min(self.max_backoff_ms)
    }
}

pub struct SidecarSupervisor {
    config: SidecarConfig,
    policy: RestartPolicy,
    client: Option<Arc<SidecarClient>>,
    restarts: u32,
}

impl SidecarSupervisor {
    pub fn new(config: SidecarConfig, policy: RestartPolicy) -> Self {
        SidecarSupervisor {
            config,
            policy,
            client: None,
            restarts: 0,
        }
    }

    pub fn restarts(&self) -> u32 {
        self.restarts
    }

    pub fn client(&self) -> Option<Arc<SidecarClient>> {
        self.client.clone()
    }

    /// Return a live client, restarting (with backoff) when needed.
    ///
    /// # Errors
    ///
    /// Returns the transport-class error from the final spawn attempt when the
    /// restart budget is exhausted.
    pub async fn ensure(&mut self) -> Result<Arc<SidecarClient>, VoicekitError> {
        if let Some(client) = &self.client {
            if client.is_alive().await {
                return Ok(Arc::clone(client));
            }
        }
        self.restart().await
    }

    /// Replace the current client, retrying at most `max_restarts` times.
    ///
    /// # Errors
    ///
    /// Returns the transport-class error from the last failed spawn, or a
    /// protocol-class handshake error surfaced by the client, once the
    /// restart budget is exhausted.
    pub async fn restart(&mut self) -> Result<Arc<SidecarClient>, VoicekitError> {
        if let Some(client) = self.client.take() {
            self.restarts += 1;
            client.kill().await;
        }
        let mut attempt: u32 = 0;
        loop {
            match SidecarClient::spawn(self.config.clone()).await {
                Ok(client) => {
                    let client = Arc::new(client);
                    self.client = Some(Arc::clone(&client));
                    return Ok(client);
                }
                Err(err) => {
                    if attempt >= self.policy.max_restarts {
                        return Err(err);
                    }
                    attempt += 1;
                    self.restarts += 1;
                    tokio::time::sleep(Duration::from_millis(self.policy.backoff_ms(attempt)))
                        .await;
                }
            }
        }
    }

    /// Reap the plugin process.
    pub async fn shutdown(&mut self) {
        if let Some(client) = self.client.take() {
            let _ = client.shutdown().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_is_exponential_and_capped() {
        let policy = RestartPolicy {
            max_restarts: 5,
            base_backoff_ms: 100,
            max_backoff_ms: 500,
        };
        assert_eq!(policy.backoff_ms(0), 0);
        assert_eq!(policy.backoff_ms(1), 100);
        assert_eq!(policy.backoff_ms(2), 200);
        assert_eq!(policy.backoff_ms(3), 400);
        assert_eq!(policy.backoff_ms(4), 500);
        assert_eq!(policy.backoff_ms(10), 500);
    }

    #[tokio::test]
    async fn crashing_plugin_stops_after_bounded_restarts() {
        let config = SidecarConfig::program(
            std::env::var("VOICEKIT_PYTHON").unwrap_or_else(|_| "python3".into()),
            vec!["-c".into(), "import sys; sys.exit(1)".into()],
        )
        .with_handshake_timeout_ms(500);
        let policy = RestartPolicy {
            max_restarts: 2,
            base_backoff_ms: 1,
            max_backoff_ms: 2,
        };
        let mut supervisor = SidecarSupervisor::new(config, policy);
        let err = match supervisor.ensure().await {
            Err(err) => err,
            Ok(_) => panic!("expected a crashing plugin to exhaust restarts"),
        };
        assert!(matches!(err, VoicekitError::Transport(_)));
        assert_eq!(supervisor.restarts(), 2);
    }
}
