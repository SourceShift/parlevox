//! The HTTP toolkit: auth, deadline, pre-first-byte retry, classification.
//!
//! [`HttpToolkit::send`] is the single request path every connector uses. It
//! injects auth once, then attempts the request until a success status is
//! seen, the failure is not retryable, or the attempt budget is spent. Retry
//! never happens after a body is handed back: once a `2xx` response exists,
//! the toolkit returns it and a later body failure propagates unchanged.

use std::sync::Arc;
use std::time::Duration;

use parlevox_core::DurationMs;
use parlevox_providers::RetryPolicy;
use tokio_util::sync::CancellationToken;

use crate::auth::Auth;
use crate::buffer::ConnectorStats;
use crate::error::ConnectorError;
use crate::stream::BodySource;
use crate::transport::{ByteStream, RequestSpec, Transport, TransportResponse};

/// An authenticated request path shared by every connector.
#[derive(Clone)]
pub struct HttpToolkit {
    transport: Arc<dyn Transport>,
    auth: Auth,
    timeout: Duration,
    retry: RetryPolicy,
    max_buffer_bytes: usize,
    stats: Arc<ConnectorStats>,
}

impl HttpToolkit {
    /// Build a toolkit.
    ///
    /// A zero `timeout` disables the per-request deadline; every other value
    /// bounds the whole request, including body streaming.
    #[must_use]
    pub fn new(
        transport: Arc<dyn Transport>,
        auth: Auth,
        timeout: Duration,
        retry: RetryPolicy,
        max_buffer_bytes: usize,
    ) -> Self {
        HttpToolkit {
            transport,
            auth,
            timeout,
            retry,
            max_buffer_bytes,
            stats: Arc::new(ConnectorStats::new()),
        }
    }

    /// Shared counters for every request and stream this toolkit creates.
    #[must_use]
    pub fn stats(&self) -> &Arc<ConnectorStats> {
        &self.stats
    }

    /// The byte bound applied to line buffers and whole-JSON bodies.
    #[must_use]
    pub fn max_buffer_bytes(&self) -> usize {
        self.max_buffer_bytes
    }

    /// Send one request, retrying only pre-first-byte retryable failures.
    ///
    /// # Errors
    ///
    /// Returns the classified [`ConnectorError`] for a transport failure, a
    /// non-success status, cancellation, or an exhausted deadline.
    pub async fn send(
        &self,
        mut request: RequestSpec,
        cancel: &CancellationToken,
    ) -> Result<HttpResponse, ConnectorError> {
        self.auth.apply(&mut request)?;
        let max_attempts = self.retry.max_attempts.max(1);
        let deadline = if self.timeout.is_zero() {
            None
        } else {
            tokio::time::Instant::now().checked_add(self.timeout)
        };
        let mut attempt: u32 = 0;
        loop {
            attempt = attempt.saturating_add(1);
            self.stats.record_attempt();
            if cancel.is_cancelled() {
                return Err(ConnectorError::Cancelled);
            }
            tracing::debug!(
                method = request.method.as_str(),
                url = request.safe_url(),
                attempt,
                "connector request"
            );
            match self.execute_once(&request, cancel, deadline).await {
                Ok(response) => return Ok(self.finish(response, deadline)),
                Err(error) => {
                    let retryable = error.is_retryable();
                    if !retryable || attempt >= max_attempts {
                        return Err(error);
                    }
                    if cancel.is_cancelled() {
                        return Err(ConnectorError::Cancelled);
                    }
                    self.stats.record_retry();
                    let delay = self.retry.delay_for(attempt.saturating_add(1));
                    tracing::debug!(attempt, ?delay, "retrying connector request");
                    if !delay.is_zero() && !sleep_or_cancel(delay, cancel, deadline).await {
                        return Err(ConnectorError::Cancelled);
                    }
                }
            }
        }
    }

    async fn execute_once(
        &self,
        request: &RequestSpec,
        cancel: &CancellationToken,
        deadline: Option<tokio::time::Instant>,
    ) -> Result<TransportResponse, ConnectorError> {
        let cancel = cancel.clone();
        let future = self.transport.execute(request.clone());
        let result = match deadline {
            Some(deadline) => tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(ConnectorError::Cancelled),
                outcome = tokio::time::timeout_at(deadline, future) => match outcome {
                    Ok(result) => result,
                    Err(_) => {
                        return Err(ConnectorError::Transient(
                            "request deadline exceeded".to_owned(),
                        ))
                    }
                },
            },
            None => tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(ConnectorError::Cancelled),
                result = future => result,
            },
        };
        let response = result?;
        if let Some(error) = ConnectorError::from_status(response.status, retry_after(&response)) {
            return Err(error);
        }
        Ok(response)
    }

    fn finish(
        &self,
        response: TransportResponse,
        deadline: Option<tokio::time::Instant>,
    ) -> HttpResponse {
        HttpResponse {
            status: response.status,
            content_type: response.content_type().map(ToOwned::to_owned),
            body: response.body,
            deadline,
            stats: Arc::clone(&self.stats),
        }
    }
}

/// Wait for `delay`, unless cancellation or the request deadline wins first.
///
/// Returns `false` when cancellation won.
async fn sleep_or_cancel(
    delay: Duration,
    cancel: &CancellationToken,
    deadline: Option<tokio::time::Instant>,
) -> bool {
    let cancel = cancel.clone();
    match deadline {
        Some(deadline) => {
            tokio::select! {
                biased;
                () = cancel.cancelled() => false,
                _ = tokio::time::timeout_at(deadline, tokio::time::sleep(delay)) => true,
            }
        }
        None => {
            tokio::select! {
                biased;
                () = cancel.cancelled() => false,
                () = tokio::time::sleep(delay) => true,
            }
        }
    }
}

/// Parse a `Retry-After` delta-seconds header. HTTP-date values are ignored.
fn retry_after(response: &TransportResponse) -> Option<DurationMs> {
    let seconds: u64 = response.header("retry-after")?.trim().parse().ok()?;
    Some(DurationMs::from_millis(seconds.saturating_mul(1000)))
}

/// A successful response head plus its body stream.
pub struct HttpResponse {
    /// HTTP status code (`2xx` for a response the toolkit returns).
    pub status: u16,
    /// The `content-type` the body should be decoded as.
    pub content_type: Option<String>,
    body: ByteStream,
    deadline: Option<tokio::time::Instant>,
    stats: Arc<ConnectorStats>,
}

impl HttpResponse {
    /// Take the body stream.
    #[must_use]
    pub fn into_body(self) -> ByteStream {
        self.body
    }

    pub(crate) fn into_body_source(self, cancel: CancellationToken) -> BodySource {
        BodySource::new(self.body, cancel, self.deadline, self.stats)
    }
}
