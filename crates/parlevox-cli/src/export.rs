//! Opt-in OTLP/HTTP trace export for the CLI demos.
//!
//! Enabled only by the `otlp` cargo feature. The runtime never installs an
//! exporter: it exposes a caller-supplied tracer hook and the application
//! decides where spans go. This module is that application-side hook — it
//! builds one OTLP/HTTP-protobuf exporter from
//! `OTEL_EXPORTER_OTLP_ENDPOINT`, installs it as the process-global tracer
//! provider exactly once, and flushes it when the demo exits.
//!
//! Installing is process-global and idempotent; flushing is best-effort, so a
//! failed export is reported on stderr once and never turns a completed turn
//! into a failed one.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Mutex, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use opentelemetry::global;
use opentelemetry::global::BoxedTracer;
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::trace::SpanExporter as SdkSpanExporter;
use opentelemetry_sdk::trace::{BatchSpanProcessor, SdkTracerProvider, SpanData};
use opentelemetry_sdk::Resource;
use parlevox_core::ParlevoxError;

/// Environment variable naming the OTLP/HTTP collector base endpoint.
pub const ENDPOINT_ENV: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";

/// Environment variable overriding the `service.name` resource attribute.
pub const SERVICE_NAME_ENV: &str = "OTEL_SERVICE_NAME";

/// Service name used when [`SERVICE_NAME_ENV`] is unset or empty.
pub const DEFAULT_SERVICE_NAME: &str = "parlevox";

/// Instrumentation scope name handed to the runtime tracer.
pub const TRACER_NAME: &str = "parlevox";

/// Upper bound on the shutdown export drain.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Per-request bound on one OTLP export.
///
/// Deliberately below [`SHUTDOWN_TIMEOUT`]: a collector that accepts the
/// connection and never answers must not outlast the drain, so the batch
/// worker is idle again by the time the process exits.
const EXPORT_TIMEOUT: Duration = Duration::from_secs(4);

/// The process-global provider installed by [`install`].
///
/// Holding the provider here keeps it alive for the whole process and makes a
/// second [`install`] an `Ok` no-op, matching the one-global-provider rule the
/// OpenTelemetry API enforces.
static PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

/// Serializes the build-and-install critical section so the global provider is
/// set exactly once even when two callers race.
static INSTALL_LOCK: Mutex<()> = Mutex::new(());

/// Resolved exporter settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceExportConfig {
    /// Collector base endpoint, resolved verbatim from
    /// [`ENDPOINT_ENV`]. The `/v1/traces` signal path is appended when the
    /// exporter is built.
    pub endpoint: String,
    /// `service.name` resource attribute.
    pub service_name: String,
}

impl TraceExportConfig {
    /// Resolve settings from the process environment.
    ///
    /// Returns `None` when [`ENDPOINT_ENV`] is absent or empty, which is the
    /// documented way to leave the exporter disabled.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// Resolve settings from an arbitrary key/value lookup.
    ///
    /// A pure function of `lookup`, so it never reads process state and is
    /// safe to exercise in tests. Returns `None` when the endpoint is absent or
    /// empty. An absent or empty [`SERVICE_NAME_ENV`] falls back to
    /// [`DEFAULT_SERVICE_NAME`].
    #[must_use]
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let endpoint = lookup(ENDPOINT_ENV)?;
        if endpoint.is_empty() {
            return None;
        }
        let service_name = lookup(SERVICE_NAME_ENV)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| DEFAULT_SERVICE_NAME.to_owned());
        Some(Self {
            endpoint,
            service_name,
        })
    }
}

/// Install the OTLP/HTTP-protobuf trace exporter as the global tracer provider.
///
/// The first call builds a [`BatchSpanProcessor`]-backed [`SdkTracerProvider`]
/// whose resource carries `service.name`, and publishes it through
/// `opentelemetry::global::set_tracer_provider`. Span end only enqueues, so a
/// turn never performs export work; the batch worker exports on its own
/// schedule, and [`flush_and_shutdown`] drains it on exit. Every later call is
/// an `Ok` no-op, so a second demo in the same process cannot install a second
/// provider or invalidate the first.
///
/// # Errors
///
/// Returns a transport-class error naming the endpoint when the exporter
/// cannot be built (for example a malformed endpoint), or a protocol-class
/// error if the installer lock is poisoned, the process is not inside a tokio
/// runtime (the batch worker needs a reactor to export), or the provider was
/// installed by a racing caller.
pub fn install(config: &TraceExportConfig) -> Result<(), ParlevoxError> {
    if PROVIDER.get().is_some() {
        return Ok(());
    }
    let _guard = INSTALL_LOCK
        .lock()
        .map_err(|_| ParlevoxError::Protocol("otlp exporter install lock is poisoned".into()))?;
    if PROVIDER.get().is_some() {
        return Ok(());
    }
    // The SDK's batch worker exports on its own std thread with
    // `futures_executor::block_on`, where no tokio reactor is current. Capture
    // the demo's runtime handle now and re-enter it around every export poll.
    let handle = tokio::runtime::Handle::try_current().map_err(|_| {
        ParlevoxError::Protocol("otlp exporter install needs a tokio runtime context".into())
    })?;
    let exporter = SpanExporter::builder()
        .with_http()
        .with_endpoint(traces_endpoint(&config.endpoint))
        .with_timeout(EXPORT_TIMEOUT)
        .build()
        .map_err(|err| {
            ParlevoxError::Transport(format!(
                "otlp exporter for {} could not be built: {err}",
                config.endpoint
            ))
        })?;
    let exporter = RuntimeBoundExporter::new(exporter, handle);
    let resource = Resource::builder()
        .with_service_name(config.service_name.clone())
        .build();
    let provider = SdkTracerProvider::builder()
        .with_resource(resource)
        .with_span_processor(BatchSpanProcessor::builder(exporter).build())
        .build();
    global::set_tracer_provider(provider.clone());
    PROVIDER
        .set(provider)
        .map_err(|_| ParlevoxError::Protocol("otlp exporter install raced".into()))
}

/// A [`SpanExporter`](SdkSpanExporter) bound to the tokio runtime it was
/// installed from.
///
/// `BatchSpanProcessor` drives exports with `futures_executor::block_on` on its
/// own std thread. The OTLP HTTP client's reqwest/hyper connect path needs a
/// tokio reactor, so each poll of the inner export future runs with a captured
/// [`tokio::runtime::Handle`] entered.
#[derive(Debug)]
struct RuntimeBoundExporter {
    inner: SpanExporter,
    handle: tokio::runtime::Handle,
}

impl RuntimeBoundExporter {
    fn new(inner: SpanExporter, handle: tokio::runtime::Handle) -> Self {
        Self { inner, handle }
    }
}

impl SdkSpanExporter for RuntimeBoundExporter {
    fn export(&self, batch: Vec<SpanData>) -> impl Future<Output = OTelSdkResult> + Send {
        RuntimeBoundFuture {
            handle: self.handle.clone(),
            inner: Box::pin(self.inner.export(batch)),
        }
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }
}

/// The future returned by [`RuntimeBoundExporter::export`].
///
/// Entering the handle on every poll (rather than once around the future) keeps
/// the reactor available for every await point the inner future reaches.
struct RuntimeBoundFuture<'a> {
    handle: tokio::runtime::Handle,
    inner: Pin<Box<dyn Future<Output = OTelSdkResult> + Send + 'a>>,
}

impl Future for RuntimeBoundFuture<'_> {
    type Output = OTelSdkResult;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let _guard = this.handle.enter();
        this.inner.as_mut().poll(cx)
    }
}

/// Append the OTLP/HTTP traces signal path to a base endpoint.
///
/// `opentelemetry-otlp` appends `/v1/traces` itself only when it reads
/// `OTEL_EXPORTER_OTLP_ENDPOINT` from the process environment; a
/// programmatically supplied endpoint is parsed verbatim. Since this module
/// owns the endpoint lookup, it must append the signal path too. One trailing
/// slash is tolerated so `http://host/` and `http://host` address the same
/// collector.
fn traces_endpoint(base: &str) -> String {
    format!("{}/v1/traces", base.trim_end_matches('/'))
}

/// The tracer a session should use once [`install`] has run.
///
/// Returns `None` before installation, which is how callers fall back to
/// [`parlevox_runtime::Session::new`] and its no-op global tracer.
#[must_use]
pub fn tracer() -> Option<BoxedTracer> {
    PROVIDER.get().map(|_| global::tracer(TRACER_NAME))
}

/// Flush and shut the installed exporter down, reporting failures once.
///
/// A no-op when [`install`] never ran. The flush and the shutdown share one
/// [`SHUTDOWN_TIMEOUT`] budget, so a black-holed collector cannot stretch the
/// drain to the sum of the two. `AlreadyShutdown` is ignored because shutdown
/// is idempotent. Any other error is printed to stderr as
/// `otlp flush failed: <error>` and is deliberately not returned: export is
/// observability, not turn semantics.
pub fn flush_and_shutdown() {
    let Some(provider) = PROVIDER.get() else {
        return;
    };
    let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
    let mut first_error = provider
        .force_flush()
        .err()
        .filter(|err| !matches!(err, OTelSdkError::AlreadyShutdown));
    let remaining = deadline.saturating_duration_since(Instant::now());
    if let Err(err) = provider.shutdown_with_timeout(remaining) {
        if !matches!(err, OTelSdkError::AlreadyShutdown) && first_error.is_none() {
            first_error = Some(err);
        }
    }
    if let Some(err) = first_error {
        eprintln!("otlp flush failed: {err}");
    }
}

#[cfg(test)]
mod tests {
    use super::{
        traces_endpoint, TraceExportConfig, DEFAULT_SERVICE_NAME, ENDPOINT_ENV, SERVICE_NAME_ENV,
    };

    #[test]
    fn traces_endpoint_appends_signal_path_without_duplicating_slash() {
        // Arrange: a bare endpoint and the same endpoint with a trailing slash.
        let bare = "http://127.0.0.1:4318";
        let slashed = "http://127.0.0.1:4318/";

        // Act.
        let from_bare = traces_endpoint(bare);
        let from_slashed = traces_endpoint(slashed);

        // Assert.
        assert_eq!(from_bare, "http://127.0.0.1:4318/v1/traces");
        assert_eq!(from_slashed, "http://127.0.0.1:4318/v1/traces");
    }

    fn lookup_from<'a>(entries: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |key| {
            entries
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    #[test]
    fn from_lookup_returns_none_when_endpoint_is_unset() {
        // Arrange: a lookup with no endpoint entry.
        let lookup = lookup_from(&[(SERVICE_NAME_ENV, "collector-test")]);

        // Act.
        let config = TraceExportConfig::from_lookup(lookup);

        // Assert.
        assert!(config.is_none());
    }

    #[test]
    fn from_lookup_returns_none_when_endpoint_is_empty() {
        // Arrange: an explicitly empty endpoint means "disabled".
        let lookup = lookup_from(&[(ENDPOINT_ENV, "")]);

        // Act.
        let config = TraceExportConfig::from_lookup(lookup);

        // Assert.
        assert!(config.is_none());
    }

    #[test]
    fn from_lookup_keeps_endpoint_verbatim_and_defaults_service_name() {
        // Arrange.
        let endpoint = "http://127.0.0.1:4318";
        let entries = [(ENDPOINT_ENV, endpoint)];
        let lookup = lookup_from(&entries);

        // Act.
        let config = TraceExportConfig::from_lookup(lookup);

        // Assert.
        let config = config.expect("a non-empty endpoint must produce a config");
        assert_eq!(config.endpoint, endpoint);
        assert_eq!(config.service_name, DEFAULT_SERVICE_NAME);
    }

    #[test]
    fn from_lookup_honours_service_name_override() {
        // Arrange.
        let lookup = lookup_from(&[
            (ENDPOINT_ENV, "http://collector.test:4318"),
            (SERVICE_NAME_ENV, "parlevox-test"),
        ]);

        // Act.
        let config = TraceExportConfig::from_lookup(lookup);

        // Assert.
        let config = config.expect("a non-empty endpoint must produce a config");
        assert_eq!(config.service_name, "parlevox-test");
    }

    #[test]
    fn from_lookup_falls_back_when_service_name_is_empty() {
        // Arrange.
        let lookup = lookup_from(&[
            (ENDPOINT_ENV, "http://collector.test:4318"),
            (SERVICE_NAME_ENV, ""),
        ]);

        // Act.
        let config = TraceExportConfig::from_lookup(lookup);

        // Assert.
        let config = config.expect("a non-empty endpoint must produce a config");
        assert_eq!(config.service_name, DEFAULT_SERVICE_NAME);
    }
}
