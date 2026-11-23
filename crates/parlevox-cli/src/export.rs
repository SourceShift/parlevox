//! Opt-in OTLP/HTTP trace and metric export for the CLI demos.
//!
//! Enabled only by the `otlp` cargo feature. The runtime never installs an
//! exporter: it exposes a caller-supplied tracer hook and meter, and the
//! application decides where telemetry goes. This module is that
//! application-side hook — it builds OTLP/HTTP-protobuf trace and metric
//! exporters from `OTEL_EXPORTER_OTLP_ENDPOINT`, installs both against one
//! shared resource exactly once, and drains them when the demo exits.
//!
//! Installing is process-global and idempotent; flushing is best-effort, so a
//! failed export is reported on stderr once and never turns a completed turn
//! into a failed one.

use std::future::Future;
use std::pin::Pin;
use std::sync::{mpsc, Mutex, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use opentelemetry::global;
use opentelemetry::global::BoxedTracer;
use opentelemetry::metrics::{Meter, MeterProvider};
use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig};
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::metrics::data::ResourceMetrics;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider, Temporality};
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

/// Instrumentation scope name handed to the runtime meter.
pub const METER_NAME: &str = "parlevox";

/// Upper bound on the shutdown export drain.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Per-request bound on one OTLP export.
///
/// Deliberately below [`SHUTDOWN_TIMEOUT`]: a collector that accepts the
/// connection and never answers must not outlast the drain, so the export
/// workers are idle again by the time the process exits.
const EXPORT_TIMEOUT: Duration = Duration::from_secs(4);

/// The process-global providers installed by [`install`].
///
/// Holding both providers here keeps them alive for the whole process and
/// makes a second [`install`] an `Ok` no-op. They are installed as a pair so
/// the tracer and the meter always share one endpoint and one resource. The
/// dedicated export runtime is stored beside them so it outlives the demo's
/// own runtime: the SDK drains spans and metrics on detached threads, and a
/// drain that outlives the caller's budget must never poll a runtime that is
/// already shutting down (tokio would panic).
#[derive(Debug)]
struct InstalledProviders {
    tracer: SdkTracerProvider,
    meter: SdkMeterProvider,
    _export_runtime: tokio::runtime::Runtime,
}

/// The process-global provider pair installed by [`install`].
static PROVIDERS: OnceLock<InstalledProviders> = OnceLock::new();

/// Serializes the build-and-install critical section so the global provider is
/// set exactly once even when two callers race.
static INSTALL_LOCK: Mutex<()> = Mutex::new(());

/// Resolved exporter settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceExportConfig {
    /// Collector base endpoint, resolved verbatim from
    /// [`ENDPOINT_ENV`]. The `/v1/traces` and `/v1/metrics` signal paths are
    /// appended when the exporters are built.
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

/// Install the OTLP/HTTP-protobuf trace and metric exporters once.
///
/// The first call builds a [`BatchSpanProcessor`]-backed [`SdkTracerProvider`]
/// and a [`PeriodicReader`]-backed [`SdkMeterProvider`] from one shared
/// resource, publishes the tracer through
/// `opentelemetry::global::set_tracer_provider`, and retains both providers
/// here. Span end only enqueues, so a turn never performs export work; the
/// batch worker exports on its own schedule and [`flush_and_shutdown`] drains
/// both providers on exit. Every later call is an `Ok` no-op, so a second demo
/// in the same process cannot install a second pair or invalidate the first.
///
/// # Errors
///
/// Returns a transport-class error naming the endpoint when an exporter cannot
/// be built (for example a malformed endpoint) or the dedicated export runtime
/// cannot start, or a protocol-class error if the installer lock is poisoned or
/// the provider pair was installed by a racing caller.
pub fn install(config: &TraceExportConfig) -> Result<(), ParlevoxError> {
    if PROVIDERS.get().is_some() {
        return Ok(());
    }
    let _guard = INSTALL_LOCK
        .lock()
        .map_err(|_| ParlevoxError::Protocol("otlp exporter install lock is poisoned".into()))?;
    if PROVIDERS.get().is_some() {
        return Ok(());
    }
    // The SDK's batch and periodic workers export on their own detached std
    // threads with `futures_executor::block_on`, where no tokio reactor is
    // current. Give those exports a runtime this module owns and keeps alive
    // for the whole process, rather than the demo's runtime: a drain that runs
    // past the caller's budget is abandoned, and if the demo runtime is torn
    // down underneath it tokio panics on the next timer registration. Entering
    // this dedicated handle around every poll decouples an abandoned export
    // from the demo's lifetime.
    let export_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("parlevox-otlp-export")
        .enable_all()
        .build()
        .map_err(|err| {
            ParlevoxError::Transport(format!("otlp export runtime could not start: {err}"))
        })?;
    let handle = export_runtime.handle().clone();
    // Build the exporters and providers inside the dedicated runtime context so
    // anything that resolves the current runtime at construction sees this one.
    let (tracer, meter) = {
        let _runtime_guard = export_runtime.enter();
        let trace_exporter = SpanExporter::builder()
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
        let metric_exporter = MetricExporter::builder()
            .with_http()
            .with_endpoint(metrics_endpoint(&config.endpoint))
            .with_timeout(EXPORT_TIMEOUT)
            .build()
            .map_err(|err| {
                ParlevoxError::Transport(format!(
                    "otlp metric exporter for {} could not be built: {err}",
                    config.endpoint
                ))
            })?;
        let resource = Resource::builder()
            .with_service_name(config.service_name.clone())
            .build();
        let tracer = SdkTracerProvider::builder()
            .with_resource(resource.clone())
            .with_span_processor(
                BatchSpanProcessor::builder(RuntimeBoundExporter::new(
                    trace_exporter,
                    handle.clone(),
                ))
                .build(),
            )
            .build();
        let meter = SdkMeterProvider::builder()
            .with_resource(resource)
            .with_reader(
                PeriodicReader::builder(RuntimeBoundMetricExporter::new(metric_exporter, handle))
                    .build(),
            )
            .build();
        (tracer, meter)
    };
    global::set_tracer_provider(tracer.clone());
    PROVIDERS
        .set(InstalledProviders {
            tracer,
            meter,
            _export_runtime: export_runtime,
        })
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

/// A [`PushMetricExporter`] bound to the tokio runtime it was installed from.
///
/// `PeriodicReader` drives exports with `futures_executor::block_on` on its own
/// std thread, exactly like `BatchSpanProcessor`; the OTLP HTTP client still
/// needs a tokio reactor. Each export poll therefore runs with the captured
/// [`tokio::runtime::Handle`] entered.
#[derive(Debug)]
struct RuntimeBoundMetricExporter {
    inner: MetricExporter,
    handle: tokio::runtime::Handle,
}

impl RuntimeBoundMetricExporter {
    fn new(inner: MetricExporter, handle: tokio::runtime::Handle) -> Self {
        Self { inner, handle }
    }
}

impl PushMetricExporter for RuntimeBoundMetricExporter {
    async fn export(&self, metrics: &ResourceMetrics) -> OTelSdkResult {
        RuntimeBoundFuture {
            handle: self.handle.clone(),
            inner: Box::pin(self.inner.export(metrics)),
        }
        .await
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn temporality(&self) -> Temporality {
        self.inner.temporality()
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
    signal_endpoint(base, "traces")
}

/// Append the OTLP/HTTP metrics signal path to a base endpoint.
///
/// Same rules as [`traces_endpoint`]: the programmatic endpoint is parsed
/// verbatim, so the `/v1/metrics` signal path must be appended here, and one
/// trailing slash is tolerated.
fn metrics_endpoint(base: &str) -> String {
    signal_endpoint(base, "metrics")
}

/// Append one OTLP/HTTP signal path to a base endpoint, tolerating one
/// trailing slash.
fn signal_endpoint(base: &str, signal: &str) -> String {
    format!("{}/v1/{signal}", base.trim_end_matches('/'))
}

/// The tracer a session should use once [`install`] has run.
///
/// Returns `None` before installation, which is how callers fall back to
/// [`parlevox_runtime::Session::new`] and its no-op global tracer.
#[must_use]
pub fn tracer() -> Option<BoxedTracer> {
    PROVIDERS.get().map(|_| global::tracer(TRACER_NAME))
}

/// The meter a session should use once [`install`] has run.
///
/// Returns `None` before installation, which is how callers fall back to a
/// session without metric instruments. The meter is created from the installed
/// [`SdkMeterProvider`], so its six turn instruments export over the same
/// endpoint and resource as the traces.
#[must_use]
pub fn meter() -> Option<Meter> {
    PROVIDERS
        .get()
        .map(|providers| providers.meter.meter(METER_NAME))
}

/// Flush and shut both installed providers down, reporting failures once.
///
/// A no-op when [`install`] never ran. Traces drain first (flush, then
/// shutdown) and meters second, all inside one [`SHUTDOWN_TIMEOUT`] budget, so
/// a black-holed collector cannot stretch the drain to the sum of the
/// individual exports. `AlreadyShutdown` is ignored because shutdown is
/// idempotent. Any other error is printed to stderr as
/// `otlp flush failed: <error>` and is deliberately not returned: export is
/// observability, not turn semantics.
pub fn flush_and_shutdown() {
    let Some(providers) = PROVIDERS.get() else {
        return;
    };
    let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
    let mut first_error = providers
        .tracer
        .force_flush()
        .err()
        .filter(|err| !matches!(err, OTelSdkError::AlreadyShutdown));
    let remaining = deadline.saturating_duration_since(Instant::now());
    if let Err(err) = providers.tracer.shutdown_with_timeout(remaining) {
        if !matches!(err, OTelSdkError::AlreadyShutdown) && first_error.is_none() {
            first_error = Some(err);
        }
    }
    let remaining = deadline.saturating_duration_since(Instant::now());
    if let Err(err) = shutdown_meter_within(&providers.meter, remaining) {
        if !matches!(err, OTelSdkError::AlreadyShutdown) && first_error.is_none() {
            first_error = Some(err);
        }
    }
    if let Some(err) = first_error {
        eprintln!("otlp flush failed: {err}");
    }
}

/// Shut the meter provider down, performing its final collect-and-export,
/// without letting an unresponsive collector outlast `remaining`.
///
/// `SdkMeterProvider::shutdown_with_timeout` in 0.33.1 ignores its timeout
/// argument: it forwards to the reader, whose own shutdown waits a fixed five
/// seconds. The metered shutdown also *exports* (it is not a pure flush), so
/// calling `force_flush` first would export twice and blow the shared budget.
/// The single shutdown therefore runs on a detached named worker thread and
/// the caller waits at most `remaining`, reporting expiry as a timeout. The
/// thread is deliberately not joined: a stuck export owns an Arc-backed
/// provider handle and must not block a process exit that already expired.
fn shutdown_meter_within(provider: &SdkMeterProvider, remaining: Duration) -> OTelSdkResult {
    let provider = provider.clone();
    let (result_tx, result_rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("otlp-meter-drain".to_owned())
        .spawn(move || {
            let _ = result_tx.send(provider.shutdown_with_timeout(remaining));
        })
        .map_err(|err| {
            OTelSdkError::InternalFailure(format!("otlp meter drain worker could not start: {err}"))
        })?;
    match result_rx.recv_timeout(remaining) {
        Ok(result) => result,
        Err(mpsc::RecvTimeoutError::Timeout) => Err(OTelSdkError::Timeout(remaining)),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(OTelSdkError::InternalFailure(
            "otlp meter drain worker exited without a result".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        metrics_endpoint, traces_endpoint, TraceExportConfig, DEFAULT_SERVICE_NAME, ENDPOINT_ENV,
        SERVICE_NAME_ENV,
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

    #[test]
    fn metrics_endpoint_appends_signal_path_without_duplicating_slash() {
        // Arrange: a bare endpoint and the same endpoint with a trailing slash.
        let bare = "http://127.0.0.1:4318";
        let slashed = "http://127.0.0.1:4318/";

        // Act.
        let from_bare = metrics_endpoint(bare);
        let from_slashed = metrics_endpoint(slashed);

        // Assert.
        assert_eq!(from_bare, "http://127.0.0.1:4318/v1/metrics");
        assert_eq!(from_slashed, "http://127.0.0.1:4318/v1/metrics");
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
