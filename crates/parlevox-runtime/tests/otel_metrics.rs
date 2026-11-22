//! Hermetic OpenTelemetry metric-set test.
//!
//! Enabled only by the `otel` feature. Nothing here touches the network, a
//! collector, or a process-global provider: the session is handed a `Meter`
//! backed by an in-memory metric exporter, so the test observes exactly the
//! metrics one fake turn produced. A session built without `with_meter` is the
//! opt-out proof: it records nothing even while a live provider exists.

#![cfg(feature = "otel")]

use opentelemetry::metrics::MeterProvider as _;
use opentelemetry::Value;
use opentelemetry_sdk::metrics::data::{
    AggregatedMetrics, HistogramDataPoint, Metric, MetricData, ResourceMetrics, SumDataPoint,
};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use parlevox_runtime::{demo_utterance, ProviderNames, ProviderRegistry, Session, SessionConfig};

/// Every metric the success path must export, in sorted order.
const EXPECTED_METRICS: [&str; 6] = [
    "parlevox.frames.dropped",
    "parlevox.playout.depth",
    "parlevox.sidecar.frames.dropped",
    "parlevox.turn.duration",
    "parlevox.turn.stage.duration",
    "parlevox.turns",
];

fn fake_session(config: SessionConfig) -> Session {
    let providers = ProviderRegistry::fakes()
        .expect("built-in fakes resolve")
        .resolve(&ProviderNames::default())
        .expect("fake providers resolve");
    Session::new(config, providers).expect("session builds")
}

fn collected(exporter: &InMemoryMetricExporter) -> Vec<ResourceMetrics> {
    exporter
        .get_finished_metrics()
        .expect("in-memory exporter returns metrics")
}

fn collected_metrics(collected: &[ResourceMetrics]) -> Vec<&Metric> {
    collected
        .iter()
        .flat_map(|resource| resource.scope_metrics())
        .flat_map(|scope| scope.metrics())
        .collect()
}

fn metric<'a>(collected: &'a [ResourceMetrics], name: &str) -> &'a Metric {
    collected_metrics(collected)
        .into_iter()
        .find(|metric| metric.name() == name)
        .unwrap_or_else(|| panic!("missing metric {name}"))
}

fn string_attribute<'a>(
    mut attributes: impl Iterator<Item = &'a opentelemetry::KeyValue>,
    key: &str,
) -> Option<&'a str> {
    attributes
        .find(|attribute| attribute.key.as_str() == key)
        .and_then(|attribute| match &attribute.value {
            Value::String(value) => Some(value.as_str()),
            _ => None,
        })
}

fn u64_sum_points(metric: &Metric) -> Vec<&SumDataPoint<u64>> {
    match metric.data() {
        AggregatedMetrics::U64(MetricData::Sum(sum)) => sum.data_points().collect(),
        other => panic!("expected a u64 sum for {}, got {other:?}", metric.name()),
    }
}

fn f64_histogram_points(metric: &Metric) -> Vec<&HistogramDataPoint<f64>> {
    match metric.data() {
        AggregatedMetrics::F64(MetricData::Histogram(histogram)) => {
            histogram.data_points().collect()
        }
        other => panic!(
            "expected an f64 histogram for {}, got {other:?}",
            metric.name()
        ),
    }
}

fn u64_histogram_points(metric: &Metric) -> Vec<&HistogramDataPoint<u64>> {
    match metric.data() {
        AggregatedMetrics::U64(MetricData::Histogram(histogram)) => {
            histogram.data_points().collect()
        }
        other => panic!(
            "expected a u64 histogram for {}, got {other:?}",
            metric.name()
        ),
    }
}

fn flush(exporter: &InMemoryMetricExporter, provider: &SdkMeterProvider) -> Vec<ResourceMetrics> {
    provider.force_flush().expect("meter provider flushes");
    collected(exporter)
}

#[tokio::test]
async fn successful_turn_records_turn_stage_and_drop_metrics() {
    // Arrange: one session handed a meter backed by an in-memory exporter.
    let exporter = InMemoryMetricExporter::default();
    let reader = PeriodicReader::builder(exporter.clone()).build();
    let provider = SdkMeterProvider::builder().with_reader(reader).build();
    let meter = provider.meter("parlevox-runtime-test");
    let mut session = fake_session(SessionConfig::default()).with_meter(meter);

    // Act: one complete fake turn, then force collection.
    let report = session
        .run_turn(demo_utterance(16_000, 1, 40))
        .await
        .expect("fake turn completes");
    assert!(
        !report.interrupted,
        "the default fake turn is uninterrupted"
    );
    let collected = flush(&exporter, &provider);

    // Assert: the collected name set is exactly the fixed six instruments.
    let mut names: Vec<&str> = collected_metrics(&collected)
        .into_iter()
        .map(|metric| metric.name())
        .collect();
    names.sort_unstable();
    assert_eq!(names, EXPECTED_METRICS, "exact metric-name set");

    // Exactly one ok turn.
    let turns = metric(&collected, "parlevox.turns");
    assert_eq!(turns.unit(), "{turn}");
    let turn_points = u64_sum_points(turns);
    assert_eq!(turn_points.len(), 1, "one turn outcome series");
    assert_eq!(
        string_attribute(turn_points[0].attributes(), "outcome"),
        Some("ok")
    );
    assert_eq!(turn_points[0].value(), 1);

    // Turn duration is one millisecond histogram sample.
    let turn_duration = metric(&collected, "parlevox.turn.duration");
    assert_eq!(turn_duration.unit(), "ms");
    let duration_points = f64_histogram_points(turn_duration);
    assert_eq!(duration_points.len(), 1);
    assert_eq!(duration_points[0].count(), 1);
    assert!(duration_points[0].sum() >= 0.0);
    assert_eq!(
        string_attribute(duration_points[0].attributes(), "outcome"),
        Some("ok")
    );

    // One stage point per stage of the sequential default path.
    let stage = metric(&collected, "parlevox.turn.stage.duration");
    assert_eq!(stage.unit(), "ms");
    let stage_points = f64_histogram_points(stage);
    assert_eq!(stage_points.len(), 5, "one point per stage");
    for name in ["vad", "stt", "llm", "tts", "playout"] {
        let point = stage_points
            .iter()
            .find(|point| string_attribute(point.attributes(), "stage") == Some(name))
            .unwrap_or_else(|| panic!("missing stage point for {name}"));
        assert_eq!(point.count(), 1, "{name} stage must be measured once");
    }

    // Drops are split by direction; the sidecar counter stands alone.
    let dropped = metric(&collected, "parlevox.frames.dropped");
    assert_eq!(dropped.unit(), "{frame}");
    let dropped_points = u64_sum_points(dropped);
    assert_eq!(dropped_points.len(), 3, "one series per drop direction");
    for direction in ["input", "output", "events"] {
        assert!(
            dropped_points.iter().any(|point| {
                string_attribute(point.attributes(), "direction") == Some(direction)
            }),
            "missing direction {direction}"
        );
    }
    let sidecar = metric(&collected, "parlevox.sidecar.frames.dropped");
    assert_eq!(sidecar.unit(), "{frame}");
    assert_eq!(u64_sum_points(sidecar).len(), 1);

    // Playout depth is sampled once, at turn end.
    let playout = metric(&collected, "parlevox.playout.depth");
    assert_eq!(playout.unit(), "{frame}");
    assert_eq!(u64_histogram_points(playout).len(), 1);
}

#[tokio::test]
async fn interrupted_turn_reports_outcome_and_skips_tts() {
    // Arrange: a turn configured to barge in after two reply deltas, so TTS
    // never starts.
    let exporter = InMemoryMetricExporter::default();
    let reader = PeriodicReader::builder(exporter.clone()).build();
    let provider = SdkMeterProvider::builder().with_reader(reader).build();
    let meter = provider.meter("parlevox-runtime-test");
    let config = SessionConfig {
        barge_in_after_deltas: Some(2),
        ..SessionConfig::default()
    };
    let mut session = fake_session(config).with_meter(meter);

    // Act
    let report = session
        .run_turn(demo_utterance(16_000, 1, 40))
        .await
        .expect("interrupted fake turn still reports");
    assert!(report.interrupted, "the barge-in must interrupt the turn");
    let collected = flush(&exporter, &provider);

    // Assert: the outcome is `interrupted`.
    let turn_points = u64_sum_points(metric(&collected, "parlevox.turns"));
    assert_eq!(turn_points.len(), 1);
    assert_eq!(
        string_attribute(turn_points[0].attributes(), "outcome"),
        Some("interrupted")
    );
    assert_eq!(turn_points[0].value(), 1);

    // Assert: no tts stage point exists, but the stages that ran did record.
    let stage_points = f64_histogram_points(metric(&collected, "parlevox.turn.stage.duration"));
    assert!(
        stage_points
            .iter()
            .all(|point| string_attribute(point.attributes(), "stage") != Some("tts")),
        "an interrupted turn must not record a tts stage"
    );
    assert!(
        stage_points
            .iter()
            .any(|point| string_attribute(point.attributes(), "stage") == Some("llm")),
        "the llm stage that ran must still be recorded"
    );

    // Assert: playout depth is still sampled exactly once at turn end.
    assert_eq!(
        u64_histogram_points(metric(&collected, "parlevox.playout.depth")).len(),
        1
    );
}

#[tokio::test]
async fn no_meter_records_nothing_and_report_matches() {
    // Arrange: a live provider and exporter that the no-meter session is never
    // handed.
    let exporter = InMemoryMetricExporter::default();
    let reader = PeriodicReader::builder(exporter.clone()).build();
    let provider = SdkMeterProvider::builder().with_reader(reader).build();
    let utterance = demo_utterance(16_000, 1, 40);

    // Act: one turn on a session with no meter.
    let mut no_meter_session = fake_session(SessionConfig::default());
    let no_meter_report = no_meter_session
        .run_turn(utterance.clone())
        .await
        .expect("no-meter fake turn completes");
    provider.force_flush().expect("meter provider flushes");

    // Assert: the exporter observed no metrics at all.
    assert!(exporter
        .get_finished_metrics()
        .unwrap_or_default()
        .is_empty());

    // Act: the same deterministic turn with a meter.
    let meter = provider.meter("parlevox-runtime-test");
    let mut meter_session = fake_session(SessionConfig::default()).with_meter(meter);
    let meter_report = meter_session
        .run_turn(utterance)
        .await
        .expect("meter-backed fake turn completes");

    // Assert: the observable turn report is identical either way.
    assert_eq!(no_meter_report, meter_report);
}
