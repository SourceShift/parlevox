//! A session with an installed [`Recorder`] writes one trace line per event
//! without changing whether the turn itself succeeds.

use std::io::Write;
use std::sync::{Arc, Mutex};

use parlevox_core::Observation;
use parlevox_runtime::{
    demo_utterance, ProviderNames, ProviderRegistry, Recorder, RecorderConfig, Session,
    SessionConfig,
};

#[derive(Clone, Default)]
struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| std::io::Error::other("poisoned"))?
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl SharedBuffer {
    fn contents(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

/// A writer that accepts the header and then fails every record write.
struct FailsAfterHeader {
    written: usize,
}

impl Write for FailsAfterHeader {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.written == 0 {
            self.written = buf.len();
            return Ok(buf.len());
        }
        Err(std::io::Error::other("disk full"))
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn fake_session() -> Session {
    let registry = ProviderRegistry::fakes().expect("fake registry");
    let providers = registry
        .resolve(&ProviderNames::default())
        .expect("fake providers resolve");
    Session::new(SessionConfig::default(), providers).expect("session")
}

#[tokio::test]
async fn recorded_turn_is_header_then_parseable_observations() {
    let buffer = SharedBuffer::default();
    let recorder = Recorder::new(
        buffer.clone(),
        RecorderConfig {
            record_transcripts: true,
            ..RecorderConfig::default()
        },
    )
    .expect("recorder");
    let mut session = fake_session().with_recorder(recorder);

    let report = session
        .run_turn(demo_utterance(16_000, 1, 40))
        .await
        .expect("the recorded turn succeeds");

    let stats = session.recorder_stats().expect("recorder installed");
    assert!(stats.records_written > 0, "no observations were recorded");
    assert_eq!(stats.dropped_cap, 0);
    assert_eq!(stats.dropped_write, 0);

    let text = buffer.contents();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(
        lines[0], r#"{"observation_schema_version":1}"#,
        "the first line is the schema header"
    );
    assert_eq!(
        lines.len() as u64,
        1 + stats.records_written,
        "one line per written record"
    );
    for line in &lines[1..] {
        serde_json::from_str::<Observation>(line).expect("every record parses");
    }
    assert_eq!(report.transcript, "hello world");
}

#[tokio::test]
async fn a_failing_writer_does_not_fail_the_turn() {
    let recorder = Recorder::new(FailsAfterHeader { written: 0 }, RecorderConfig::default())
        .expect("the header itself succeeds");
    let mut session = fake_session().with_recorder(recorder);

    let result = session.run_turn(demo_utterance(16_000, 1, 40)).await;

    assert!(result.is_ok(), "recording must never fail the turn");
    let stats = session.recorder_stats().expect("recorder installed");
    assert!(
        stats.dropped_write > 0,
        "every failed write must be counted, got {stats:?}"
    );
}

#[tokio::test]
async fn a_tiny_cap_counts_drops_and_leaves_the_turn_intact() {
    let recorder = Recorder::new(
        SharedBuffer::default(),
        RecorderConfig {
            max_records: 1,
            ..RecorderConfig::default()
        },
    )
    .expect("recorder");
    let mut session = fake_session().with_recorder(recorder);

    let report = session
        .run_turn(demo_utterance(16_000, 1, 40))
        .await
        .expect("a capped recorder does not fail the turn");

    let stats = session.recorder_stats().expect("recorder installed");
    assert_eq!(stats.records_written, 1, "the cap bounds written records");
    assert!(stats.dropped_cap > 0, "cap hits are counted, got {stats:?}");
    assert_eq!(report.transcript, "hello world");
}
