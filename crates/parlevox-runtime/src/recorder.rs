//! Opt-in, bounded, secret-redacting JSONL observation recorder.
//!
//! A [`Recorder`] writes the trace-file contract documented in
//! [`parlevox_core::observation`]: one header line carrying
//! [`parlevox_core::OBSERVATION_SCHEMA_VERSION`], then one compact
//! [`Observation`] per line. It is installed on a [`crate::Session`] with
//! [`crate::Session::with_recorder`]; a session without one records nothing and
//! takes no extra clone or format on the event path.
//!
//! # Bounded and non-blocking
//!
//! Every write is bounded by [`RecorderConfig::max_records`] and
//! [`RecorderConfig::max_bytes`]. A cap hit or a write failure is counted in
//! [`RecorderStats`] and never returned to the turn: recording cannot fail or
//! end a turn. Writes are synchronous, so an operating system blocked on a full
//! disk can still add latency; that is the residual, documented cost of not
//! spawning a writer thread.
//!
//! # Secret redaction
//!
//! Every recorded transcript, delta and failure message is scanned for
//! credential-shaped substrings and each match is replaced with `[redacted]`.
//! A PEM private key is redacted as a block, not as a header: the replacement
//! runs from the `-----BEGIN … PRIVATE KEY` header through the first matching
//! `-----END … PRIVATE KEY-----` marker, so the base64 body never reaches the
//! trace. Without an END marker, the header is replaced together with every
//! following non-empty line made only of base64 bytes. Provider configuration
//! and API-key values are never passed to the recorder, so there is no second
//! path by which key material could reach a trace.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use parlevox_core::{
    Observation, ObservationKind, ObservationSequencer, ParlevoxError, OBSERVATION_SCHEMA_VERSION,
};
use serde::Serialize;

use crate::events::{observation_kind, SessionEvent, TimedSessionEvent};
use parlevox_core::TurnEventKind;

/// Replacement written in place of a credential-shaped match.
pub const REDACTED: &str = "[redacted]";

/// Bounds and transcript policy for a [`Recorder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecorderConfig {
    /// Maximum number of observation records written after the header.
    pub max_records: u64,
    /// Maximum number of record bytes written after the header.
    pub max_bytes: u64,
    /// Whether text-bearing transcript and delta records keep their text.
    ///
    /// When `false`, a final transcript is recorded as the text-free
    /// [`ObservationKind::Control`] for [`TurnEventKind::FinalTranscript`] and
    /// partial/delta records are omitted and counted in
    /// [`RecorderStats::transcripts_suppressed`]. The transcript field is
    /// absent, never blank.
    pub record_transcripts: bool,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        RecorderConfig {
            max_records: 100_000,
            max_bytes: 16 * 1024 * 1024,
            record_transcripts: false,
        }
    }
}

/// Cumulative counters for one [`Recorder`].
///
/// The accounting identity is
/// `records_written + dropped_cap + dropped_write + transcripts_suppressed`
/// equals the number of events offered to [`Recorder::record`], because an
/// event is either written, dropped for a cap, dropped for a write failure, or
/// suppressed by transcript gating.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecorderStats {
    /// Records written after the header.
    pub records_written: u64,
    /// Record bytes written after the header.
    pub bytes_written: u64,
    /// Records refused because `max_records` or `max_bytes` was reached.
    pub dropped_cap: u64,
    /// Records whose write failed.
    pub dropped_write: u64,
    /// Credential-shaped matches replaced with [`REDACTED`].
    pub redacted: u64,
    /// Text-bearing records omitted by transcript gating.
    pub transcripts_suppressed: u64,
}

/// Writes one bounded JSONL observation trace.
pub struct Recorder {
    writer: Box<dyn Write + Send + Sync>,
    config: RecorderConfig,
    sequencer: ObservationSequencer,
    stats: RecorderStats,
}

/// The header line's only field, serialized from the single version constant.
#[derive(Serialize)]
struct TraceHeader {
    observation_schema_version: u64,
}

impl Recorder {
    /// Create a recorder over `writer` and write the header line.
    ///
    /// # Errors
    ///
    /// Returns a transport-class error when the header cannot be rendered or
    /// written. A recorder without a valid header would produce a trace a
    /// reader must refuse, so construction fails instead of degrading silently.
    pub fn new<W>(writer: W, config: RecorderConfig) -> Result<Self, ParlevoxError>
    where
        W: Write + Send + Sync + 'static,
    {
        let mut writer: Box<dyn Write + Send + Sync> = Box::new(writer);
        let header = serde_json::to_string(&TraceHeader {
            observation_schema_version: OBSERVATION_SCHEMA_VERSION,
        })
        .map_err(|err| ParlevoxError::Transport(format!("cannot render trace header: {err}")))?;
        let mut line = header.into_bytes();
        line.push(b'\n');
        writer
            .write_all(&line)
            .map_err(|err| ParlevoxError::Transport(format!("cannot write trace header: {err}")))?;
        Ok(Recorder {
            writer,
            config,
            sequencer: ObservationSequencer::new(parlevox_core::MonoClock::new()),
            stats: RecorderStats::default(),
        })
    }

    /// Create a recorder that writes to `path`, truncating any existing file,
    /// and write the header line.
    ///
    /// # Errors
    ///
    /// Returns a transport-class error naming the path when the file cannot be
    /// created or the header cannot be written.
    pub fn to_path(path: impl AsRef<Path>, config: RecorderConfig) -> Result<Self, ParlevoxError> {
        let path = path.as_ref();
        let file = File::create(path).map_err(|err| {
            ParlevoxError::Transport(format!(
                "cannot create trace file {}: {err}",
                path.display()
            ))
        })?;
        Self::new(file, config)
    }

    /// Record one timed session event under `turn` and `generation`.
    ///
    /// The observation's `t_ms` is the timed event's monotonic timestamp; its
    /// `event_id` comes from the recorder's own sequencer. A cap hit, a write
    /// failure, or transcript suppression is counted and the call returns; it
    /// never panics and never propagates an error.
    pub fn record(&mut self, event: &TimedSessionEvent, turn: u64, generation: u64) {
        let Some(kind) = self.prepare_kind(&event.event) else {
            return;
        };
        if self.stats.records_written >= self.config.max_records {
            self.stats.dropped_cap = self.stats.dropped_cap.saturating_add(1);
            return;
        }
        let line = match serde_json::to_string(&self.stamp(event.t_ms, turn, generation, kind)) {
            Ok(line) => line,
            Err(_) => {
                self.stats.dropped_write = self.stats.dropped_write.saturating_add(1);
                return;
            }
        };
        let mut bytes = line.into_bytes();
        bytes.push(b'\n');
        let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if self.stats.bytes_written.saturating_add(len) > self.config.max_bytes {
            self.stats.dropped_cap = self.stats.dropped_cap.saturating_add(1);
            return;
        }
        match self.writer.write_all(&bytes) {
            Ok(()) => {
                self.stats.records_written = self.stats.records_written.saturating_add(1);
                self.stats.bytes_written = self.stats.bytes_written.saturating_add(len);
            }
            Err(_) => {
                self.stats.dropped_write = self.stats.dropped_write.saturating_add(1);
            }
        }
    }

    /// Cumulative counters.
    #[must_use]
    pub fn stats(&self) -> RecorderStats {
        self.stats
    }

    /// Apply transcript gating and redaction, returning `None` when the event
    /// must not be recorded.
    fn prepare_kind(&mut self, event: &SessionEvent) -> Option<ObservationKind> {
        match event {
            SessionEvent::ReplyDelta { .. } if !self.config.record_transcripts => {
                self.stats.transcripts_suppressed =
                    self.stats.transcripts_suppressed.saturating_add(1);
                return None;
            }
            SessionEvent::TranscriptFinal { .. } if !self.config.record_transcripts => {
                return Some(ObservationKind::Control {
                    event: TurnEventKind::FinalTranscript,
                });
            }
            _ => {}
        }
        Some(self.redact_kind(observation_kind(event)))
    }

    /// Replace credential-shaped substrings in every text-bearing kind.
    fn redact_kind(&mut self, kind: ObservationKind) -> ObservationKind {
        match kind {
            ObservationKind::FinalTranscript { text } => {
                let text = self.redact(&text);
                ObservationKind::FinalTranscript { text }
            }
            ObservationKind::PartialTranscript { text } => {
                let text = self.redact(&text);
                ObservationKind::PartialTranscript { text }
            }
            ObservationKind::ReplyDelta { text } => {
                let text = self.redact(&text);
                ObservationKind::ReplyDelta { text }
            }
            ObservationKind::Failure { class, message } => {
                let message = self.redact(&message);
                ObservationKind::Failure { class, message }
            }
            other => other,
        }
    }

    fn redact(&mut self, text: &str) -> String {
        let (redacted, matches) = redact_credentials(text);
        self.stats.redacted = self.stats.redacted.saturating_add(matches);
        redacted
    }

    fn stamp(
        &mut self,
        t_ms: u64,
        turn: u64,
        generation: u64,
        kind: ObservationKind,
    ) -> Observation {
        let mut observation = self.sequencer.stamp(turn, generation, kind);
        // The event's stamp is the one observed in the turn loop, not the
        // recorder's clock read at write time.
        observation.t_ms = t_ms;
        observation
    }
}

/// Replace every credential-shaped substring in `text`.
///
/// The shapes are the ones the repository forbids from being committed:
/// `sk-…`, `ghp_…`, `github_pat_…`, `AKIA…`, `xox[bpa]-…` and a PEM
/// `-----BEGIN … PRIVATE KEY` block, each with the minimum trailing run length
/// the guard uses. A PEM block counts as one match however many bytes it spans.
/// Returns the redacted text and the number of matches.
#[must_use]
pub fn redact_credentials(text: &str) -> (String, u64) {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut matches = 0u64;
    let mut index = 0usize;
    while index < bytes.len() {
        if let Some(end) = credential_end(bytes, index) {
            out.push_str(REDACTED);
            matches += 1;
            index = end;
        } else if let Some(character) = text[index..].chars().next() {
            out.push(character);
            index += character.len_utf8();
        } else {
            break;
        }
    }
    (out, matches)
}

fn credential_end(bytes: &[u8], start: usize) -> Option<usize> {
    if let Some(end) = pem_block_end(bytes, start) {
        return Some(end);
    }
    // `sk-` followed by at least 24 key characters.
    if starts_with(bytes, start, b"sk-") {
        return run_end(bytes, start + 3, is_key_char, 24);
    }
    // `github_pat_` followed by at least 30 identifier characters. Checked
    // before `ghp_` because the two prefixes are disjoint but ordered for
    // clarity.
    if starts_with(bytes, start, b"github_pat_") {
        return run_end(bytes, start + 11, is_pat_char, 30);
    }
    // `ghp_` followed by at least 30 alphanumeric characters.
    if starts_with(bytes, start, b"ghp_") {
        return run_end(bytes, start + 4, is_alnum, 30);
    }
    // `AKIA` followed by at least 16 uppercase alphanumerics.
    if starts_with(bytes, start, b"AKIA") {
        return run_end(bytes, start + 4, is_upper_alnum, 16);
    }
    // `xox[bpa]-` followed by at least 10 key characters.
    if starts_with(bytes, start, b"xox")
        && (starts_with(bytes, start + 3, b"b-")
            || starts_with(bytes, start + 3, b"p-")
            || starts_with(bytes, start + 3, b"a-"))
    {
        return run_end(bytes, start + 5, is_key_char, 10);
    }
    None
}

/// The end of a PEM `-----BEGIN … PRIVATE KEY` block, when one starts at
/// `start`.
///
/// The guard shape is `-----BEGIN [A-Z ]*PRIVATE KEY`. Once that header is
/// matched the whole block is replaced: the span runs through the first
/// matching `-----END … PRIVATE KEY-----` marker, so the base64 body and the
/// END marker are redacted with the header. When no END marker exists, the
/// header's closing dashes and line terminator are replaced together with every
/// following non-empty line made only of base64 bytes; the first line with any
/// other content ends the span, and the result is never shorter than the header
/// itself.
fn pem_block_end(bytes: &[u8], start: usize) -> Option<usize> {
    let header_end = pem_header_end(bytes, start)?;
    match pem_end_marker_end(bytes, header_end) {
        Some(end) => Some(end),
        None => Some(pem_body_end_without_end(bytes, header_end)),
    }
}

/// The end of a PEM `-----BEGIN … PRIVATE KEY` header, when one starts at
/// `start`.
///
/// The algorithm phrase between the fixed prefix and the literal may be empty,
/// and it can contain only uppercase ASCII and spaces, so the scan stops at the
/// first byte outside that alphabet. Anchoring on the trailing literal keeps
/// ordinary prose such as `-----BEGIN and then lowercase text` from being
/// redacted.
fn pem_header_end(bytes: &[u8], start: usize) -> Option<usize> {
    const BEGIN: &[u8] = b"-----BEGIN ";
    const PRIVATE_KEY: &[u8] = b"PRIVATE KEY";
    if !starts_with(bytes, start, BEGIN) {
        return None;
    }
    let body = start + BEGIN.len();
    let mut end = body;
    while end < bytes.len() && is_pem_header_byte(bytes[end]) {
        end += 1;
    }
    // Mirror the guard regex's greedy `[A-Z ]*`: take the last literal that the
    // run can end with, so trailing header bytes are never left unredacted.
    bytes[body..end]
        .windows(PRIVATE_KEY.len())
        .rposition(|window| window == PRIVATE_KEY)
        .map(|offset| body + offset + PRIVATE_KEY.len())
}

/// The offset just past the first `-----END … PRIVATE KEY-----` marker at or
/// after `from`, when one exists.
///
/// The marker reads `-----END [A-Z ]*PRIVATE KEY-----`, matching the BEGIN
/// header's alphabet. A trailing space before the closing dashes is tolerated
/// because over-redacting is the safe direction for a secret.
fn pem_end_marker_end(bytes: &[u8], from: usize) -> Option<usize> {
    const END: &[u8] = b"-----END ";
    const PRIVATE_KEY: &[u8] = b"PRIVATE KEY";
    const CLOSING: &[u8] = b"-----";
    let mut search = from;
    while search < bytes.len() {
        let offset = bytes[search..]
            .windows(END.len())
            .position(|window| window == END)?;
        let marker = search + offset;
        let body = marker + END.len();
        let mut end = body;
        while end < bytes.len() && is_pem_header_byte(bytes[end]) {
            end += 1;
        }
        let mut trimmed = end;
        while trimmed > body && bytes[trimmed - 1] == b' ' {
            trimmed -= 1;
        }
        if trimmed >= body + PRIVATE_KEY.len()
            && &bytes[trimmed - PRIVATE_KEY.len()..trimmed] == PRIVATE_KEY
            && starts_with(bytes, trimmed, CLOSING)
        {
            return Some(trimmed + CLOSING.len());
        }
        search = marker + 1;
    }
    None
}

/// The end of a PEM block whose END marker is missing: the header's closing
/// dashes and line terminator, then every following non-empty line whose
/// content is base64-only.
///
/// The span never includes prose after a lone header on the same line, so a
/// matched header in ordinary text is redacted without swallowing its
/// neighbours.
fn pem_body_end_without_end(bytes: &[u8], header_end: usize) -> usize {
    let mut end = header_end;
    while end < bytes.len() && bytes[end] == b'-' {
        end += 1;
    }
    // Look past trailing spaces and tabs, but only commit when a line
    // terminator confirms this is a header line rather than same-line prose.
    let mut terminator = end;
    while terminator < bytes.len() && matches!(bytes[terminator], b' ' | b'\t') {
        terminator += 1;
    }
    if terminator < bytes.len() && bytes[terminator] == b'\r' {
        terminator += 1;
    }
    if terminator >= bytes.len() || bytes[terminator] != b'\n' {
        return end;
    }
    let mut cursor = terminator + 1;
    loop {
        let line_start = cursor;
        let mut line_end = line_start;
        while line_end < bytes.len() && bytes[line_end] != b'\n' {
            line_end += 1;
        }
        let mut content_end = line_end;
        if content_end > line_start && bytes[content_end - 1] == b'\r' {
            content_end -= 1;
        }
        let content = &bytes[line_start..content_end];
        if content.is_empty() || !content.iter().all(|byte| is_base64_byte(*byte)) {
            return end;
        }
        end = content_end;
        if line_end >= bytes.len() {
            return end;
        }
        cursor = line_end + 1;
    }
}

fn is_pem_header_byte(byte: u8) -> bool {
    byte.is_ascii_uppercase() || byte == b' '
}

fn is_base64_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'+' || byte == b'/' || byte == b'='
}

/// The end of a run of `allowed` bytes at `start`, when it reaches `minimum`.
fn run_end(bytes: &[u8], start: usize, allowed: fn(u8) -> bool, minimum: usize) -> Option<usize> {
    let mut end = start;
    while end < bytes.len() && allowed(bytes[end]) {
        end += 1;
    }
    if end.saturating_sub(start) >= minimum {
        Some(end)
    } else {
        None
    }
}

fn starts_with(bytes: &[u8], start: usize, pattern: &[u8]) -> bool {
    bytes
        .get(start..start + pattern.len())
        .is_some_and(|candidate| candidate == pattern)
}

fn is_alnum(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
}

fn is_upper_alnum(byte: u8) -> bool {
    byte.is_ascii_digit() || byte.is_ascii_uppercase()
}

fn is_key_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-'
}

fn is_pat_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use parlevox_core::{AudioFrame, ErrorClass, Timestamp};
    use proptest::prelude::*;

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
        fn lines(&self) -> Vec<String> {
            String::from_utf8(self.0.lock().unwrap().clone())
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect()
        }
    }

    fn config(max_records: u64, max_bytes: u64, record_transcripts: bool) -> RecorderConfig {
        RecorderConfig {
            max_records,
            max_bytes,
            record_transcripts,
        }
    }

    fn timed(t_ms: u64, event: SessionEvent) -> TimedSessionEvent {
        TimedSessionEvent { t_ms, event }
    }

    fn final_transcript(text: &str) -> TimedSessionEvent {
        timed(
            5,
            SessionEvent::TranscriptFinal {
                turn: 1,
                text: text.into(),
            },
        )
    }

    fn observation(line: &str) -> Observation {
        serde_json::from_str(line).expect("a record line parses as an Observation")
    }

    #[test]
    fn header_is_first_and_every_line_parses() {
        let buffer = SharedBuffer::default();
        let mut recorder =
            Recorder::new(buffer.clone(), config(100, 1_000_000, true)).expect("recorder");
        recorder.record(&timed(0, SessionEvent::TurnStarted { turn: 1 }), 1, 0);
        recorder.record(&final_transcript("hello"), 1, 0);
        recorder.record(
            &timed(
                9,
                SessionEvent::Failed {
                    class: ErrorClass::Fatal,
                    message: "boom".into(),
                },
            ),
            1,
            0,
        );

        let lines = buffer.lines();
        assert_eq!(lines[0], r#"{"observation_schema_version":1}"#);
        assert_eq!(lines.len(), 4);
        let records: Vec<Observation> = lines[1..].iter().map(|line| observation(line)).collect();
        assert_eq!(records[0].t_ms, 0);
        assert_eq!(records[1].t_ms, 5);
        assert_eq!(
            records[1].kind,
            ObservationKind::FinalTranscript {
                text: "hello".into()
            }
        );
        assert_eq!(records[2].t_ms, 9);
        assert_eq!(recorder.stats().records_written, 3);
        assert_eq!(recorder.stats().dropped_cap, 0);
    }

    #[test]
    fn record_cap_drops_and_counts() {
        let buffer = SharedBuffer::default();
        let mut recorder = Recorder::new(buffer.clone(), config(2, 1_000_000, true)).expect("rec");
        for turn in 1..=5 {
            recorder.record(&timed(turn, SessionEvent::TurnStarted { turn }), turn, 0);
        }
        assert_eq!(recorder.stats().records_written, 2);
        assert_eq!(recorder.stats().dropped_cap, 3);
        assert_eq!(buffer.lines().len(), 3, "header plus two records");
    }

    #[test]
    fn byte_cap_drops_and_counts_and_stays_under_the_cap() {
        let buffer = SharedBuffer::default();
        let mut recorder = Recorder::new(buffer.clone(), config(1_000, 60, true)).expect("rec");
        for turn in 1..=10 {
            recorder.record(
                &timed(
                    turn,
                    SessionEvent::TranscriptFinal {
                        turn,
                        text: "a reasonably long transcript".into(),
                    },
                ),
                turn,
                0,
            );
        }
        let stats = recorder.stats();
        assert!(
            stats.bytes_written <= 60,
            "bytes {} exceed cap",
            stats.bytes_written
        );
        assert!(stats.dropped_cap > 0);
        assert_eq!(stats.records_written + stats.dropped_cap, 10);
    }

    #[test]
    fn write_failure_drops_and_counts() {
        struct Failing;
        impl Write for Failing {
            fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk full"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        // The header is written first and must succeed, so use a writer that
        // fails only after the header.
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

        // Sanity: the always-failing writer fails construction.
        assert!(Recorder::new(Failing, config(10, 1_000_000, true)).is_err());

        let mut recorder =
            Recorder::new(FailsAfterHeader { written: 0 }, config(10, 1_000_000, true))
                .expect("header succeeds");
        recorder.record(&timed(1, SessionEvent::TurnStarted { turn: 1 }), 1, 0);
        assert_eq!(recorder.stats().records_written, 0);
        assert_eq!(recorder.stats().dropped_write, 1);
    }

    #[test]
    fn transcripts_are_absent_without_the_flag() {
        let buffer = SharedBuffer::default();
        let mut recorder =
            Recorder::new(buffer.clone(), config(100, 1_000_000, false)).expect("rec");
        recorder.record(&final_transcript("secret sentence"), 1, 0);
        recorder.record(
            &timed(
                6,
                SessionEvent::ReplyDelta {
                    turn: 1,
                    text: "streamed".into(),
                },
            ),
            1,
            0,
        );

        let lines = buffer.lines();
        assert_eq!(lines.len(), 2, "header plus the degraded final record");
        let record = observation(&lines[1]);
        assert_eq!(
            record.kind,
            ObservationKind::Control {
                event: TurnEventKind::FinalTranscript
            }
        );
        assert!(
            !lines[1].contains("secret"),
            "transcript text must be absent"
        );
        assert_eq!(recorder.stats().transcripts_suppressed, 1);
        assert_eq!(recorder.stats().records_written, 1);
    }

    #[test]
    fn planted_secret_is_redacted() {
        // Arrange: a complete PEM block (header, three base64 body lines, END
        // marker) next to a key-shaped fixture, in a recorded transcript and a
        // failure message. Every fixture value carries `test` so the
        // repository's commit gate accepts it under its value-scoped fixture
        // exemption.
        let planted_key = "sk-test-value-never-shown-0123456789abcdef";
        let begin = "-----BEGIN TEST PRIVATE KEY-----";
        let body_one = "MIIBVAIBADANBgkqhkiG9w0BAQEFAASCAT4wggE6AgEAAkEA";
        let body_two = "TESTBASE64LINE0123456789abcdefghijklmnopqrstuvwxyz";
        let body_three = "Zm9vYmFyVGVzdEJvZHlMaW5lVGhyZWUxMjM0NTY3ODkw";
        let end = "-----END TEST PRIVATE KEY-----";
        let planted_pem = format!("{begin}\n{body_one}\n{body_two}\n{body_three}\n{end}");
        let planted_text = format!("before {planted_key} then\n{planted_pem}\nafter");
        let buffer = SharedBuffer::default();
        let mut recorder =
            Recorder::new(buffer.clone(), config(100, 1_000_000, true)).expect("rec");

        // Act
        recorder.record(&final_transcript(&planted_text), 1, 0);
        recorder.record(
            &timed(
                8,
                SessionEvent::Failed {
                    class: ErrorClass::Transient,
                    message: format!("auth failed for {planted_key}"),
                },
            ),
            1,
            0,
        );

        // Assert: no part of the planted argument reaches the record buffer.
        let text = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
        for secret in [planted_key, begin, body_one, body_two, body_three, end] {
            assert!(
                !text.contains(secret),
                "the planted credential part must not appear byte-for-byte"
            );
        }
        assert!(text.contains(REDACTED), "the match is replaced: {text}");
        assert!(text.contains("before"), "leading text survives: {text}");
        assert!(text.contains("after"), "trailing text survives: {text}");
        assert_eq!(recorder.stats().redacted, 3);
        // Quote the redacted record with --nocapture; the raw input is never
        // printed.
        println!("redacted records:\n{text}");
    }

    #[test]
    fn pem_without_end_redacts_base64_body() {
        // Arrange: a header and base64 body with no END marker, followed by an
        // ordinary line.
        let begin = "-----BEGIN TEST PRIVATE KEY-----";
        let body_one = "MIIBVAIBADANBgkqhkiG9w0BAQEFAASCAT4wggE6AgEAAkEA";
        let body_two = "TESTBASE64LINE0123456789abcdefghijklmnopqrstuvwxyz";
        let input = format!("header\n{begin}\n{body_one}\n{body_two}\nfollowing prose stays");

        // Act
        let (text, matches) = redact_credentials(&input);

        // Assert: the header and every base64 line are gone; following prose is
        // not swallowed.
        assert_eq!(matches, 1, "the block is one match: {text}");
        assert!(text.contains(REDACTED), "the block is replaced: {text}");
        assert!(!text.contains(begin));
        assert!(!text.contains(body_one));
        assert!(!text.contains(body_two));
        assert!(
            text.contains("following prose stays"),
            "prose after the block survives: {text}"
        );
        println!("no-END redaction:\n{text}");
    }

    #[test]
    fn redaction_covers_every_guard_shape() {
        // Every matched value carries `test`, `fake` or `dummy` so the
        // repository's commit gate accepts it under its value-scoped fixture
        // exemption. Labels name the shape without echoing its value.
        let shapes = [
            ("openai_key", "sk-test-key-0123456789abcdefghij"),
            ("github_pat", "ghp_TESTabcdefghijklmnopqrstuvwx0123"),
            (
                "github_fine_grained_pat",
                "github_pat_FAKEabcdefghijklmnopqrstuvwx0123",
            ),
            ("aws_access_key", "AKIATESTTESTTESTTEST"),
            ("slack_token", "xoxb-test-0123456789"),
            ("private_key", "-----BEGIN TEST PRIVATE KEY-----"),
        ];
        for (name, shape) in shapes {
            let (text, matches) = redact_credentials(&format!("prefix {shape} suffix"));
            assert_eq!(matches, 1, "{name} not redacted: {text}");
            assert!(!text.contains(shape), "{name} value must be absent: {text}");
            assert!(text.contains(REDACTED), "{name} is replaced: {text}");
            println!("{name}: matches={matches}; redacted={text}");
        }
        assert_eq!(
            redact_credentials("a task-list is fine"),
            ("a task-list is fine".into(), 0)
        );
        // The PEM guard is anchored on the trailing literal: a `-----BEGIN`
        // that is not followed by an uppercase algorithm phrase and `PRIVATE
        // KEY` is ordinary text and must survive untouched.
        assert_eq!(
            redact_credentials("-----BEGIN and then lowercase prose"),
            ("-----BEGIN and then lowercase prose".into(), 0)
        );
        // Same-line prose after a lone header is not part of the header and
        // must survive the no-END fallback byte-intact.
        assert_eq!(
            redact_credentials("prefix -----BEGIN TEST PRIVATE KEY----- suffix"),
            ("prefix [redacted] suffix".into(), 1)
        );
    }

    proptest! {
        #[test]
        fn recorder_accounting_invariant(
            events in proptest::collection::vec(0u8..6, 0..40),
            max_records in 0u64..20,
            max_bytes in 0u64..400,
            record_transcripts in proptest::bool::ANY,
        ) {
            let buffer = SharedBuffer::default();
            let mut recorder = Recorder::new(buffer.clone(), config(max_records, max_bytes, record_transcripts))
                .expect("recorder");
            let mut offered = 0u64;
            for (index, which) in events.iter().enumerate() {
                let turn = index as u64;
                let event = match which {
                    0 => SessionEvent::TurnStarted { turn },
                    1 => SessionEvent::TranscriptFinal { turn, text: "hello world".into() },
                    2 => SessionEvent::ReplyDelta { turn, text: "delta".into() },
                    3 => SessionEvent::TurnDone {
                        turn,
                        audio_frames: 0,
                        dropped_frames: 0,
                        dropped_ms: 0,
                        dropped_events: 0,
                        sidecar_dropped_frames: 0,
                        interrupted: false,
                    },
                    4 => SessionEvent::Failed { class: ErrorClass::Fatal, message: "oops".into() },
                    _ => SessionEvent::TurnStarted { turn },
                };
                recorder.record(&TimedSessionEvent { t_ms: turn, event }, turn, 0);
                offered += 1;
            }
            let stats = recorder.stats();
            prop_assert_eq!(
                stats.records_written
                    + stats.dropped_cap
                    + stats.dropped_write
                    + stats.transcripts_suppressed,
                offered
            );
            prop_assert!(stats.records_written <= max_records);
            prop_assert!(stats.bytes_written <= max_bytes);
            let lines = buffer.lines();
            prop_assert_eq!(lines[0].as_str(), r#"{"observation_schema_version":1}"#);
            prop_assert_eq!(lines.len() as u64, 1 + stats.records_written);
            for line in &lines[1..] {
                prop_assert!(serde_json::from_str::<Observation>(line).is_ok());
            }
        }
    }

    #[test]
    fn audio_and_terminal_events_map_to_their_kinds() {
        let frame = AudioFrame::new(Timestamp::from_millis(0), 16_000, 1, vec![0i16; 160]);
        let before = observation_kind(&SessionEvent::AudioOut { turn: 2, frame });
        assert_eq!(
            before,
            ObservationKind::AudioFrameQueued {
                sample_rate: 16_000,
                channels: 1,
                pcm_samples: 160
            }
        );
        assert_eq!(
            observation_kind(&SessionEvent::TurnDone {
                turn: 2,
                audio_frames: 1,
                dropped_frames: 0,
                dropped_ms: 0,
                dropped_events: 0,
                sidecar_dropped_frames: 0,
                interrupted: false
            }),
            ObservationKind::ReplyFinished
        );
        assert_eq!(
            observation_kind(&SessionEvent::TurnStarted { turn: 2 }),
            ObservationKind::Control {
                event: TurnEventKind::Start
            }
        );
    }
}
