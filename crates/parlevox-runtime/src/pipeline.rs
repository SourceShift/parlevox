//! Sentence-level pipelining primitives for the opt-in TTS mode.
//!
//! The default turn synthesizes the whole reply once. The opt-in
//! [`crate::SessionConfig::pipelined_tts`] mode instead splits the reply on
//! sentence boundaries as deltas arrive, queues the completed sentences in a
//! bounded FIFO, and synthesizes them one at a time while the LLM is still
//! producing the rest of the reply.
//!
//! Three pieces make that possible without a second task or a new dependency:
//!
//! * [`SentenceSplitter`] partitions a streamed reply byte-exactly. Every byte
//!   pushed in is eventually emitted by [`SentenceSplitter::push_delta`] or
//!   [`SentenceSplitter::flush`], in order, so `concat(emitted) == reply` — the
//!   property that makes per-sentence audio comparable to whole-reply audio for
//!   a compositional synthesizer.
//! * [`SentenceQueue`] is the bounded producer-to-synthesizer handoff. It drops
//!   the incoming sentence under overload (never the head) because evicting the
//!   head would speak the remaining sentences out of reply order; every drop is
//!   counted.
//! * [`PipelineStats`] exposes the queue's cumulative counters and depth, the
//!   same shape as the session's other bounded handoffs.

use std::collections::VecDeque;

use parlevox_core::{ParlevoxError, PushOutcome};

/// Terminal punctuation that ends a sentence.
///
/// Abbreviation handling is deliberately out of scope: the split is a latency
/// heuristic, not a linguistic parse, and a wrong split only changes which
/// synthesis request the text lands in — never the concatenated audio.
fn is_terminator(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?')
}

/// Accumulates streamed deltas and emits completed sentences in reply order.
///
/// A sentence ends at the first run of terminal punctuation (`.`/`!`/`?`) that
/// falls within the first `max_chars` characters and follows at least one
/// non-whitespace, non-terminator character. A run with no sentence content
/// before it is a fragment, not a boundary: the `..` that opens `.. Next.` stays
/// with the sentence that follows it. When no terminator arrives within
/// `max_chars` characters, the splitter force-splits on the last whitespace
/// boundary at or before `max_chars`, so an unpunctuated reply still pipelines.
///
/// Release is prompt and has bounded lookahead. A unit is emitted as soon as a
/// non-whitespace character follows its trailing whitespace, so a sentence
/// whose terminator is followed by more text is released in the same delta.
/// When a terminator run reaches the end of the buffer, the ownership of the
/// whitespace that may follow is still undecided, so the unit is held across
/// that delta boundary rather than released. The lookahead is bounded and
/// never waits for stream exhaustion: the next delta that supplies a
/// non-whitespace character releases it, [`SentenceSplitter::flush`] releases
/// it if the reply ends first, and the `max_chars` hold releases it if the
/// buffer grows past the bound. In the common case the unit therefore waits at
/// most one delta. Holding the run keeps whitespace attached to the sentence
/// it follows instead of turning into a whitespace-only request; because every
/// held byte is emitted by that later delta or by `flush`, byte coverage stays
/// exact. Whitespace that arrives after a sentence was already released
/// becomes leading text of the next unit: byte coverage wins over the "never
/// start with whitespace" preference.
///
/// The invariant that the rest of the pipelined path relies on: concatenating
/// everything returned by [`SentenceSplitter::push_delta`] and
/// [`SentenceSplitter::flush`] reproduces the pushed reply byte for byte.
#[derive(Debug)]
pub struct SentenceSplitter {
    buffer: String,
    max_chars: usize,
}

impl SentenceSplitter {
    /// Create a splitter whose force-split bound is `max_chars` characters.
    ///
    /// # Errors
    ///
    /// Returns [`ParlevoxError::Protocol`] when `max_chars` is zero: a zero
    /// bound has no whitespace boundary to split on and could not make
    /// progress.
    pub fn new(max_chars: usize) -> Result<Self, ParlevoxError> {
        if max_chars == 0 {
            return Err(ParlevoxError::Protocol(
                "sentence splitter max_chars must be at least 1 (got 0)".into(),
            ));
        }
        Ok(SentenceSplitter {
            // A sentence is rarely longer than a few dozen characters, so a
            // modest reserve avoids reallocation without over-allocating for a
            // pathological bound.
            buffer: String::with_capacity(max_chars.min(256)),
            max_chars,
        })
    }

    /// The configured maximum sentence length, in characters.
    pub fn max_chars(&self) -> usize {
        self.max_chars
    }

    /// Append `delta` and return every sentence it completed, in order.
    ///
    /// A sentence whose terminator run ends the buffered text is held, not
    /// returned, until a later delta supplies a non-whitespace character,
    /// until [`SentenceSplitter::flush`], or until the
    /// [`SentenceSplitter::max_chars`] hold is exceeded. That lookahead is
    /// bounded and never waits for stream exhaustion; in the common case the
    /// unit waits at most one delta. See the type-level documentation.
    pub fn push_delta(&mut self, delta: &str) -> Vec<String> {
        self.buffer.push_str(delta);
        let mut completed = Vec::new();
        while let Some(sentence) = self.take_ready() {
            completed.push(sentence);
        }
        completed
    }

    /// Return the unterminated remainder at end of reply, if any.
    ///
    /// The remainder is emitted whole even when it is shorter than a sentence;
    /// dropping it would break the byte-exact partition.
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            None
        } else {
            // Move the buffer out instead of cloning it: the splitter is done
            // with this reply once `flush` is called.
            Some(std::mem::take(&mut self.buffer))
        }
    }

    /// Emit one ready prefix, or `None` when the buffer holds no complete unit.
    fn take_ready(&mut self) -> Option<String> {
        let mut char_count = 0usize;
        let mut has_content = false;
        let mut in_terminator_run = false;
        // Char index where the first eligible terminator run starts, plus the
        // byte offset just past that run. A run is eligible only when sentence
        // content precedes it; a leading run like the `..` of `.. Next.` is not
        // a boundary.
        let mut terminator: Option<(usize, usize)> = None;

        for (char_index, (byte_index, ch)) in self.buffer.char_indices().enumerate() {
            char_count = char_index + 1;
            if is_terminator(ch) {
                if let Some((_, run_end)) = terminator.as_mut() {
                    if in_terminator_run {
                        // Extend the run so `...` is one boundary.
                        *run_end = byte_index + ch.len_utf8();
                    }
                } else if has_content {
                    terminator = Some((char_index, byte_index + ch.len_utf8()));
                    in_terminator_run = true;
                }
            } else {
                in_terminator_run = false;
                if !ch.is_whitespace() {
                    has_content = true;
                }
            }
        }

        if let Some((terminator_index, run_end)) = terminator {
            if terminator_index < self.max_chars {
                let end = self.whitespace_end(run_end);
                if end < self.buffer.len() {
                    // A non-whitespace character follows the buffered
                    // whitespace, so that whitespace belongs to this sentence
                    // and the unit is complete. This includes the case where
                    // the terminator run ended the previous delta: the delta
                    // that supplied the following text releases it now.
                    return Some(self.take_prefix(end));
                }
                // Only whitespace follows and it reaches the end of the
                // buffer (or the run itself ends the buffer). The whitespace
                // ownership is undecided, so hold the unit for at most one
                // delta: a later non-whitespace character, flush, or the
                // `max_chars` bound ends the hold. Releasing immediately would
                // strand the following whitespace in its own request.
                if char_count > self.max_chars {
                    return Some(self.take_prefix(end));
                }
                return None;
            }
            // The terminator is past the length bound; fall through to a
            // forced split so one long clause cannot defeat the bound.
        }

        if char_count > self.max_chars {
            return self.force_split();
        }
        None
    }

    /// Extend `terminator_end` through the whitespace already buffered after
    /// the punctuation, so a sentence never begins with whitespace when the
    /// next character is known.
    fn whitespace_end(&self, terminator_end: usize) -> usize {
        let mut end = terminator_end;
        for (_, ch) in self.buffer[terminator_end..].char_indices() {
            if ch.is_whitespace() {
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        end
    }

    /// Split at the last whitespace within `max_chars` characters, or hard-split
    /// at exactly `max_chars` characters when no whitespace is available.
    fn force_split(&mut self) -> Option<String> {
        let mut boundary = 0usize;
        let mut last_whitespace = None;
        for (char_index, (byte_index, ch)) in self.buffer.char_indices().enumerate() {
            if char_index >= self.max_chars {
                break;
            }
            // Byte offset just past this character; always a char boundary.
            boundary = byte_index + ch.len_utf8();
            if ch.is_whitespace() {
                last_whitespace = Some(boundary);
            }
        }
        // `boundary` is non-zero because `max_chars >= 1` and the buffer is
        // known to hold more than `max_chars` characters, so this always makes
        // progress.
        let end = last_whitespace.unwrap_or(boundary);
        Some(self.take_prefix(end))
    }

    fn take_prefix(&mut self, end: usize) -> String {
        let rest = self.buffer.split_off(end);
        std::mem::replace(&mut self.buffer, rest)
    }
}

/// A snapshot of the bounded sentence queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineStats {
    /// Sentences accepted into the queue for its whole lifetime.
    pub sentences_queued: u64,
    /// Sentences rejected because the bounded queue was full.
    pub sentences_dropped: u64,
    /// Synthesis requests that ran to completion.
    pub sentences_synthesized: u64,
    /// Sentences currently waiting in the queue.
    pub depth: usize,
    /// Configured maximum number of waiting sentences.
    pub capacity: usize,
}

/// Bounded producer-to-synthesizer handoff for completed sentences.
///
/// The queue is strictly FIFO and uses
/// [`DropNewest`](parlevox_core::OverflowPolicy::DropNewest) semantics:
/// when it is full the incoming sentence is dropped and counted. Evicting the
/// head instead would synthesize the remaining sentences in an order that no
/// longer matches the reply, so drop-the-newest is the only ordering-safe
/// policy here.
pub struct SentenceQueue {
    capacity: usize,
    queue: VecDeque<String>,
    queued: u64,
    dropped: u64,
    synthesized: u64,
}

impl SentenceQueue {
    /// Create a queue with the given sentence capacity.
    ///
    /// A capacity of zero is well defined here — every push is dropped and
    /// counted — because [`crate::Session::new`] rejects it for a real
    /// session with a protocol error.
    pub fn new(capacity: usize) -> Self {
        SentenceQueue {
            capacity,
            queue: VecDeque::with_capacity(capacity.min(1024)),
            queued: 0,
            dropped: 0,
            synthesized: 0,
        }
    }

    /// Queue one sentence, dropping the incoming text when full.
    ///
    /// The returned [`PushOutcome`] lets the producer observe the drop, matching
    /// the contract of every other bounded queue in the runtime.
    pub fn push(&mut self, sentence: String) -> PushOutcome {
        if self.capacity == 0 || self.queue.len() >= self.capacity {
            self.dropped = self.dropped.saturating_add(1);
            return PushOutcome::Dropped;
        }
        self.queue.push_back(sentence);
        self.queued = self.queued.saturating_add(1);
        PushOutcome::Accepted
    }

    /// Take the oldest queued sentence, preserving reply order.
    pub fn pop(&mut self) -> Option<String> {
        self.queue.pop_front()
    }

    /// Drop every sentence still waiting at the start of a new turn.
    ///
    /// Cumulative counters are preserved: they describe the session, not one
    /// turn, exactly like the audio and event ring counters.
    pub fn clear_pending(&mut self) {
        self.queue.clear();
    }

    /// Record one synthesis request that ran to completion.
    pub fn record_synthesized(&mut self) {
        self.synthesized = self.synthesized.saturating_add(1);
    }

    /// Number of sentences waiting to be synthesized.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether no sentence is waiting to be synthesized.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Configured maximum number of waiting sentences.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Depth and cumulative counters for this handoff.
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            sentences_queued: self.queued,
            sentences_dropped: self.dropped,
            sentences_synthesized: self.synthesized,
            depth: self.queue.len(),
            capacity: self.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split_all(max_chars: usize, deltas: &[&str]) -> Vec<String> {
        let mut splitter = SentenceSplitter::new(max_chars).unwrap();
        let mut out = Vec::new();
        for delta in deltas {
            out.extend(splitter.push_delta(delta));
        }
        if let Some(rest) = splitter.flush() {
            out.push(rest);
        }
        out
    }

    #[test]
    fn splitter_emits_completed_sentences_with_their_trailing_whitespace() {
        let sentences = split_all(240, &["Hello", " there! ", "How", " are you?"]);
        assert_eq!(
            sentences,
            vec!["Hello there! ".to_string(), "How are you?".to_string()]
        );
        assert_eq!(sentences.concat(), "Hello there! How are you?");
    }

    #[test]
    fn splitter_holds_a_terminator_that_ends_a_delta_until_the_next_delta() {
        // The terminator ends the buffer, so whitespace ownership is still
        // undecided: hold the unit for one delta, then release it as soon as
        // the next delta supplies non-whitespace text.
        let mut splitter = SentenceSplitter::new(240).unwrap();
        assert!(splitter.push_delta("One.").is_empty());
        assert_eq!(splitter.push_delta(" Two."), vec!["One. ".to_string()]);
        assert_eq!(splitter.flush(), Some("Two.".to_string()));
    }

    #[test]
    fn splitter_releases_a_held_terminator_at_flush() {
        // A reply that ends on a terminator is emitted by flush, and trailing
        // whitespace that arrived in a later delta rides with it.
        let mut splitter = SentenceSplitter::new(240).unwrap();
        assert!(splitter.push_delta("Hello.").is_empty());
        assert!(splitter.push_delta(" ").is_empty());
        assert_eq!(splitter.flush(), Some("Hello. ".to_string()));
    }

    #[test]
    fn splitter_hold_is_bounded_to_one_delta() {
        // Constraint 3: the bounded lookahead is at most one delta. The very
        // next delta that supplies a non-whitespace character releases the
        // held unit; it never waits for stream exhaustion.
        let mut splitter = SentenceSplitter::new(240).unwrap();
        assert!(splitter.push_delta("Hi!").is_empty());
        assert_eq!(splitter.push_delta(" there"), vec!["Hi! ".to_string()]);
        assert_eq!(splitter.flush(), Some("there".to_string()));
    }

    #[test]
    fn splitter_holds_trailing_whitespace_until_the_next_character() {
        // P7: whitespace arriving in its own delta belongs to the sentence it
        // follows, not to a whitespace-only request.
        let mut splitter = SentenceSplitter::new(240).unwrap();
        assert!(splitter.push_delta("Hello. ").is_empty());
        assert!(splitter.push_delta(" ").is_empty());
        assert_eq!(splitter.flush(), Some("Hello.  ".to_string()));
    }

    #[test]
    fn splitter_releases_a_held_whitespace_run_at_the_bound() {
        // The trailing-whitespace hold is bounded: once the buffer exceeds
        // `max_chars`, the whitespace is released with its sentence.
        let mut splitter = SentenceSplitter::new(5).unwrap();
        assert!(splitter.push_delta("Hi! ").is_empty());
        assert_eq!(splitter.push_delta("  "), vec!["Hi!   ".to_string()]);
        assert_eq!(splitter.flush(), None);
    }

    #[test]
    fn splitter_merges_a_leading_terminator_run_into_the_next_sentence() {
        // P8: the `..` that opens `.. Next.` is a fragment, not a sentence; it
        // rides with the preceding terminator run and the next sentence.
        let mut splitter = SentenceSplitter::new(240).unwrap();
        assert!(splitter.push_delta("Wait.").is_empty());
        assert_eq!(
            splitter.push_delta(".. Next."),
            vec!["Wait... ".to_string()]
        );
        assert_eq!(splitter.flush(), Some("Next.".to_string()));
    }

    #[test]
    fn splitter_keeps_every_byte_across_lenient_boundaries() {
        let sentences = split_all(240, &["One.", " Two.", "Wait.", ".. Next."]);
        assert_eq!(
            sentences,
            vec![
                "One. ".to_string(),
                "Two.".to_string(),
                "Wait... ".to_string(),
                "Next.".to_string(),
            ]
        );
        assert_eq!(sentences.concat(), "One. Two.Wait... Next.");
    }

    #[test]
    fn splitter_force_splits_at_a_whitespace_boundary() {
        let sentences = split_all(10, &["aaaa bbbb cccc dddd"]);
        assert_eq!(
            sentences,
            vec!["aaaa bbbb ".to_string(), "cccc dddd".to_string()]
        );
        assert!(sentences.iter().all(|s| s.chars().count() <= 10));
    }

    #[test]
    fn splitter_hard_splits_when_no_whitespace_is_available() {
        let sentences = split_all(4, &["abcdefghij"]);
        assert_eq!(
            sentences,
            vec!["abcd".to_string(), "efgh".to_string(), "ij".to_string()]
        );
    }

    #[test]
    fn splitter_never_emits_an_empty_sentence() {
        let sentences = split_all(4, &["", "Hi!", "", " ", "Ok."]);
        assert!(sentences.iter().all(|s| !s.is_empty()));
        assert_eq!(sentences.concat(), "Hi! Ok.");
    }

    #[test]
    fn splitter_rejects_a_zero_bound() {
        let err = SentenceSplitter::new(0).unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        assert!(err.to_string().contains("max_chars"));
    }

    #[test]
    fn queue_drops_the_incoming_sentence_and_counts_it() {
        let mut queue = SentenceQueue::new(2);
        assert_eq!(queue.push("one".into()), PushOutcome::Accepted);
        assert_eq!(queue.push("two".into()), PushOutcome::Accepted);
        assert_eq!(queue.push("three".into()), PushOutcome::Dropped);
        // The head is preserved: reply order is never rewritten by a drop.
        assert_eq!(queue.pop(), Some("one".into()));
        assert_eq!(queue.pop(), Some("two".into()));
        assert_eq!(queue.pop(), None);

        let stats = queue.stats();
        assert_eq!(stats.sentences_queued, 2);
        assert_eq!(stats.sentences_dropped, 1);
        assert_eq!(stats.capacity, 2);
    }

    #[test]
    fn zero_capacity_queue_drops_and_counts_every_sentence() {
        let mut queue = SentenceQueue::new(0);
        assert_eq!(queue.push("x".into()), PushOutcome::Dropped);
        let stats = queue.stats();
        assert_eq!(stats.sentences_queued, 0);
        assert_eq!(stats.sentences_dropped, 1);
        assert_eq!(stats.depth, 0);
    }

    #[test]
    fn clear_pending_preserves_cumulative_counters() {
        let mut queue = SentenceQueue::new(4);
        queue.push("one".into());
        queue.record_synthesized();
        queue.clear_pending();
        let stats = queue.stats();
        assert!(queue.is_empty());
        assert_eq!(stats.depth, 0);
        assert_eq!(stats.sentences_queued, 1);
        assert_eq!(stats.sentences_synthesized, 1);
    }
}
