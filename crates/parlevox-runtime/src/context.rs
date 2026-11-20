//! Rust-owned, bounded conversation context.
//!
//! A [`ChatContext`] stores the messages a session has already seen. Both
//! bounds are explicit and enforced on every push: a message count
//! (`max_messages`) and a character budget (`max_chars`). Loss from either
//! bound is counted — [`ContextStats::evicted_messages`],
//! [`ContextStats::evicted_chars`] and [`ContextStats::truncated_chars`] — so a
//! caller can tell "the model forgot" from "the model was never told".
//!
//! The budgets count `char`s, not model tokens. That is an intentional
//! estimate: exact accounting needs a tokenizer and a model-specific vocabulary
//! (both excluded by the kickoff), so the character budget is a transport-safe
//! proxy rather than a promise about prompt length.
//!
//! Storage is a [`VecDeque`]: the common eviction path pops the oldest
//! non-system message, which is usually the front, and `VecDeque` makes that
//! `O(1)` while still supporting the rarer middle removal needed when system
//! messages are retained ahead of older turns.

use std::collections::VecDeque;

use parlevox_core::ParlevoxError;
use parlevox_providers::{ChatMessage, Role};

/// Most messages a [`ChatContext`] will ever retain at once.
const RESERVE_LIMIT: usize = 64;

/// A snapshot of a [`ChatContext`]'s depth and cumulative loss counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextStats {
    /// Messages currently stored.
    pub messages: usize,
    /// Sum of `text.chars().count()` over the stored messages.
    pub chars: usize,
    /// Messages removed to make room, for the context's whole lifetime.
    pub evicted_messages: u64,
    /// Characters removed with those messages.
    pub evicted_chars: u64,
    /// Characters removed by truncating a single oversized message.
    pub truncated_chars: u64,
}

/// Bounded conversation history with counted, non-silent loss.
#[derive(Debug, Clone)]
pub struct ChatContext {
    messages: VecDeque<ChatMessage>,
    max_messages: usize,
    max_chars: usize,
    live_chars: usize,
    evicted_messages: u64,
    evicted_chars: u64,
    truncated_chars: u64,
}

impl ChatContext {
    /// Build an empty context.
    ///
    /// # Errors
    ///
    /// Returns [`ParlevoxError::Protocol`] when `max_messages < 2` or
    /// `max_chars < 1`; the message names the rejected bound.
    pub fn new(max_messages: usize, max_chars: usize) -> Result<Self, ParlevoxError> {
        if max_messages < 2 {
            return Err(ParlevoxError::Protocol(format!(
                "context max_messages must be at least 2 (got {max_messages})"
            )));
        }
        if max_chars < 1 {
            return Err(ParlevoxError::Protocol(
                "context max_chars must be at least 1 (got 0)".into(),
            ));
        }
        Ok(ChatContext {
            // A hostile or accidental huge bound must not force one giant
            // allocation up front; grow on demand instead.
            messages: VecDeque::with_capacity(max_messages.min(RESERVE_LIMIT)),
            max_messages,
            max_chars,
            live_chars: 0,
            evicted_messages: 0,
            evicted_chars: 0,
            truncated_chars: 0,
        })
    }

    /// Append a system message, evicting older content if either bound needs it.
    pub fn push_system(&mut self, text: impl Into<String>) {
        self.push(Role::System, text.into());
    }

    /// Append a user message, evicting older content if either bound needs it.
    pub fn push_user(&mut self, text: impl Into<String>) {
        self.push(Role::User, text.into());
    }

    /// Append an assistant message, evicting older content if either bound needs it.
    pub fn push_assistant(&mut self, text: impl Into<String>) {
        self.push(Role::Assistant, text.into());
    }

    /// An owned snapshot, oldest first.
    ///
    /// A `VecDeque` is not contiguous, so a caller that needs a slice (for
    /// example a [`parlevox_providers::GenerateRequest`]) gets one allocation
    /// rather than borrowing the internal buffer.
    pub fn messages(&self) -> Vec<ChatMessage> {
        let mut snapshot = Vec::with_capacity(self.messages.len());
        snapshot.extend(self.messages.iter().cloned());
        snapshot
    }

    /// Depth and loss counters for this context.
    pub fn stats(&self) -> ContextStats {
        ContextStats {
            messages: self.messages.len(),
            chars: self.live_chars,
            evicted_messages: self.evicted_messages,
            evicted_chars: self.evicted_chars,
            truncated_chars: self.truncated_chars,
        }
    }

    /// Drop every message and reset all counters.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.live_chars = 0;
        self.evicted_messages = 0;
        self.evicted_chars = 0;
        self.truncated_chars = 0;
    }

    /// Number of stored messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether no messages are stored.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    fn push(&mut self, role: Role, text: String) {
        let original_chars = char_count(&text);
        let mut text = text;
        let mut kept_chars = original_chars;

        if original_chars > self.max_chars {
            // Keep the first `max_chars` characters on a char boundary. A
            // `char` may be several bytes, so the cut is located by char
            // index, never by raw byte offset.
            let end = text
                .char_indices()
                .nth(self.max_chars)
                .map_or(text.len(), |(index, _)| index);
            text.truncate(end);
            kept_chars = self.max_chars;
            self.truncated_chars = self
                .truncated_chars
                .saturating_add(to_u64(original_chars.saturating_sub(kept_chars)));
        }

        // Evict oldest non-system messages first; only when none remain does a
        // system message lose its place. The loop ends because the incoming
        // message now fits the char bound and an empty context has no chars.
        while self.messages.len() >= self.max_messages
            || self.live_chars.saturating_add(kept_chars) > self.max_chars
        {
            if !self.evict_oldest() {
                break;
            }
        }

        self.live_chars = self.live_chars.saturating_add(kept_chars);
        self.messages.push_back(ChatMessage { role, text });
    }

    /// Remove the oldest non-system message, or the oldest message when every
    /// entry is a system message. Returns `false` when the context is empty.
    fn evict_oldest(&mut self) -> bool {
        let index = self
            .messages
            .iter()
            .position(|message| message.role != Role::System)
            .or(if self.messages.is_empty() {
                None
            } else {
                Some(0)
            });
        let Some(index) = index else {
            return false;
        };
        let Some(removed) = self.messages.remove(index) else {
            return false;
        };
        let chars = char_count(&removed.text);
        self.live_chars = self.live_chars.saturating_sub(chars);
        self.evicted_messages = self.evicted_messages.saturating_add(1);
        self.evicted_chars = self.evicted_chars.saturating_add(to_u64(chars));
        true
    }
}

fn char_count(text: &str) -> usize {
    text.chars().count()
}

/// Widen a `usize` counter to `u64` without an unchecked `as` cast.
fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(context: &ChatContext) -> Vec<String> {
        context
            .messages()
            .into_iter()
            .map(|message| message.text)
            .collect()
    }

    #[test]
    fn user_messages_are_evicted_before_system_messages() {
        let mut context = ChatContext::new(4, 1_000).unwrap();
        context.push_system("sys");
        context.push_user("u1");
        context.push_assistant("a1");
        context.push_user("u2");
        context.push_user("u3");

        assert_eq!(texts(&context), ["sys", "a1", "u2", "u3"]);
        assert_eq!(context.stats().evicted_messages, 1);
    }

    #[test]
    fn system_messages_are_evicted_once_no_non_system_remains() {
        let mut context = ChatContext::new(2, 1_000).unwrap();
        context.push_system("s1");
        context.push_system("s2");
        context.push_system("s3");

        assert_eq!(texts(&context), ["s2", "s3"]);
        assert_eq!(context.stats().evicted_messages, 1);
    }

    #[test]
    fn char_budget_evicts_whole_messages_and_counts_them() {
        let mut context = ChatContext::new(10, 10).unwrap();
        context.push_user("12345");
        context.push_user("67890");
        context.push_user("abcde");

        assert_eq!(texts(&context), ["67890", "abcde"]);
        let stats = context.stats();
        assert_eq!(stats.messages, 2);
        assert_eq!(stats.chars, 10);
        assert_eq!(stats.evicted_messages, 1);
        assert_eq!(stats.evicted_chars, 5);
        assert_eq!(stats.truncated_chars, 0);
    }

    #[test]
    fn oversized_message_is_truncated_on_a_char_boundary() {
        let mut context = ChatContext::new(4, 5).unwrap();
        context.push_user("hello world");

        assert_eq!(texts(&context), ["hello"]);
        assert_eq!(context.stats().truncated_chars, 6);

        // Multi-byte characters must not be split mid-codepoint.
        let mut unicode = ChatContext::new(4, 3).unwrap();
        unicode.push_user("aé🙂b");
        assert_eq!(texts(&unicode), ["aé🙂"]);
        assert_eq!(unicode.stats().truncated_chars, 1);
    }

    #[test]
    fn clear_empties_messages_and_resets_counters() {
        let mut context = ChatContext::new(2, 10).unwrap();
        context.push_user("123456");
        context.push_assistant("789012");
        assert!(context.stats().evicted_messages > 0 || context.stats().truncated_chars > 0);

        context.clear();
        assert!(context.is_empty());
        assert_eq!(context.len(), 0);
        assert_eq!(
            context.stats(),
            ContextStats {
                messages: 0,
                chars: 0,
                evicted_messages: 0,
                evicted_chars: 0,
                truncated_chars: 0,
            }
        );
    }

    #[test]
    fn new_rejects_bounds_that_cannot_hold_a_turn() {
        let err = ChatContext::new(1, 10).unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        assert!(err.to_string().contains("max_messages"));

        let err = ChatContext::new(2, 0).unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        assert!(err.to_string().contains("max_chars"));
    }

    mod properties {
        use super::*;
        use proptest::prelude::*;

        fn arb_role() -> impl Strategy<Value = Role> {
            prop_oneof![Just(Role::System), Just(Role::User), Just(Role::Assistant)]
        }

        fn arb_text() -> impl Strategy<Value = String> {
            prop::collection::vec(any::<char>(), 0..40)
                .prop_map(|chars| chars.into_iter().collect())
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn bounds_hold_and_characters_are_conserved(
                max_messages in 2usize..8,
                max_chars in 1usize..64,
                pushes in prop::collection::vec((arb_role(), arb_text()), 0..32),
            ) {
                let mut context = ChatContext::new(max_messages, max_chars).unwrap();
                let mut pushed_chars = 0u64;
                for (role, text) in pushes {
                    pushed_chars = pushed_chars.saturating_add(to_u64(char_count(&text)));
                    match role {
                        Role::System => context.push_system(text),
                        Role::User => context.push_user(text),
                        Role::Assistant => context.push_assistant(text),
                    }
                    let stats = context.stats();
                    prop_assert!(stats.messages <= max_messages);
                    prop_assert!(stats.chars <= max_chars);
                    let live: usize = context
                        .messages()
                        .iter()
                        .map(|message| char_count(&message.text))
                        .sum();
                    prop_assert_eq!(live, stats.chars);
                    // Every character pushed is either live, evicted with a
                    // whole message, or truncated away. Nothing vanishes.
                    prop_assert_eq!(
                        pushed_chars,
                        to_u64(stats.chars)
                            .saturating_add(stats.evicted_chars)
                            .saturating_add(stats.truncated_chars)
                    );
                }
            }
        }
    }
}
