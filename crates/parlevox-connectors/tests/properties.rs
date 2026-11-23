//! Property tests for the parsers, the bounded queue and auth redaction.
//!
//! These are the tier-3 checks: arbitrary bytes must never panic, chunk
//! splits must not change an event sequence, and buffer accounting must
//! conserve items.

#![cfg(feature = "http")]

use parlevox_connectors::{
    Auth, BoundedQueue, ConnectorError, NdjsonParser, RequestSpec, Secret, SseEvent, SseParser,
};
use proptest::prelude::*;

fn parse_sse_whole(data: &[u8], max_buffer_bytes: usize) -> Result<Vec<SseEvent>, ConnectorError> {
    let mut parser = SseParser::new(max_buffer_bytes);
    let mut events = parser.push(data)?;
    events.extend(parser.finish()?);
    Ok(events)
}

fn parse_sse_chunked(
    data: &[u8],
    splits: &[usize],
    max_buffer_bytes: usize,
) -> Result<Vec<SseEvent>, ConnectorError> {
    let mut parser = SseParser::new(max_buffer_bytes);
    let mut events = Vec::new();
    let mut start = 0usize;
    for split in splits {
        let end = start.saturating_add(*split).min(data.len());
        events.extend(parser.push(&data[start..end])?);
        start = end;
    }
    events.extend(parser.push(&data[start..])?);
    events.extend(parser.finish()?);
    Ok(events)
}

proptest! {
    #[test]
    fn proptest_sse_event_sequence_is_invariant_to_chunk_splits(
        data in proptest::collection::vec(any::<u8>(), 0..256),
        splits in proptest::collection::vec(0usize..64, 0..8),
    ) {
        let max_buffer_bytes = 1 << 16;
        let whole = parse_sse_whole(&data, max_buffer_bytes);
        let chunked = parse_sse_chunked(&data, &splits, max_buffer_bytes);
        prop_assert_eq!(whole, chunked);
    }

    #[test]
    fn proptest_sse_and_ndjson_parsers_never_panic_on_arbitrary_bytes(
        data in proptest::collection::vec(any::<u8>(), 0..512),
        chunk in 1usize..64,
    ) {
        // A deliberately small bound exercises the capacity path.
        let mut sse = SseParser::new(64);
        for piece in data.chunks(chunk) {
            let _ = sse.push(piece);
        }
        let _ = sse.finish();

        let mut ndjson = NdjsonParser::new(64);
        for piece in data.chunks(chunk) {
            let _ = ndjson.push(piece);
        }
        let _ = ndjson.finish();
    }

    #[test]
    fn proptest_bounded_queue_conserves_pushed_equals_delivered_plus_dropped(
        capacity in 0usize..8,
        items in proptest::collection::vec(any::<u16>(), 0..64),
        pops in proptest::collection::vec(any::<bool>(), 0..64),
    ) {
        let mut queue = BoundedQueue::new(capacity);
        let mut pushed = 0u64;
        let mut dropped = 0u64;
        let mut delivered = 0u64;
        let mut pop_now = pops.into_iter();
        for item in items {
            pushed = pushed.saturating_add(1);
            if queue.push(item).is_some() {
                dropped = dropped.saturating_add(1);
            }
            if pop_now.next().unwrap_or(false) && queue.pop().is_some() {
                delivered = delivered.saturating_add(1);
            }
        }
        while queue.pop().is_some() {
            delivered = delivered.saturating_add(1);
        }
        prop_assert_eq!(pushed, delivered.saturating_add(dropped));
    }

    #[test]
    fn proptest_auth_debug_never_contains_the_secret(suffix in "[a-z0-9]{4,32}") {
        // A credential-shaped value: the random part can never be a substring
        // of the redaction marker itself.
        let secret = format!("sk-test-{suffix}");
        let token = Secret::new(secret.clone());
        let auths = vec![
            Auth::bearer(token.clone()),
            Auth::custom_header("x-api-key", token.clone()).expect("token is a valid header value"),
            Auth::query_template("api_key", "key={key}", token).expect("template has a placeholder"),
        ];
        for auth in auths {
            let debug = format!("{auth:?}");
            prop_assert!(!debug.contains(&secret), "auth debug leaked: {}", debug);
            let mut request = RequestSpec::post("https://example.invalid/v1");
            auth.apply(&mut request).expect("valid auth applies");
            let debug = format!("{request:?}");
            prop_assert!(!debug.contains(&secret), "request debug leaked: {}", debug);
        }
    }
}
