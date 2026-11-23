//! Integration tests for the opt-in adaptive endpointing decorator.
//!
//! The [`ScriptedVad`] double below implements the real [`VadProvider`] trait
//! and returns a scripted transition sequence on every `analyze` call. Time is
//! data in the transitions and in the [`AudioChunk`] length, never a clock, so
//! every test is deterministic and no test sleeps.
//!
//! At a sample rate of 1000 Hz with one channel, one PCM sample is exactly one
//! millisecond, which keeps the trailing-silence arithmetic in the tests
//! readable: [`audio_ms`] builds an utterance of the requested duration.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use parlevox_core::{ErrorClass, ParlevoxError};
use parlevox_providers::{
    AdaptiveEndpointingVad, AudioChunk, EndpointingPolicy, EndpointingStats, FallbackVad, Modality,
    ProviderManifest, ProviderStream, VadEvent, VadProvider, VadTransition,
};
use tokio_util::sync::CancellationToken;

type MakeErr = fn() -> ParlevoxError;

/// One scripted item: a transition, or an error made fresh per `analyze` call.
enum Item {
    Transition(VadTransition),
    Error(MakeErr),
}

/// A VAD double whose script is fixed and replayed on every call.
struct ScriptedVad {
    manifest: ProviderManifest,
    items: Vec<Item>,
}

impl ScriptedVad {
    fn scripted(name: &str, items: Vec<Item>) -> Arc<dyn VadProvider> {
        Arc::new(ScriptedVad {
            manifest: ProviderManifest::new(name, Modality::Vad),
            items,
        })
    }
}

#[async_trait]
impl VadProvider for ScriptedVad {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn analyze(
        &self,
        _audio: AudioChunk,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream<VadTransition>, ParlevoxError> {
        let items = self
            .items
            .iter()
            .map(|item| match item {
                Item::Transition(transition) => Ok(*transition),
                Item::Error(make) => Err(make()),
            })
            .collect::<Vec<Result<VadTransition, ParlevoxError>>>();
        Ok(Box::pin(futures_util::stream::iter(items)))
    }
}

fn transition(event: VadEvent, timestamp_ms: u64) -> VadTransition {
    VadTransition {
        event,
        timestamp_ms,
    }
}

fn speech_start(timestamp_ms: u64) -> Item {
    Item::Transition(transition(VadEvent::SpeechStart, timestamp_ms))
}

fn speech_end(timestamp_ms: u64) -> Item {
    Item::Transition(transition(VadEvent::SpeechEnd, timestamp_ms))
}

fn policy(
    min_trailing_silence_ms: u64,
    adapt_ms_per_speech_second: u64,
    max_trailing_silence_ms: u64,
) -> EndpointingPolicy {
    EndpointingPolicy {
        min_trailing_silence_ms,
        adapt_ms_per_speech_second,
        max_trailing_silence_ms,
    }
}

/// An utterance of exactly `ms` milliseconds at 1000 Hz, one channel.
fn audio_ms(ms: u64) -> AudioChunk {
    let samples = usize::try_from(ms).expect("test durations fit in usize");
    AudioChunk::new(1000, 1, vec![0; samples])
}

/// Analyze `audio` and drain the wrapper's stream.
async fn run(
    vad: &AdaptiveEndpointingVad,
    audio: AudioChunk,
) -> (Vec<VadTransition>, Option<ParlevoxError>) {
    let stream = vad
        .analyze(audio, CancellationToken::new())
        .await
        .expect("the wrapper builds a stream without polling the inner provider");
    drain(stream).await
}

async fn drain(
    mut stream: ProviderStream<VadTransition>,
) -> (Vec<VadTransition>, Option<ParlevoxError>) {
    let mut items = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(item) => items.push(item),
            Err(error) => return (items, Some(error)),
        }
    }
    (items, None)
}

#[tokio::test]
async fn default_policy_is_a_pass_through() {
    // Arrange: the all-zero default falls through the decorator untouched.
    let inner = ScriptedVad::scripted("scripted-vad", vec![speech_start(0), speech_end(500)]);
    let vad = AdaptiveEndpointingVad::new(inner, EndpointingPolicy::default());

    // Act
    let (items, error) = run(&vad, audio_ms(500)).await;

    // Assert
    assert!(
        error.is_none(),
        "no error escapes the default policy: {error:?}"
    );
    assert_eq!(
        items,
        vec![
            transition(VadEvent::SpeechStart, 0),
            transition(VadEvent::SpeechEnd, 500),
        ],
        "the default policy emits the pair unchanged in event and timestamp_ms"
    );
    assert_eq!(
        vad.stats(),
        EndpointingStats {
            accepted: 1,
            rejected: 0,
        },
        "the default policy accepts the sole SpeechEnd"
    );

    // The manual Debug names the decorator and the inner provider, never the
    // wrapper's object graph, and formats without touching the inner provider.
    let rendered = format!("{vad:?}");
    assert!(
        rendered.starts_with("AdaptiveEndpointingVad"),
        "Debug names the decorator: {rendered}"
    );
    assert!(
        rendered.contains("scripted-vad"),
        "Debug shows the inner provider name, not the object: {rendered}"
    );
}

#[tokio::test]
async fn early_speech_end_is_dropped_then_synthesized_at_audio_end() {
    // Arrange: 150ms of trailing silence is below the 200ms minimum.
    let inner = ScriptedVad::scripted("scripted-vad", vec![speech_start(0), speech_end(100)]);
    let vad = AdaptiveEndpointingVad::new(inner, policy(200, 0, 1000));

    // Act
    let (items, error) = run(&vad, audio_ms(250)).await;

    // Assert
    assert!(
        error.is_none(),
        "no error escapes when the early end is replaced: {error:?}"
    );
    assert_eq!(
        items,
        vec![
            transition(VadEvent::SpeechStart, 0),
            transition(VadEvent::SpeechEnd, 250),
        ],
        "the dropped end is replaced by one end at the audio duration"
    );
    assert_eq!(
        items
            .iter()
            .filter(|item| item.event == VadEvent::SpeechEnd)
            .count(),
        1,
        "exactly one SpeechEnd is emitted"
    );
    assert_eq!(
        vad.stats(),
        EndpointingStats {
            accepted: 1,
            rejected: 1,
        },
        "the reject and the synthesized accept are both counted"
    );
}

#[tokio::test]
async fn adaptive_allowance_changes_the_verdict() {
    // Arrange: the same SpeechEnd@1000 is scripted for both utterances; only
    // the speech length (and therefore the adaptive allowance) differs.
    let same_end = transition(VadEvent::SpeechEnd, 1000);
    let adaptive = policy(100, 200, 2000);

    let short_inner = ScriptedVad::scripted(
        "short-vad",
        vec![speech_start(400), Item::Transition(same_end)],
    );
    let short = AdaptiveEndpointingVad::new(short_inner, adaptive);

    let long_inner = ScriptedVad::scripted(
        "long-vad",
        vec![speech_start(0), Item::Transition(same_end)],
    );
    let long = AdaptiveEndpointingVad::new(long_inner, adaptive);

    let control_inner = ScriptedVad::scripted(
        "control-vad",
        vec![speech_start(400), Item::Transition(same_end)],
    );
    let control = AdaptiveEndpointingVad::new(control_inner, policy(100, 0, 2000));

    // Act
    let (short_items, short_error) = run(&short, audio_ms(1150)).await;
    let (long_items, long_error) = run(&long, audio_ms(2000)).await;
    let (control_items, control_error) = run(&control, audio_ms(1150)).await;

    // Assert
    assert!(
        short_error.is_none(),
        "the short utterance returns no error: {short_error:?}"
    );
    assert!(
        long_error.is_none(),
        "the long utterance returns no error: {long_error:?}"
    );
    assert!(
        control_error.is_none(),
        "the control utterance returns no error: {control_error:?}"
    );
    assert_eq!(
        short_items,
        vec![
            transition(VadEvent::SpeechStart, 400),
            transition(VadEvent::SpeechEnd, 1150),
        ],
        "SpeechEnd@1000 is rejected for the short utterance (150ms trailing < 220ms allowance)"
    );
    assert!(
        !short_items.contains(&same_end),
        "the short utterance's original SpeechEnd@1000 is dropped"
    );
    assert_eq!(
        short.stats(),
        EndpointingStats {
            accepted: 1,
            rejected: 1,
        },
        "the short utterance rejects one end and accepts the synthesized one"
    );
    assert_eq!(
        long_items,
        vec![
            transition(VadEvent::SpeechStart, 0),
            transition(VadEvent::SpeechEnd, 1000),
        ],
        "the same SpeechEnd@1000 is accepted for the longer utterance"
    );
    assert_eq!(
        long.stats(),
        EndpointingStats {
            accepted: 1,
            rejected: 0,
        },
        "the longer utterance accepts its end"
    );
    assert_eq!(
        control_items,
        vec![
            transition(VadEvent::SpeechStart, 400),
            transition(VadEvent::SpeechEnd, 1000),
        ],
        "with adapt=0 the short utterance's end is accepted, so the adaptive term flipped the verdict"
    );
    assert_eq!(
        control.stats(),
        EndpointingStats {
            accepted: 1,
            rejected: 0,
        },
        "the control policy accepts the same short utterance"
    );
}

#[tokio::test]
async fn max_trailing_silence_caps_the_allowance() {
    // Arrange: the uncapped requirement is 100 + 10000 = 10100ms of trailing
    // silence, but the 300ms cap admits an end with only 500ms.
    let inner = ScriptedVad::scripted("scripted-vad", vec![speech_start(0), speech_end(10_000)]);
    let capped = AdaptiveEndpointingVad::new(inner, policy(100, 1000, 300));

    let uncapped_inner =
        ScriptedVad::scripted("uncapped-vad", vec![speech_start(0), speech_end(10_000)]);
    let uncapped = AdaptiveEndpointingVad::new(uncapped_inner, policy(100, 1000, 20_000));

    // Act
    let (capped_items, capped_error) = run(&capped, audio_ms(10_500)).await;
    let (uncapped_items, uncapped_error) = run(&uncapped, audio_ms(10_500)).await;

    // Assert
    assert!(
        capped_error.is_none(),
        "the capped policy returns no error: {capped_error:?}"
    );
    assert!(
        uncapped_error.is_none(),
        "the uncapped policy returns no error: {uncapped_error:?}"
    );
    assert_eq!(
        capped_items,
        vec![
            transition(VadEvent::SpeechStart, 0),
            transition(VadEvent::SpeechEnd, 10_000),
        ],
        "the 300ms cap admits the end at 500ms of trailing silence"
    );
    assert_eq!(
        capped.stats(),
        EndpointingStats {
            accepted: 1,
            rejected: 0,
        },
        "the capped allowance accepts the end"
    );
    assert_eq!(
        uncapped_items,
        vec![
            transition(VadEvent::SpeechStart, 0),
            transition(VadEvent::SpeechEnd, 10_500),
        ],
        "without the cap the same 500ms trailing silence is rejected"
    );
    assert_eq!(
        uncapped.stats(),
        EndpointingStats {
            accepted: 1,
            rejected: 1,
        },
        "the uncapped policy rejects the end before synthesizing one"
    );
}

#[tokio::test]
async fn two_speech_pairs_are_each_evaluated() {
    // Arrange: the adaptive term is chosen so pair 2 is accepted only if t0
    // resets to 5000 after pair 1's accepted end. Without the reset the pair's
    // speech would be 5100ms, giving an allowance of 610ms that rejects its
    // 400ms of trailing silence.
    let inner = ScriptedVad::scripted(
        "scripted-vad",
        vec![
            speech_start(0),
            speech_end(200),
            speech_start(5000),
            speech_end(5100),
        ],
    );
    let vad = AdaptiveEndpointingVad::new(inner, policy(100, 100, 1000));

    // Act
    let (items, error) = run(&vad, audio_ms(5500)).await;

    // Assert
    assert!(
        error.is_none(),
        "both accepted pairs pass through without an error: {error:?}"
    );
    assert_eq!(
        items,
        vec![
            transition(VadEvent::SpeechStart, 0),
            transition(VadEvent::SpeechEnd, 200),
            transition(VadEvent::SpeechStart, 5000),
            transition(VadEvent::SpeechEnd, 5100),
        ],
        "both pairs pass through unchanged, so t0 reset after the first accepted end"
    );
    assert_eq!(
        vad.stats(),
        EndpointingStats {
            accepted: 2,
            rejected: 0,
        },
        "each accepted end is counted once"
    );
}

#[tokio::test]
async fn inner_error_is_forwarded_without_synthesized_end() {
    // Arrange: a start followed by a transitory provider error.
    let inner = ScriptedVad::scripted(
        "scripted-vad",
        vec![
            speech_start(0),
            Item::Error(|| ParlevoxError::provider("scripted-vad", "analysis failed")),
        ],
    );
    let vad = AdaptiveEndpointingVad::new(inner, policy(200, 0, 1000));
    let mut stream = vad
        .analyze(audio_ms(1000), CancellationToken::new())
        .await
        .expect("the wrapper builds a stream");

    // Act
    let first = stream.next().await.expect("the start is forwarded");
    let errored = stream.next().await.expect("the inner error is forwarded");
    let after = stream.next().await;

    // Assert
    assert_eq!(
        first.expect("the start is an Ok item"),
        transition(VadEvent::SpeechStart, 0)
    );
    let error = errored.expect_err("the scripted error is forwarded");
    let ParlevoxError::Provider {
        provider,
        message,
        class,
        ..
    } = &error
    else {
        panic!("expected the scripted provider error, got {error}");
    };
    assert_eq!(provider, "scripted-vad", "the error keeps its provider");
    assert_eq!(message, "analysis failed", "the error keeps its message");
    assert_eq!(*class, ErrorClass::Transient, "the error keeps its class");
    assert!(after.is_none(), "no synthesized end follows an inner error");
    assert_eq!(
        vad.stats(),
        EndpointingStats {
            accepted: 0,
            rejected: 0,
        },
        "an error is neither accepted nor rejected"
    );
}

#[tokio::test]
async fn cancelled_ends_the_stream_immediately() {
    // Arrange: a start followed by an inner cancellation.
    let inner = ScriptedVad::scripted(
        "scripted-vad",
        vec![speech_start(0), Item::Error(|| ParlevoxError::Cancelled)],
    );
    let vad = AdaptiveEndpointingVad::new(inner, policy(200, 0, 1000));
    let mut stream = vad
        .analyze(audio_ms(1000), CancellationToken::new())
        .await
        .expect("the wrapper builds a stream");

    // Act
    let first = stream.next().await.expect("the start is forwarded");
    let errored = stream.next().await.expect("the cancellation is forwarded");
    let after = stream.next().await;

    // Assert
    assert_eq!(
        first.expect("the start is an Ok item"),
        transition(VadEvent::SpeechStart, 0)
    );
    assert!(
        matches!(errored, Err(ParlevoxError::Cancelled)),
        "cancellation is forwarded unchanged, got {errored:?}"
    );
    assert!(
        after.is_none(),
        "cancellation ends the stream with no synthesized end"
    );
    assert_eq!(
        vad.stats(),
        EndpointingStats {
            accepted: 0,
            rejected: 0,
        },
        "a cancellation is neither accepted nor rejected"
    );
}

#[tokio::test]
async fn wraps_fallback_vad_and_inherits_manifest() {
    // Arrange: a chain whose primary is "primary-vad".
    let primary = ScriptedVad::scripted("primary-vad", vec![speech_start(0), speech_end(500)]);
    let backup = ScriptedVad::scripted("backup-vad", vec![speech_start(0)]);
    let chain = FallbackVad::new(vec![primary, backup]).expect("chain is non-empty");

    // Act: the exact shape the acceptance criteria name.
    let wrapped: Arc<dyn VadProvider> = Arc::new(AdaptiveEndpointingVad::new(
        Arc::new(chain),
        EndpointingPolicy::default(),
    ));
    let stream = wrapped
        .analyze(audio_ms(500), CancellationToken::new())
        .await
        .expect("the wrapped chain builds a stream");
    let (items, error) = drain(stream).await;

    // Assert
    assert_eq!(
        wrapped.manifest().name,
        "primary-vad",
        "the wrapper inherits the chain primary's manifest"
    );
    assert!(
        error.is_none(),
        "the wrapped chain returns no error under the default policy: {error:?}"
    );
    assert_eq!(
        items,
        vec![
            transition(VadEvent::SpeechStart, 0),
            transition(VadEvent::SpeechEnd, 500),
        ],
        "the wrapped chain passes the pair through under the default policy"
    );
}

/// The effective allowance computed independently of the decorator's code.
fn reference_effective(policy: EndpointingPolicy, start: u64, end: u64) -> u64 {
    let speech_ms = end.saturating_sub(start);
    let adaptive = speech_ms
        .saturating_mul(policy.adapt_ms_per_speech_second)
        .saturating_div(1000);
    policy
        .min_trailing_silence_ms
        .saturating_add(adaptive)
        .min(policy.max_trailing_silence_ms)
}

#[tokio::test]
async fn policy_grid_matches_reference_model() {
    // Arrange: a deterministic sweep of the policy space, checked against a
    // reference saturating model. `start`/`end`/`duration` are fixed so every
    // cell differs only by its policy.
    const MIN: [u64; 3] = [0, 100, 300];
    const ADAPT: [u64; 3] = [0, 50, 200];
    const MAX: [u64; 3] = [0, 500, 1000];
    let start: u64 = 200;
    let end: u64 = 900;
    let duration: u64 = 1000;

    let mut accepted_total = 0_u64;
    let mut rejected_total = 0_u64;
    let mut reject_cases = 0_u64;
    let mut cases = 0_u64;

    // Act + Assert: each cell must match the reference accept/reject verdict.
    for &min in &MIN {
        for &adapt in &ADAPT {
            for &max in &MAX {
                let policy = policy(min, adapt, max);
                let effective = reference_effective(policy, start, end);
                let expected_accept = duration.saturating_sub(end) >= effective;

                let inner =
                    ScriptedVad::scripted("grid-vad", vec![speech_start(start), speech_end(end)]);
                let vad = AdaptiveEndpointingVad::new(inner, policy);
                let (items, error) = run(&vad, audio_ms(duration)).await;
                assert!(error.is_none(), "policy {policy:?} yields no error");

                if expected_accept {
                    assert_eq!(
                        items,
                        vec![
                            transition(VadEvent::SpeechStart, start),
                            transition(VadEvent::SpeechEnd, end),
                        ],
                        "policy {policy:?} accepts the end in the reference model"
                    );
                    assert_eq!(
                        vad.stats(),
                        EndpointingStats {
                            accepted: 1,
                            rejected: 0,
                        },
                        "an accepted end increments only accepted for {policy:?}"
                    );
                } else {
                    reject_cases += 1;
                    assert_eq!(
                        items,
                        vec![
                            transition(VadEvent::SpeechStart, start),
                            transition(VadEvent::SpeechEnd, duration),
                        ],
                        "policy {policy:?} rejects the end and synthesizes one at the audio duration"
                    );
                    assert_eq!(
                        vad.stats(),
                        EndpointingStats {
                            accepted: 1,
                            rejected: 1,
                        },
                        "a rejected end plus the synthesized accept are counted for {policy:?}"
                    );
                }

                accepted_total += vad.stats().accepted;
                rejected_total += vad.stats().rejected;
                cases += 1;
            }
        }
    }

    assert_eq!(cases, 27, "the grid covers every policy combination");
    assert_eq!(
        accepted_total, 27,
        "every stream ends with exactly one accepted end"
    );
    assert_eq!(
        rejected_total, reject_cases,
        "the rejected total matches the reference partition"
    );
    assert!(
        reject_cases > 0 && reject_cases < cases,
        "the grid exercises both verdicts, saw {reject_cases} rejections"
    );

    // The all-zero policy is the default and a pure pass-through.
    assert_eq!(
        policy(0, 0, 0),
        EndpointingPolicy::default(),
        "the all-zero policy is the default"
    );
    let inner = ScriptedVad::scripted("default-vad", vec![speech_start(start), speech_end(end)]);
    let vad = AdaptiveEndpointingVad::new(inner, EndpointingPolicy::default());
    let (items, error) = run(&vad, audio_ms(duration)).await;
    assert!(
        error.is_none(),
        "the default policy returns no error in the grid sweep: {error:?}"
    );
    assert_eq!(
        items,
        vec![
            transition(VadEvent::SpeechStart, start),
            transition(VadEvent::SpeechEnd, end),
        ],
        "the all-zero default policy passes the pair through"
    );
    assert_eq!(
        vad.stats(),
        EndpointingStats {
            accepted: 1,
            rejected: 0,
        }
    );
}
