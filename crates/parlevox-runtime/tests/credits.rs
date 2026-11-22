//! Behavioural checks for the opt-in playout credit signal.
//!
//! The ring still drops rather than throttles. These tests prove the two things
//! the signal is allowed to do — resolve `Granted` when slack opens and
//! `Cancelled` when the caller's token fires — and the one thing it must never
//! do: block a turn.

use std::time::Duration;

use parlevox_core::{AudioFrame, AudioRing, CreditWait, OverflowPolicy, PushOutcome, Timestamp};
use parlevox_runtime::{
    demo_utterance, ProviderNames, ProviderRegistry, Session, SessionConfig, SessionCredits,
};
use tokio_util::sync::CancellationToken;

fn frame() -> AudioFrame {
    AudioFrame::new(Timestamp::from_millis(0), 16_000, 1, vec![1; 16])
}

#[tokio::test]
async fn credit_wait_granted_when_slack_opens() {
    let (mut ring, handle) = AudioRing::new_with_credits(1, OverflowPolicy::DropOldest);
    assert_eq!(ring.push(frame()), PushOutcome::Accepted);
    assert_eq!(handle.snapshot().slack, 0, "the ring starts full");

    let credits = SessionCredits::from_handle(handle);
    let token = CancellationToken::new();
    let wait_token = token.clone();
    let waiter = tokio::spawn(async move { credits.wait(&wait_token).await });
    // Let the waiter register against a full ring before any slack opens.
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished(), "there is no slack yet");

    assert!(ring.pop().is_some());
    let outcome = tokio::time::timeout(Duration::from_secs(5), waiter)
        .await
        .expect("waiter resolves when slack opens")
        .expect("waiter task did not panic");
    assert_eq!(outcome, CreditWait::Granted);
}

#[tokio::test]
async fn credit_wait_cancelled_when_token_fires_first() {
    let (mut ring, handle) = AudioRing::new_with_credits(1, OverflowPolicy::DropOldest);
    assert_eq!(ring.push(frame()), PushOutcome::Accepted);

    let credits = SessionCredits::from_handle(handle.clone());
    let token = CancellationToken::new();
    let wait_token = token.clone();
    let waiter = tokio::spawn(async move { credits.wait(&wait_token).await });
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished(), "slack never opened");

    token.cancel();
    let outcome = tokio::time::timeout(Duration::from_secs(5), waiter)
        .await
        .expect("cancellation resolves the wait")
        .expect("waiter task did not panic");
    assert_eq!(outcome, CreditWait::Cancelled);

    let stats = handle.stats();
    assert_eq!(stats.cancelled, 1);
    assert_eq!(stats.granted, 0);
}

/// A genuine two-worker race between registration and the pop that opens
/// slack. `loom` is out of reach here (the workspace forbids new dependencies),
/// so this is a probabilistic stress test: half the rounds let the waiter
/// register first, half let the pop run first, and the multi-thread runtime
/// additionally lets the two genuinely overlap. A lost wakeup leaves the await
/// (and therefore the loop) stuck until the overall timeout fails the test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_pop_never_loses_a_waiting_consumer() {
    const ROUNDS: usize = 5_000;
    let completed = tokio::time::timeout(Duration::from_secs(30), async {
        for round in 0..ROUNDS {
            let (mut ring, handle) = AudioRing::new_with_credits(1, OverflowPolicy::DropOldest);
            assert_eq!(ring.push(frame()), PushOutcome::Accepted, "round {round}");

            let credits = SessionCredits::from_handle(handle);
            let token = CancellationToken::new();
            let wait_token = token.clone();
            let waiter = tokio::spawn(async move { credits.wait(&wait_token).await });

            // Alternate the interleaving: even rounds register first, odd
            // rounds pop first. Both orderings must resolve `Granted`.
            if round % 2 == 0 {
                tokio::task::yield_now().await;
            }
            assert!(ring.pop().is_some(), "round {round}");

            let outcome = waiter.await.expect("waiter task did not panic");
            assert_eq!(outcome, CreditWait::Granted, "round {round}");
        }
    })
    .await;

    assert!(
        completed.is_ok(),
        "a concurrent pop lost the waiter: the credit wait never resolved"
    );
}

#[tokio::test]
async fn outstanding_credit_waiter_does_not_block_a_turn() {
    let providers = ProviderRegistry::fakes()
        .expect("in-process fakes register")
        .resolve(&ProviderNames::default())
        .expect("in-process fakes resolve");
    let config = SessionConfig {
        // Slack is permanently zero, so the waiter can never be granted and
        // must stay pending for the whole turn.
        output_ring_capacity: 0,
        ..SessionConfig::default()
    };
    let mut session = Session::new(config, providers).expect("session builds");

    let credits = session.credits();
    let token = CancellationToken::new();
    let wait_token = token.clone();
    let waiter = tokio::spawn(async move { credits.wait(&wait_token).await });
    tokio::task::yield_now().await;
    assert!(!waiter.is_finished(), "capacity zero never opens slack");

    let report = session
        .run_turn(demo_utterance(16_000, 1, 40))
        .await
        .expect("a turn completes while a credit waiter is pending");
    assert_eq!(report.audio_frames, 0, "capacity zero drops every frame");
    assert!(report.dropped_frames > 0, "the drops are still counted");
    assert!(
        !waiter.is_finished(),
        "an outstanding credit waiter must not block run_turn"
    );
    assert_eq!(session.credit_stats().granted, 0);

    token.cancel();
    let outcome = tokio::time::timeout(Duration::from_secs(5), waiter)
        .await
        .expect("cancellation resolves the pending wait")
        .expect("waiter task did not panic");
    assert_eq!(outcome, CreditWait::Cancelled);
    assert_eq!(session.credit_stats().cancelled, 1);
}
