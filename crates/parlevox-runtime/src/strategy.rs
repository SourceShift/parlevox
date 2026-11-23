//! Session-level, ordered turn-start strategies.
//!
//! A [`TurnStrategy`] is consulted once per final transcript, at the point
//! between the empty-final rejection and the commit of the turn. The list is
//! ordered and consulted first-to-last; the first strategy that returns
//! [`TurnDecision::Accept`] or [`TurnDecision::Reject`] decides, and every
//! later strategy is skipped. A list whose strategies all abstain, an empty
//! list, and no list at all all mean "accept", which is the pre-strategy
//! behaviour byte-for-byte.
//!
//! The list only shapes the *turn start*. Utterance boundaries — when a turn
//! is considered over — are owned by the caller and the VAD decorator, so
//! endpoint/stop strategies are deliberately not part of this surface.
//!
//! Consultation is synchronous and short: [`TurnStrategy::decide`] must not
//! block, await, or spawn. The runtime calls it while driving a turn and holds
//! the session mutably across the call.

use parlevox_core::TurnState;

/// What the session sees when it is about to commit a turn.
#[derive(Debug, Clone, Copy)]
pub struct TurnContext<'a> {
    /// The final transcript the STT provider produced.
    pub transcript: &'a str,
    /// Machine state at the decision point (`UserSpeaking` after a VAD end, or
    /// `Listening` for a manually injected final).
    pub state: TurnState,
    /// Index of the *candidate* turn ([`TurnMachine::turn_index`] + 1).
    ///
    /// [`TurnMachine::turn_index`]: parlevox_core::TurnMachine::turn_index
    pub turn_index: u64,
}

/// One strategy's verdict at the turn-start decision point.
///
/// [`TurnDecision::Abstain`] defers to the next strategy (and ultimately to the
/// default [`TurnDecision::Accept`]); it does *not* hold the turn open. A
/// deployment that needs to wait or answer with a backchannel must express that
/// as [`TurnDecision::Reject`] today. The enum is `#[non_exhaustive]` so a
/// future verdict (such as a hold or backchannel decision) can be added without
/// breaking downstream code: a new variant is additive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TurnDecision {
    /// Commit the turn and run the reply.
    Accept,
    /// Do not commit the turn: no user message, no LLM call, no TTS call, and
    /// no audio are produced.
    Reject,
    /// Defer to the next strategy (and to the default [`TurnDecision::Accept`]).
    Abstain,
}

/// One pluggable turn-start strategy.
///
/// Implementations are stored in a [`TurnStrategies`] list and consulted in
/// order. `decide` takes `&mut self` so a strategy can keep whatever state it
/// needs; it must return promptly and must not perform I/O.
///
/// The `Send + Sync` bounds keep [`Session`]'s auto traits intact, because the
/// strategy list lives inside the session: [`Session`] must stay `Sync` for
/// callers that share it across threads.
///
/// [`Session`]: crate::session::Session
pub trait TurnStrategy: Send + Sync {
    /// Stable name, used in diagnostics and [`TurnStrategies::names`].
    fn name(&self) -> &'static str;

    /// Return this strategy's verdict for `context`.
    ///
    /// Return [`TurnDecision::Abstain`] to let a later strategy decide. The
    /// first `Accept` or `Reject` stops consultation for the turn.
    fn decide(&mut self, context: &TurnContext<'_>) -> TurnDecision;
}

/// Cumulative per-session consultation counters.
///
/// Every counter is zero while [`Session::with_turn_strategies`] has not been
/// called. [`StrategyStats::consulted`] always equals the sum of
/// [`StrategyStats::accepted`], [`StrategyStats::rejected`] and
/// [`StrategyStats::abstained`].
///
/// [`Session::with_turn_strategies`]: crate::session::Session::with_turn_strategies
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StrategyStats {
    /// Strategies invoked (one per strategy reached).
    pub consulted: u64,
    /// Consultations that returned [`TurnDecision::Accept`].
    pub accepted: u64,
    /// Consultations that returned [`TurnDecision::Reject`].
    pub rejected: u64,
    /// Consultations that returned [`TurnDecision::Abstain`].
    pub abstained: u64,
    /// Final transcripts that were rejected (a turn was not committed).
    pub turns_rejected: u64,
}

/// An ordered list of turn-start strategies.
///
/// Install it with [`Session::with_turn_strategies`]; without one the session
/// behaves exactly as before and [`Session::strategy_stats`] stays zero.
///
/// [`Session::with_turn_strategies`]: crate::session::Session::with_turn_strategies
/// [`Session::strategy_stats`]: crate::session::Session::strategy_stats
#[derive(Default)]
pub struct TurnStrategies {
    strategies: Vec<Box<dyn TurnStrategy>>,
}

impl TurnStrategies {
    /// Create an empty list. An empty list is inert: it accepts.
    #[must_use]
    pub fn new() -> Self {
        TurnStrategies {
            strategies: Vec::new(),
        }
    }

    /// Append one strategy to the end of the consultation order.
    pub fn push(&mut self, strategy: Box<dyn TurnStrategy>) {
        self.strategies.push(strategy);
    }

    /// Number of installed strategies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strategies.len()
    }

    /// Whether the list has no strategies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strategies.is_empty()
    }

    /// Strategy names in consultation order, for diagnostics.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.strategies
            .iter()
            .map(|strategy| strategy.name())
            .collect()
    }

    /// Consult the list and return the effective verdict plus the counters for
    /// this consultation.
    ///
    /// First-to-last, stopping at the first `Accept` or `Reject`. When every
    /// strategy abstains — or the list is empty — the effective verdict is
    /// [`TurnDecision::Accept`]. The returned counters satisfy
    /// `consulted == accepted + rejected + abstained`.
    pub(crate) fn resolve(&mut self, context: &TurnContext<'_>) -> (TurnDecision, StrategyStats) {
        let mut stats = StrategyStats::default();
        for strategy in &mut self.strategies {
            stats.consulted = stats.consulted.saturating_add(1);
            match strategy.decide(context) {
                TurnDecision::Accept => {
                    stats.accepted = stats.accepted.saturating_add(1);
                    return (TurnDecision::Accept, stats);
                }
                TurnDecision::Reject => {
                    stats.rejected = stats.rejected.saturating_add(1);
                    return (TurnDecision::Reject, stats);
                }
                TurnDecision::Abstain => {
                    stats.abstained = stats.abstained.saturating_add(1);
                }
            }
        }
        (TurnDecision::Accept, stats)
    }
}
