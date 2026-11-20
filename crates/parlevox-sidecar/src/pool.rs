//! A bounded, opt-in warm pool of handshaken plugin processes.
//!
//! A [`SidecarPool`] is composed by the caller; the runtime never knows a pool
//! exists. Acquiring a lease returns an already-handshaken process when one is
//! idle and healthy, and otherwise spawns a fresh one. Releasing runs the
//! additive `reset` handshake before the process is pooled.
//!
//! The pool only bounds *idle* growth: concurrent leases can still hold more
//! live processes than [`PoolConfig::max_idle`]. An idle entry older than
//! [`PoolConfig::max_age`] is shut down instead of handed out, and a process
//! whose acquire-time probe reset fails is killed, counted, and replaced by a
//! fresh spawn — it is never handed out dirty.
//!
//! A process is also retired after it has served [`PoolConfig::max_reuses`]
//! pooled returns, so a plugin that accumulates state across `reset` calls is
//! replaced instead of being reused forever.
//!
//! The two bounds read `0` in opposite ways, deliberately: a `max_idle` of `0`
//! disables pooling, so every release shuts its process down, while a
//! `max_reuses` of `0` means **unlimited**, so no process is ever retired for
//! its reuse count.
//!
//! Reset runs twice per reuse: once on release, so only a process that proved
//! it can clear itself enters the idle queue, and once on acquire, which is the
//! only authoritative liveness and cleanliness check before handover. That is
//! deliberate, not redundant: the two round-trips are far cheaper than an
//! interpreter start.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parlevox_core::ParlevoxError;
use tokio::sync::Mutex;

use crate::client::SidecarClient;
use crate::config::SidecarConfig;

/// Idle-process bounds for a [`SidecarPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PoolConfig {
    /// Maximum number of idle processes kept alive. `0` disables pooling, so
    /// every release shuts its process down. That is the opposite of
    /// [`PoolConfig::max_reuses`], where `0` means unlimited.
    pub max_idle: usize,
    /// Maximum time a process may sit idle. Age is checked at lease time: a
    /// process older than this is shut down rather than handed out.
    pub max_age: Duration,
    /// Maximum number of pooled returns a single process may serve before it
    /// is retired instead of pooled again. Unlike [`PoolConfig::max_idle`],
    /// where `0` disables pooling, `0` here means **unlimited**.
    pub max_reuses: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            max_idle: 2,
            max_age: Duration::from_secs(60),
            max_reuses: 64,
        }
    }
}

impl PoolConfig {
    /// Build a configuration with an explicit idle cap and age ceiling.
    ///
    /// [`PoolConfig::max_reuses`] keeps its default; use
    /// [`PoolConfig::with_max_reuses`] to change it.
    #[must_use]
    pub fn new(max_idle: usize, max_age: Duration) -> Self {
        PoolConfig {
            max_idle,
            max_age,
            ..PoolConfig::default()
        }
    }

    /// Return this configuration with a different [`PoolConfig::max_idle`].
    #[must_use = "builder methods return a modified config; chain or assign the result"]
    pub fn with_max_idle(mut self, max_idle: usize) -> Self {
        self.max_idle = max_idle;
        self
    }

    /// Return this configuration with a different [`PoolConfig::max_age`].
    #[must_use = "builder methods return a modified config; chain or assign the result"]
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = max_age;
        self
    }

    /// Return this configuration with a different [`PoolConfig::max_reuses`].
    ///
    /// `0` means unlimited, so the process is never retired for its reuse
    /// count. That is the opposite of [`PoolConfig::max_idle`], where `0`
    /// disables pooling instead.
    #[must_use = "builder methods return a modified config; chain or assign the result"]
    pub fn with_max_reuses(mut self, max_reuses: u64) -> Self {
        self.max_reuses = max_reuses;
        self
    }
}

/// One idle client plus the instant it became idle.
///
/// `since` is measured from release, not spawn: the age ceiling is about how
/// long a handshaken process has been parked, not how old the process is.
#[derive(Debug)]
#[non_exhaustive]
pub struct IdleEntry<T> {
    /// The idle client itself.
    pub client: T,
    /// The instant the client became idle, used to enforce the age ceiling.
    pub since: Instant,
    /// Accepted pooled returns this process has served. A fresh spawn starts at
    /// `0`; each accepted [`IdleQueue::push`] increments it by one.
    pub reuses: u64,
}

impl<T> IdleEntry<T> {
    /// Pair a client with the instant it became idle.
    ///
    /// The reuse count starts at `0`; use [`IdleEntry::with_reuses`] to restore
    /// a process's history when it is offered back to the queue.
    #[must_use]
    pub fn new(client: T, since: Instant) -> Self {
        IdleEntry {
            client,
            since,
            reuses: 0,
        }
    }

    /// Return this entry with a different [`IdleEntry::reuses`] count.
    #[must_use = "builder methods return a modified entry; chain or assign the result"]
    pub fn with_reuses(mut self, reuses: u64) -> Self {
        self.reuses = reuses;
        self
    }
}

/// Why [`IdleQueue::push`] refused an offered entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RefusalReason {
    /// Pooling is disabled (`max_idle == 0`) or the entry is already at or past
    /// the age ceiling. This keeps the historical `shutdown` fate. A zero
    /// *reuse* ceiling is not this variant: `max_reuses == 0` means unlimited,
    /// so the retirement check never fires.
    Configuration,
    /// The process has served [`PoolConfig::max_reuses`] pooled returns, so it
    /// is retired rather than pooled again. Only a nonzero ceiling retires:
    /// `max_reuses == 0` means unlimited, the opposite of `max_idle == 0`,
    /// which disables pooling.
    Retirement,
}

/// What [`IdleQueue::push`] did with an offered entry.
///
/// Both fields carry clients the caller must shut down; neither is retained by
/// the queue.
#[derive(Debug)]
#[must_use = "the outcome carries clients the caller must shut down"]
#[non_exhaustive]
pub struct PushOutcome<T> {
    /// Entries displaced to stay within the idle cap. At most one.
    pub displaced: Vec<T>,
    /// The offered entry, returned unchanged when the queue refuses it because
    /// pooling is disabled, it is already past the age ceiling, or it has
    /// reached its reuse ceiling.
    pub refused: Option<IdleEntry<T>>,
    /// Why the entry was refused. `Some` exactly when [`PushOutcome::refused`]
    /// is `Some`, because both are set by the same refusal branch.
    refusal: Option<RefusalReason>,
}

impl<T> PushOutcome<T> {
    /// Why the offered entry was refused, or `None` when it was accepted.
    #[must_use]
    pub fn refusal_reason(&self) -> Option<RefusalReason> {
        self.refusal
    }
}

/// What [`IdleQueue::take`] found.
#[must_use = "the outcome carries clients the caller must shut down"]
#[non_exhaustive]
pub struct TakeOutcome<T> {
    /// The oldest live entry, when one exists.
    pub entry: Option<IdleEntry<T>>,
    /// Entries that aged out; the caller must shut them down. They are never
    /// returned in `entry`.
    pub expired: Vec<T>,
}

/// A bounded FIFO of idle clients with an age ceiling.
///
/// Entries are pushed in release order, so `since` is non-decreasing and the
/// expired entries always form a prefix. The queue is generic over the client
/// payload so its cap and expiry rules can be tested without spawning a
/// process.
pub struct IdleQueue<T> {
    entries: VecDeque<IdleEntry<T>>,
    max_idle: usize,
    max_age: Duration,
    /// Per-process reuse ceiling. `0` means unlimited: a process is never
    /// retired for its reuse count. This is the opposite of `max_idle`, where
    /// `0` disables pooling instead.
    max_reuses: u64,
}

impl<T> IdleQueue<T> {
    /// Create an empty queue bounded by an idle cap and an age ceiling.
    ///
    /// The reuse ceiling defaults to `0`, meaning unlimited; use
    /// [`IdleQueue::with_max_reuses`] to bound it. Note the opposing senses:
    /// `max_idle == 0` disables pooling, while `max_reuses == 0` means no
    /// process is ever retired for its reuse count.
    #[must_use]
    pub fn new(max_idle: usize, max_age: Duration) -> Self {
        IdleQueue {
            entries: VecDeque::new(),
            max_idle,
            max_age,
            max_reuses: 0,
        }
    }

    /// Return this queue with a per-process reuse ceiling. `0` means unlimited
    /// (never retired for its reuse count), unlike an idle cap of `0`, which
    /// disables pooling.
    #[must_use = "builder methods return a modified queue; chain or assign the result"]
    pub fn with_max_reuses(mut self, max_reuses: u64) -> Self {
        self.max_reuses = max_reuses;
        self
    }

    /// Number of entries currently parked in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the queue currently parks no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Offer `entry` to the queue at `now`.
    ///
    /// This is the single home of the refusal rule. An entry is first refused
    /// for [`RefusalReason::Configuration`] when pooling is disabled or it is
    /// already at or past the age ceiling, then for
    /// [`RefusalReason::Retirement`] when it has reached the reuse ceiling. An
    /// accepted entry has its [`IdleEntry::reuses`] count incremented and may
    /// still displace the oldest one to respect the idle cap. Every client the
    /// queue did not retain is returned for the caller to shut down.
    #[must_use = "the outcome carries clients the caller must shut down"]
    pub fn push(&mut self, mut entry: IdleEntry<T>, now: Instant) -> PushOutcome<T> {
        if self.max_idle == 0 || now.saturating_duration_since(entry.since) >= self.max_age {
            return PushOutcome {
                displaced: Vec::new(),
                refused: Some(entry),
                refusal: Some(RefusalReason::Configuration),
            };
        }
        if self.max_reuses != 0 && entry.reuses >= self.max_reuses {
            return PushOutcome {
                displaced: Vec::new(),
                refused: Some(entry),
                refusal: Some(RefusalReason::Retirement),
            };
        }
        entry.reuses = entry.reuses.saturating_add(1);
        if self.entries.len() >= self.max_idle {
            let displaced: Vec<T> = self
                .entries
                .pop_front()
                .map(|entry| entry.client)
                .into_iter()
                .collect();
            self.entries.push_back(entry);
            return PushOutcome {
                displaced,
                refused: None,
                refusal: None,
            };
        }
        self.entries.push_back(entry);
        PushOutcome {
            displaced: Vec::new(),
            refused: None,
            refusal: None,
        }
    }

    /// Take the oldest live entry at `now`, dropping (and returning) every
    /// aged-out entry so an expired process is never handed out.
    #[must_use = "the outcome carries the entry and every expired client"]
    pub fn take(&mut self, now: Instant) -> TakeOutcome<T> {
        let mut expired = Vec::new();
        while let Some(front) = self.entries.front() {
            if now.saturating_duration_since(front.since) < self.max_age {
                break;
            }
            if let Some(entry) = self.entries.pop_front() {
                expired.push(entry.client);
            }
        }
        TakeOutcome {
            entry: self.entries.pop_front(),
            expired,
        }
    }

    /// Remove every entry; the caller owns shutting them down.
    #[must_use = "the caller owns shutting down the drained entries"]
    pub fn drain(&mut self) -> Vec<IdleEntry<T>> {
        self.entries.drain(..).collect()
    }
}

/// Monotonic pool counters.
#[derive(Debug, Default)]
pub struct PoolStats {
    spawned: AtomicU64,
    reused: AtomicU64,
    reset_failures: AtomicU64,
    release_refused: AtomicU64,
    expired: AtomicU64,
    evicted: AtomicU64,
    shutdown: AtomicU64,
    retired: AtomicU64,
}

/// A point-in-time copy of [`PoolStats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PoolStatsSnapshot {
    /// Fresh processes spawned because no idle entry was usable.
    pub spawned: u64,
    /// Idle processes handed out after a successful probe reset.
    pub reused: u64,
    /// Probe or release resets that failed; each such process was killed.
    pub reset_failures: u64,
    /// Releases refused because a [`PoolLease::client`] clone was still alive
    /// when the lease was released. Each such process was killed, never pooled.
    pub release_refused: u64,
    /// Idle entries dropped because they were past `max_age`.
    pub expired: u64,
    /// Idle entries displaced because the [`PoolConfig::max_idle`] cap was
    /// reached. A release refused by configuration is not counted here.
    pub evicted: u64,
    /// Processes shut down without being pooled: a non-reset-capable release,
    /// a release the configuration refuses (pooling disabled or the age
    /// ceiling), or [`SidecarPool::shutdown`]. A retirement is counted
    /// separately in [`PoolStatsSnapshot::retired`].
    pub shutdown: u64,
    /// Processes shut down because they reached [`PoolConfig::max_reuses`]
    /// pooled returns. A retirement is not counted in
    /// [`PoolStatsSnapshot::shutdown`].
    pub retired: u64,
}

impl PoolStats {
    /// Read every counter as a point-in-time snapshot.
    #[must_use]
    pub fn snapshot(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            spawned: self.spawned.load(Ordering::Relaxed),
            reused: self.reused.load(Ordering::Relaxed),
            reset_failures: self.reset_failures.load(Ordering::Relaxed),
            release_refused: self.release_refused.load(Ordering::Relaxed),
            expired: self.expired.load(Ordering::Relaxed),
            evicted: self.evicted.load(Ordering::Relaxed),
            shutdown: self.shutdown.load(Ordering::Relaxed),
            retired: self.retired.load(Ordering::Relaxed),
        }
    }

    fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// A bounded pool of handshaken plugin processes.
///
/// Use it through an [`Arc`]: [`SidecarPool::acquire`] hands each lease a
/// handle back to the pool so [`PoolLease::release`] can return the process.
pub struct SidecarPool {
    config: SidecarConfig,
    queue: Mutex<IdleQueue<Arc<SidecarClient>>>,
    stats: PoolStats,
}

impl SidecarPool {
    /// Create an empty pool that will spawn processes from `config` and bound
    /// its idle set with `pool_config`.
    #[must_use]
    pub fn new(config: SidecarConfig, pool_config: PoolConfig) -> Self {
        SidecarPool {
            config,
            queue: Mutex::new(
                IdleQueue::new(pool_config.max_idle, pool_config.max_age)
                    .with_max_reuses(pool_config.max_reuses),
            ),
            stats: PoolStats::default(),
        }
    }

    /// Counters accumulated since the pool was created.
    #[must_use]
    pub fn stats(&self) -> PoolStatsSnapshot {
        self.stats.snapshot()
    }

    /// Lease a process from the pool, spawning one when no idle entry is
    /// usable.
    ///
    /// # Errors
    ///
    /// Returns the transport or protocol error from a fresh spawn when the
    /// pool is empty and the process cannot be started or handshaken. A failed
    /// probe reset is not returned: that process is killed, counted in
    /// [`PoolStatsSnapshot::reset_failures`], and replaced by a fresh spawn.
    pub async fn acquire(self: &Arc<Self>) -> Result<PoolLease, ParlevoxError> {
        let (entry, expired) = {
            let mut queue = self.queue.lock().await;
            let outcome = queue.take(Instant::now());
            (outcome.entry, outcome.expired)
        };
        for client in expired {
            PoolStats::bump(&self.stats.expired);
            client.kill().await;
        }

        if let Some(entry) = entry {
            let reuses = entry.reuses;
            let client = entry.client;
            match client.reset().await {
                Ok(()) => {
                    PoolStats::bump(&self.stats.reused);
                    return Ok(PoolLease::new(client, Arc::clone(self), reuses));
                }
                Err(err) => {
                    tracing::warn!(
                        target: "parlevox_sidecar",
                        error = %err,
                        "pooled plugin failed its acquire probe; replacing it"
                    );
                    PoolStats::bump(&self.stats.reset_failures);
                    client.kill().await;
                }
            }
        }

        let client = Arc::new(SidecarClient::spawn(self.config.clone()).await?);
        PoolStats::bump(&self.stats.spawned);
        Ok(PoolLease::new(client, Arc::clone(self), 0))
    }

    /// Shut down every idle process. Leases still outstanding are not touched;
    /// dropping one kills its process.
    pub async fn shutdown(&self) {
        let drained = {
            let mut queue = self.queue.lock().await;
            queue.drain()
        };
        for entry in drained {
            PoolStats::bump(&self.stats.shutdown);
            if entry.client.shutdown().await.is_err() {
                entry.client.kill().await;
            }
        }
    }

    /// Return one lease's process to the pool, or shut it down.
    ///
    /// `reuses` is how many pooled returns the process has already served, so
    /// the queue alone decides whether this return retires it.
    async fn release_client(
        &self,
        client: Arc<SidecarClient>,
        reuses: u64,
    ) -> Result<(), ParlevoxError> {
        if !client.supports_reset() {
            // Today's lifecycle for a plugin built before the capability:
            // one process per session.
            PoolStats::bump(&self.stats.shutdown);
            return client.shutdown().await;
        }
        match client.reset().await {
            Ok(()) => {
                let now = Instant::now();
                let entry = IdleEntry::new(client, now).with_reuses(reuses);
                let outcome = {
                    let mut queue = self.queue.lock().await;
                    queue.push(entry, now)
                };
                // Capture the typed reason before moving `outcome.refused`.
                let reason = outcome.refusal_reason();
                for victim in outcome.displaced {
                    PoolStats::bump(&self.stats.evicted);
                    if victim.shutdown().await.is_err() {
                        victim.kill().await;
                    }
                }
                if let Some(entry) = outcome.refused {
                    // The queue refused this release, so the process is shut
                    // down without ever being pooled. The reason selects the
                    // counter: reaching the reuse ceiling is a retirement,
                    // everything else keeps the historical `shutdown` fate. A
                    // missing reason cannot occur while `refused` is `Some`,
                    // but it is mapped to `shutdown` rather than panicking.
                    match reason {
                        Some(RefusalReason::Retirement) => {
                            PoolStats::bump(&self.stats.retired);
                        }
                        Some(RefusalReason::Configuration) | None => {
                            PoolStats::bump(&self.stats.shutdown);
                        }
                    }
                    if entry.client.shutdown().await.is_err() {
                        entry.client.kill().await;
                    }
                }
                Ok(())
            }
            Err(err) => {
                PoolStats::bump(&self.stats.reset_failures);
                client.kill().await;
                Err(err)
            }
        }
    }
}

/// One session's exclusive use of a pooled process.
///
/// Call [`PoolLease::release`] to return it; dropping without releasing is a
/// best-effort kill so a forgotten lease cannot leak a process.
pub struct PoolLease {
    client: Arc<SidecarClient>,
    pool: Arc<SidecarPool>,
    /// Pooled returns this process has already served, carried from its idle
    /// entry (or `0` for a fresh spawn) so a release can be classified.
    reuses: u64,
    released: bool,
}

impl PoolLease {
    fn new(client: Arc<SidecarClient>, pool: Arc<SidecarPool>, reuses: u64) -> Self {
        PoolLease {
            client,
            pool,
            reuses,
            released: false,
        }
    }

    /// The process this lease owns. Adapters may clone the handle; release the
    /// lease only once they have all been dropped.
    #[must_use]
    pub fn client(&self) -> &Arc<SidecarClient> {
        &self.client
    }

    /// Reset and return the process to the pool.
    ///
    /// The release runs on a detached task, so this future is cancellation-safe:
    /// if it is dropped before it completes, the reset, pooling, or kill still
    /// runs to completion in the background and the process is reset, pooled,
    /// or killed and counted exactly as an uninterrupted release would. The
    /// caller then observes no `Result`, but no process is leaked.
    ///
    /// A lease is exclusive: every clone returned by [`PoolLease::client`] must
    /// be dropped before this is called. If any live clone remains, the release
    /// is refused rather than pooling a process the caller may still be using.
    /// The refused process is killed, never pooled, and counted in
    /// [`PoolStatsSnapshot::release_refused`]; `Drop` cannot kill it a second
    /// time because the lease is already marked released.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when a [`PoolLease::client`] clone is
    /// still alive at release time, naming the outstanding handles and the
    /// `client()` fix; that process is killed and counted in
    /// [`PoolStatsSnapshot::release_refused`]. Returns a protocol-class error
    /// when a request route is still open, or a transport-class error when the
    /// process is dead or does not answer. A failed release kills and counts the
    /// process rather than pooling it. Returns a transport-class error when
    /// called outside a Tokio runtime or when the detached release task cannot
    /// be joined.
    pub async fn release(mut self) -> Result<(), ParlevoxError> {
        // Resolve the runtime before touching the client: a caller outside
        // Tokio still receives the transport error it always has, rather than a
        // refusal whose kill could not be spawned.
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| ParlevoxError::Transport("release requires a Tokio runtime".into()))?;

        // Check the live handle count before cloning. This method's own clone
        // for the detached task is made only after the decision below, so it
        // cannot inflate the count.
        let live = Arc::strong_count(&self.client);
        if live > 1 {
            // Refuse the release and kill the process instead of pooling it:
            // the caller still holds a handle into a process it handed back, so
            // the pool cannot trust the process's state. Mark the lease released
            // first so `Drop` cannot spawn a second, uncounted kill.
            self.released = true;
            PoolStats::bump(&self.pool.stats.release_refused);
            let client = Arc::clone(&self.client);
            runtime.spawn(async move {
                client.kill().await;
            });
            return Err(ParlevoxError::Protocol(format!(
                "lease released while {} handle(s) to the sidecar client were still \
                 alive; drop every clone returned by `client()` before calling `release()`",
                live - 1
            )));
        }

        // Take ownership before spawning: the task must outlive this future so
        // dropping the future detaches the release instead of aborting it.
        let pool = Arc::clone(&self.pool);
        let client = Arc::clone(&self.client);
        let reuses = self.reuses;
        let detached = runtime.spawn(async move { pool.release_client(client, reuses).await });
        // Only mark the lease handed over once the task exists; if anything
        // above failed, `Drop` still owns the process and kills it.
        self.released = true;
        match detached.await {
            Ok(result) => result,
            Err(err) => Err(ParlevoxError::Transport(format!(
                "pooled release task failed: {err}"
            ))),
        }
    }
}

impl Drop for PoolLease {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        let client = Arc::clone(&self.client);
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                client.kill().await;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(client: u32, since: Instant) -> IdleEntry<u32> {
        IdleEntry::new(client, since)
    }

    #[test]
    fn push_never_exceeds_the_idle_cap_and_evicts_at_most_one() {
        let mut queue = IdleQueue::new(2, Duration::from_secs(60));
        let base = Instant::now();
        let mut largest = 0usize;

        for index in 0..32u32 {
            let now = base + Duration::from_millis(u64::from(index));
            let outcome = queue.push(entry(index, now), now);
            assert!(
                outcome.displaced.len() <= 1,
                "one push displaced {} entries",
                outcome.displaced.len()
            );
            assert!(
                outcome.refused.is_none(),
                "a live entry must not be refused"
            );
            largest = largest.max(queue.len());
        }

        assert!(largest <= 2, "idle length reached {largest}");
        assert_eq!(queue.len(), 2);
        // The survivors are the two most recently pushed.
        let first = queue.take(base + Duration::from_secs(1));
        let second = queue.take(base + Duration::from_secs(1));
        assert_eq!(first.entry.map(|e| e.client), Some(30));
        assert_eq!(second.entry.map(|e| e.client), Some(31));
    }

    #[test]
    fn push_evicts_the_oldest_entry_when_the_cap_is_reached() {
        let mut queue = IdleQueue::new(1, Duration::from_secs(60));
        let base = Instant::now();
        assert!(queue.push(entry(1, base), base).displaced.is_empty());
        let outcome = queue.push(entry(2, base), base);
        assert_eq!(outcome.displaced, vec![1]);
        assert!(outcome.refused.is_none());
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn take_never_hands_out_an_expired_entry() {
        let mut queue = IdleQueue::new(4, Duration::from_millis(10));
        let base = Instant::now();
        assert!(queue.push(entry(1, base), base).refused.is_none());

        let outcome = queue.take(base + Duration::from_millis(11));
        assert!(outcome.entry.is_none());
        assert_eq!(outcome.expired, vec![1]);
        assert!(queue.is_empty());
    }

    #[test]
    fn take_returns_live_entries_in_release_order() {
        let mut queue = IdleQueue::new(4, Duration::from_secs(60));
        let base = Instant::now();
        assert!(queue.push(entry(1, base), base).refused.is_none());
        assert!(queue
            .push(entry(2, base + Duration::from_millis(1)), base)
            .refused
            .is_none());

        let now = base + Duration::from_millis(2);
        assert_eq!(queue.take(now).entry.map(|e| e.client), Some(1));
        assert_eq!(queue.take(now).entry.map(|e| e.client), Some(2));
        assert!(queue.take(now).entry.is_none());
    }

    #[test]
    fn push_shuts_down_an_already_expired_entry_and_a_disabled_pool() {
        let base = Instant::now();
        let mut zero_age = IdleQueue::new(4, Duration::from_millis(1));
        let refused = zero_age.push(entry(7, base), base + Duration::from_millis(2));
        assert!(refused.displaced.is_empty());
        assert_eq!(refused.refused.map(|entry| entry.client), Some(7));

        let mut disabled = IdleQueue::new(0, Duration::from_secs(60));
        let refused = disabled.push(entry(8, base), base);
        assert!(refused.displaced.is_empty());
        assert_eq!(refused.refused.map(|entry| entry.client), Some(8));
        assert!(disabled.is_empty());
    }

    #[test]
    fn push_counts_each_accepted_return_as_one_reuse() {
        let mut queue = IdleQueue::new(4, Duration::from_secs(60));
        let base = Instant::now();
        assert!(queue.push(entry(1, base), base).refused.is_none());

        let taken = queue.take(base).entry.expect("the entry is pooled");
        assert_eq!(taken.reuses, 1, "an accepted return counts as one reuse");

        assert!(queue.push(taken, base).refused.is_none());
        let taken = queue.take(base).entry.expect("the entry is pooled");
        assert_eq!(taken.reuses, 2, "each accepted return increments the count");
    }

    #[test]
    fn push_retires_an_entry_at_the_reuse_ceiling() {
        let mut queue = IdleQueue::new(4, Duration::from_secs(60)).with_max_reuses(2);
        let base = Instant::now();

        // Two accepted returns fit under the ceiling and are counted.
        assert!(queue.push(entry(1, base), base).refused.is_none());
        let taken = queue.take(base).entry.expect("the entry is pooled");
        assert_eq!(taken.reuses, 1);
        assert!(queue.push(taken, base).refused.is_none());
        let taken = queue.take(base).entry.expect("the entry is pooled");
        assert_eq!(taken.reuses, 2);

        // The next return reaches the ceiling and is refused as a retirement.
        let outcome = queue.push(taken, base);
        assert_eq!(outcome.refusal_reason(), Some(RefusalReason::Retirement));
        assert!(outcome.displaced.is_empty());
        assert_eq!(outcome.refused.map(|entry| entry.client), Some(1));
        assert!(queue.is_empty());
    }

    #[test]
    fn push_reports_a_configuration_refusal_reason() {
        let base = Instant::now();
        let mut disabled = IdleQueue::new(0, Duration::from_secs(60));
        let refused = disabled.push(entry(1, base), base);
        assert_eq!(refused.refusal_reason(), Some(RefusalReason::Configuration));
        assert_eq!(refused.refused.map(|entry| entry.client), Some(1));
    }

    #[test]
    fn drain_removes_every_entry() {
        let mut queue = IdleQueue::new(4, Duration::from_secs(60));
        let base = Instant::now();
        assert!(queue.push(entry(1, base), base).refused.is_none());
        assert!(queue.push(entry(2, base), base).refused.is_none());
        assert_eq!(queue.drain().len(), 2);
        assert!(queue.is_empty());
    }
}
