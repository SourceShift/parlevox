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
pub struct PoolConfig {
    /// Maximum number of idle processes kept alive. `0` disables pooling, so
    /// every release shuts its process down.
    pub max_idle: usize,
    /// Maximum time a process may sit idle. Age is checked at lease time: a
    /// process older than this is shut down rather than handed out.
    pub max_age: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            max_idle: 2,
            max_age: Duration::from_secs(60),
        }
    }
}

impl PoolConfig {
    /// Build a configuration with an explicit idle cap and age ceiling.
    #[must_use]
    pub fn new(max_idle: usize, max_age: Duration) -> Self {
        PoolConfig { max_idle, max_age }
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
}

/// One idle client plus the instant it became idle.
///
/// `since` is measured from release, not spawn: the age ceiling is about how
/// long a handshaken process has been parked, not how old the process is.
#[derive(Debug)]
pub struct IdleEntry<T> {
    /// The idle client itself.
    pub client: T,
    /// The instant the client became idle, used to enforce the age ceiling.
    pub since: Instant,
}

impl<T> IdleEntry<T> {
    /// Pair a client with the instant it became idle.
    #[must_use]
    pub fn new(client: T, since: Instant) -> Self {
        IdleEntry { client, since }
    }
}

/// What [`IdleQueue::push`] did with an offered entry.
///
/// Both fields carry clients the caller must shut down; neither is retained by
/// the queue.
#[derive(Debug)]
#[must_use = "the outcome carries clients the caller must shut down"]
pub struct PushOutcome<T> {
    /// Entries displaced to stay within the idle cap. At most one.
    pub displaced: Vec<T>,
    /// The offered entry, returned unchanged when the queue refuses it because
    /// pooling is disabled or it is already past the age ceiling.
    pub refused: Option<IdleEntry<T>>,
}

/// What [`IdleQueue::take`] found.
#[must_use = "the outcome carries clients the caller must shut down"]
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
}

impl<T> IdleQueue<T> {
    /// Create an empty queue bounded by an idle cap and an age ceiling.
    #[must_use]
    pub fn new(max_idle: usize, max_age: Duration) -> Self {
        IdleQueue {
            entries: VecDeque::new(),
            max_idle,
            max_age,
        }
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
    /// This is the single home of the refusal rule: an entry is refused when
    /// pooling is disabled or it is already at or past the age ceiling. An
    /// accepted entry may still displace the oldest one to respect the idle
    /// cap. Every client the queue did not retain is returned for the caller to
    /// shut down.
    #[must_use = "the outcome carries clients the caller must shut down"]
    pub fn push(&mut self, entry: IdleEntry<T>, now: Instant) -> PushOutcome<T> {
        if self.max_idle == 0 || now.saturating_duration_since(entry.since) >= self.max_age {
            return PushOutcome {
                displaced: Vec::new(),
                refused: Some(entry),
            };
        }
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
            };
        }
        self.entries.push_back(entry);
        PushOutcome {
            displaced: Vec::new(),
            refused: None,
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
    expired: AtomicU64,
    evicted: AtomicU64,
    shutdown: AtomicU64,
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
    /// Idle entries dropped because they were past `max_age`.
    pub expired: u64,
    /// Idle entries displaced because the [`PoolConfig::max_idle`] cap was
    /// reached. A release refused by configuration is not counted here.
    pub evicted: u64,
    /// Processes shut down without being pooled: a non-reset-capable release,
    /// a release the configuration refuses (an idle cap or age ceiling of
    /// zero), or [`SidecarPool::shutdown`].
    pub shutdown: u64,
}

impl PoolStats {
    /// Read every counter as a point-in-time snapshot.
    #[must_use]
    pub fn snapshot(&self) -> PoolStatsSnapshot {
        PoolStatsSnapshot {
            spawned: self.spawned.load(Ordering::Relaxed),
            reused: self.reused.load(Ordering::Relaxed),
            reset_failures: self.reset_failures.load(Ordering::Relaxed),
            expired: self.expired.load(Ordering::Relaxed),
            evicted: self.evicted.load(Ordering::Relaxed),
            shutdown: self.shutdown.load(Ordering::Relaxed),
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
            queue: Mutex::new(IdleQueue::new(pool_config.max_idle, pool_config.max_age)),
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
            let client = entry.client;
            match client.reset().await {
                Ok(()) => {
                    PoolStats::bump(&self.stats.reused);
                    return Ok(PoolLease::new(client, Arc::clone(self)));
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
        Ok(PoolLease::new(client, Arc::clone(self)))
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
    async fn release_client(&self, client: Arc<SidecarClient>) -> Result<(), ParlevoxError> {
        if !client.supports_reset() {
            // Today's lifecycle for a plugin built before the capability:
            // one process per session.
            PoolStats::bump(&self.stats.shutdown);
            return client.shutdown().await;
        }
        match client.reset().await {
            Ok(()) => {
                let now = Instant::now();
                let outcome = {
                    let mut queue = self.queue.lock().await;
                    queue.push(IdleEntry::new(client, now), now)
                };
                for victim in outcome.displaced {
                    PoolStats::bump(&self.stats.evicted);
                    if victim.shutdown().await.is_err() {
                        victim.kill().await;
                    }
                }
                if let Some(entry) = outcome.refused {
                    // The configuration refuses this release, so the process is
                    // shut down without ever being pooled. That is the
                    // documented meaning of `shutdown`, not a cap displacement.
                    PoolStats::bump(&self.stats.shutdown);
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
    released: bool,
}

impl PoolLease {
    fn new(client: Arc<SidecarClient>, pool: Arc<SidecarPool>) -> Self {
        PoolLease {
            client,
            pool,
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
    /// # Errors
    ///
    /// Returns a protocol-class error when a request route is still open, or a
    /// transport-class error when the process is dead or does not answer. A
    /// failed release kills and counts the process rather than pooling it.
    /// Returns a transport-class error when called outside a Tokio runtime or
    /// when the detached release task cannot be joined.
    pub async fn release(mut self) -> Result<(), ParlevoxError> {
        // Take ownership before spawning: the task must outlive this future so
        // dropping the future detaches the release instead of aborting it.
        let pool = Arc::clone(&self.pool);
        let client = Arc::clone(&self.client);
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| ParlevoxError::Transport("release requires a Tokio runtime".into()))?;
        let detached = runtime.spawn(async move { pool.release_client(client).await });
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
    fn drain_removes_every_entry() {
        let mut queue = IdleQueue::new(4, Duration::from_secs(60));
        let base = Instant::now();
        assert!(queue.push(entry(1, base), base).refused.is_none());
        assert!(queue.push(entry(2, base), base).refused.is_none());
        assert_eq!(queue.drain().len(), 2);
        assert!(queue.is_empty());
    }
}
