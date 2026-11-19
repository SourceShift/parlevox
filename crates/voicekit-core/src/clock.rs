//! Monotonic time for everything that measures a duration.
//!
//! Wall-clock time (`SystemTime`) is deliberately absent here: it can jump
//! backwards on NTP correction, which turns endpointing delays and latency
//! metrics into nonsense. Wall-clock timestamps belong to the event log, not
//! to the control path.

use std::time::Instant;

/// Milliseconds since a clock epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DurationMs(u64);

impl DurationMs {
    pub const ZERO: DurationMs = DurationMs(0);

    pub fn from_millis(ms: u64) -> Self {
        DurationMs(ms)
    }

    pub fn from_secs_f64(secs: f64) -> Self {
        debug_assert!(secs.is_finite() && secs >= 0.0, "duration must be finite and non-negative");
        DurationMs((secs.max(0.0) * 1000.0).round() as u64)
    }

    pub fn as_millis(self) -> u64 {
        self.0
    }

    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1000.0
    }

    pub fn saturating_sub(self, other: DurationMs) -> DurationMs {
        DurationMs(self.0.saturating_sub(other.0))
    }
}

/// A point in time on the session's monotonic timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(DurationMs);

impl Timestamp {
    pub fn from_millis(ms: u64) -> Self {
        Timestamp(DurationMs::from_millis(ms))
    }

    pub fn since(self, earlier: Timestamp) -> DurationMs {
        self.0.saturating_sub(earlier.0)
    }
}

/// Session-scoped monotonic clock. Cheap to copy; all copies share an epoch.
#[derive(Debug, Clone, Copy)]
pub struct MonoClock {
    epoch: Instant,
}

impl Default for MonoClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonoClock {
    pub fn new() -> Self {
        MonoClock { epoch: Instant::now() }
    }

    pub fn now(&self) -> Timestamp {
        Timestamp(DurationMs::from_millis(
            self.epoch.elapsed().as_millis() as u64,
        ))
    }

    /// Measure a block with this clock. Useful in traces and tests.
    pub fn measure<T>(&self, f: impl FnOnce() -> T) -> (T, Timestamp) {
        let value = f();
        (value, self.now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subtraction_saturates_instead_of_wrapping() {
        let later = Timestamp::from_millis(10);
        let earlier = Timestamp::from_millis(50);
        assert_eq!(later.since(earlier), DurationMs::ZERO);
    }

    #[test]
    fn clock_advances_monotonically() {
        let clock = MonoClock::new();
        let a = clock.now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = clock.now();
        assert!(b >= a);
    }

    #[test]
    fn secs_conversion_rounds() {
        assert_eq!(DurationMs::from_secs_f64(0.0205).as_millis(), 21);
        assert_eq!(DurationMs::from_secs_f64(0.0), DurationMs::ZERO);
    }
}
