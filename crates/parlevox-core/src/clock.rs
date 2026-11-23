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

    /// Convert seconds to a whole millisecond duration, rounding to nearest.
    ///
    /// Returns `None` for `NaN` and negative values, which have no meaningful
    /// monotonic-duration interpretation. Values too large for `u64`
    /// milliseconds (about 584 million years) saturate to `u64::MAX`; infinity
    /// saturates the same way.
    pub fn try_from_secs_f64(secs: f64) -> Option<Self> {
        if secs.is_nan() || secs < 0.0 {
            return None;
        }
        // Rust saturates float-to-integer casts, so +inf and huge finite
        // values both land on `u64::MAX` instead of wrapping.
        Some(DurationMs((secs * 1000.0).round() as u64))
    }

    /// Convert seconds to a whole millisecond duration, rounding to nearest.
    ///
    /// `NaN` and negative inputs are clamped to [`DurationMs::ZERO`]; values
    /// that overflow `u64` milliseconds saturate to `u64::MAX`.
    pub fn from_secs_f64(secs: f64) -> Self {
        Self::try_from_secs_f64(secs).unwrap_or(DurationMs::ZERO)
    }

    pub fn as_millis(self) -> u64 {
        self.0
    }

    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1000.0
    }

    #[must_use]
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
        MonoClock {
            epoch: Instant::now(),
        }
    }

    /// Milliseconds elapsed on this clock since its epoch.
    ///
    /// `Instant::elapsed` measures u128 milliseconds; a saturating conversion
    /// cannot lose a realistic duration (u64 ms is about 584 million years) and
    /// avoids an unchecked narrowing cast.
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    pub fn now(&self) -> Timestamp {
        Timestamp(DurationMs::from_millis(self.elapsed_ms()))
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

    #[test]
    fn secs_conversion_rejects_nan_and_negative() {
        assert_eq!(DurationMs::try_from_secs_f64(f64::NAN), None);
        assert_eq!(DurationMs::try_from_secs_f64(-0.001), None);
        assert_eq!(DurationMs::try_from_secs_f64(f64::NEG_INFINITY), None);
        assert_eq!(DurationMs::from_secs_f64(f64::NAN), DurationMs::ZERO);
        assert_eq!(DurationMs::from_secs_f64(-0.001), DurationMs::ZERO);
    }

    #[test]
    fn secs_conversion_saturates_at_the_u64_ceiling() {
        let saturated = DurationMs::from_millis(u64::MAX);
        assert_eq!(
            DurationMs::try_from_secs_f64(f64::INFINITY),
            Some(saturated)
        );
        assert_eq!(DurationMs::try_from_secs_f64(f64::MAX), Some(saturated));
        assert_eq!(DurationMs::try_from_secs_f64(1e17), Some(saturated));
        assert_eq!(DurationMs::from_secs_f64(f64::INFINITY), saturated);
        assert_eq!(DurationMs::from_secs_f64(1e17), saturated);
    }
}
