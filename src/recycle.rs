// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! How long a camera frame's DMA buffer stays intact after capture.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

/// Window for a stream whose descriptors do not identify its buffers (one
/// fd for every frame): 4 buffers at 30 FPS.
pub const DEFAULT_RECYCLE_WINDOW_NS: u64 = 100_000_000;

/// Frames of history used to count the pool and estimate the period. Covers
/// every buffer of the largest pool the camera service allows (32) twice.
const POOL_HISTORY: usize = 64;

/// Frames observed before the learned window replaces the default.
const MIN_FRAMES: usize = POOL_HISTORY / 2;

/// Consecutive stamps further apart than this are a gap, not a frame period.
const MAX_PERIOD_NS: u64 = 1_000_000_000;

/// Lower bound on the safety margin: the frame is converted after the age
/// check, and the ISP begins rewriting a buffer shortly before the stamp
/// arithmetic says it does.
const MIN_MARGIN_NS: u64 = 5_000_000;

/// Minimum time between warnings about skipped frames.
const SKIP_LOG_INTERVAL: Duration = Duration::from_secs(10);

/// How long after its stamp a `CameraFrame`'s DMA buffer stays intact.
///
/// The camera re-queues each capture buffer as soon as the frame is
/// published, behind the buffers already queued, so with `N` buffers at
/// frame period `T` the driver starts overwriting it about `(N - 1) T` after
/// the frame's end-of-frame stamp. `N` is the number of distinct DMA-BUF
/// file descriptors the publishing process cycles through, and `T` the
/// median stamp interval. Every received frame must be observed, including
/// those the model does not convert, or a model slower than the camera would
/// see only some of the buffers. A camera restart (new pid) starts learning
/// again.
#[derive(Debug, Default)]
pub struct RecycleWindow {
    pid: Option<u32>,
    handles: VecDeque<i64>,
    periods: VecDeque<u64>,
    last_stamp: Option<u64>,
    learned: Option<CameraPool>,
}

/// Capture pool as inferred from the frame stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CameraPool {
    pub buffers: usize,
    pub period_ns: u64,
}

impl CameraPool {
    /// Age after capture beyond which a frame may be partly overwritten,
    /// less a margin of a quarter period (at least `MIN_MARGIN_NS`).
    pub fn window_ns(&self) -> u64 {
        let span = (self.buffers.saturating_sub(1) as u64).saturating_mul(self.period_ns);
        let margin = (self.period_ns / 4).max(MIN_MARGIN_NS);
        span.saturating_sub(margin)
    }
}

impl RecycleWindow {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a frame from process `pid` in buffer `handle`, stamped
    /// `stamp_ns`. Returns the pool when the learned estimate changes.
    pub fn observe(&mut self, pid: u32, handle: i64, stamp_ns: u64) -> Option<CameraPool> {
        if self.pid != Some(pid) {
            *self = Self {
                pid: Some(pid),
                ..Self::default()
            };
        }
        if let Some(last) = self.last_stamp {
            let period = stamp_ns.saturating_sub(last);
            if period > 0 && period <= MAX_PERIOD_NS {
                push_bounded(&mut self.periods, period);
            }
        }
        self.last_stamp = Some(stamp_ns);
        push_bounded(&mut self.handles, handle);

        let pool = self.estimate()?;
        let changed = self.learned.is_none_or(|old| {
            old.buffers != pool.buffers
                || old.period_ns.abs_diff(pool.period_ns) > pool.period_ns / 10
        });
        if changed {
            self.learned = Some(pool);
            return Some(pool);
        }
        None
    }

    fn estimate(&self) -> Option<CameraPool> {
        if self.handles.len() < MIN_FRAMES || self.periods.len() < MIN_FRAMES / 2 {
            return None;
        }
        let mut periods: Vec<u64> = self.periods.iter().copied().collect();
        periods.sort_unstable();
        let pool = CameraPool {
            buffers: self.distinct_handles(),
            period_ns: periods[periods.len() / 2],
        };
        (pool.buffers >= 2).then_some(pool)
    }

    fn distinct_handles(&self) -> usize {
        let mut handles: Vec<i64> = self.handles.iter().copied().collect();
        handles.sort_unstable();
        handles.dedup();
        handles.len()
    }

    /// Current window, once learned. While learning, the buffers and the
    /// shortest interval seen so far, which can only understate the true
    /// window, and zero (every frame is too old) until two buffers have been
    /// seen. A stream that shows one descriptor for `MIN_FRAMES` frames gets
    /// `DEFAULT_RECYCLE_WINDOW_NS`.
    pub fn window_ns(&self) -> u64 {
        if let Some(pool) = self.learned {
            return pool.window_ns();
        }
        let buffers = self.distinct_handles();
        if buffers < 2 && self.handles.len() >= MIN_FRAMES {
            return DEFAULT_RECYCLE_WINDOW_NS;
        }
        match self.periods.iter().copied().min() {
            Some(period_ns) if buffers >= 2 => CameraPool { buffers, period_ns }.window_ns(),
            _ => 0,
        }
    }

    /// True when a frame stamped `stamp_ns` is too old at `now_ns` to read.
    pub fn is_stale(&self, stamp_ns: u64, now_ns: u64) -> bool {
        now_ns.saturating_sub(stamp_ns) > self.window_ns()
    }
}

fn push_bounded<T>(q: &mut VecDeque<T>, v: T) {
    if q.len() == POOL_HISTORY {
        q.pop_front();
    }
    q.push_back(v);
}

/// Counts skipped frames so the warning is logged at most once per
/// `SKIP_LOG_INTERVAL`.
#[derive(Debug, Default)]
pub struct SkipLog {
    skipped: u64,
    last_log: Option<Instant>,
}

impl SkipLog {
    /// Count one skipped frame. Returns the number skipped since the last
    /// warning when one is due now.
    pub fn record(&mut self, now: Instant) -> Option<u64> {
        self.skipped += 1;
        let due = self
            .last_log
            .is_none_or(|last| now.duration_since(last) >= SKIP_LOG_INTERVAL);
        if !due {
            return None;
        }
        self.last_log = Some(now);
        Some(std::mem::take(&mut self.skipped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;
    const PERIOD: u64 = 33_333_333;

    fn feed(w: &mut RecycleWindow, pid: u32, buffers: i64, frames: u64) -> Option<CameraPool> {
        let mut last = None;
        for k in 0..frames {
            if let Some(p) = w.observe(pid, 40 + (k as i64 % buffers), 1_000 * MS + k * PERIOD) {
                last = Some(p);
            }
        }
        last
    }

    #[test]
    fn partial_window_while_learning_understates_the_pool() {
        let mut w = RecycleWindow::new();
        w.observe(1, 40, 1_000 * MS);
        assert_eq!(w.window_ns(), 0, "one buffer seen: every frame is too old");
        assert!(w.is_stale(1_000 * MS, 1_000 * MS + 1));
        w.observe(1, 41, 1_000 * MS + PERIOD);
        assert_eq!(w.window_ns(), PERIOD - PERIOD / 4);
        feed(&mut w, 1, 4, 4);
        assert_eq!(w.window_ns(), 3 * PERIOD - PERIOD / 4);
        assert!(w.learned.is_none());
    }

    #[test]
    fn partial_window_uses_the_shortest_interval() {
        let mut w = RecycleWindow::new();
        w.observe(1, 40, 0);
        w.observe(1, 41, 3 * PERIOD);
        w.observe(1, 42, 4 * PERIOD);
        assert_eq!(w.window_ns(), 2 * PERIOD - PERIOD / 4);
    }

    #[test]
    fn learns_pool_depth_and_period() {
        let mut w = RecycleWindow::new();
        let pool = feed(&mut w, 1, 6, 64).expect("learned");
        assert_eq!(pool.buffers, 6);
        assert_eq!(pool.period_ns, PERIOD);
        assert_eq!(w.window_ns(), 5 * PERIOD - PERIOD / 4);
    }

    #[test]
    fn four_buffers_at_sixty_fps_is_under_fifty_ms() {
        let pool = CameraPool {
            buffers: 4,
            period_ns: 16_666_667,
        };
        assert_eq!(pool.window_ns(), 3 * 16_666_667 - 5_000_000);
    }

    #[test]
    fn reports_a_change_once() {
        let mut w = RecycleWindow::new();
        assert!(feed(&mut w, 1, 4, 64).is_some());
        assert!(feed(&mut w, 1, 4, 64).is_none());
    }

    #[test]
    fn camera_restart_relearns() {
        let mut w = RecycleWindow::new();
        feed(&mut w, 1, 4, 64);
        assert!(w.observe(2, 99, 5_000 * MS).is_none());
        assert_eq!(w.learned, None);
        assert_eq!(w.window_ns(), 0);
        assert_eq!(feed(&mut w, 2, 8, 64).map(|p| p.buffers), Some(8));
    }

    #[test]
    fn dropped_frames_do_not_move_the_median_period() {
        let mut w = RecycleWindow::new();
        let mut stamp = 1_000 * MS;
        for k in 0..64u64 {
            stamp += if k % 5 == 4 { 2 * PERIOD } else { PERIOD };
            w.observe(1, 40 + (k as i64 % 4), stamp);
        }
        assert_eq!(w.learned.map(|p| p.period_ns), Some(PERIOD));
    }

    #[test]
    fn gaps_and_steps_are_not_periods() {
        let mut w = RecycleWindow::new();
        feed(&mut w, 1, 4, 64);
        w.observe(1, 40, 1);
        w.observe(1, 41, 10_000_000 * MS);
        assert_eq!(w.learned.map(|p| p.period_ns), Some(PERIOD));
    }

    #[test]
    fn single_descriptor_stream_gets_default_once_seen_long_enough() {
        let mut w = RecycleWindow::new();
        assert!(feed(&mut w, 1, 1, (MIN_FRAMES - 1) as u64).is_none());
        assert_eq!(w.window_ns(), 0);
        assert!(feed(&mut w, 1, 1, 1).is_none());
        assert_eq!(w.window_ns(), DEFAULT_RECYCLE_WINDOW_NS);
    }

    #[test]
    fn skip_log_is_rate_limited_and_counts() {
        let mut log = SkipLog::default();
        let t0 = Instant::now();
        assert_eq!(log.record(t0), Some(1));
        assert_eq!(log.record(t0 + Duration::from_secs(1)), None);
        assert_eq!(log.record(t0 + Duration::from_secs(2)), None);
        assert_eq!(log.record(t0 + SKIP_LOG_INTERVAL), Some(3));
    }
}
