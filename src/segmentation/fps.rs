use std::{
    fmt,
    time::{Duration, SystemTime},
};

use log::debug;

const NSEC_PER_SEC: i64 = 1_000_000_000;

/// A simple struct to calculate the average FPS over the last N frames.  To
/// use, call the `update` method once per frame which returns the average FPS
/// over the last N frames.
pub struct Fps<const N: usize> {
    previous: SystemTime,
    history: [f32; N],
    index: usize,
}

impl<const N: usize> Default for Fps<N> {
    fn default() -> Self {
        Fps {
            previous: SystemTime::now(),
            history: [0.0; N],
            index: 0,
        }
    }
}

impl<const N: usize> fmt::Display for Fps<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "now: {:?} previous: {:?} history: {:?}",
            SystemTime::now(),
            self.previous,
            self.history
        )
    }
}

impl<const N: usize> Fps<N> {
    /// Update the FPS calculation and return the average FPS over the last N
    /// calls to this function.  Over the first N calls to this function the
    /// value will ramp up over N frames to the true average FPS.
    pub fn update(&mut self) -> f32 {
        let timestamp = SystemTime::now();
        let frame_time = timestamp
            .duration_since(self.previous)
            .unwrap_or(Duration::from_secs(0));
        self.previous = timestamp;
        self.history[self.index] = 1.0 / frame_time.as_secs_f32();
        self.index = (self.index + 1) % N;
        let fps = self.history.iter().sum::<f32>() / N as f32;
        if self.index == 0 {
            log::debug!(
                "FPS AVG: {:.2} MIN: {:.2} MAX: {:.2}",
                fps,
                self.history
                    .iter()
                    .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap(),
                self.history
                    .iter()
                    .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
            );
        }

        fps
    }
}
