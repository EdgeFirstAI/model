// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

pub mod args;
pub mod buildmsgs;
pub mod fps;
pub mod letterbox;
pub mod masks;
pub mod model;
pub mod recycle;
pub mod runtime;

/// Newtype wrapper to bridge `edgefirst_tracker::DetectionBox` for
/// `edgefirst_hal::decoder::DetectBox`.
///
/// Owns the box by value: `ByteTrack` retains the most recent box per
/// tracklet, so a borrowed wrapper would dangle across frames.
#[derive(Clone, Copy, Debug)]
pub struct TrackerBox(pub edgefirst_hal::decoder::DetectBox);

impl edgefirst_tracker::DetectionBox for TrackerBox {
    fn bbox(&self) -> [f32; 4] {
        [
            self.0.bbox.xmin,
            self.0.bbox.ymin,
            self.0.bbox.xmax,
            self.0.bbox.ymax,
        ]
    }

    fn score(&self) -> f32 {
        self.0.score
    }

    fn label(&self) -> usize {
        self.0.label
    }
}

use crate::buildmsgs::*;
use args::{Args, LabelSetting};
use async_pidfd::PidFd;
use edgefirst_schemas::{self, builtin_interfaces::Time, edgefirst_msgs::CameraFrame};
use log::{error, info, trace, warn};
use nix::{
    sys::time::TimeValLike,
    time::{ClockId, clock_gettime},
};
use pidfd_getfd::{GetFdFlags, get_file_from_pidfd};
use recycle::RecycleWindow;
use std::{fs::File, os::fd::AsRawFd, time::Duration};
use tokio::sync::mpsc::{Receiver, error::TryRecvError};

use zenoh::{
    Session,
    bytes::{Encoding, ZBytes},
    handlers::FifoChannelHandler,
    pubsub::Subscriber,
    sample::Sample,
    time::{NTP64, Timestamp},
};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ModelTypeActual {
    pub segment_output_ind: Option<usize>,
    pub detection: bool,
    pub detection_with_mask: bool,
}

/// Camera frame received from Zenoh with the DMA-BUF fd imported into this process.
pub struct ResolvedCameraFrame {
    frame: CameraFrame<Vec<u8>>,
    plane_fd: i32,
    stride: u32,
    offset: u32,
    width: u32,
    height: u32,
    format: String,
    _fd_guard: File,
}

impl ResolvedCameraFrame {
    pub fn stamp(&self) -> Time {
        self.frame.stamp()
    }

    pub fn frame_id(&self) -> &str {
        self.frame.frame_id()
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn format(&self) -> &str {
        &self.format
    }

    pub fn fd(&self) -> i32 {
        self.plane_fd
    }

    pub fn stride(&self) -> u32 {
        self.stride
    }

    pub fn offset(&self) -> u32 {
        self.offset
    }
}

pub async fn heart_beat(
    session: Session,
    args: Args,
    sub_camera: Subscriber<FifoChannelHandler<Sample>>,
    mut rx: Receiver<bool>,
    stream_dims: (f64, f64),
) -> Subscriber<FifoChannelHandler<Sample>> {
    let model_path = args.model.clone();

    let status = format!("Loading Model: {}", model_path.to_string_lossy());

    loop {
        match rx.try_recv() {
            Ok(_) => return sub_camera,
            Err(TryRecvError::Disconnected) => return sub_camera,
            Err(_) => (),
        }
        heart_beat_loop(
            &session,
            &args,
            &sub_camera,
            stream_dims,
            &model_path,
            &status,
        )
        .await;
    }
}

async fn heart_beat_loop(
    session: &Session,
    args: &Args,
    sub_camera: &Subscriber<FifoChannelHandler<Sample>>,
    stream_dims: (f64, f64),
    model_path: &std::path::Path,
    status: &str,
) {
    let mut recycle = RecycleWindow::new();
    let Some(frame) = wait_for_camera_frame(sub_camera, Duration::from_millis(100), &mut recycle)
    else {
        return;
    };
    trace!("Received camera frame");

    if !args.mask_topic.is_empty() {
        let mask = build_segmentation_msg(frame.stamp(), None, 0, None);
        let msg = ZBytes::from(mask.into_cdr());
        let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

        match session
            .put(&args.mask_topic, msg)
            .encoding(enc)
            .timestamp(zenoh_timestamp(session, frame.stamp()))
            .await
        {
            Ok(_) => (),
            Err(e) => {
                error!("Error sending message on {}: {:?}", args.mask_topic, e)
            }
        }
    }

    if !args.detect_topic.is_empty() {
        let (msg, enc) = build_detect_msg_and_encode_(
            &[],
            &[],
            &[],
            frame.stamp(),
            frame.frame_id(),
            time_from_ns(0u32),
            time_from_ns(0u32),
            time_from_ns(0u32),
        );

        match session
            .put(&args.detect_topic, msg)
            .encoding(enc)
            .timestamp(zenoh_timestamp(session, frame.stamp()))
            .await
        {
            Ok(_) => (),
            Err(e) => {
                error!("Error sending message on {}: {:?}", args.detect_topic, e)
            }
        }
    }

    let model_info_msg = build_model_info_msg(frame.stamp(), None, model_path, false, false);
    let msg = ZBytes::from(model_info_msg.into_cdr());
    let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/ModelInfo");

    match session
        .put(&args.info_topic, msg)
        .encoding(enc)
        .timestamp(zenoh_timestamp(session, frame.stamp()))
        .await
    {
        Ok(_) => (),
        Err(e) => {
            error!("Error sending message on {}: {:?}", args.info_topic, e)
        }
    }

    if args.visualization {
        let (msg, enc) = build_image_annotations_msg_and_encode_(
            &[],
            &[],
            &[],
            frame.stamp(),
            stream_dims,
            status,
            LabelSetting::Index,
        );

        match session
            .put(&args.visual_topic, msg)
            .encoding(enc)
            .timestamp(zenoh_timestamp(session, frame.stamp()))
            .await
        {
            Ok(_) => trace!("Sent message on {}", args.visual_topic),
            Err(e) => {
                error!("Error sending message on {}: {:?}", args.visual_topic, e)
            }
        }
    }
}

/// Converts a message `Time` to nanoseconds since the Unix epoch. Pre-epoch
/// stamps (negative `sec`) clamp to zero.
pub fn time_to_ns(stamp: Time) -> u64 {
    if stamp.sec < 0 {
        return 0;
    }
    stamp.sec as u64 * 1_000_000_000 + stamp.nanosec as u64
}

/// Current `CLOCK_REALTIME` in nanoseconds since the Unix epoch, the clock
/// camera frame stamps are expressed in. Zero before the epoch.
pub fn realtime_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

/// Builds the Zenoh sample timestamp for a message carrying `stamp`, so the
/// sample timestamp and `header.stamp` denote the same instant (to NTP64
/// resolution, about 0.23 ns). Uses the session's ZenohId so the sample is
/// attributable to this producer.
pub fn zenoh_timestamp(session: &Session, stamp: Time) -> Timestamp {
    Timestamp::new(ntp64_from_time(stamp), session.zid().into())
}

fn ntp64_from_time(stamp: Time) -> NTP64 {
    NTP64::from(Duration::from_nanos(time_to_ns(stamp)))
}

/// Camera stamp timeline fed to the tracker. The tracker expires tracks by
/// comparing stamps, so a backward clock step would keep lost tracks alive
/// until the clock caught up again; callers reset the tracker instead.
#[derive(Debug, Default)]
pub struct StampTimeline {
    last: Option<u64>,
}

impl StampTimeline {
    /// Records `stamp_ns` and returns how far it stepped back from the
    /// previous stamp, or `None` when time did not move backward.
    pub fn observe(&mut self, stamp_ns: u64) -> Option<u64> {
        let step = self
            .last
            .filter(|&last| stamp_ns < last)
            .map(|last| last - stamp_ns);
        self.last = Some(stamp_ns);
        step
    }
}

pub fn get_curr_time() -> u64 {
    match clock_gettime(ClockId::CLOCK_MONOTONIC) {
        Ok(t) => t.num_nanoseconds() as u64,
        Err(e) => {
            error!("Could not get Monotonic clock time: {e:?}");
            0
        }
    }
}

/// Wait for the newest camera frame and import its DMA-BUF. Every queued
/// frame, not only the newest, is fed to `recycle` so the camera's buffer
/// pool can be learned.
pub fn wait_for_camera_frame(
    sub_camera: &Subscriber<FifoChannelHandler<Sample>>,
    timeout: Duration,
    recycle: &mut RecycleWindow,
) -> Option<ResolvedCameraFrame> {
    let mut newest = None;
    for sample in sub_camera.drain() {
        if let Some(frame) = decode_camera_frame(&sample, recycle) {
            newest = Some(frame);
        }
    }
    let frame = match newest {
        Some(v) => v,
        None => match sub_camera.recv_timeout(timeout) {
            Ok(Some(sample)) => decode_camera_frame(&sample, recycle)?,
            Ok(None) => {
                warn!(
                    "timeout receiving camera frame on {}",
                    sub_camera.key_expr()
                );
                return None;
            }
            Err(e) => {
                error!(
                    "error receiving camera frame on {}: {:?}",
                    sub_camera.key_expr(),
                    e
                );
                return None;
            }
        },
    };

    match resolve_camera_frame_fd(frame) {
        Ok(v) => Some(v),
        Err(e) => {
            error!("Failed to import camera DMA-BUF fd: {e:?}");
            None
        }
    }
}

/// Deserialize a camera sample and record its buffer in `recycle`.
fn decode_camera_frame(
    sample: &Sample,
    recycle: &mut RecycleWindow,
) -> Option<CameraFrame<Vec<u8>>> {
    let frame = match CameraFrame::from_cdr(sample.payload().to_bytes().to_vec()) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to deserialize CameraFrame: {e:?}");
            return None;
        }
    };
    let tensor = frame.tensor();
    if let Some(plane) = tensor.plane_at(0)
        && let Some(pool) = recycle.observe(tensor.pid(), plane.handle, time_to_ns(frame.stamp()))
    {
        info!(
            "{}: camera cycles {} buffers at {:.1} FPS; frames older than {:.0} ms are skipped",
            sample.key_expr(),
            pool.buffers,
            1e9 / pool.period_ns as f64,
            pool.window_ns() as f64 * 1e-6
        );
    }
    Some(frame)
}

fn camera_frame_invalid(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into())
}

fn camera_frame_u32(name: &str, value: u64) -> Result<u32, std::io::Error> {
    u32::try_from(value).map_err(|_| {
        camera_frame_invalid(format!(
            "CameraFrame tensor {name} {value} does not fit in u32"
        ))
    })
}

fn camera_frame_nonzero_u32(name: &str, value: u64) -> Result<u32, std::io::Error> {
    let dim = camera_frame_u32(name, value)?;
    if dim == 0 {
        return Err(camera_frame_invalid(format!(
            "CameraFrame tensor {name} is 0"
        )));
    }
    Ok(dim)
}

fn resolve_camera_frame_fd(
    frame: CameraFrame<Vec<u8>>,
) -> Result<ResolvedCameraFrame, std::io::Error> {
    let (pid, handle, stride, offset, width, height, format) = {
        let tensor = frame.tensor();
        let plane0 = tensor
            .plane_at(0)
            .ok_or_else(|| camera_frame_invalid("CameraFrame tensor has no plane 0"))?;
        let height = tensor
            .shape_at(0)
            .ok_or_else(|| camera_frame_invalid("CameraFrame tensor missing height (shape[0])"))?;
        let width = tensor
            .shape_at(1)
            .ok_or_else(|| camera_frame_invalid("CameraFrame tensor missing width (shape[1])"))?;
        (
            tensor.pid(),
            plane0.handle,
            plane0.stride,
            plane0.offset,
            width,
            height,
            tensor.format().to_owned(),
        )
    };

    let pid_i32 = i32::try_from(pid)
        .map_err(|_| camera_frame_invalid(format!("CameraFrame tensor pid {pid} exceeds i32")))?;
    let pidfd = match PidFd::from_pid(pid_i32) {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error getting PID {pid:?}, please check if the camera process is running: {e:?}"
            );
            return Err(e);
        }
    };

    let target_fd = i32::try_from(handle).map_err(|_| {
        camera_frame_invalid(format!("CameraFrame plane handle {handle} exceeds i32"))
    })?;

    let fd = match get_file_from_pidfd(pidfd.as_raw_fd(), target_fd, GetFdFlags::empty()) {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error getting Camera DMA file descriptor, please check if current process is running with same permissions as camera: {e:?}"
            );
            return Err(e);
        }
    };

    Ok(ResolvedCameraFrame {
        plane_fd: fd.as_raw_fd(),
        stride: camera_frame_u32("stride", stride)?,
        offset: camera_frame_u32("offset", offset)?,
        width: camera_frame_nonzero_u32("width", width)?,
        height: camera_frame_nonzero_u32("height", height)?,
        format,
        _fd_guard: fd,
        frame,
    })
}

// If the receiver is empty, waits for the next message, otherwise returns the
// most recent message on this receiver. If the receiver is closed, returns None
pub(crate) async fn drain_recv<T>(rx: &mut Receiver<T>) -> Option<T> {
    let mut msg = match rx.try_recv() {
        Err(TryRecvError::Empty) => {
            return rx.recv().await;
        }
        Err(_) => {
            return None;
        }
        Ok(v) => v,
    };
    while let Ok(v) = rx.try_recv() {
        msg = v;
    }
    Some(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn time(sec: i32, nanosec: u32) -> Time {
        Time { sec, nanosec }
    }

    #[test]
    fn test_time_to_ns_combines_fields() {
        assert_eq!(time_to_ns(time(0, 0)), 0);
        assert_eq!(
            time_to_ns(time(1_790_000_000, 123_456_789)),
            1_790_000_000_123_456_789
        );
    }

    #[test]
    fn test_time_to_ns_clamps_pre_epoch() {
        assert_eq!(time_to_ns(time(-1, 999_999_999)), 0);
    }

    #[test]
    fn test_ntp64_from_time_round_trips_within_resolution() {
        for stamp in [
            time(0, 1),
            time(1_748_544_498, 0),
            time(1_790_000_000, 123_456_789),
            time(1_790_000_000, 999_999_999),
        ] {
            let decoded = ntp64_from_time(stamp).to_duration().as_nanos() as i128;
            let expected = time_to_ns(stamp) as i128;
            assert!(
                (decoded - expected).abs() <= 1,
                "{stamp:?} decoded to {decoded}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_stamp_timeline_reports_backward_step() {
        let mut timeline = StampTimeline::default();
        assert_eq!(timeline.observe(1_000), None);
        assert_eq!(timeline.observe(2_000), None);
        assert_eq!(timeline.observe(500), Some(1_500));
        assert_eq!(timeline.observe(600), None);
    }

    #[test]
    fn test_stamp_timeline_ignores_forward_step_and_repeats() {
        let mut timeline = StampTimeline::default();
        assert_eq!(timeline.observe(1_000), None);
        assert_eq!(timeline.observe(1_000), None);
        assert_eq!(timeline.observe(41_054_973_000_000_000), None);
    }
}
