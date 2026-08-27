// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

pub mod args;
pub mod buildmsgs;
pub mod fps;
pub mod letterbox;
pub mod masks;
pub mod model;
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
use log::{error, trace, warn};
use nix::{
    sys::time::TimeValLike,
    time::{ClockId, clock_gettime},
};
use pidfd_getfd::{GetFdFlags, get_file_from_pidfd};
use std::{fs::File, os::fd::AsRawFd, time::Duration};
use tokio::sync::mpsc::{Receiver, error::TryRecvError};

use zenoh::{
    Session,
    bytes::{Encoding, ZBytes},
    handlers::FifoChannelHandler,
    pubsub::Subscriber,
    sample::Sample,
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
        self.frame.width()
    }

    pub fn height(&self) -> u32 {
        self.frame.height()
    }

    pub fn format(&self) -> &str {
        self.frame.format()
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
    let Some(frame) = wait_for_camera_frame(sub_camera, Duration::from_millis(100)) else {
        return;
    };
    trace!("Received camera frame");

    if !args.mask_topic.is_empty() {
        let mask = build_segmentation_msg(frame.stamp(), None, 0, None);
        let msg = ZBytes::from(mask.into_cdr());
        let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

        match session.put(&args.mask_topic, msg).encoding(enc).await {
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

        match session.put(&args.detect_topic, msg).encoding(enc).await {
            Ok(_) => (),
            Err(e) => {
                error!("Error sending message on {}: {:?}", args.detect_topic, e)
            }
        }
    }

    let model_info_msg = build_model_info_msg(frame.stamp(), None, model_path, false, false);
    let msg = ZBytes::from(model_info_msg.into_cdr());
    let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/ModelInfo");

    match session.put(&args.info_topic, msg).encoding(enc).await {
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

        match session.put(&args.visual_topic, msg).encoding(enc).await {
            Ok(_) => trace!("Sent message on {}", args.visual_topic),
            Err(e) => {
                error!("Error sending message on {}: {:?}", args.visual_topic, e)
            }
        }
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

pub fn wait_for_camera_frame(
    sub_camera: &Subscriber<FifoChannelHandler<Sample>>,
    timeout: Duration,
) -> Option<ResolvedCameraFrame> {
    let sample = if let Some(v) = sub_camera.drain().last() {
        v
    } else {
        match sub_camera.recv_timeout(timeout) {
            Ok(msg) => match msg {
                Some(v) => v,
                None => {
                    warn!(
                        "timeout receiving camera frame on {}",
                        sub_camera.key_expr()
                    );
                    return None;
                }
            },
            Err(e) => {
                error!(
                    "error receiving camera frame on {}: {:?}",
                    sub_camera.key_expr(),
                    e
                );
                return None;
            }
        }
    };

    let cdr = sample.payload().to_bytes();
    let frame = match CameraFrame::from_cdr(cdr.to_vec()) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to deserialize CameraFrame: {e:?}");
            return None;
        }
    };

    match resolve_camera_frame_fd(frame) {
        Ok(v) => Some(v),
        Err(e) => {
            error!("Failed to import camera DMA-BUF fd: {e:?}");
            None
        }
    }
}

fn resolve_camera_frame_fd(
    frame: CameraFrame<Vec<u8>>,
) -> Result<ResolvedCameraFrame, std::io::Error> {
    let planes = frame.planes();
    let plane0 = planes.first().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "CameraFrame has no planes")
    })?;

    let pidfd = match PidFd::from_pid(frame.pid() as i32) {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error getting PID {:?}, please check if the camera process is running: {:?}",
                frame.pid(),
                e
            );
            return Err(e);
        }
    };

    let fd = match get_file_from_pidfd(pidfd.as_raw_fd(), plane0.fd, GetFdFlags::empty()) {
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
        stride: plane0.stride,
        offset: plane0.offset,
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
