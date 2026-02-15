// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

pub mod args;
pub mod buildmsgs;
pub mod fps;
pub mod masks;
pub mod model;
pub mod tflite_model;

#[cfg(feature = "rtm")]
pub mod rtm_model;

/// Newtype wrapper to bridge `edgefirst_tracker::DetectionBox` for
/// `edgefirst_hal::decoder::DetectBox`.
pub struct TrackerBox<'a>(pub &'a edgefirst_hal::decoder::DetectBox);

impl std::fmt::Debug for TrackerBox<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl edgefirst_tracker::DetectionBox for TrackerBox<'_> {
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
use edgefirst_schemas::{
    self,
    edgefirst_msgs::{DmaBuffer, Mask, ModelInfo},
    schema_registry::SchemaType,
    serde_cdr,
};
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

pub async fn heart_beat(
    session: Session,
    args: Args,
    sub_camera: Subscriber<FifoChannelHandler<Sample>>,
    mut rx: Receiver<bool>,
    stream_dims: (f64, f64),
) -> Subscriber<FifoChannelHandler<Sample>> {
    let model_path = args.model.clone();

    let mut model_info_msg =
        build_model_info_msg(time_from_ns(0u32), None, &model_path, false, false);
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
            &mut model_info_msg,
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
    model_info_msg: &mut ModelInfo,
    status: &str,
) {
    let Some(mut dma_buf) = wait_for_camera_frame(sub_camera, Duration::from_millis(100)) else {
        return;
    };
    trace!("Received camera frame");

    let Ok(_fd) = update_dmabuf_with_pidfd(&mut dma_buf) else {
        return;
    };

    trace!("Opened DMA buffer from camera");

    let mask = build_segmentation_msg(dma_buf.header.stamp.clone(), None, 0);
    let msg = ZBytes::from(serde_cdr::serialize(&mask).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema(Mask::SCHEMA_NAME);

    match session.put(&args.mask_topic, msg).encoding(enc).await {
        Ok(_) => (),
        Err(e) => {
            error!("Error sending message on {}: {:?}", args.mask_topic, e)
        }
    }

    let (msg, enc) = build_detect_msg_and_encode_(
        &[],
        &[],
        &[],
        dma_buf.header.clone(),
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

    model_info_msg.header.stamp = dma_buf.header.stamp.clone();
    let msg = ZBytes::from(serde_cdr::serialize(&model_info_msg).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema(ModelInfo::SCHEMA_NAME);

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
            dma_buf.header.stamp.clone(),
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
) -> Option<DmaBuffer> {
    let dma_buf = if let Some(v) = sub_camera.drain().last() {
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
    match serde_cdr::deserialize(&dma_buf.payload().to_bytes()) {
        Ok(v) => Some(v),
        Err(e) => {
            error!("Failed to deserialize message: {e:?}");
            None
        }
    }
}

pub fn update_dmabuf_with_pidfd(dma_buf: &mut DmaBuffer) -> Result<File, std::io::Error> {
    let pidfd: PidFd = match PidFd::from_pid(dma_buf.pid as i32) {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error getting PID {:?}, please check if the camera process is running: {:?}",
                dma_buf.pid, e
            );
            return Err(e);
        }
    };

    let fd = match get_file_from_pidfd(pidfd.as_raw_fd(), dma_buf.fd, GetFdFlags::empty()) {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error getting Camera DMA file descriptor, please check if current process is running with same permissions as camera: {e:?}"
            );
            return Err(e);
        }
    };
    dma_buf.pid = std::process::id();
    dma_buf.fd = fd.as_raw_fd();
    Ok(fd)
}

// If the receiver is empty, waits for the next message, otherwise returns the
// most recent message on this receiver. If the receiver is closed, returns None
async fn drain_recv<T>(rx: &mut Receiver<T>) -> Option<T> {
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
