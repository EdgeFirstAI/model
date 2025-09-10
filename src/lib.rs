pub mod args;
pub mod buildmsgs;
pub mod fps;
pub mod image;
pub mod kalman;
pub mod masks;
pub mod model;
pub mod nms;
pub mod tflite_model;
pub mod tracker;

#[cfg(feature = "rtm")]
pub mod rtm_model;

use crate::{buildmsgs::*, model::ModelErrorKind, tracker::*};
use args::{Args, LabelSetting};
use async_pidfd::PidFd;
use cdr::{CdrLe, Infinite};
use edgefirst_schemas::{
    self,
    edgefirst_msgs::{DmaBuf, ModelInfo},
};
use image::Image;
use log::{debug, error, info, trace, warn};
use model::{DetectBox, Model, ModelError, SupportedModel};
use ndarray::{Array1, Array3};
use nix::{
    sys::time::TimeValLike,
    time::{ClockId, clock_gettime},
};
use pidfd_getfd::{GetFdFlags, get_file_from_pidfd};
use std::{fs::File, io, os::fd::AsRawFd, time::Duration};
use tokio::sync::mpsc::{Receiver, error::TryRecvError};
use tracing::{info_span, instrument};
use uuid::Uuid;
use zenoh::{
    Session,
    bytes::{Encoding, ZBytes},
    handlers::FifoChannelHandler,
    pubsub::Subscriber,
    sample::Sample,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ModelType {
    pub segment_output_ind: Option<usize>,
    pub detection: bool,
    pub detection_with_mask: bool,
}

#[instrument(skip_all)]
pub fn run_detection(
    model: &mut SupportedModel,
    labels: &[String],
    boxes: &mut Vec<DetectBox>,
    tracker: &mut ByteTrack,
    timestamp: u64,
    args: &Args,
) -> (Vec<BoxWithTrack>, Option<Array3<f32>>) {
    let mut protos = None;
    if let Err(e) = model.decode_outputs(boxes, &mut protos) {
        error!("Failed to read bounding boxes from model: {e:?}");
        return (Vec::new(), None);
    };

    let mut new_boxes = Vec::new();
    if args.track {
        info_span!("tracker").in_scope(|| {
            let _ = tracker.update(args, boxes, timestamp);
            let tracks = tracker.get_tracklets();
            for track in tracks {
                let detect_box: DetectBox = track.get_predicted_location();
                let box_2d =
                    detectbox_to_boxwithtrack(args, &detect_box, labels, timestamp, Some(track));
                new_boxes.push(box_2d);
            }
        });
    } else {
        for detect_box in boxes.iter_mut() {
            let box_2d = detectbox_to_boxwithtrack(args, detect_box, labels, timestamp, None);
            new_boxes.push(box_2d);
        }
    }
    (new_boxes, protos)
}
#[derive(Debug, Clone, Default)]
pub struct BoxWithTrack {
    #[doc = " left-most coordinate of the bounding box."]
    pub xmin: f64,
    #[doc = " top-most coordinate of the bounding box."]
    pub ymin: f64,
    #[doc = " right-most coordinate of the bounding box."]
    pub xmax: f64,
    #[doc = " bottom-most coordinate of the bounding box."]
    pub ymax: f64,
    #[doc = " model-specific score for this detection, higher implies more confidence."]
    pub score: f64,
    #[doc = " index of label for this detection"]
    pub index: usize,
    #[doc = " Optional mask coefficients for computing instanced masks"]
    pub mask_coeff: Option<Array1<f32>>,
    #[doc = " label for this detection"]
    pub label: String,
    #[doc = " timestamp"]
    pub ts: u64,
    #[doc = " tracking information for this detection"]
    pub track: Option<Track>,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Track {
    #[doc = " track UUID for this detection"]
    pub uuid: Uuid,
    #[doc = " number of detects this track has been seen for"]
    pub count: i32,
    #[doc = " when this track was first added"]
    pub created: u64,
    #[doc = " when this track was first added"]
    pub last_updated: u64,
}
fn detectbox_to_boxwithtrack(
    args: &Args,
    b: &DetectBox,
    labels: &[String],
    ts: u64,
    track: Option<&Tracklet>,
) -> BoxWithTrack {
    let label_ind = b.label as i32 + args.label_offset;
    let label = match labels.get(label_ind as usize) {
        Some(s) => s.clone(),
        None => b.label.to_string(),
    };

    trace!("Created box with label {label}");
    let track_info = track.as_ref().map(|v| Track {
        uuid: v.id,
        count: v.count,
        created: v.created,
        last_updated: v.last_updated,
    });

    BoxWithTrack {
        xmin: b.xmin as f64,
        ymin: b.ymin as f64,
        xmax: b.xmax as f64,
        ymax: b.ymax as f64,
        score: b.score as f64,
        mask_coeff: b.mask_coeff.clone(),
        label,
        ts,
        index: b.label,
        track: track_info,
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
    let model_type = ModelType {
        segment_output_ind: None,
        detection: false,
        detection_with_mask: false,
    };
    let mut model_info_msg =
        build_model_info_msg(time_from_ns(0u32), None, &model_path, &model_type);
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
    trace!("Recieved camera frame");

    let Ok(_fd) = update_dmabuf_with_pidfd(&mut dma_buf) else {
        return;
    };

    trace!("Opened DMA buffer from camera");

    let curr_time = get_curr_time();

    let mask = build_segmentation_msg(dma_buf.header.stamp.clone(), None, 0);
    let msg = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&mask, Infinite).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

    match session.put(&args.mask_topic, msg).encoding(enc).await {
        Ok(_) => (),
        Err(e) => {
            error!("Error sending message on {}: {:?}", args.mask_topic, e)
        }
    }

    let (msg, enc) = build_detect_msg_and_encode(
        &Vec::new(),
        dma_buf.header.stamp.clone(),
        time_from_ns(0u32),
        time_from_ns(curr_time),
    );

    match session.put(&args.detect_topic, msg).encoding(enc).await {
        Ok(_) => (),
        Err(e) => {
            error!("Error sending message on {}: {:?}", args.detect_topic, e)
        }
    }

    model_info_msg.header.stamp = dma_buf.header.stamp.clone();
    let msg = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&model_info_msg, Infinite).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/ModelInfo");

    match session.put(&args.info_topic, msg).encoding(enc).await {
        Ok(_) => (),
        Err(e) => {
            error!("Error sending message on {}: {:?}", args.info_topic, e)
        }
    }

    if args.visualization {
        let (msg, enc) = build_image_annotations_msg_and_encode(
            &Vec::new(),
            dma_buf.header.stamp.clone(),
            stream_dims.0,
            stream_dims.1,
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
) -> Option<DmaBuf> {
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
    match cdr::deserialize(&dma_buf.payload().to_bytes()) {
        Ok(v) => Some(v),
        Err(e) => {
            error!("Failed to deserialize message: {e:?}");
            None
        }
    }
}

pub fn update_dmabuf_with_pidfd(dma_buf: &mut DmaBuf) -> Result<File, std::io::Error> {
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
    dma_buf.fd = fd.as_raw_fd();
    Ok(fd)
}

pub fn identify_model<M: Model>(model: &M) -> Result<ModelType, ModelError> {
    if model.input_count().is_ok_and(|f| f != 1) {
        error!(
            "Model has {} inputs but expected only 1",
            model.input_count()?
        );
    }

    if let Ok(metadata) = model.get_model_metadata()
        && let Some(config) = &metadata.config
    {
        debug!("Metadata: {metadata:?}");
        let mut model_type = ModelType {
            segment_output_ind: None,
            detection: false,
            detection_with_mask: false,
        };

        for output in &config.outputs {
            match output {
                model::ConfigOutput::Detection(_)
                | model::ConfigOutput::Boxes(_)
                | model::ConfigOutput::Scores(_) => model_type.detection = true,
                model::ConfigOutput::Segmentation(segmentation)
                    if matches!(segmentation.decoder, model::Decoder::Yolov8) =>
                {
                    model_type.detection = true;
                    model_type.detection_with_mask = true;
                }
                model::ConfigOutput::Segmentation(segmentation) => {
                    // model_type.segment_output_ind = Some(segmentation.index)
                    model_type.segment_output_ind = match (0..model.output_count()?).find_map(
                        |index| {
                            if model.output_shape(index).ok()? == segmentation.shape {
                                Some(index)
                            } else {
                                None
                            }
                        },
                    ) {
                        Some(index) => Some(index),
                        None => {
                            return Err(ModelError::new(
                                ModelErrorKind::Decoding,
                                format!(
                                    "Cannot find output with shape {:?} as specified in metadata",
                                    segmentation.shape
                                ),
                            ));
                        }
                    }
                }
                _ => {}
            }
        }

        return Ok(model_type);
    }
    let output_count = model.output_count()?;
    info!("output_count {output_count:?}");
    let mut segmentation_index = Vec::new();
    let mut model_type = ModelType {
        segment_output_ind: None,
        detection: false,
        detection_with_mask: false,
    };
    // first: check if segmentation -> segmentation if output has 4 dimensions and
    // the W/H is greater than 8
    // this criteria is wider than the current criteria for MPK segmentation, but
    // still ensures that detection outputs won't be mistaken for segmentation
    // output.
    for i in 0..output_count {
        let shape = model.output_shape(i)?;
        info!("output_shape[{i}] {shape:?}");
        if shape.len() != 4 {
            continue;
        }
        if shape[1] < 8 {
            continue;
        }
        if shape[2] < 8 {
            continue;
        }
        info!("segmentation output shape: {shape:?}");
        segmentation_index.push(i);
    }
    if segmentation_index.len() > 1 {
        error!("Found more than 1 valid segmentation output tensors");
    }
    if !segmentation_index.is_empty() {
        model_type.segment_output_ind = Some(segmentation_index[0]);
        info!("Model has segmentation output");
    }

    // if there are any leftover outputs, assume it is a detection model `boxes`
    if segmentation_index.len() < output_count {
        model_type.detection = true;
        info!("Model has detection output");
    }

    Ok(model_type)
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

impl TryFrom<&DmaBuf> for Image {
    type Error = io::Error;

    fn try_from(dma_buf: &DmaBuf) -> Result<Self, io::Error> {
        let pidfd: PidFd = PidFd::from_pid(dma_buf.pid as i32)?;
        let fd = get_file_from_pidfd(pidfd.as_raw_fd(), dma_buf.fd, GetFdFlags::empty())?;
        let fourcc = dma_buf.fourcc.into();
        // println!("src fourcc: {:?}", fourcc);
        Ok(Image {
            fd: fd.into(),
            width: dma_buf.width,
            height: dma_buf.height,
            format: fourcc,
        })
    }
}
