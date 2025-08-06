mod args;
mod buildmsgs;
mod fps;
mod image;
mod kalman;
mod masks;
mod model;
mod nms;
mod tflite_model;
mod tracker;

#[cfg(feature = "rtm")]
mod rtm_model;

use crate::{buildmsgs::*, tracker::*};
use args::{Args, LabelSetting};
use async_pidfd::PidFd;
use cdr::{CdrLe, Infinite};
use clap::Parser;
use edgefirst_schemas::{self, edgefirst_msgs::DmaBuf, sensor_msgs::CameraInfo};
use image::{Image, ImageManager};
use log::{error, info, trace, warn};
use masks::{mask_compress_thread, mask_thread};
use model::{DetectBox, Model, ModelError, SupportedModel};
use nix::{
    sys::time::TimeValLike,
    time::{clock_gettime, ClockId},
};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    io,
    os::fd::AsRawFd,
    process::ExitCode,
    time::{Duration, Instant},
};
use tflite_model::{TFLiteLib, DEFAULT_NPU_DELEGATE_PATH};
use tokio::sync::mpsc::{self, error::TryRecvError, Receiver};
use tracing::{info_span, instrument, level_filters::LevelFilter};
use tracing_subscriber::{layer::SubscriberExt as _, Layer as _, Registry};
use tracy_client::frame_mark;
use uuid::Uuid;
use zenoh::{
    bytes::{Encoding, ZBytes},
    handlers::FifoChannelHandler,
    pubsub::Subscriber,
    sample::Sample,
    Session,
};

#[cfg(feature = "rtm")]
use rtm_model::RtmModel;

struct ModelType {
    segment_output_ind: Option<usize>,
    detection: bool,
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    args.tracy.then(tracy_client::Client::start);

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let stdout_log = tracing_subscriber::fmt::layer()
        .pretty()
        .with_filter(env_filter);

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let journald = match tracing_journald::layer() {
        Ok(journald) => Some(journald.with_filter(env_filter)),
        Err(_) => None,
    };

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    let tracy = match args.tracy {
        true => Some(tracing_tracy::TracyLayer::default().with_filter(env_filter)),
        false => None,
    };

    let subscriber = Registry::default()
        .with(stdout_log)
        .with(journald)
        .with(tracy);
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
    tracing_log::LogTracer::init().unwrap();

    let mut first_run = true;
    let session = zenoh::open(args.clone()).await.unwrap();

    let stream_width: f64;
    let stream_height: f64;
    if args.visualization {
        let info_sub = session
            .declare_subscriber(&args.camera_info_topic)
            .await
            .unwrap();
        info!("Declared subscriber on {:?}", &args.camera_info_topic);
        match info_sub.recv_timeout(Duration::from_secs(10)) {
            Ok(v) => {
                match cdr::deserialize::<CameraInfo>(&v.unwrap().payload().to_bytes()) {
                    Ok(v) => {
                        stream_width = v.width as f64;
                        stream_height = v.height as f64;
                        info!("Found stream resolution: {stream_width}x{stream_height}");
                    }
                    Err(e) => {
                        warn!("Failed to deserialize camera info message: {e:?}");
                        warn!("Cannot determine stream resolution, using normalized coordinates");
                        stream_width = 1.0;
                        stream_height = 1.0;
                    }
                };
            }
            Err(e) => {
                warn!("Failed to receive on {:?}: {:?}", args.camera_info_topic, e);
                warn!("Cannot determine stream resolution, using normalized coordinates");
                stream_width = 1.0;
                stream_height = 1.0;
            }
        }
        drop(info_sub);
    } else {
        stream_width = 1.0;
        stream_height = 1.0;
    }

    let sub_camera = session
        .declare_subscriber(&args.camera_topic)
        .await
        .unwrap();
    info!("Declared subscriber on {:?}", &args.camera_topic);

    let (tx, rx) = mpsc::channel(50);
    let heartbeat = tokio::spawn(heart_beat(
        session.clone(),
        args.clone(),
        sub_camera,
        rx,
        (stream_width, stream_height),
    ));

    let model_data = match std::fs::read(&args.model) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not read model file {:?}: {:?}", args.model, e);
            return ExitCode::FAILURE;
        }
    };

    let mut _tflite = None;

    let mut model: SupportedModel<'_> = match args.model.extension() {
        Some(v) if v == "tflite" => {
            _tflite = match TFLiteLib::new() {
                Ok(v) => Some(v),
                Err(e) => {
                    error!("Could not load TensorFlowLite API: {e:?}");
                    return ExitCode::FAILURE;
                }
            };
            let delegate = if &args.engine.to_lowercase() == "npu" {
                Some(DEFAULT_NPU_DELEGATE_PATH)
            } else {
                None
            };

            let mut model = match _tflite
                .as_ref()
                .unwrap()
                .load_model_from_mem_with_delegate(model_data, delegate)
            {
                Ok(v) => v,
                Err(e) => {
                    error!("Could not load TFLite model: {e:?}");
                    return ExitCode::FAILURE;
                }
            };
            model.setup_context(&args);
            model.into()
        }
        Some(v) if v == "rtm" => {
            #[cfg(feature = "rtm")]
            {
                let mut model =
                    match RtmModel::load_model_from_mem_with_engine(model_data, &args.engine) {
                        Ok(v) => v,
                        Err(e) => {
                            error!("Could not load RTM model: {e:?}");
                            return ExitCode::FAILURE;
                        }
                    };
                model.setup_context(&args);
                model.into()
            }

            #[cfg(not(feature = "rtm"))]
            {
                error!("RTM model support is not enabled in this build");
                return ExitCode::FAILURE;
            }
        }
        Some(v) => {
            error!("Unsupported model extension: {v:?}");
            return ExitCode::FAILURE;
        }
        None => {
            error!("No model extension: {:?}", args.model);
            return ExitCode::FAILURE;
        }
    };
    info!("Loaded model");

    let model_type = match identify_model(&model) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not identify model type: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    info!("identified model type");
    drop(tx);

    let publ_model_info = session
        .declare_publisher(args.info_topic.clone())
        .await
        .unwrap();
    let publ_detect = session
        .declare_publisher(args.detect_topic.clone())
        .await
        .unwrap();
    let publ_mask = session
        .declare_publisher(args.mask_topic.clone())
        .await
        .unwrap();

    let publ_mask_compressed = match args.mask_compression {
        true => Some(
            session
                .declare_publisher(args.mask_compressed_topic.clone())
                .await
                .unwrap(),
        ),
        false => None,
    };

    let publ_visual = match args.visualization {
        true => Some(
            session
                .declare_publisher(args.visual_topic.clone())
                .await
                .unwrap(),
        ),
        false => None,
    };

    let (mask_tx, mask_rx) = mpsc::channel(50);

    let mask_compress_tx = if let Some(publ_mask_compressed) = publ_mask_compressed {
        let (mask_compress_tx, mask_compress_rx) = mpsc::channel(50);
        tokio::spawn(mask_compress_thread(
            mask_compress_rx,
            publ_mask_compressed,
            args.mask_compression_level,
        ));
        Some(mask_compress_tx)
    } else {
        None
    };

    tokio::spawn(mask_thread(
        mask_rx,
        args.mask_classes.clone(),
        publ_mask,
        mask_compress_tx,
    ));

    let mut model_info_msg =
        build_model_info_msg(time_from_ns(0u32), Some(&model), &args.model, &model_type);
    info!("built model_info_msg");

    let sub_camera = heartbeat.await.unwrap();

    let model_name = match args.model.as_path().file_name() {
        Some(v) => String::from(v.to_string_lossy()),
        None => {
            warn!("Cannot determine model file basename");
            String::from("unknown_model_file")
        }
    };
    info!("got model_name {model_name}");
    let model_labels = match model.labels() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not get model labels: {e:?}");
            Vec::new()
        }
    };
    info!("got model_labels {model_labels:?}");
    let mut tracker = ByteTrack::new();
    let mut detect_boxes: Vec<DetectBox> = vec![DetectBox::default(); args.max_boxes];
    let timeout = Duration::from_millis(100);
    let mut fps = fps::Fps::<90>::default();

    let img_mgr = match ImageManager::new() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open G2D: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    info!("Opened G2D with version {}", img_mgr.version());

    loop {
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
                        continue;
                    }
                },
                Err(e) => {
                    error!(
                        "error receiving camera frame on {}: {:?}",
                        sub_camera.key_expr(),
                        e
                    );
                    continue;
                }
            }
        };

        let dma_buf: DmaBuf = cdr::deserialize(&dma_buf.payload().to_bytes()).unwrap();
        trace!("Recieved camera frame");

        match model.load_frame_dmabuf(&dma_buf, &img_mgr, model::Preprocessing::Raw) {
            Ok(_) => trace!("Loaded frame into model"),
            Err(e) => error!("Could not load frame into model: {e:?}"),
        }

        let model_start = Instant::now();

        if let Err(e) = model.run_model() {
            error!("Failed to run model: {e:?}");
            return ExitCode::FAILURE;
        }
        let model_duration = model_start.elapsed().as_nanos();
        trace!("Ran model: {:.3} ms", model_duration as f32 / 1_000_000.0);

        if let Some(i) = model_type.segment_output_ind {
            let masks = build_segmentation_msg(dma_buf.header.stamp.clone(), Some(&model), i);
            match mask_tx.send(masks).await {
                Ok(_) => {}
                Err(e) => {
                    error! {"Cannot send to mask publishing thread {e:?}"};
                }
            }
        }

        if model_type.detection {
            let timestamp = dma_buf.header.stamp.nanosec as u64
                + dma_buf.header.stamp.sec as u64 * 1_000_000_000;
            let new_boxes = run_detection(
                &model,
                &model_labels,
                &mut detect_boxes,
                &mut tracker,
                timestamp,
                &args,
            );
            if first_run {
                info!(
                    "Successfully recieved camera frames and run model, found {:?} boxes",
                    new_boxes.len()
                );
                first_run = false;
            } else {
                trace!(
                    "Detected {:?} boxes. FPS: {}",
                    new_boxes.len(),
                    fps.update()
                );
            }
            let curr_time = match clock_gettime(ClockId::CLOCK_MONOTONIC) {
                Ok(t) => t.num_nanoseconds() as u64,
                Err(e) => {
                    error!("Could not get Monotonic clock time: {e:?}");
                    0
                }
            };

            let (msg, enc) = build_detect_msg_and_encode(
                &new_boxes,
                dma_buf.header.stamp.clone(),
                time_from_ns(model_duration as u32),
                time_from_ns(curr_time as u32),
            );

            match publ_detect.put(msg).encoding(enc).await {
                Ok(_) => trace!("Sent Detect message on {}", publ_detect.key_expr()),
                Err(e) => {
                    error!(
                        "Error sending message on {}: {:?}",
                        publ_detect.key_expr(),
                        e
                    )
                }
            }

            if publ_visual.is_some() {
                let publ_visual = publ_visual.as_ref().unwrap();
                let (msg, enc) = build_image_annotations_msg_and_encode(
                    &new_boxes,
                    dma_buf.header.stamp.clone(),
                    stream_width,
                    stream_height,
                    &model_name,
                    args.labels,
                );

                match publ_visual.put(msg).encoding(enc).await {
                    Ok(_) => trace!("Sent message on {}", publ_detect.key_expr()),
                    Err(e) => {
                        error!(
                            "Error sending message on {}: {:?}",
                            publ_detect.key_expr(),
                            e
                        )
                    }
                }
            }
        }

        model_info_msg.header.stamp = dma_buf.header.stamp.clone();
        let msg = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&model_info_msg, Infinite).unwrap());
        let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/ModelInfo");

        match publ_model_info.put(msg).encoding(enc).await {
            Ok(_) => (),
            Err(e) => {
                error!(
                    "Error sending message on {}: {:?}",
                    publ_model_info.key_expr(),
                    e
                )
            }
        }

        args.tracy.then(frame_mark);
    }
}

#[instrument(skip_all)]
fn run_detection(
    model: &SupportedModel,
    labels: &[String],
    boxes: &mut [DetectBox],
    tracker: &mut ByteTrack,
    timestamp: u64,
    args: &Args,
) -> Vec<BoxWithTrack> {
    let n_boxes = match model.boxes(boxes) {
        Ok(n_boxes) => n_boxes,
        Err(e) => {
            error!("Failed to read bounding boxes from model: {e:?}");
            return Vec::new();
        }
    };
    let mut new_boxes = Vec::new();
    if args.track {
        info_span!("tracker").in_scope(|| {
            let _ = tracker.update(args, &mut boxes[0..n_boxes], timestamp);
            let tracks = tracker.get_tracklets();
            for track in tracks {
                let detect_box: DetectBox = track.get_predicted_location();
                let box_2d =
                    detectbox_to_boxwithtrack(args, &detect_box, labels, timestamp, Some(track));
                new_boxes.push(box_2d);
            }
        });
    } else {
        for detect_box in boxes[0..n_boxes].iter() {
            let box_2d = detectbox_to_boxwithtrack(args, detect_box, labels, timestamp, None);
            new_boxes.push(box_2d);
        }
    }
    new_boxes
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
    });

    BoxWithTrack {
        xmin: b.xmin as f64,
        ymin: b.ymin as f64,
        xmax: b.xmax as f64,
        ymax: b.ymax as f64,
        score: b.score as f64,
        label,
        ts,
        index: b.label,
        track: track_info,
    }
}

async fn heart_beat(
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
    };
    let mut model_info_msg =
        build_model_info_msg(time_from_ns(0u32), None, &model_path, &model_type);
    let status = format!("Loading Model: {}", model_path.to_string_lossy());

    loop {
        match rx.try_recv() {
            Ok(_) => return sub_camera,
            Err(e) => match e {
                TryRecvError::Disconnected => return sub_camera,
                TryRecvError::Empty => (),
            },
        }
        let sample = if let Some(s) = sub_camera.drain().last() {
            s
        } else {
            match sub_camera.recv_timeout(Duration::from_millis(100)) {
                Ok(Some(v)) => v,
                Ok(None) => {
                    error!(
                        "Timeout receiving camera frame on {}",
                        sub_camera.key_expr()
                    );
                    continue;
                }
                Err(e) => {
                    error!(
                        "error receiving camera frame on {}: {:?}",
                        sub_camera.key_expr(),
                        e
                    );
                    continue;
                }
            }
        };
        let mut dma_buf: DmaBuf = match cdr::deserialize(&sample.payload().to_bytes()) {
            Ok(v) => v,
            Err(e) => {
                error!("Failed to deserialize message: {e:?}");
                continue;
            }
        };
        trace!("Recieved camera frame");

        let pidfd: PidFd = match PidFd::from_pid(dma_buf.pid as i32) {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "Error getting PID {:?}, please check if the camera process is running: {:?}",
                    dma_buf.pid, e
                );
                continue;
            }
        };
        let fd = match get_file_from_pidfd(pidfd.as_raw_fd(), dma_buf.fd, GetFdFlags::empty()) {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "Error getting Camera DMA file descriptor, please check if current process is running with same permissions as camera: {e:?}"
                );
                continue;
            }
        };
        dma_buf.fd = fd.as_raw_fd();
        trace!("Opened DMA buffer from camera");
        let curr_time = match clock_gettime(ClockId::CLOCK_MONOTONIC) {
            Ok(t) => t.num_nanoseconds() as u64,
            Err(e) => {
                error!("Could not get Monotonic clock time: {e:?}");
                0
            }
        };

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
                &status,
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
}

fn identify_model<M: Model>(model: &M) -> Result<ModelType, ModelError> {
    let output_count = model.output_count()?;
    info!("output_count {output_count:?}");
    let mut segmentation_index = Vec::new();
    let mut model_type = ModelType {
        segment_output_ind: None,
        detection: false,
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
