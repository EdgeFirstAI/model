mod args;
mod buildmsgs;
mod fps;
mod kalman;
mod tracker;

use crate::{buildmsgs::*, tracker::*};
use args::{Args, LabelSetting};
use async_pidfd::PidFd;
use cdr::{CdrLe, Infinite};
use clap::Parser;
use edgefirst_schemas::{
    self,
    edgefirst_msgs::{DmaBuf, Mask},
    sensor_msgs::CameraInfo,
};
use log::{debug, error, info, trace, warn};
use nix::{
    sys::time::TimeValLike,
    time::{clock_gettime, ClockId},
};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    fs,
    os::fd::AsRawFd,
    process::Command,
    sync::mpsc::{self, Receiver, TryRecvError},
    time::{Duration, Instant},
};
use tracing::{info_span, instrument, Instrument};
use tracing_subscriber::{layer::SubscriberExt as _, Layer as _, Registry};
use tracy_client::frame_mark;
use uuid::Uuid;
use vaal::{self, Context, VAALBox};
use zenoh::{
    bytes::{Encoding, ZBytes},
    handlers::FifoChannelHandler,
    pubsub::{Publisher, Subscriber},
    sample::Sample,
    Session,
};

struct ModelType {
    segment_output_ind: Option<i32>,
    detection: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    args.tracy.then(tracy_client::Client::start);

    let stdout_log = tracing_subscriber::fmt::layer()
        .pretty()
        .with_filter(args.rust_log);

    let journald = match tracing_journald::layer() {
        Ok(journald) => Some(journald.with_filter(args.rust_log)),
        Err(_) => None,
    };

    let tracy = match args.tracy {
        true => Some(tracing_tracy::TracyLayer::default().with_filter(args.rust_log)),
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
                        info!(
                            "Found stream resolution: {}x{}",
                            stream_width, stream_height
                        );
                    }
                    Err(e) => {
                        warn!("Failed to deserialize camera info message: {:?}", e);
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

    let (tx, rx) = mpsc::channel();
    let heartbeat = tokio::spawn(heart_beat(
        session.clone(),
        args.clone(),
        sub_camera,
        rx,
        (stream_width, stream_height),
    ));

    let mut backbone = match Context::new(&args.engine) {
        Ok(v) => {
            debug!("Opened VAAL Context on {}", args.engine);
            v
        }
        Err(e) => {
            error!("Could not open VAAL Context on {}, {:?}", args.engine, e);
            return;
        }
    };
    let filename = match args.model.to_str() {
        Some(v) => v,
        None => {
            error!(
                "Cannot use file {:?}, please use only utf8 characters in file path",
                args.model
            );
            return;
        }
    };
    match backbone.load_model_file(filename) {
        Ok(_) => info!("Loaded backbone model {:?}", filename),
        Err(e) => {
            error!("Could not load model file {}: {:?}", filename, e);
            return;
        }
    }
    let mut decoder = None;
    let model_type;
    if args.decoder_model.is_some() {
        let decoder_device = "cpu";
        let mut decoder_ctx = match Context::new(decoder_device) {
            Ok(v) => {
                debug!("Opened VAAL Context on {}", decoder_device);
                v
            }
            Err(e) => {
                error!("Could not open VAAL Context on {}, {:?}", decoder_device, e);
                return;
            }
        };
        setup_context(&mut decoder_ctx, &args);
        let decoder_file = match args.decoder_model.as_ref().unwrap().to_str() {
            Some(v) => v,
            None => {
                error!(
                    "Cannot use file {:?}, please use only utf8 characters in file path",
                    args.decoder_model.as_ref().unwrap()
                );
                return;
            }
        };
        match decoder_ctx.load_model_file(decoder_file) {
            Ok(_) => info!("Loaded decoder model {:?}", decoder_file),
            Err(e) => {
                error!("Could not load decoder file {}: {:?}", decoder_file, e);
                return;
            }
        }
        model_type = match identify_model(&decoder_ctx) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not identify model type: {:?}", e);
                return;
            }
        };
        decoder = Some(decoder_ctx);
    } else {
        model_type = match identify_model(&backbone) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not identify model type: {:?}", e);
                return;
            }
        };
        setup_context(&mut backbone, &args);
    }

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

    let (mask_tx, mask_rx) = mpsc::channel();
    tokio::spawn(mask_thread(mask_rx, publ_mask, publ_mask_compressed));

    let mut model_info_msg = build_model_info_msg(
        time_from_ns(0u32),
        Some(&mut backbone),
        decoder.as_mut(),
        &args.model,
        &model_type,
    );
    let sub_camera = heartbeat.await.unwrap();

    let model_name = match args.model.as_path().file_name() {
        Some(v) => String::from(v.to_string_lossy()),
        None => {
            warn!("Cannot determine model file basename");
            String::from("unknown_model_file")
        }
    };
    let mut tracker = ByteTrack::new();
    let mut vaal_boxes: Vec<vaal::VAALBox> = Vec::with_capacity(args.max_boxes as usize);
    let timeout = Duration::from_millis(100);
    let mut fps = fps::Fps::<90>::default();

    loop {
        let _ = sub_camera.drain();
        let mut dma_buf: DmaBuf = match sub_camera.recv_timeout(timeout) {
            Ok(msg) => match msg {
                Some(v) => match cdr::deserialize(&v.payload().to_bytes()) {
                    Ok(v) => v,
                    Err(e) => {
                        error!(
                            "Failed to deserialize message on {}: {:?}",
                            sub_camera.key_expr(),
                            e
                        );
                        continue;
                    }
                },
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
        };
        trace!("Recieved camera frame");

        let fd = match info_span!("dma_buf_open").in_scope(|| {
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
                    "Error getting Camera DMA file descriptor, please check if current process is running with same permissions as camera: {:?}",
                    e
                );
                    return Err(e);
                }
            };

            Ok(fd)
        }) {
            Ok(v) => v,
            Err(_) => continue,
        };
        dma_buf.fd = fd.as_raw_fd();

        trace!("Opened DMA buffer from camera");
        let model_start = Instant::now();

        match run_model(&dma_buf, &backbone, &mut decoder) {
            Ok(boxes) => boxes,
            Err(e) => {
                error!("Failed to run model: {:?}", e);
                return;
            }
        };
        let model_duration = model_start.elapsed().as_nanos();
        let model = if decoder.is_some() {
            decoder.as_ref().unwrap()
        } else {
            &backbone
        };

        if let Some(i) = model_type.segment_output_ind {
            let masks = build_segmentation_msg(dma_buf.header.stamp.clone(), Some(model), i);
            match mask_tx.send(masks) {
                Ok(_) => {}
                Err(e) => {
                    error! {"Cannot send to mask publishing thread {:?}", e};
                }
            }
        }

        if model_type.detection {
            let mut new_boxes = Vec::new();
            let timestamp = dma_buf.header.stamp.nanosec as u64
                + dma_buf.header.stamp.sec as u64 * 1_000_000_000;
            run_detection(
                model,
                &mut vaal_boxes,
                &mut tracker,
                &mut new_boxes,
                timestamp,
                &args,
            );
            if first_run {
                info!(
                    "Successfully recieved camera frames and run model, found {:?} boxes",
                    vaal_boxes.len()
                );
                first_run = false;
            } else {
                trace!(
                    "Detected {:?} boxes. FPS: {}",
                    vaal_boxes.len(),
                    fps.update()
                );
            }
            let curr_time = match clock_gettime(ClockId::CLOCK_MONOTONIC) {
                Ok(t) => t.num_nanoseconds() as u64,
                Err(e) => {
                    error!("Could not get Monotonic clock time: {:?}", e);
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
    model: &Context,
    boxes: &mut Vec<VAALBox>,
    tracker: &mut ByteTrack,
    new_boxes: &mut Vec<Box2D>,
    timestamp: u64,
    args: &Args,
) {
    let n_boxes = match model.boxes(boxes, boxes.capacity()) {
        Ok(len) => len,
        Err(e) => {
            return error!("Failed to read bounding boxes from model: {:?}", e);
        }
    };
    if args.track {
        info_span!("tracker").in_scope(|| {
            let _ = tracker.update(args, &mut boxes[0..n_boxes], timestamp);
            let tracks = tracker.get_tracklets();
            for track in tracks {
                let vaal_box = track.get_predicted_location();
                let box_2d = vaalbox_to_box2d(args, &vaal_box, model, timestamp, Some(track));
                new_boxes.push(box_2d);
            }
        });
    } else {
        for vaal_box in boxes.iter().take(n_boxes) {
            let box_2d = vaalbox_to_box2d(args, vaal_box, model, timestamp, None);
            new_boxes.push(box_2d);
        }
    }
}

#[inline(always)]
#[instrument(skip_all)]
fn run_model(
    dma_buf: &DmaBuf,
    backbone: &vaal::Context,
    decoder: &mut Option<vaal::Context>,
) -> Result<(), String> {
    info_span!("load_frame").in_scope(|| {
        match backbone.load_frame_dmabuf(
            None,
            dma_buf.fd,
            dma_buf.fourcc,
            dma_buf.width as i32,
            dma_buf.height as i32,
            None,
            0,
        ) {
            Err(vaal::Error::VAALError(e)) => {
                //possible vaal error that we can handle
                let poss_err =
                    "attempted an operation which is unsupported on the current platform";
                if e == poss_err {
                    error!(
                        "Attemping to clear cache,\
                            likely due to g2d alloc fail,\
                            this should be fixed in VAAL"
                    );
                    match clear_cached_memory() {
                        Ok(()) => error!("Cleared cached memory"),
                        Err(()) => error!("Could not clear cached memory"),
                    }
                } else {
                    return Err(format!("Could not load frame {:?}", e));
                }

                match backbone.load_frame_dmabuf(
                    None,
                    dma_buf.fd,
                    dma_buf.fourcc,
                    dma_buf.width as i32,
                    dma_buf.height as i32,
                    None,
                    0,
                ) {
                    Err(e) => return Err(format!("Could not load frame {:?}", e)),
                    Ok(_) => {
                        trace!("Loaded frame into model")
                    }
                };
            }
            Err(e) => return Err(format!("Could not load frame {:?}", e)),
            Ok(_) => {
                trace!("Loaded frame into model");
            }
        }
        Ok(())
    })?;

    info_span!("run_backbone").in_scope(|| {
        if let Err(e) = backbone.run_model() {
            return Err(format!("Failed to run model: {}", e));
        }
        trace!("Ran model inference");
        Ok(())
    })?;

    info_span!("run_decoder").in_scope(|| {
        if decoder.is_some() {
            let decoder_: &mut Context = decoder.as_mut().unwrap();
            let model = match decoder_.model() {
                Ok(model) => model,
                Err(e) => return Err(format!("Failed get decoder model: {:?}", e)),
            };

            let inputs_idx = match model.inputs() {
                Ok(inputs) => inputs,
                Err(e) => return Err(format!("Failed get decoder model input: {:?}", e)),
            };

            let context = decoder_.dvrt_context().unwrap();

            let mut in_1_idx = inputs_idx[1];
            let mut in_2_idx = inputs_idx[0];

            let out_1 = backbone.output_tensor(0).unwrap();
            let in_1 = context.tensor_index(in_1_idx as usize).unwrap();

            let out_2 = backbone.output_tensor(1).unwrap();

            let out_1_shape = out_1.shape();

            let in_1_shape = in_1.shape();

            if out_1_shape[1] != in_1_shape[1] && out_1_shape[2] != in_1_shape[2] {
                std::mem::swap(&mut in_2_idx, &mut in_1_idx);
            }

            let in_1 = context.tensor_index_mut(in_1_idx as usize).unwrap();

            if let Err(e) = out_1.dequantize(in_1) {
                return Err(format!(
                    "Failed to copy backbone out_1 ({:?}) to decoder in_1 ({:?}): {}",
                    out_1.tensor_type(),
                    in_1.tensor_type(),
                    e
                ));
            }
            trace!("Copied backdone out_1 to decoder in_1");

            let in_2 = context.tensor_index_mut(in_2_idx as usize).unwrap();

            if let Err(e) = out_2.dequantize(in_2) {
                return Err(format!(
                    "Failed to copy backbone out_2 ({:?}) to decoder in_2 ({:?}): {}",
                    out_2.tensor_type(),
                    in_2.tensor_type(),
                    e
                ));
            }
            trace!("Copied backdone out_2 to decoder in_2");

            if let Err(e) = decoder_.run_model() {
                return Err(format!("Failed to run decoder model: {:?}", e));
            }
            trace!("Ran decoder model inference");
        }

        Ok(())
    })?;

    Ok(())
}

fn setup_context(context: &mut Context, args: &Args) {
    context
        .parameter_seti("max_detection", &[args.max_boxes])
        .unwrap();
    context
        .parameter_setf("score_threshold", &[args.threshold])
        .unwrap();
    context
        .parameter_setf("iou_threshold", &[args.iou])
        .unwrap();
    context.parameter_sets("nms_type", "standard").unwrap();
}

/*
    This function clears cached memory pages
*/
fn clear_cached_memory() -> Result<(), ()> {
    match Command::new("sync").output() {
        Ok(output) => {
            match output.status.code() {
                Some(0) => {}
                _ => {
                    error!("sync command Failed");
                    error!("stdout {:?}", output.stdout);
                    error!("stderr {:?}", output.stderr);
                    return Err(());
                }
            };
        }
        Err(e) => {
            error!("Unable to run sync");
            error!("{:?}", e);
            return Err(());
        }
    };
    fs::write("/proc/sys/vm/drop_caches", "1").unwrap();
    Ok(())
}

pub struct Box2D {
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
    pub index: i32,
    #[doc = " label for this detection"]
    pub label: String,
    #[doc = " timestamp"]
    pub ts: u64,
    #[doc = " tracking information for this detection"]
    pub track: Option<Track>,
}

pub struct Track {
    #[doc = " track UUID for this detection"]
    pub uuid: Uuid,
    #[doc = " number of detects this track has been seen for"]
    pub count: i32,
    #[doc = " when this track was first added"]
    pub created: u64,
}
fn vaalbox_to_box2d(
    args: &Args,
    b: &VAALBox,
    model: &Context,
    ts: u64,
    track: Option<&Tracklet>,
) -> Box2D {
    let label_ind = b.label + args.label_offset;
    let label = match model.label(label_ind) {
        Ok(s) => String::from(s),
        Err(_) => b.label.to_string(),
    };

    trace!("Created box with label {}", label);
    let track_info = track.as_ref().map(|v| Track {
        uuid: v.id,
        count: v.count,
        created: v.created,
    });

    Box2D {
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
    rx: Receiver<bool>,
    stream_dims: (f64, f64),
) -> Subscriber<FifoChannelHandler<Sample>> {
    let model_path = args.model.clone();
    let model_type = ModelType {
        segment_output_ind: None,
        detection: false,
    };
    let mut model_info_msg =
        build_model_info_msg(time_from_ns(0u32), None, None, &model_path, &model_type);
    let status = format!("Loading Model: {}", model_path.to_string_lossy());

    loop {
        match rx.try_recv() {
            Ok(_) => return sub_camera,
            Err(e) => match e {
                TryRecvError::Disconnected => return sub_camera,
                TryRecvError::Empty => (),
            },
        }
        let _ = sub_camera.drain();
        let mut dma_buf: DmaBuf = match sub_camera.recv_timeout(Duration::from_millis(100)) {
            Ok(v) => match cdr::deserialize(&v.unwrap().payload().to_bytes()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
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
                    "Error getting Camera DMA file descriptor, please check if current process is running with same permissions as camera: {:?}",
                    e
                );
                continue;
            }
        };
        dma_buf.fd = fd.as_raw_fd();
        trace!("Opened DMA buffer from camera");
        let curr_time = match clock_gettime(ClockId::CLOCK_MONOTONIC) {
            Ok(t) => t.num_nanoseconds() as u64,
            Err(e) => {
                error!("Could not get Monotonic clock time: {:?}", e);
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

fn identify_model(model: &Context) -> Result<ModelType, vaal::Error> {
    let output_count = model.output_count()?;
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
        let output = match model.output_tensor(i) {
            Some(v) => v,
            None => {
                continue;
            }
        };
        let shape = output.shape();
        if shape.len() != 4 {
            continue;
        }
        if shape[1] < 8 {
            continue;
        }
        if shape[2] < 8 {
            continue;
        }
        segmentation_index.push(i);
    }
    if segmentation_index.len() > 1 {
        error!("Found more than 1 valid segmentation output tensors");
    }
    if !segmentation_index.is_empty() {
        model_type.segment_output_ind = Some(segmentation_index[0]);
        info!("Model has segmentation output");
    }

    // if there are any leftover outputs, assume it is a detection model and
    // vaal_boxes will decode it
    if segmentation_index.len() < output_count as usize {
        model_type.detection = true;
        info!("Model has detection output");
    }

    Ok(model_type)
}

async fn mask_thread(
    rx: Receiver<Mask>,
    publ_mask: Publisher<'_>,
    publ_mask_compressed: Option<Publisher<'_>>,
) {
    loop {
        let msg = match rx.recv() {
            Ok(v) => v,
            Err(_) => return,
        };

        let mut msg_compressed = match publ_mask_compressed.is_some() {
            true => Some(msg.clone()),
            false => None,
        };

        let mask_span = info_span!("mask_publish");
        let mask_task = async {
            let buf = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&msg, Infinite).unwrap());
            let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");

            match publ_mask.put(buf).encoding(enc).await {
                Ok(_) => trace!("Sent Mask message on {}", publ_mask.key_expr()),
                Err(e) => {
                    error!("Error sending message on {}: {:?}", publ_mask.key_expr(), e)
                }
            }
        }
        .instrument(mask_span);

        let mask_compressed_span = info_span!("mask_compressed_publish");
        let mask_compressed_task = async {
            if let Some(msg_compressed) = &mut msg_compressed {
                let publ_mask_compressed = publ_mask_compressed.as_ref().unwrap();
                let compression_start = Instant::now();
                msg_compressed.mask = zstd::bulk::compress(&msg_compressed.mask, 1).unwrap();
                msg_compressed.encoding = "zstd".to_string();
                trace!("Compression takes {:?}", compression_start.elapsed());

                let serialization_start = Instant::now();
                let buf =
                    ZBytes::from(cdr::serialize::<_, _, CdrLe>(&msg_compressed, Infinite).unwrap());
                let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Mask");
                trace!("Serialization takes {:?}", serialization_start.elapsed());

                let publ_start = Instant::now();
                match publ_mask_compressed.put(buf).encoding(enc).await {
                    Ok(_) => trace!("Sent Mask message on {}", publ_mask_compressed.key_expr()),
                    Err(e) => {
                        error!(
                            "Error sending message on {}: {:?}",
                            publ_mask_compressed.key_expr(),
                            e
                        )
                    }
                }
                trace!("Msg sending took {:?}", publ_start.elapsed());
            }
        }
        .instrument(mask_compressed_span);

        tokio::join!(mask_task, mask_compressed_task);
    }
}
