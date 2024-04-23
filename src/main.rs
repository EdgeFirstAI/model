// use serde::{Deserialize, Serialize};

mod buildmsgs;
mod kalman;
mod setup;
mod tracker;

use crate::{buildmsgs::*, setup::*, tracker::*};
use async_pidfd::PidFd;
use async_std::task::spawn;
use clap::Parser;
use edgefirst_schemas::{self, edgefirst_msgs::DmaBuf, sensor_msgs::CameraInfo};
use log::{debug, error, info, trace, warn};
use nix::{
    sys::time::TimeValLike,
    time::{clock_gettime, ClockId},
};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    fs,
    os::fd::AsRawFd,
    path::PathBuf,
    process::Command,
    str::FromStr,
    sync::mpsc::{self, Receiver, TryRecvError},
    time::{Duration, Instant},
};
use uuid::Uuid;
use vaal::{self, Context, VAALBox};
use zenoh::{
    config::Config,
    prelude::{r#async::*, sync::SyncResolve},
    publication::Publisher,
    subscriber::FlumeSubscriber,
};

mod fps;

const NSEC_PER_MSEC: i64 = 1_000_000;

#[async_std::main]
async fn main() {
    let mut s = Settings::parse();
    validate_settings(&mut s);
    env_logger::init();
    let mut first_run = true;

    let mut config = Config::default();

    let mode = WhatAmI::from_str(&s.mode).unwrap();
    config.set_mode(Some(mode)).unwrap();
    config.connect.endpoints = s.connect.iter().map(|v| v.parse().unwrap()).collect();
    config.listen.endpoints = s.listen.iter().map(|v| v.parse().unwrap()).collect();
    let _ = config.scouting.multicast.set_enabled(Some(true));
    let _ = config
        .scouting
        .multicast
        .set_interface(Some("lo".to_string()));
    let _ = config.scouting.gossip.set_enabled(Some(true));
    let session = match zenoh::open(config.clone()).res_async().await {
        Ok(v) => v,
        Err(e) => {
            error!("Error while opening Zenoh session: {:?}", e);
            return;
        }
    }
    .into_arc();
    info!("Opened Zenoh session");

    let publ_detect = match session
        .declare_publisher(s.detect_topic.clone())
        .res_async()
        .await
    {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error while declaring detection publisher {}: {:?}",
                s.detect_topic, e
            );
            return;
        }
    };

    let publ_model_info = match session
        .declare_publisher(s.info_topic.clone())
        .res_async()
        .await
    {
        Ok(v) => v,
        Err(e) => {
            error!(
                "Error while declaring detection publisher {}: {:?}",
                s.info_topic, e
            );
            return;
        }
    };

    let publ_visual = if s.visualization {
        match session
            .declare_publisher(s.visual_topic.clone())
            .res_async()
            .await
        {
            Ok(v) => Some(v),
            Err(e) => {
                error!(
                    "Error while declaring detection publisher {}: {:?}",
                    s.detect_topic, e
                );
                return;
            }
        }
    } else {
        None
    };

    let stream_width: f64;
    let stream_height: f64;
    if s.visualization {
        let info_sub = session
            .declare_subscriber(&s.camera_info_topic)
            .res_async()
            .await
            .unwrap();
        info!("Declared subscriber on {:?}", &s.camera_info_topic);
        match info_sub.recv_timeout(Duration::from_secs(10)) {
            Ok(v) => {
                match cdr::deserialize::<CameraInfo>(&v.payload.contiguous()) {
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
                warn!("Failed to receive on {:?}: {:?}", s.camera_info_topic, e);
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
    let sub_camera: FlumeSubscriber<'_> = session
        .declare_subscriber(&s.camera_topic)
        .res_async()
        .await
        .unwrap();
    info!("Declared subscriber on {:?}", &s.camera_topic);

    let (tx, rx) = mpsc::channel();
    let heartbeat = spawn(heart_beat(
        sub_camera,
        publ_detect.clone(),
        publ_model_info.clone(),
        publ_visual.clone(),
        rx,
        s.model.clone(),
        stream_width,
        stream_height,
    ));

    let mut backbone = match Context::new(&s.engine) {
        Ok(v) => {
            debug!("Opened VAAL Context on {}", s.engine);
            v
        }
        Err(e) => {
            error!("Could not open VAAL Context on {}, {:?}", s.engine, e);
            return;
        }
    };
    let filename = match s.model.to_str() {
        Some(v) => v,
        None => {
            error!(
                "Cannot use file {:?}, please use only utf8 characters in file path",
                s.model
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

    if s.decoder_model.is_some() {
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
        setup_context(&mut decoder_ctx, &s);
        let decoder_file = match s.decoder_model.as_ref().unwrap().to_str() {
            Some(v) => v,
            None => {
                error!(
                    "Cannot use file {:?}, please use only utf8 characters in file path",
                    s.decoder_model.as_ref().unwrap()
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
        decoder = Some(decoder_ctx);
    } else {
        setup_context(&mut backbone, &s);
    }

    drop(tx);

    let mut model_info_msg = build_model_info_msg(
        time_from_ns(0u32),
        Some(&mut backbone),
        decoder.as_mut(),
        &s.model,
    );
    let sub_camera = heartbeat.await;

    let model_name = match s.model.as_path().file_name() {
        Some(v) => String::from(v.to_string_lossy()),
        None => {
            warn!("Cannot determine model file basename");
            String::from("unknown_model_file")
        }
    };
    let mut tracker = ByteTrack::new();
    let mut vaal_boxes: Vec<vaal::VAALBox> = Vec::with_capacity(s.max_boxes as usize);
    let timeout = Duration::from_millis(100);
    let mut fps = fps::Fps::<90>::default();

    loop {
        let _ = sub_camera.drain();
        let mut dma_buf: DmaBuf = match sub_camera.recv_timeout(timeout) {
            Ok(v) => match cdr::deserialize(&v.payload.contiguous()) {
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
        let model_start = Instant::now();
        let n_boxes = match run_model(
            &dma_buf,
            &backbone,
            &mut decoder,
            &mut vaal_boxes,
            fps.update(),
        ) {
            Ok(boxes) => boxes,
            Err(e) => {
                error!("Failed to run model: {:?}", e);
                return;
            }
        };
        let model_duration = model_start.elapsed().as_nanos();

        if first_run {
            info!(
                "Successfully recieved camera frames and run model, found {:?} boxes",
                n_boxes
            );
            first_run = false;
        } else {
            trace!("Detected {:?} boxes", n_boxes);
        }
        let model = if decoder.is_some() {
            decoder.as_ref().unwrap()
        } else {
            &backbone
        };
        let timestamp =
            dma_buf.header.stamp.nanosec as u64 + dma_buf.header.stamp.sec as u64 * 1_000_000_000;
        let tracks = if s.track {
            tracker.update(&s, &mut vaal_boxes[0..n_boxes], timestamp)
        } else {
            vec![None; n_boxes]
        };

        let mut new_boxes: Vec<Box2D> = Vec::new();
        for (vaal_box, track_info) in vaal_boxes.iter().take(n_boxes).zip(tracks.iter()) {
            // when tracking is turned on, only send results for tracked boxes
            if s.track && track_info.is_none() {
                continue;
            }
            new_boxes.push(vaalbox_to_box2d(&s, vaal_box, model, timestamp, track_info));
        }
        let curr_time = match clock_gettime(ClockId::CLOCK_MONOTONIC) {
            Ok(t) => t.num_nanoseconds() as u64,
            Err(e) => {
                error!("Could not get Monotonic clock time: {:?}", e);
                0
            }
        };
        let detect = build_detect_msg_and_encode(
            &new_boxes,
            dma_buf.header.stamp.clone(),
            time_from_ns(model_duration),
            time_from_ns(curr_time),
        );
        match publ_detect.put(detect).res_async().await {
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
            let annotations = build_image_annotations_msg_and_encode(
                &new_boxes,
                dma_buf.header.stamp.clone(),
                stream_width,
                stream_height,
                &model_name,
                s.labels,
            );
            match publ_visual.put(annotations).res_async().await {
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

        let model_info =
            update_model_info_msg_and_encode(dma_buf.header.stamp.clone(), &mut model_info_msg);
        match publ_model_info.put(model_info).res_sync() {
            Ok(_) => (),
            Err(e) => {
                error!(
                    "Error sending message on {}: {:?}",
                    publ_model_info.key_expr(),
                    e
                )
            }
        }
    }
}

#[inline(always)]
fn run_model(
    dma_buf: &DmaBuf,
    backbone: &vaal::Context,
    decoder: &mut Option<vaal::Context>,
    boxes: &mut Vec<VAALBox>,
    fps: i32,
) -> Result<usize, String> {
    let start = vaal::clock_now();
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
            let poss_err = "attempted an operation which is unsupported on the current platform";
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
    };
    let load_ns = vaal::clock_now() - start;

    let start = vaal::clock_now();
    if let Err(e) = backbone.run_model() {
        return Err(format!("Failed to run model: {}", e));
    }
    let model_ns = vaal::clock_now() - start;
    trace!("Ran model inference");

    let mut copy_ns;
    let decoder_ns;
    let boxes_ns;
    let n_boxes;

    let start = vaal::clock_now();
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
        copy_ns = vaal::clock_now() - start;
        trace!("Copied backdone out_1 to decoder in_1");

        let start = vaal::clock_now();
        let in_2 = context.tensor_index_mut(in_2_idx as usize).unwrap();

        if let Err(e) = out_2.dequantize(in_2) {
            return Err(format!(
                "Failed to copy backbone out_2 ({:?}) to decoder in_2 ({:?}): {}",
                out_2.tensor_type(),
                in_2.tensor_type(),
                e
            ));
        }
        copy_ns += vaal::clock_now() - start;
        trace!("Copied backdone out_2 to decoder in_2");

        let start = vaal::clock_now();
        if let Err(e) = decoder_.run_model() {
            return Err(format!("Failed to run decoder model: {:?}", e));
        }
        decoder_ns = vaal::clock_now() - start;
        trace!("Ran decoder model inference");

        let start = vaal::clock_now();
        n_boxes = match decoder_.boxes(boxes, boxes.capacity()) {
            Ok(len) => len,
            Err(e) => {
                return Err(format!("Failed to read bounding boxes from model: {:?}", e));
            }
        };
        boxes_ns = vaal::clock_now() - start;
        trace!("Read bounding boxes from model");
        trace!(
            "Model: FPS: {:>3}, load: {:>2.3} ms, infer: {:>2.3} ms, copy: {:>2.3} ms, decode: {:>2.3} ms, boxes: {:>2.3} ms",
            fps,
            load_ns as f64 / NSEC_PER_MSEC as f64,
            model_ns as f64 / NSEC_PER_MSEC as f64,
            copy_ns as f64 / NSEC_PER_MSEC as f64,
            decoder_ns as f64 / NSEC_PER_MSEC as f64,
            boxes_ns as f64 / NSEC_PER_MSEC as f64,
        )
    } else {
        n_boxes = match backbone.boxes(boxes, boxes.capacity()) {
            Ok(len) => len,
            Err(e) => {
                return Err(format!("Failed to read bounding boxes from model: {:?}", e));
            }
        };
        boxes_ns = vaal::clock_now() - start;
        trace!("Read bounding boxes from model");
        trace!(
            "Model: FPS: {:>3}, load: {:>2.3} ms, infer: {:>2.3} ms, boxes: {:>2.3} ms",
            fps,
            load_ns as f64 / NSEC_PER_MSEC as f64,
            model_ns as f64 / NSEC_PER_MSEC as f64,
            boxes_ns as f64 / NSEC_PER_MSEC as f64,
        );
    }

    Ok(n_boxes)
}

fn setup_context(context: &mut Context, s: &Settings) {
    context
        .parameter_seti("max_detection", &[s.max_boxes])
        .unwrap();

    context
        .parameter_setf("score_threshold", &[s.threshold])
        .unwrap();

    context.parameter_setf("iou_threshold", &[s.iou]).unwrap();
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
    s: &Settings,
    b: &VAALBox,
    model: &Context,
    ts: u64,
    track: &Option<TrackInfo>,
) -> Box2D {
    let label_ind = b.label + s.label_offset;
    let label = match model.label(label_ind) {
        Ok(s) => String::from(s),
        Err(_) => b.label.to_string(),
    };

    trace!("Created box with label {}", label);
    let track_info = match track {
        None => None,
        Some(v) => Some(Track {
            uuid: v.uuid,
            count: v.count,
            created: v.created,
        }),
    };
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

async fn heart_beat<'a>(
    sub_camera: FlumeSubscriber<'a>,
    publ_detect: Publisher<'_>,
    publ_model_info: Publisher<'_>,
    publ_visual: Option<Publisher<'_>>,
    rx: Receiver<bool>,
    model_path: PathBuf,
    stream_width: f64,
    stream_height: f64,
) -> FlumeSubscriber<'a> {
    let mut model_info_msg = build_model_info_msg(time_from_ns(0u32), None, None, &model_path);
    let msg = format!("Loading Model: {}", model_path.to_string_lossy());
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
            Ok(v) => match cdr::deserialize(&v.payload.contiguous()) {
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
        let detect = build_detect_msg_and_encode(
            &Vec::new(),
            dma_buf.header.stamp.clone(),
            time_from_ns(0u32),
            time_from_ns(curr_time),
        );

        match publ_detect.put(detect).res_sync() {
            Ok(_) => (),
            Err(e) => {
                error!(
                    "Error sending message on {}: {:?}",
                    publ_detect.key_expr(),
                    e
                )
            }
        }

        let model_info =
            update_model_info_msg_and_encode(dma_buf.header.stamp.clone(), &mut model_info_msg);
        match publ_model_info.put(model_info).res_sync() {
            Ok(_) => (),
            Err(e) => {
                error!(
                    "Error sending message on {}: {:?}",
                    publ_model_info.key_expr(),
                    e
                )
            }
        }

        if publ_visual.is_some() {
            let publ_visual = publ_visual.as_ref().unwrap();
            let annotations = build_image_annotations_msg_and_encode(
                &Vec::new(),
                dma_buf.header.stamp.clone(),
                stream_width,
                stream_height,
                &msg,
                LabelSetting::Index,
            );

            match publ_visual.put(annotations).res_async().await {
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
}
