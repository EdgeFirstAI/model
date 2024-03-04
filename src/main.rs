// use serde::{Deserialize, Serialize};
use crate::setup::*;
mod setup;
use async_pidfd::PidFd;
use cdr::{CdrLe, Infinite};
use clap::Parser;
use log::{error, info, trace, warn};
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{
    error::Error,
    fs,
    os::fd::AsRawFd,
    process::Command,
    str::FromStr,
    time::{Duration, SystemTime},
};
use vaal::{self, Context, VAALBox};
use zenoh::{config::Config, prelude::r#async::*};
use zenoh_ros_type::{
    builtin_interfaces::Time,
    deepview_msgs::DeepviewDMABuf,
    foxglove_msgs::{
        point_annotation_type::LINE_LOOP, FoxgloveColor, FoxgloveImageAnnotations, FoxglovePoint2,
        FoxglovePointAnnotations, FoxgloveTextAnnotations,
    },
    sensor_msgs::CameraInfo,
};

const NSEC_PER_MSEC: i64 = 1_000_000;
const NSEC_PER_SEC: i64 = 1_000_000_000;

#[async_std::main]
async fn main() {
    let s = Settings::parse();
    env_logger::init();
    let mut first_run = true;

    let mut backbone = Context::new(&s.engine).unwrap();
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
    backbone.load_model_file(filename).unwrap();

    info!("Loaded backbone model {:?}", s.model);

    let mut decoder = None;

    if s.decoder_model.is_some() {
        let decoder_device = "cpu";
        let mut decoder_ctx = Context::new(decoder_device).unwrap();

        setup_context(&mut decoder_ctx, &s);
        let decoder_file = s.decoder_model.as_ref().unwrap();
        decoder_ctx
            .load_model_file(decoder_file.to_str().unwrap())
            .unwrap();
        info!("Loaded decoder model {}", decoder_file.to_str().unwrap());
        decoder = Some(decoder_ctx);
    } else {
        setup_context(&mut backbone, &s);
    }

    let mut config = Config::default();

    let mode = WhatAmI::from_str(&s.mode).unwrap();
    config.set_mode(Some(mode)).unwrap();
    config.connect.endpoints = s.connect.iter().map(|v| v.parse().unwrap()).collect();
    config.listen.endpoints = s.listen.iter().map(|v| v.parse().unwrap()).collect();
    let _ = config.scouting.multicast.set_enabled(Some(false));
    let _ = config.scouting.gossip.set_enabled(Some(false));
    let _ = config.scouting.set_timeout(Some(0));
    let session = zenoh::open(config.clone()).res_async().await.unwrap();
    info!("Opened Zenoh session");

    let info_sub = session
        .declare_subscriber(&s.info_topic)
        .res()
        .await
        .unwrap();
    info!("Declared subscriber on {:?}", &s.info_topic);

    let stream_width: f64;
    let stream_height: f64;
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
        Err(_) => {
            warn!("Timeout on {:?}", s.info_topic);
            warn!("Cannot determine stream resolution, using normalized coordinates");
            stream_width = 1.0;
            stream_height = 1.0;
        }
    }
    drop(info_sub);

    let subscriber = session
        .declare_subscriber(&s.camera_topic)
        .res()
        .await
        .unwrap();
    info!("Declared subscriber on {:?}", &s.camera_topic);

    let mut vaal_boxes: Vec<vaal::VAALBox> = Vec::with_capacity(s.max_boxes as usize);
    loop {
        let _ = subscriber.drain();
        let mut dma_buf: DeepviewDMABuf = match subscriber.recv_timeout(Duration::from_secs(1)) {
            Ok(v) => match cdr::deserialize(&v.payload.contiguous()) {
                Ok(v) => v,
                Err(e) => {
                    error!("Failed to deserialize message: {:?}", e);
                    continue;
                }
            },

            Err(_) => {
                error!(
                    "Timeout receiving camera frames from {:?}",
                    subscriber.key_expr().as_str()
                );
                continue;
            }
        };
        trace!("Recieved camera frame");

        let pidfd: PidFd = match PidFd::from_pid(dma_buf.src_pid as i32) {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "Error getting PID {:?}, please check if the camera process is running: {:?}",
                    dma_buf.src_pid, e
                );
                continue;
            }
        };
        let fd = match get_file_from_pidfd(pidfd.as_raw_fd(), dma_buf.dma_fd, GetFdFlags::empty()) {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "Error getting Camera DMA file descriptor, please check if current process is running with same permissions as camera: {:?}",
                    e
                );
                continue;
            }
        };
        dma_buf.dma_fd = fd.as_raw_fd();
        trace!("Opened DMA buffer from camera");

        let boxes = match run_model(
            &s,
            &dma_buf,
            &backbone,
            &mut decoder,
            &mut vaal_boxes,
            stream_width,
            stream_height,
        ) {
            Ok(boxes) => boxes,
            Err(e) => {
                error!("Failed to run model: {:?}", e);
                return;
            }
        };
        if first_run {
            info!(
                "Successfully recieved camera frames and run model, found {:?} boxes",
                boxes.len()
            );
            first_run = false;
        } else {
            trace!("Detected {:?} boxes", boxes.len());
        }

        let msg = build_image_annotations_msg(
            &boxes,
            dma_buf.header.stamp.clone(),
            stream_width,
            stream_height,
        );
        match msg {
            Ok(m) => {
                let encoded = cdr::serialize::<_, _, CdrLe>(&m, Infinite).unwrap();
                session
                    .put(&s.detect_topic, encoded)
                    .encoding(Encoding::WithSuffix(
                        KnownEncoding::AppOctetStream,
                        "foxglove_msgs/msg/ImageAnnotations".into(),
                    ))
                    .res_async()
                    .await
                    .unwrap();
            }
            Err(e) => error!("{e:?}"),
        }
    }
}

#[inline(always)]
fn run_model(
    s: &Settings,
    dma_buf: &DeepviewDMABuf,
    backbone: &vaal::Context,
    decoder: &mut Option<vaal::Context>,
    boxes: &mut Vec<vaal::VAALBox>,
    stream_width: f64,
    stream_height: f64,
) -> Result<Vec<Box2D>, String> {
    let fps = update_fps();
    let start = vaal::clock_now();
    match backbone.load_frame_dmabuf(
        None,
        dma_buf.dma_fd,
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
                return Err(String::from("Could not clear cache exiting"));
            }

            if let Err(e) = backbone.load_frame_dmabuf(
                None,
                dma_buf.dma_fd,
                dma_buf.fourcc,
                dma_buf.width as i32,
                dma_buf.height as i32,
                None,
                0,
            ) {
                return Err(format!("Could not load frame {:?}", e));
            }
            trace!("Loaded frame into model");
        }
        Err(_) => return Err(String::from("load_frame_dmabuf error")),
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
            let temp = in_2_idx;
            in_2_idx = in_1_idx;
            in_1_idx = temp;
        }

        let in_1 = context.tensor_index_mut(in_1_idx as usize).unwrap();

        if let Err(e) = out_1.dequantize(in_1) {
            return Err(format!(
                "Failed to copy backbone out_1 ({:?}) to decoder in_1 ({:?}):{}",
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
                "Failed to copy backbone out_2 ({:?}) to decoder in_2 ({:?}):{}",
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

    let model = if decoder.is_some() {
        decoder.as_ref().unwrap()
    } else {
        backbone
    };

    let mut new_boxes: Vec<Box2D> = Vec::new();
    for vaal_box in boxes.iter().take(n_boxes) {
        new_boxes.push(vaalbox_to_box2d(
            s,
            vaal_box,
            model,
            stream_width,
            stream_height,
        ));
    }

    Ok(new_boxes)
}

fn build_image_annotations_msg(
    boxes: &Vec<Box2D>,
    timestamp: Time,
    stream_width: f64,
    stream_height: f64,
) -> Result<FoxgloveImageAnnotations, Box<dyn Error>> {
    let mut annotations = FoxgloveImageAnnotations {
        circles: Vec::new(),
        points: Vec::new(),
        texts: Vec::new(),
    };
    let white = FoxgloveColor {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    let transparent = FoxgloveColor {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    for b in boxes.iter() {
        let outline_colors = vec![white.clone(), white.clone(), white.clone(), white.clone()];
        let points = vec![
            FoxglovePoint2 {
                x: b.xmin,
                y: b.ymin,
            },
            FoxglovePoint2 {
                x: b.xmax,
                y: b.ymin,
            },
            FoxglovePoint2 {
                x: b.xmax,
                y: b.ymax,
            },
            FoxglovePoint2 {
                x: b.xmin,
                y: b.ymax,
            },
        ];
        let points = FoxglovePointAnnotations {
            timestamp: timestamp.clone(),
            type_: LINE_LOOP,
            points,
            outline_color: white.clone(),
            outline_colors,
            fill_color: transparent.clone(),
            thickness: 2.0,
        };

        let text = FoxgloveTextAnnotations {
            timestamp: timestamp.clone(),
            text: b.label.clone(),
            position: FoxglovePoint2 {
                x: b.xmin as f64,
                y: b.ymin as f64,
            },
            font_size: 0.02 * stream_width.max(stream_height),
            text_color: white.clone(),
            background_color: transparent.clone(),
        };
        annotations.points.push(points);
        annotations.texts.push(text);
    }
    Ok(annotations)
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
                Some(code) if code == 0 => {}
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

fn update_fps() -> i32 {
    static mut PREVIOUS_TIME: Option<SystemTime> = None;
    static mut FPS_HISTORY: [i32; 30] = [0; 30];
    static mut FPS_INDEX: usize = 0;

    let timestamp = SystemTime::now();
    let frame_time = match unsafe { PREVIOUS_TIME } {
        Some(prev_time) => timestamp.duration_since(prev_time).unwrap(),
        None => timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap(),
    };
    unsafe {
        PREVIOUS_TIME = Some(timestamp);
    };
    unsafe {
        FPS_HISTORY[FPS_INDEX] = (NSEC_PER_SEC as u128 / frame_time.as_nanos()) as i32;
    };
    unsafe {
        FPS_INDEX = (FPS_INDEX + 1) % 30;
    };

    let mut fps = 0;
    unsafe {
        for fps_history in &FPS_HISTORY {
            fps += fps_history;
        }
    }
    fps /= 30;
    fps
}

pub struct Box2D {
    #[doc = " left-most pixel coordinate of the bounding box."]
    pub xmin: f64,
    #[doc = " top-most pixel coordinate of the bounding box."]
    pub ymin: f64,
    #[doc = " right-most pixel coordinate of the bounding box."]
    pub xmax: f64,
    #[doc = " bottom-most pixel coordinate of the bounding box."]
    pub ymax: f64,
    #[doc = " model-specific score for this detection, higher implies more confidence."]
    pub score: f64,
    #[doc = " label for this detection"]
    pub label: String,
}

fn vaalbox_to_box2d(
    s: &Settings,
    b: &VAALBox,
    model: &Context,
    stream_width: f64,
    stream_height: f64,
) -> Box2D {
    let label_ind = b.label + s.label_offset;
    let label = match s.labels {
        LabelSetting::Index => label_ind.to_string(),
        LabelSetting::Score => format!("{:.2}", b.score),
        LabelSetting::Label => match model.label(label_ind) {
            Ok(s) => String::from(s),
            Err(_) => b.label.to_string(),
        },
        LabelSetting::LabelScore => {
            format!(
                "{:?} {:?}",
                match model.label(label_ind) {
                    Ok(s) => String::from(s),
                    Err(_) => label_ind.to_string(),
                },
                format!("{:.2}", b.score)
            )
        }
    };

    Box2D {
        xmin: b.xmin as f64 * stream_width,
        ymin: b.ymin as f64 * stream_height,
        xmax: b.xmax as f64 * stream_width,
        ymax: b.ymax as f64 * stream_height,
        score: b.score as f64,
        label,
    }
}
