// use serde::{Deserialize, Serialize};
use crate::setup::*;
mod setup;
use async_pidfd::PidFd;
use cdr::{CdrLe, Infinite};
use clap::Parser;
use pidfd_getfd::{get_file_from_pidfd, GetFdFlags};
use std::{error::Error, fs, os::fd::AsRawFd, process::Command, str::FromStr, time::Duration};
use vaal::{self, Context, VAALBox};
use zenoh::{config::Config, prelude::r#async::*};
use zenoh_ros_type::{
    builtin_interfaces::Time,
    deepview_msgs::DeepviewDMABuf,
    foxglove_msgs::{
        point_annotation_type::LINE_LOOP, FoxgloveColor, FoxgloveImageAnnotations, FoxglovePoint2,
        FoxglovePointAnnotations, FoxgloveTextAnnotations,
    },
};
#[macro_export]
macro_rules! log {
	($verbose: expr, $( $args:expr ),*) => {
		if $verbose {eprintln!( $( $args ),* );}
	}
}
#[async_std::main]
async fn main() {
    let s = Settings::parse();

    let mut backbone = Context::new(&s.engine).unwrap();
    backbone.load_model_file(s.model.to_str().unwrap()).unwrap();
    if s.verbose {
        eprintln!("Loaded backbone model {}", s.model.to_str().unwrap());
    }
    let mut decoder = None;

    if s.decoder_model.is_some() {
        let decoder_device = "cpu";
        let mut _decoder = Context::new(decoder_device).unwrap();

        setup_context(&mut _decoder, &s);
        let decoder_file = s.decoder_model.as_ref().unwrap();
        _decoder
            .load_model_file(decoder_file.to_str().unwrap())
            .unwrap();
        if s.verbose {
            eprintln!("Loaded decoder model {}", decoder_file.to_str().unwrap());
        }
        decoder = Some(_decoder);
    } else {
        setup_context(&mut backbone, &s);
    }

    let mut config = Config::default();

    let mode = WhatAmI::from_str(&s.mode).unwrap();
    config.set_mode(Some(mode)).unwrap();
    // config.connect.endpoints = s.endpoints.iter().map(|v|
    // v.parse().unwrap()).collect();

    let session = zenoh::open(config).res_async().await.unwrap();

    log!(s.verbose, "Opened Zenoh session");

    let subscriber = session
        .declare_subscriber(&s.camera_topic)
        .res()
        .await
        .unwrap();

    let res_topic: Vec<OwnedKeyExpr> = vec!["width", "height"]
        .into_iter()
        .map(|x| keyexpr::new(&s.res_topic).unwrap().join(x).unwrap())
        .collect();
    let width_sub = session
        .declare_subscriber(&res_topic[0])
        .res()
        .await
        .unwrap();
    let height_sub = session
        .declare_subscriber(&res_topic[1])
        .res()
        .await
        .unwrap();
    log!(s.verbose, "Declared subscriber on {:?}", &s.camera_topic);

    let mut err = false;
    let mut stream_width = match width_sub.recv_timeout(Duration::from_secs(2)) {
        Ok(v) => match i32::try_from(v.value) {
            Ok(val) => val,
            Err(_) => {
                err = true;
                1
            }
        },
        Err(_) => {
            err = true;
            1
        }
    } as f64;

    let mut stream_height = match height_sub.recv_timeout(Duration::from_secs(2)) {
        Ok(v) => match i32::try_from(v.value) {
            Ok(val) => val,
            Err(_) => {
                err = true;
                1
            }
        },
        Err(_) => {
            err = true;
            1
        }
    } as f64;
    if err {
        eprintln!("Cannot determine stream resolution, using normalized coordinates");
        stream_height = 1.0;
        stream_width = 1.0;
    };
    drop(height_sub);
    drop(width_sub);
    let mut vaal_boxes: Vec<vaal::VAALBox> = Vec::with_capacity(s.max_boxes as usize);
    loop {
        let dma_buf: DeepviewDMABuf = match subscriber.recv_timeout(Duration::from_secs(1)) {
            Ok(v) => cdr::deserialize(&mut v.payload.contiguous())
                .expect("Failed to deserialize message"),
            Err(e) => {
                eprintln!("Error when recv camera frame {:?}", e);
                continue;
            }
        };
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
                eprintln!("{:?}", e);
                return;
            }
        };
        log!(s.verbose, "Detected {:?} boxes", boxes.len());
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
                    .res_async()
                    .await
                    .unwrap();
            }
            Err(e) => eprintln!("{e:?}"),
        }
    }
}

#[inline(always)]
fn run_model(
    s: &Settings,
    dma_fd: &DeepviewDMABuf,
    backbone: &vaal::Context,
    decoder: &mut Option<vaal::Context>,
    boxes: &mut Vec<vaal::VAALBox>,
    stream_width: f64,
    stream_height: f64,
) -> Result<Vec<Box2D>, String> {
    let pidfd: PidFd = match PidFd::from_pid(dma_fd.src_pid as i32) {
        Ok(v) => v,
        Err(e) => return Err(e.to_string()),
    };
    let fd = match get_file_from_pidfd(pidfd.as_raw_fd(), dma_fd.dma_fd, GetFdFlags::empty()) {
        Ok(v) => v,
        Err(e) => return Err(e.to_string()),
    };
    match backbone.load_frame_dmabuf(
        None,
        fd.as_raw_fd(),
        dma_fd.fourcc,
        dma_fd.width as i32,
        dma_fd.height as i32,
        None,
        0,
    ) {
        Err(vaal::Error::VAALError(e)) => {
            //possible vaal error that we can handle
            let poss_err = "attempted an operation which is unsupported on the current platform";
            if e == poss_err {
                eprintln!(
                    "Attemping to clear cache,\
						   likely due to g2d alloc fail,\
						   this should be fixed in VAAL"
                );
                match clear_cached_memory() {
                    Ok(()) => eprintln!("Cleared cached memory"),
                    Err(()) => eprintln!("Could not clear cached memory"),
                }
            } else {
                panic!("Could not clear cache exiting");
            }

            if let Err(e) = backbone.load_frame_dmabuf(
                None,
                dma_fd.dma_fd,
                dma_fd.fourcc,
                dma_fd.width as i32,
                dma_fd.height as i32,
                None,
                0,
            ) {
                panic!("{:?}", e);
            }
        }
        Err(_) => panic!("load_frame_dmabuf error"),
        Ok(_) => {}
    };

    if let Err(e) = backbone.run_model() {
        return Err(format!("failed to run backbone: {}", e));
    }
    let n_boxes;

    if decoder.is_some() {
        let decoder_: &mut Context = decoder.as_mut().unwrap();
        let model = match decoder_.model() {
            Ok(model) => model,
            Err(e) => return Err(e.to_string()),
        };

        let inputs_idx = match model.inputs() {
            Ok(inputs) => inputs,
            Err(e) => return Err(e.to_string()),
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
            eprintln!(
                "failed to copy backbone out_1 ({:?}) to decoder in_1 ({:?}):{}",
                out_1.tensor_type(),
                in_1.tensor_type(),
                e
            );
        }

        let in_2 = context.tensor_index_mut(in_2_idx as usize).unwrap();

        if let Err(e) = out_2.dequantize(in_2) {
            eprintln!(
                "failed to copy backbone out_2 ({:?}) to decoder in_2 ({:?}):{}",
                out_2.tensor_type(),
                in_2.tensor_type(),
                e
            );
        }

        if let Err(e) = decoder_.run_model() {
            return Err(e.to_string());
        }

        n_boxes = match decoder_.boxes(boxes, boxes.capacity()) {
            Ok(len) => len,
            Err(e) => {
                return Err(format!("failed to read bounding boxes from model: {:?}", e));
            }
        };
    } else {
        n_boxes = match backbone.boxes(boxes, boxes.capacity()) {
            Ok(len) => len,
            Err(e) => {
                return Err(format!("failed to read bounding boxes from model: {:?}", e));
            }
        };
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
                    eprintln!("sync command failed");
                    eprintln!("stdout {:?}", output.stdout);
                    eprintln!("stderr {:?}", output.stderr);
                    return Err(());
                }
            };
        }
        Err(e) => {
            eprintln!("Unable to run sync");
            eprintln!("{:?}", e);
            return Err(());
        }
    };
    fs::write("/proc/sys/vm/drop_caches", "1").unwrap();
    Ok(())
}

pub struct Box2D {
    #[doc = " left-most normalized coordinate of the bounding box."]
    pub xmin: f64,
    #[doc = " top-most normalized coordinate of the bounding box."]
    pub ymin: f64,
    #[doc = " right-most normalized coordinate of the bounding box."]
    pub xmax: f64,
    #[doc = " bottom-most normalized coordinate of the bounding box."]
    pub ymax: f64,
    #[doc = " model-specific score for this detection, higher implies more confidence."]
    pub score: f64,
    #[doc = " label index for this detection"]
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
