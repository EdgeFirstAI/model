// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use cdr::{CdrLe, Infinite};
use clap::Parser;
use edgefirst_model::{
    args::Args,
    buildmsgs::{
        build_detect_msg_and_encode, build_image_annotations_msg_and_encode, build_model_info_msg,
        build_segmentation_msg, time_from_ns,
    },
    get_curr_time, heart_beat, identify_model,
    image::ImageManager,
    masks::{mask_compress_thread, mask_thread},
    model::{self, DetectBox, Model, SupportedModel},
    run_detection,
    tflite_model::{DEFAULT_NPU_DELEGATE_PATH, TFLiteLib},
    tracker::ByteTrack,
    wait_for_camera_frame,
};
use edgefirst_schemas::sensor_msgs::CameraInfo;
use log::{error, info, trace, warn};
use std::{
    process::ExitCode,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{Layer, Registry, layer::SubscriberExt};
use tracy_client::frame_mark;
use zenoh::bytes::{Encoding, ZBytes};

#[tokio::main]
pub async fn main() -> ExitCode {
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
                use edgefirst_model::rtm_model::RtmModel;

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
    info!("identified model type: {model_type:?}");
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
    let mut fps = edgefirst_model::fps::Fps::<90>::default();

    let img_mgr = match ImageManager::new() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open G2D: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    info!("Opened G2D with version {}", img_mgr.version());

    loop {
        let Some(dma_buf) = wait_for_camera_frame(&sub_camera, timeout) else {
            continue;
        };
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

        if model_type.detection {
            let timestamp = dma_buf.header.stamp.nanosec as u64
                + dma_buf.header.stamp.sec as u64 * 1_000_000_000;
            let (new_boxes, _protos) = run_detection(
                &mut model,
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
            let curr_time = get_curr_time();

            // let boxes: Vec<DetectBox2D> = new_boxes.iter().map(|b| b.into()).collect();

            // if model_type.detection_with_mask
            //     && let Some(protos) = _protos
            // {
            //     new_boxes.sort_by_key(|a| {
            //         if let Some(track) = a.track {
            //             track.last_updated
            //         } else {
            //             a.ts
            //         }
            //     });
            // let masks = decode_masks(&new_boxes, protos.view());
            // let masks = masks.into_iter().filter_map(|x| {
            //     x.map(|v| Mask {
            //         height: v.0.shape()[0] as u32,
            //         width: v.0.shape()[1] as u32,
            //         length: 1,
            //         encoding: "".to_string(),
            //         mask: v.0.into_raw_vec_and_offset().0,
            //         boxed: true,
            //     })
            // });
            // }

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
        if let Some(i) = model_type.segment_output_ind {
            let masks = build_segmentation_msg(dma_buf.header.stamp.clone(), Some(&model), i);
            match mask_tx.send(masks).await {
                Ok(_) => {}
                Err(e) => {
                    error! {"Cannot send to mask publishing thread {e:?}"};
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
