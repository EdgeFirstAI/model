// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use clap::Parser;
use edgefirst_hal::decoder::DecoderBuilder;
use edgefirst_model::{
    args::Args,
    buildmsgs::{
        build_detect_msg_and_encode_, build_image_annotations_msg_and_encode_,
        build_model_info_msg, build_segmentation_msg_, time_from_ns,
    },
    heart_beat,
    masks::{mask_compress_thread, mask_thread},
    model::{self, Model, SupportedModel, guess_model_config},
    tflite_model::{DEFAULT_NPU_DELEGATE_PATH, TFLiteLib},
    update_dmabuf_with_pidfd, wait_for_camera_frame,
};
use edgefirst_schemas::{
    edgefirst_msgs::ModelInfo, schema_registry::SchemaType, sensor_msgs::CameraInfo, serde_cdr,
};
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
                match serde_cdr::deserialize::<CameraInfo>(&v.unwrap().payload().to_bytes()) {
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

    let mut tracker = edgefirst_tracker::bytetrack::ByteTrack::new();
    tracker.track_extra_lifespan = (args.track_extra_lifespan * 1_000_000_000.0) as u64;
    tracker.track_high_conf = args.track_high_conf;
    tracker.track_iou = args.track_iou;
    tracker.track_update = args.track_update;

    let mut decoder_builder = DecoderBuilder::new()
        .with_score_threshold(args.threshold)
        .with_iou_threshold(args.iou);
    if let Some(path) = args.edgefirst_config {
        let config = match std::fs::read_to_string(&path) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not read edgefirst config file: {e:?}");
                return ExitCode::FAILURE;
            }
        };
        match path.extension() {
            Some(v) if v == "yaml" || v == "yml" => {
                decoder_builder = decoder_builder.with_config_yaml_str(config);
            }
            Some(v) if v == "json" => {
                decoder_builder = decoder_builder.with_config_json_str(config);
            }
            Some(v) => {
                error!(
                    "Unsupported edgefirst config file extension {}",
                    v.display()
                );
                return ExitCode::FAILURE;
            }
            None => {
                error!("No edgefirst config file extension");
                return ExitCode::FAILURE;
            }
        }
    } else if let Some(yaml) = model.get_model_metadata().unwrap().config_yaml {
        decoder_builder = decoder_builder.with_config_yaml_str(yaml);
    } else {
        warn!(
            "No edgefirst config provided and none found in model metadata, guessing config based on model shape"
        );
        let output_count = match model.output_count() {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get model output count: {e:?}");
                return ExitCode::FAILURE;
            }
        };

        let output_shapes: Result<Vec<Vec<usize>>, _> =
            (0..output_count).map(|i| model.output_shape(i)).collect();

        let output_shapes = match output_shapes {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get model output shapes: {e:?}");
                return ExitCode::FAILURE;
            }
        };

        let output_quants = (0..output_count)
            .map(|i| model.output_quantization(i))
            .collect::<Result<Vec<_>, _>>();

        let output_quants = match output_quants {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get model output quantization: {e:?}");
                return ExitCode::FAILURE;
            }
        };

        let config = guess_model_config(&output_shapes, &output_quants);
        info!("Model has shape: {:?}", output_shapes);
        if let Some(cfg) = config {
            info!(
                "A config file was not provided. Guessed model config based on model shape: {:?}",
                cfg,
            );

            decoder_builder = decoder_builder.with_config(cfg);
        } else {
            error!("Could not guess model config from output shapes: {output_shapes:?}");
            return ExitCode::FAILURE;
        }
    }

    let decoder = match decoder_builder.build() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not build decoder: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    let model_type_ = decoder.model_type();
    let (has_box, has_seg, _has_instance_seg) = match model_type_ {
        edgefirst_hal::decoder::configs::ModelType::ModelPackSegDet { .. } => (true, true, false),
        edgefirst_hal::decoder::configs::ModelType::ModelPackSegDetSplit { .. } => (true, true, false),
        edgefirst_hal::decoder::configs::ModelType::ModelPackDet { .. } => (true, false, false),
        edgefirst_hal::decoder::configs::ModelType::ModelPackDetSplit { .. } => (true, false, false),
        edgefirst_hal::decoder::configs::ModelType::ModelPackSeg { .. } => (false, true, false),
        edgefirst_hal::decoder::configs::ModelType::YoloDet { .. } => (true, false, false),
        edgefirst_hal::decoder::configs::ModelType::YoloSegDet { .. } => (true, false, true),
        edgefirst_hal::decoder::configs::ModelType::YoloSplitDet { .. } => (true, false, false),
        edgefirst_hal::decoder::configs::ModelType::YoloSplitSegDet { .. } => (true, false, true),
        edgefirst_hal::decoder::configs::ModelType::YoloEndToEndDet { .. } => (true, false, false),
        edgefirst_hal::decoder::configs::ModelType::YoloEndToEndSegDet { .. } => (true, false, true),
    };

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

    let mut model_info_msg = build_model_info_msg(
        time_from_ns(0u32),
        Some(&model),
        &args.model,
        has_box,
        has_seg | _has_instance_seg,
    );
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

    let timeout = Duration::from_millis(100);
    let mut fps = edgefirst_model::fps::Fps::<90>::default();

    let mut img_proc = match edgefirst_hal::image::ImageProcessor::new() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open ImageProcessor: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    let mut output_boxes = Vec::with_capacity(50);
    let mut output_masks = Vec::with_capacity(50);
    let mut output_tracks = Vec::with_capacity(50);
    loop {
        let Some(mut dma_buf) = wait_for_camera_frame(&sub_camera, timeout) else {
            continue;
        };
        trace!("Received camera frame");

        let input_start = Instant::now();
        // the _fd needs to remain valid while `dma_buf`` is used
        let _fd = match update_dmabuf_with_pidfd(&mut dma_buf) {
            Ok(fd) => fd,
            Err(e) => {
                error!("Could not update dma_buf with pidfd. Are you running with sudo? {e:?}");
                return ExitCode::FAILURE;
            }
        };

        match model.load_frame_dmabuf_(&dma_buf, &mut img_proc, model::Preprocessing::Raw) {
            Ok(_) => trace!("Loaded frame into model"),
            Err(e) => error!("Could not load frame into model: {e:?}"),
        }

        let input_duration = input_start.elapsed().as_nanos();
        trace!("Load input: {:.3} ms", input_duration as f32 / 1_000_000.0);

        let model_start = Instant::now();

        if let Err(e) = model.run_model() {
            error!("Failed to run model: {e:?}");
            return ExitCode::FAILURE;
        }
        let model_duration = model_start.elapsed().as_nanos();
        trace!("Ran model: {:.3} ms", model_duration as f32 / 1_000_000.0);
        let output_start = Instant::now();

        let res = model.decode_outputs(&decoder, &mut output_boxes, &mut output_masks);

        if res.is_ok() && args.track {
            use edgefirst_model::TrackerBox;
            use edgefirst_tracker::Tracker;
            let timestamp = dma_buf.header.stamp.nanosec as u64
                + dma_buf.header.stamp.sec as u64 * 1_000_000_000;
            let wrapped: Vec<_> = output_boxes.iter().map(TrackerBox).collect();
            let tracks = tracker.update(&wrapped, timestamp);
            output_tracks.clear();
            output_tracks.extend(tracks.into_iter().flatten());
        }

        if let Err(e) = res {
            error!("Failed to decode model outputs: {e:?}");
            return ExitCode::FAILURE;
        }

        let output_duration = output_start.elapsed();

        if first_run {
            info!("First run complete. Found {} boxes", output_boxes.len());
            first_run = false;
        }

        if has_seg {
            let masks = build_segmentation_msg_(dma_buf.header.stamp.clone(), &output_masks);
            match mask_tx.send(masks).await {
                Ok(_) => {}
                Err(e) => {
                    error!("Cannot send to mask publishing thread {e:?}");
                }
            }
        }

        if has_box {
            let (msg, enc) = build_detect_msg_and_encode_(
                &output_boxes,
                &output_tracks,
                &model_labels,
                dma_buf.header.clone(),
                time_from_ns(input_duration),
                time_from_ns(model_duration),
                time_from_ns(output_duration.as_nanos()),
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
        }

        if has_box && let Some(publ_visual) = publ_visual.as_ref() {
            let (msg, enc) = build_image_annotations_msg_and_encode_(
                &output_boxes,
                &output_tracks,
                &model_labels,
                dma_buf.header.stamp.clone(),
                (stream_width, stream_height),
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

        model_info_msg.header.stamp = dma_buf.header.stamp.clone();
        let msg = ZBytes::from(serde_cdr::serialize(&model_info_msg).unwrap());
        let enc = Encoding::APPLICATION_CDR.with_schema(ModelInfo::SCHEMA_NAME);

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
        fps.update();

        args.tracy.then(frame_mark);
    }
}
