// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use clap::Parser;
use edgefirst_hal::decoder::DecoderBuilder;
use edgefirst_hal::image::{
    Crop, Flip, ImageProcessor, ImageProcessorTrait, PLANAR_RGB, PLANAR_RGB_INT8, RGB, Rotation,
};
use edgefirst_hal::tensor::{TensorMapTrait, TensorTrait};
use edgefirst_model::{
    args::Args,
    buildmsgs::{
        build_detect_msg_and_encode_, build_image_annotations_msg_and_encode_,
        build_model_info_msg, build_model_output_msg, build_segmentation_msg_, time_from_ns,
    },
    heart_beat,
    masks::mask_thread,
    model::{ModelContext, decode_outputs, dmabuf_to_tensor_image, guess_model_config},
    runtime, update_dmabuf_with_pidfd, wait_for_camera_frame,
};
use edgefirst_schemas::{
    edgefirst_msgs::{Model as ModelMsg, ModelInfo},
    schema_registry::SchemaType,
    sensor_msgs::CameraInfo,
    serde_cdr,
};
use log::{error, info, trace, warn};
use std::{
    process::ExitCode,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use tokio::sync::mpsc;
use tracing::info_span;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{Layer, Registry, layer::SubscriberExt};
use tracy_client::frame_mark;
use zenoh::bytes::{Encoding, ZBytes};

/// Global shutdown flag for graceful termination.
/// This is critical for coverage instrumentation - LLVM uses atexit() handlers
/// to flush profraw files, so the process must exit cleanly (not via SIGKILL).
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_signal(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    unsafe {
        libc::signal(
            libc::SIGTERM,
            handle_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGINT,
            handle_signal as *const () as libc::sighandler_t,
        );
    }
}

#[tokio::main]
pub async fn main() -> ExitCode {
    install_signal_handlers();

    let args = Args::parse();

    args.tracy.then(tracy_client::Client::start);

    let env_filter = || {
        tracing_subscriber::EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy()
    };

    let stdout_log = tracing_subscriber::fmt::layer()
        .pretty()
        .with_filter(env_filter());

    let journald = tracing_journald::layer()
        .ok()
        .map(|j| j.with_filter(env_filter()));

    let tracy = if args.tracy {
        Some(tracing_tracy::TracyLayer::default().with_filter(env_filter()))
    } else {
        None
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

    // ── Create runtime ──────────────────────────────────────────────────
    let mut runtime = match runtime::create_runtime(&args.model, &args.delegate) {
        Ok(r) => r,
        Err(e) => {
            error!("Could not create runtime: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let info = runtime.metadata().clone();

    let in_shape = runtime.input_shape(0);
    let in_h = in_shape.get(1).copied().unwrap_or(0);
    let in_w = in_shape.get(2).copied().unwrap_or(0);

    info!("Labels: {:?}", info.labels);

    // ── Build ModelContext ────────────────────────────────────────────────
    let model_ctx = ModelContext {
        input_shapes: (0..runtime.input_count())
            .map(|i| runtime.input_shape(i).to_vec())
            .collect(),
        input_types: (0..runtime.input_count())
            .map(|i| runtime.input_dtype(i))
            .collect(),
        output_shapes: (0..runtime.output_count())
            .map(|i| runtime.output_shape(i).to_vec())
            .collect(),
        output_types: (0..runtime.output_count())
            .map(|i| runtime.output_dtype(i))
            .collect(),
        labels: info.labels.clone(),
        name: info.name.clone().unwrap_or_default(),
    };

    let mut tracker = edgefirst_tracker::bytetrack::ByteTrack::new();
    tracker.track_extra_lifespan = (args.track_extra_lifespan * 1_000_000_000.0) as u64;
    tracker.track_high_conf = args.track_high_conf;
    tracker.track_iou = args.track_iou;
    tracker.track_update = args.track_update;

    // ── Build decoder ────────────────────────────────────────────────────
    let mut decoder_builder = DecoderBuilder::new()
        .with_score_threshold(args.threshold)
        .with_iou_threshold(args.iou);
    if let Some(path) = args.edgefirst_config() {
        let config = match std::fs::read_to_string(path) {
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
    } else if let Some(yaml) = &info.config_yaml {
        decoder_builder = decoder_builder.with_config_yaml_str(yaml.clone());
    } else {
        warn!("No edgefirst config provided, guessing config based on model shape");

        // For quantized YOLO models (e.g. ARA-2 DVM), box coordinate outputs
        // are dequantized to pixel-space values.  The HAL decoder expects
        // normalised [0,1] coordinates, so fold the 1/input_dim division
        // into the quantization scale for outputs that look like box coords
        // (shape contains a dimension == 4).
        let input_dim = in_w.max(in_h) as f32;
        let output_quants: Vec<Option<(f32, i32)>> = (0..runtime.output_count())
            .map(|i| {
                runtime.output_quantization(i).map(|q| {
                    let is_box_output = model_ctx.output_shapes[i].contains(&4);
                    let scale = if is_box_output && input_dim > 1.0 {
                        q.scale / input_dim
                    } else {
                        q.scale
                    };
                    (scale, q.zero_point)
                })
            })
            .collect();

        let config = guess_model_config(&model_ctx.output_shapes, &output_quants);
        info!("Model has shape: {:?}", model_ctx.output_shapes);
        if let Some(cfg) = config {
            info!("Guessed model config: {:?}", cfg);
            decoder_builder = decoder_builder.with_config(cfg);
        } else {
            error!(
                "Could not guess model config from output shapes: {:?}",
                model_ctx.output_shapes
            );
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
    let (has_box, has_seg, has_instance_seg) = {
        use edgefirst_hal::decoder::configs::ModelType::*;
        match model_type_ {
            ModelPackSegDet { .. } | ModelPackSegDetSplit { .. } => (true, true, false),
            ModelPackDet { .. } | ModelPackDetSplit { .. } => (true, false, false),
            ModelPackSeg { .. } => (false, true, false),
            YoloDet { .. }
            | YoloSplitDet { .. }
            | YoloEndToEndDet { .. }
            | YoloSplitEndToEndDet { .. } => (true, false, false),
            YoloSegDet { .. }
            | YoloSplitSegDet { .. }
            | YoloEndToEndSegDet { .. }
            | YoloSplitEndToEndSegDet { .. } => (true, false, true),
        }
    };

    drop(tx);

    let publ_model_info = session
        .declare_publisher(args.info_topic.clone())
        .await
        .unwrap();

    let publ_detect = if !args.detect_topic.is_empty() {
        info!("Legacy detect topic enabled: {}", args.detect_topic);
        Some(
            session
                .declare_publisher(args.detect_topic.clone())
                .await
                .unwrap(),
        )
    } else {
        info!("Legacy detect topic disabled (empty DETECT_TOPIC)");
        None
    };

    let publ_output = session
        .declare_publisher(args.output_topic.clone())
        .await
        .unwrap();

    let publ_visual = if args.visualization {
        Some(
            session
                .declare_publisher(args.visual_topic.clone())
                .await
                .unwrap(),
        )
    } else {
        None
    };

    let mask_tx = if !args.mask_topic.is_empty() {
        info!("Legacy mask topic enabled: {}", args.mask_topic);
        let publ_mask = session
            .declare_publisher(args.mask_topic.clone())
            .await
            .unwrap();
        let (mask_tx, mask_rx) = mpsc::channel(50);
        tokio::spawn(mask_thread(mask_rx, publ_mask));
        Some(mask_tx)
    } else {
        info!("Legacy mask topic disabled (empty MASK_TOPIC)");
        None
    };

    let mut model_info_msg = build_model_info_msg(
        time_from_ns(0u32),
        Some(&model_ctx),
        &args.model,
        has_box,
        has_seg | has_instance_seg,
    );
    info!("built model_info_msg");

    let sub_camera = heartbeat.await.unwrap();

    let model_name = args
        .model
        .file_name()
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_else(|| {
            warn!("Cannot determine model file basename");
            String::from("unknown_model_file")
        });
    info!("got model_name {model_name}");

    if !args.classes.is_empty() {
        info!("Class filter active: {:?}", args.classes);
    }

    let timeout = Duration::from_millis(100);
    let mut fps = edgefirst_model::fps::Fps::<90>::default();

    // ── ImageProcessor and destination image ─────────────────────────────
    let mut img_proc = match tokio::task::spawn_blocking(ImageProcessor::new)
        .await
        .unwrap()
    {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open ImageProcessor: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    let input_fourcc = runtime.input_fourcc(0);
    // G2D outputs interleaved RGB. For planar input models (ARA-2),
    // deinterleaving happens in the copy-to-input-tensor step below.
    let is_planar = matches!(input_fourcc, f if f == PLANAR_RGB || f == PLANAR_RGB_INT8);
    let dst_fourcc = if is_planar { RGB } else { input_fourcc };
    let mut dst_image = match img_proc.create_image(in_w, in_h, dst_fourcc) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not create destination image: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    info!("Destination image: {}x{} {:?}", in_w, in_h, dst_fourcc);

    let mut output_boxes = Vec::with_capacity(50);
    let mut output_masks = Vec::with_capacity(50);
    let mut output_tracks = Vec::with_capacity(50);
    while !SHUTDOWN.load(Ordering::SeqCst) {
        let Some(mut dma_buf) = ({
            let _span = info_span!("wait_for_camera_frame").entered();
            wait_for_camera_frame(&sub_camera, timeout)
        }) else {
            continue;
        };
        trace!("Received camera frame");

        // the _fd needs to remain valid while `dma_buf` is used
        let _fd = match update_dmabuf_with_pidfd(&mut dma_buf) {
            Ok(fd) => fd,
            Err(e) => {
                error!("Could not update dma_buf with pidfd: {e:?}");
                return ExitCode::FAILURE;
            }
        };

        let src_image = match dmabuf_to_tensor_image(&dma_buf) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not create source image: {e:?}");
                continue;
            }
        };

        let preprocess_start = std::time::Instant::now();

        {
            let _span = info_span!("preprocess").entered();
            if let Err(e) = img_proc.convert(
                &src_image,
                &mut dst_image,
                Rotation::None,
                Flip::None,
                Crop::no_crop(),
            ) {
                error!("Image conversion failed: {e:?}");
                continue;
            }
        }

        // Write preprocessed pixels to runtime's input tensor
        {
            let _span = info_span!("load_input").entered();
            let src_map = match dst_image.tensor().map() {
                Ok(v) => v,
                Err(e) => {
                    error!("Could not map destination image: {e:?}");
                    continue;
                }
            };
            let pixels = src_map.as_slice();
            let input = runtime.input_tensor(0);
            let mut dst_map = match input.map() {
                Ok(v) => v,
                Err(e) => {
                    error!("Could not map input tensor: {e:?}");
                    continue;
                }
            };
            let dst = dst_map.as_mut_slice();
            if is_planar {
                // Deinterleave RGB → planar CHW and convert uint8 → int8
                // via XOR 0x80 when input_fourcc is PLANAR_RGB_INT8
                let plane_size = in_w * in_h;
                let xor_mask = if input_fourcc == PLANAR_RGB_INT8 {
                    0x80u8
                } else {
                    0x00u8
                };
                for i in 0..plane_size {
                    dst[i] = pixels[i * 3] ^ xor_mask;
                    dst[plane_size + i] = pixels[i * 3 + 1] ^ xor_mask;
                    dst[2 * plane_size + i] = pixels[i * 3 + 2] ^ xor_mask;
                }
            } else {
                dst[..pixels.len()].copy_from_slice(pixels);
            }
        }

        let timing = {
            let _span = info_span!("invoke").entered();
            match runtime.invoke() {
                Ok(t) => t,
                Err(e) => {
                    error!("Failed to run model: {e:?}");
                    return ExitCode::FAILURE;
                }
            }
        };
        // Fold preprocessing time into input_time (no separate preprocess field)
        let preprocess_time = preprocess_start.elapsed();
        let input_duration = (preprocess_time + timing.input_time).as_nanos();
        let model_duration = timing.model_time.as_nanos();
        let output_duration = timing.output_time.as_nanos();

        let decode_start = std::time::Instant::now();
        output_boxes.clear();
        output_masks.clear();
        output_tracks.clear();
        let res = {
            let _span = info_span!("decode_outputs").entered();
            decode_outputs(
                runtime.as_ref(),
                &decoder,
                &mut output_boxes,
                &mut output_masks,
            )
        };

        if let Err(e) = res {
            error!("Failed to decode model outputs: {e:?}");
            continue;
        }

        if args.track {
            let _span = info_span!("tracker_update").entered();
            use edgefirst_model::TrackerBox;
            use edgefirst_tracker::Tracker;
            let timestamp = dma_buf.header.stamp.nanosec as u64
                + dma_buf.header.stamp.sec as u64 * 1_000_000_000;
            let wrapped: Vec<_> = output_boxes.iter().map(TrackerBox).collect();
            let tracks = tracker.update(&wrapped, timestamp);
            output_tracks.extend(tracks.into_iter().flatten());
        }

        if !args.classes.is_empty() {
            let keep: Vec<bool> = output_boxes
                .iter()
                .map(|b| {
                    info.labels
                        .get(b.label)
                        .map(|name| args.classes.iter().any(|c| c == name))
                        .unwrap_or(false)
                })
                .collect();

            let mut i = 0;
            output_boxes.retain(|_| {
                let k = keep[i];
                i += 1;
                k
            });

            if has_instance_seg {
                let mut i = 0;
                output_masks.retain(|_| {
                    let k = keep.get(i).copied().unwrap_or(false);
                    i += 1;
                    k
                });
            }

            let mut i = 0;
            output_tracks.retain(|_| {
                let k = keep.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
        }

        let decode_duration = decode_start.elapsed();

        if first_run {
            info!("First run complete. Found {} boxes", output_boxes.len());
            first_run = false;
        }

        let _pub_span = info_span!("zenoh_publish").entered();
        if has_seg && let Some(mask_tx) = mask_tx.as_ref() {
            let masks = build_segmentation_msg_(dma_buf.header.stamp.clone(), &output_masks);
            if let Err(e) = mask_tx.send(masks).await {
                error!("Cannot send to mask publishing thread {e:?}");
            }
        }

        if has_box && let Some(publ_detect) = publ_detect.as_ref() {
            let (msg, enc) = build_detect_msg_and_encode_(
                &output_boxes,
                &output_tracks,
                &info.labels,
                dma_buf.header.clone(),
                time_from_ns(input_duration),
                time_from_ns(model_duration),
                time_from_ns(decode_duration.as_nanos()),
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
                &info.labels,
                dma_buf.header.stamp.clone(),
                (stream_width, stream_height),
                &model_name,
                args.labels,
            );

            match publ_visual.put(msg).encoding(enc).await {
                Ok(_) => trace!("Sent message on {}", publ_visual.key_expr()),
                Err(e) => {
                    error!(
                        "Error sending message on {}: {:?}",
                        publ_visual.key_expr(),
                        e
                    )
                }
            }
        }

        let model_output = build_model_output_msg(
            &output_boxes,
            &output_tracks,
            &info.labels,
            &output_masks,
            dma_buf.header.clone(),
            input_duration,
            model_duration,
            output_duration,
            decode_duration.as_nanos(),
            has_instance_seg,
        );
        let msg = ZBytes::from(serde_cdr::serialize(&model_output).unwrap());
        let enc = Encoding::APPLICATION_CDR.with_schema(ModelMsg::SCHEMA_NAME);

        match publ_output.put(msg).encoding(enc).await {
            Ok(_) => trace!("Sent Model message on {}", publ_output.key_expr()),
            Err(e) => {
                error!(
                    "Error sending message on {}: {:?}",
                    publ_output.key_expr(),
                    e
                )
            }
        }

        model_info_msg.header.stamp = dma_buf.header.stamp.clone();
        let msg = ZBytes::from(serde_cdr::serialize(&model_info_msg).unwrap());
        let enc = Encoding::APPLICATION_CDR.with_schema(ModelInfo::SCHEMA_NAME);

        if let Err(e) = publ_model_info.put(msg).encoding(enc).await {
            error!(
                "Error sending message on {}: {:?}",
                publ_model_info.key_expr(),
                e
            );
        }
        fps.update();

        args.tracy.then(frame_mark);
    }

    info!("Shutting down gracefully");
    ExitCode::SUCCESS
}
