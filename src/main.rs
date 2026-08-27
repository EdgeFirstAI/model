// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use clap::Parser;
use edgefirst_hal::decoder::DecoderBuilder;
use edgefirst_hal::image::{
    ComputeBackend, Flip, ImageProcessor, ImageProcessorConfig, ImageProcessorTrait, Rotation,
};
use edgefirst_hal::tensor::{
    CpuAccess, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorTrait,
};
use edgefirst_model::{
    args::Args,
    buildmsgs::{
        build_detect_msg_and_encode_, build_image_annotations_msg_and_encode_,
        build_model_info_msg, build_model_output_msg, build_segmentation_msg_, time_from_ns,
    },
    heart_beat,
    letterbox::LetterboxTransform,
    masks::mask_thread,
    model::{ModelContext, camera_frame_to_tensor_dyn, decode_outputs, guess_model_config},
    runtime, wait_for_camera_frame,
};
use edgefirst_schemas::sensor_msgs::CameraInfo;
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

/// Copy a preprocessed image into the runtime's CPU staging input tensor.
///
/// For planar-input models (ARA-2) the interleaved RGB is deinterleaved into
/// `CHW` planes, applying the `u8 → i8` quantization XOR when the model input
/// is signed. For packed-input models the pixels are copied verbatim. Only
/// used when the runtime does not support DMA-BUF zero-copy input.
fn populate_input_tensor(
    runtime: &mut dyn runtime::Runtime,
    staging: &TensorDyn,
    in_w: usize,
    in_h: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let src = staging.as_u8().ok_or("staging tensor is not u8-typed")?;
    let src_map = src.map()?;
    let pixels = src_map.as_slice();

    let input_dtype = runtime.input_dtype(0);
    let is_planar = matches!(
        runtime.input_pixel_format(0),
        PixelFormat::PlanarRgb | PixelFormat::PlanarRgba
    );

    let input = runtime.input_tensor(0);
    let mut dst_map = input.map()?;
    let dst = dst_map.as_mut_slice();

    if is_planar {
        // Deinterleave RGB → planar CHW; XOR 0x80 converts u8 → i8 for
        // signed-quantized ARA-2 inputs.
        let plane_size = in_w * in_h;
        let xor_mask = if input_dtype == DType::I8 {
            0x80u8
        } else {
            0x00u8
        };
        for i in 0..plane_size {
            dst[i] = pixels[i * 3] ^ xor_mask;
            dst[plane_size + i] = pixels[i * 3 + 1] ^ xor_mask;
            dst[2 * plane_size + i] = pixels[i * 3 + 2] ^ xor_mask;
        }
    } else if pixels.len() != dst.len() {
        return Err(format!(
            "input tensor size mismatch: staging={} bytes, input_tensor={} bytes \
             — check input_pixel_format vs staging buffer geometry",
            pixels.len(),
            dst.len()
        )
        .into());
    } else {
        dst.copy_from_slice(pixels);
    }
    Ok(())
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
        info!("Declared subscriber on {:?}", args.camera_info_topic);
        match info_sub.recv_timeout(Duration::from_secs(10)) {
            Ok(v) => {
                match CameraInfo::from_cdr(v.unwrap().payload().to_bytes()) {
                    Ok(v) => {
                        stream_width = v.width() as f64;
                        stream_height = v.height() as f64;
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
    info!("Declared subscriber on {:?}", args.camera_topic);

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

    let mut tracker = edgefirst_tracker::ByteTrackBuilder::new()
        .track_extra_lifespan((args.track_extra_lifespan * 1_000_000_000.0) as u64)
        .track_high_conf(args.threshold)
        .track_iou(args.track_iou)
        .track_update(args.track_update)
        .build::<edgefirst_model::TrackerBox>();

    if args.track && args.track_score >= args.threshold {
        warn!(
            "--track-score ({}) >= --threshold ({}); tracker will see no extra candidates",
            args.track_score, args.threshold
        );
    }

    // ── Build decoder ────────────────────────────────────────────────────
    let decoder_score = if args.track {
        args.track_score
    } else {
        args.threshold
    };
    let mut decoder_builder = DecoderBuilder::new()
        .with_score_threshold(decoder_score)
        .with_iou_threshold(args.iou)
        .with_input_dims(in_w, in_h);
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
    } else if let Some(cfg) = &info.config {
        decoder_builder = match cfg {
            edgefirst_model::runtime::EmbeddedConfig::Yaml(s) => {
                decoder_builder.with_config_yaml_str(s.clone())
            }
            edgefirst_model::runtime::EmbeddedConfig::Json(s) => {
                decoder_builder.with_config_json_str(s.clone())
            }
        };
    } else {
        warn!("No edgefirst config provided, guessing config based on model shape");

        // The decoder was given the model input dimensions via
        // `with_input_dims`, so it normalises pixel-space box coordinates
        // itself — no manual `1/input_dim` scale fold is needed here.
        let output_quants: Vec<Option<(f32, i32)>> = (0..runtime.output_count())
            .map(|i| {
                runtime
                    .output_quantization(i)
                    .map(|q| (q.scale, q.zero_point))
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
            | YoloSegDet2Way { .. }
            | YoloSplitSegDet { .. }
            | YoloEndToEndSegDet { .. }
            | YoloSplitEndToEndSegDet { .. } => (true, false, true),
            PerScale => (true, false, false),
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
    // Force the G2D backend. The HAL's threaded OpenGL backend drives its
    // GL worker with `tokio::mpsc::blocking_send`, which panics when called
    // from this service's tokio runtime. G2D is a synchronous i.MX 2D
    // accelerator (with a CPU fallback) and has no async coupling, so it is
    // safe to drive directly from the single-threaded inference loop.
    let mut img_proc = match tokio::task::spawn_blocking(|| {
        ImageProcessor::with_config(ImageProcessorConfig {
            backend: ComputeBackend::G2d,
            ..Default::default()
        })
    })
    .await
    .unwrap()
    {
        Ok(v) => v,
        Err(e) => {
            error!("Could not open ImageProcessor: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    let input_fmt = runtime.input_pixel_format(0);
    let is_planar = matches!(input_fmt, PixelFormat::PlanarRgb | PixelFormat::PlanarRgba);

    // Probe for a DMA-BUF zero-copy input. When available, the HAL writes
    // preprocessed pixels straight into the delegate's input buffer; otherwise
    // a CPU staging buffer is allocated and the input tensor is populated
    // explicitly each frame.
    let (mut preprocess_dst, dma_zero_copy): (TensorDyn, bool) =
        match runtime.import_input_image(&img_proc, in_w, in_h) {
            Some(Ok(t)) => {
                info!("DMA-BUF zero-copy input enabled");
                (t, true)
            }
            Some(Err(e)) => {
                error!("DMA-BUF input import failed: {e:?}");
                return ExitCode::FAILURE;
            }
            None => {
                // G2D / OpenGL emit packed pixels; planar-input models get an
                // interleaved RGB staging buffer and are deinterleaved later.
                let staging_fmt = if is_planar {
                    PixelFormat::Rgb
                } else {
                    input_fmt
                };
                match img_proc.create_image(
                    in_w,
                    in_h,
                    staging_fmt,
                    DType::U8,
                    None,
                    CpuAccess::ReadWrite,
                ) {
                    Ok(v) => {
                        info!("CPU staging input: {in_w}x{in_h} {staging_fmt:?}");
                        (v, false)
                    }
                    Err(e) => {
                        error!("Could not create staging image: {e:?}");
                        return ExitCode::FAILURE;
                    }
                }
            }
        };

    let mut output_boxes = Vec::with_capacity(50);
    let mut output_masks = Vec::with_capacity(50);
    let mut output_tracks = Vec::with_capacity(50);
    while !SHUTDOWN.load(Ordering::SeqCst) {
        let Some(frame) = ({
            let _span = info_span!("wait_for_camera_frame").entered();
            wait_for_camera_frame(&sub_camera, timeout)
        }) else {
            continue;
        };
        trace!("Received camera frame");

        let src_image = match camera_frame_to_tensor_dyn(&img_proc, &frame) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not create source image: {e:?}");
                continue;
            }
        };

        // Aspect-preserving fit of the camera frame into the model input.
        let letterbox = LetterboxTransform::compute(
            frame.width() as usize,
            frame.height() as usize,
            in_w,
            in_h,
        );

        let preprocess_start = std::time::Instant::now();

        {
            let _span = info_span!("preprocess").entered();
            if let Err(e) = img_proc.convert(
                &src_image,
                &mut preprocess_dst,
                Rotation::None,
                Flip::None,
                letterbox.crop(),
            ) {
                error!("Image conversion failed: {e:?}");
                continue;
            }
        }

        // CPU staging path: copy / deinterleave the preprocessed image into
        // the runtime input tensor. The DMA-BUF path is already populated by
        // the convert above.
        if !dma_zero_copy {
            let _span = info_span!("load_input").entered();
            if let Err(e) = populate_input_tensor(runtime.as_mut(), &preprocess_dst, in_w, in_h) {
                error!("Could not populate input tensor: {e:?}");
                continue;
            }
        }

        if let Err(e) = runtime.sync_input_for_device() {
            error!("Failed to sync input for device: {e:?}");
            continue;
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

        let sync_outputs_start = std::time::Instant::now();
        if let Err(e) = runtime.sync_outputs_for_cpu() {
            error!("Failed to sync outputs for cpu: {e:?}");
            continue;
        }
        let sync_outputs_time = sync_outputs_start.elapsed();
        // Fold preprocessing time into input_time (no separate preprocess
        // field). The post-invoke sync_outputs_for_cpu now carries the
        // DMA-BUF cache flush and (on copy fallbacks) the output staging
        // work — add it to output_duration so the published telemetry
        // covers everything between invoke and decode.
        let preprocess_time = preprocess_start.elapsed();
        let input_duration = (preprocess_time + timing.input_time).as_nanos();
        let model_duration = timing.model_time.as_nanos();
        let output_duration = (timing.output_time + sync_outputs_time).as_nanos();

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

        // Map decoded coordinates from the letterboxed model canvas back to
        // the camera frame. Boxes and instance-mask regions are unlettered;
        // a full-canvas semantic mask is cropped to the content region.
        for b in output_boxes.iter_mut() {
            letterbox.unletter_box(&mut b.bbox);
        }
        if has_instance_seg {
            for m in output_masks.iter_mut() {
                letterbox.unletter_instance_mask(m);
            }
        } else if has_seg {
            for m in output_masks.iter_mut() {
                letterbox.crop_semantic_mask(m);
            }
        }

        if args.track {
            let _span = info_span!("tracker_update").entered();
            use edgefirst_model::TrackerBox;
            use edgefirst_tracker::Tracker;
            let stamp = frame.stamp();
            let timestamp = stamp.nanosec as u64 + stamp.sec as u64 * 1_000_000_000;
            let wrapped: Vec<_> = output_boxes.iter().map(|b| TrackerBox(*b)).collect();
            let track_results = tracker.update(&wrapped, timestamp);

            // Keep only detections that received a track assignment
            let keep: Vec<bool> = track_results.iter().map(|t| t.is_some()).collect();

            let mut j = 0;
            output_boxes.retain(|_| {
                let k = keep[j];
                j += 1;
                k
            });

            if has_instance_seg {
                let mut j = 0;
                output_masks.retain(|_| {
                    let k = keep.get(j).copied().unwrap_or(false);
                    j += 1;
                    k
                });
            }

            // All remaining detections have Some(TrackInfo), so flatten is safe
            output_tracks.extend(track_results.into_iter().flatten());
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
            let masks = build_segmentation_msg_(frame.stamp(), &output_masks);
            if let Err(e) = mask_tx.send(masks).await {
                error!("Cannot send to mask publishing thread {e:?}");
            }
        }

        if has_box && let Some(publ_detect) = publ_detect.as_ref() {
            let (msg, enc) = build_detect_msg_and_encode_(
                &output_boxes,
                &output_tracks,
                &info.labels,
                frame.stamp(),
                frame.frame_id(),
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
                frame.stamp(),
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
            frame.stamp(),
            frame.frame_id(),
            input_duration,
            model_duration,
            output_duration,
            decode_duration.as_nanos(),
            has_instance_seg,
        );
        let msg = ZBytes::from(model_output.into_cdr());
        let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Model");

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

        if let Err(e) = model_info_msg.set_stamp(frame.stamp()) {
            error!("Failed to update ModelInfo stamp: {e:?}");
        }
        let msg = ZBytes::from(model_info_msg.as_cdr());
        let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/ModelInfo");

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
