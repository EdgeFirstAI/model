// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use clap::Parser;
use edgefirst_hal::decoder::DecoderBuilder;
use edgefirst_hal::image::{Crop, Flip, ImageProcessor, ImageProcessorTrait, RGB, RGBA, Rotation};
use edgefirst_hal::tensor::{TensorMapTrait, TensorTrait};
use edgefirst_model::{
    args::Args,
    buildmsgs::{
        build_detect_msg_and_encode_, build_image_annotations_msg_and_encode_,
        build_model_info_msg, build_model_output_msg, build_segmentation_msg_, time_from_ns,
    },
    heart_beat,
    masks::mask_thread,
    model::{Metadata, ModelContext, decode_outputs, dmabuf_to_tensor_image, guess_model_config},
    update_dmabuf_with_pidfd, wait_for_camera_frame,
};
use edgefirst_schemas::{
    edgefirst_msgs::{Model as ModelMsg, ModelInfo},
    schema_registry::SchemaType,
    sensor_msgs::CameraInfo,
    serde_cdr,
};
use edgefirst_tflite::{Delegate, Interpreter, Library, TensorType};
use log::{error, info, trace, warn};
use std::{
    process::ExitCode,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tracing::info_span;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{Layer, Registry, layer::SubscriberExt};
use tracy_client::frame_mark;
use zenoh::bytes::{Encoding, ZBytes};

fn tflite_type_to_datatype(tt: TensorType) -> edgefirst_hal::decoder::configs::DataType {
    use edgefirst_hal::decoder::configs::DataType;
    match tt {
        TensorType::Float32 => DataType::Float32,
        TensorType::Float16 => DataType::Float16,
        TensorType::Float64 => DataType::Float64,
        TensorType::Int8 => DataType::Int8,
        TensorType::UInt8 => DataType::UInt8,
        TensorType::Int16 => DataType::Int16,
        TensorType::UInt16 => DataType::UInt16,
        TensorType::Int32 => DataType::Int32,
        TensorType::UInt32 => DataType::UInt32,
        TensorType::Int64 => DataType::Int64,
        TensorType::UInt64 => DataType::UInt64,
        _ => DataType::Raw,
    }
}

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

    // ── Load TFLite library ──────────────────────────────────────────────
    let lib = match Library::new() {
        Ok(v) => v,
        Err(e) => {
            error!("Could not load TensorFlow Lite library: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    let tflite_model = match edgefirst_tflite::Model::from_bytes(&lib, model_data) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not load TFLite model: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    // ── Load delegate and probe features ─────────────────────────────────
    let mut use_camera_adaptor = false;
    let use_dmabuf;

    let delegate = if !args.delegate.is_empty() {
        match Delegate::load(&args.delegate) {
            Ok(d) => {
                info!("Delegate loaded: {}", args.delegate);

                if d.has_camera_adaptor()
                    && let Some(adaptor) = d.camera_adaptor()
                {
                    if let Err(e) = adaptor.set_format(0, "rgba") {
                        warn!("CameraAdaptor set_format failed: {e:?}");
                    } else {
                        use_camera_adaptor = true;
                        info!("CameraAdaptor: enabled (RGBA -> RGB on NPU)");
                    }
                }

                use_dmabuf = d.has_dmabuf();
                if use_dmabuf {
                    info!("DMA-BUF: available");
                }

                Some(d)
            }
            Err(e) => {
                error!("Could not load delegate {}: {e:?}", args.delegate);
                return ExitCode::FAILURE;
            }
        }
    } else {
        info!("No delegate specified, using CPU inference");
        use_dmabuf = false;
        None
    };

    // ── Build interpreter ────────────────────────────────────────────────
    let mut builder = match Interpreter::builder(&lib) {
        Ok(b) => b,
        Err(e) => {
            error!("Could not create interpreter builder: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(d) = delegate {
        builder = builder.delegate(d);
    }
    let mut interpreter = match builder.build(&tflite_model) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not build interpreter: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    info!("Loaded model");

    // ── Inspect input tensor ─────────────────────────────────────────────
    let (in_h, in_w, input_type, _input_quant) = {
        let inputs = match interpreter.inputs() {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get input tensors: {e:?}");
                return ExitCode::FAILURE;
            }
        };
        let input = &inputs[0];
        let shape = input.shape().unwrap();
        let tt = input.tensor_type();
        let qp = input.quantization_params();
        info!(
            "Input: {} (scale={}, zp={})",
            input, qp.scale, qp.zero_point
        );
        (shape[1], shape[2], tt, qp)
    };

    // ── Extract metadata and labels ──────────────────────────────────────
    let metadata = {
        let m = edgefirst_tflite::metadata::Metadata::from_model_bytes(tflite_model.data());
        let mut meta = Metadata {
            name: m.name,
            version: m.version,
            description: m.description,
            author: m.author,
            license: m.license,
            config_yaml: None,
        };

        // Extract config YAML from model zip archive
        if let Ok(mut z) = zip::ZipArchive::new(std::io::Cursor::new(tflite_model.data())) {
            for name in [
                "edgefirst.yaml",
                "edgefirst.yml",
                "config.yaml",
                "config.yml",
            ] {
                if let Ok(mut f) = z.by_name(name)
                    && f.is_file()
                {
                    let mut yaml = String::new();
                    if let Err(e) = std::io::Read::read_to_string(&mut f, &mut yaml) {
                        error!("Error reading {name}: {e:?}");
                    }
                    meta.config_yaml = Some(yaml);
                    break;
                }
            }
        }
        meta
    };

    let model_labels = {
        let mut labels = Vec::new();
        if let Ok(mut z) = zip::ZipArchive::new(std::io::Cursor::new(tflite_model.data()))
            && let Ok(mut f) = z.by_name("labels.txt")
            && f.is_file()
        {
            let mut txt = String::new();
            if let Err(e) = std::io::Read::read_to_string(&mut f, &mut txt) {
                error!("Error reading labels.txt: {e:?}");
            }
            labels = txt.lines().map(|l| l.to_string()).collect();
        }
        labels
    };
    info!("Labels: {model_labels:?}");

    // ── Build ModelContext ────────────────────────────────────────────────
    let model_ctx = {
        let inputs = interpreter.inputs().unwrap();
        let outputs = interpreter.outputs().unwrap();

        let input_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|t| t.shape().unwrap_or_default())
            .collect();
        let input_types: Vec<edgefirst_hal::decoder::configs::DataType> = inputs
            .iter()
            .map(|t| tflite_type_to_datatype(t.tensor_type()))
            .collect();
        let output_shapes: Vec<Vec<usize>> = outputs
            .iter()
            .map(|t| t.shape().unwrap_or_default())
            .collect();
        let output_types: Vec<edgefirst_hal::decoder::configs::DataType> = outputs
            .iter()
            .map(|t| tflite_type_to_datatype(t.tensor_type()))
            .collect();

        ModelContext {
            input_shapes,
            input_types,
            output_shapes,
            output_types,
            labels: model_labels.clone(),
            name: metadata.name.clone().unwrap_or_default(),
        }
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
    } else if let Some(yaml) = &metadata.config_yaml {
        decoder_builder = decoder_builder.with_config_yaml_str(yaml.clone());
    } else {
        warn!("No edgefirst config provided, guessing config based on model shape");

        let output_quants: Vec<Option<(f32, i32)>> = {
            let outputs = interpreter.outputs().unwrap();
            outputs
                .iter()
                .map(|t| {
                    let qp = t.quantization_params();
                    Some((qp.scale, qp.zero_point))
                })
                .collect()
        };

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
    let (has_box, has_seg, has_instance_seg) = match model_type_ {
        edgefirst_hal::decoder::configs::ModelType::ModelPackSegDet { .. } => (true, true, false),
        edgefirst_hal::decoder::configs::ModelType::ModelPackSegDetSplit { .. } => {
            (true, true, false)
        }
        edgefirst_hal::decoder::configs::ModelType::ModelPackDet { .. } => (true, false, false),
        edgefirst_hal::decoder::configs::ModelType::ModelPackDetSplit { .. } => {
            (true, false, false)
        }
        edgefirst_hal::decoder::configs::ModelType::ModelPackSeg { .. } => (false, true, false),
        edgefirst_hal::decoder::configs::ModelType::YoloDet { .. } => (true, false, false),
        edgefirst_hal::decoder::configs::ModelType::YoloSegDet { .. } => (true, false, true),
        edgefirst_hal::decoder::configs::ModelType::YoloSplitDet { .. } => (true, false, false),
        edgefirst_hal::decoder::configs::ModelType::YoloSplitSegDet { .. } => (true, false, true),
        edgefirst_hal::decoder::configs::ModelType::YoloEndToEndDet { .. } => (true, false, false),
        edgefirst_hal::decoder::configs::ModelType::YoloEndToEndSegDet { .. } => {
            (true, false, true)
        }
        edgefirst_hal::decoder::configs::ModelType::YoloSplitEndToEndDet { .. } => {
            (true, false, false)
        }
        edgefirst_hal::decoder::configs::ModelType::YoloSplitEndToEndSegDet { .. } => {
            (true, false, true)
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

    let publ_visual = match args.visualization {
        true => Some(
            session
                .declare_publisher(args.visual_topic.clone())
                .await
                .unwrap(),
        ),
        false => None,
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

    let model_name = match args.model.as_path().file_name() {
        Some(v) => String::from(v.to_string_lossy()),
        None => {
            warn!("Cannot determine model file basename");
            String::from("unknown_model_file")
        }
    };
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

    let dst_format = if use_camera_adaptor { RGBA } else { RGB };
    let mut dst_image = match img_proc.create_image(in_w, in_h, dst_format) {
        Ok(v) => v,
        Err(e) => {
            error!("Could not create destination image: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    info!("Destination image: {}x{} {:?}", in_w, in_h, dst_format);

    // ── Set up DMA-BUF binding (persistent for application lifetime) ─────
    let dmabuf_handle = if use_dmabuf {
        let delegate_ref = interpreter.delegate(0).expect("delegate not found");
        let dmabuf = delegate_ref
            .dmabuf()
            .expect("DMA-BUF probed but not available");
        let buf_size = if use_camera_adaptor {
            in_h * in_w * 4 // RGBA
        } else {
            let inputs = interpreter.inputs().unwrap();
            inputs[0].byte_size()
        };
        match dmabuf.request(0, edgefirst_tflite::dmabuf::Ownership::Delegate, buf_size) {
            Ok((handle, _desc)) => {
                if let Err(e) = dmabuf.bind_to_tensor(handle, 0) {
                    error!("Could not bind DMA-BUF to input tensor: {e:?}");
                    return ExitCode::FAILURE;
                }
                info!("DMA-BUF bound to input tensor (size={buf_size})");
                Some(handle)
            }
            Err(e) => {
                warn!("DMA-BUF request failed, falling back to CPU: {e:?}");
                None
            }
        }
    } else {
        None
    };

    let mut output_boxes = Vec::with_capacity(50);
    let mut output_masks = Vec::with_capacity(50);
    let mut output_tracks = Vec::with_capacity(50);
    loop {
        let Some(mut dma_buf) = ({
            let _span = info_span!("wait_for_camera_frame").entered();
            wait_for_camera_frame(&sub_camera, timeout)
        }) else {
            continue;
        };
        trace!("Received camera frame");

        let input_start = Instant::now();
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

        // Write preprocessed pixels to input tensor
        {
            let _span = info_span!("load_input").entered();
            let map = match dst_image.tensor().map() {
                Ok(v) => v,
                Err(e) => {
                    error!("Could not map destination image: {e:?}");
                    continue;
                }
            };
            let pixels = map.as_slice();

            if use_camera_adaptor && let Some(handle) = dmabuf_handle {
                // Tier 1: CameraAdaptor + DMA-BUF — write RGBA to DMA-BUF
                let delegate_ref = interpreter.delegate(0).unwrap();
                let dmabuf = delegate_ref.dmabuf().unwrap();

                let fd = dmabuf.fd(handle).unwrap();
                let buf_size = pixels.len();
                let ptr = unsafe {
                    nix::libc::mmap(
                        std::ptr::null_mut(),
                        buf_size,
                        nix::libc::PROT_READ | nix::libc::PROT_WRITE,
                        nix::libc::MAP_SHARED,
                        fd,
                        0,
                    )
                };
                if ptr == nix::libc::MAP_FAILED {
                    error!("Failed to mmap DMA-BUF");
                    continue;
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr.cast::<u8>(), buf_size);
                    nix::libc::munmap(ptr, buf_size);
                }
                if let Err(e) = dmabuf.sync_for_device(handle) {
                    error!("DMA-BUF sync_for_device failed: {e:?}");
                    continue;
                }
            } else {
                // Tier 2/3: CPU type-convert into input tensor
                let mut inputs = match interpreter.inputs_mut() {
                    Ok(v) => v,
                    Err(e) => {
                        error!("Could not get mutable inputs: {e:?}");
                        continue;
                    }
                };
                let input = &mut inputs[0];
                let result = match input_type {
                    TensorType::Float32 => {
                        let float_data: Vec<f32> =
                            pixels.iter().map(|&v| f32::from(v) / 255.0).collect();
                        input.copy_from_slice(&float_data)
                    }
                    TensorType::UInt8 => input.copy_from_slice(pixels),
                    TensorType::Int8 => {
                        #[allow(clippy::cast_possible_wrap)]
                        let i8_data: Vec<i8> =
                            pixels.iter().map(|&v| v.wrapping_sub(128) as i8).collect();
                        input.copy_from_slice(&i8_data)
                    }
                    _ => {
                        error!("Unsupported input type: {input_type:?}");
                        return ExitCode::FAILURE;
                    }
                };
                if let Err(e) = result {
                    error!("Could not write to input tensor: {e:?}");
                    continue;
                }

                // Tier 2: sync DMA-BUF if available
                if let Some(handle) = dmabuf_handle {
                    let delegate_ref = interpreter.delegate(0).unwrap();
                    let dmabuf = delegate_ref.dmabuf().unwrap();
                    if let Err(e) = dmabuf.sync_for_device(handle) {
                        error!("DMA-BUF sync_for_device failed: {e:?}");
                        continue;
                    }
                }
            }
        }

        let input_duration = input_start.elapsed().as_nanos();
        trace!("Load input: {:.3} ms", input_duration as f32 / 1_000_000.0);

        let model_start = Instant::now();
        {
            let _span = info_span!("invoke").entered();
            if let Err(e) = interpreter.invoke() {
                error!("Failed to run model: {e:?}");
                return ExitCode::FAILURE;
            }
        }
        let model_duration = model_start.elapsed().as_nanos();
        trace!("Ran model: {:.3} ms", model_duration as f32 / 1_000_000.0);

        // Sync DMA-BUF output back to CPU
        if let Some(handle) = dmabuf_handle {
            let delegate_ref = interpreter.delegate(0).unwrap();
            let dmabuf = delegate_ref.dmabuf().unwrap();
            if let Err(e) = dmabuf.sync_for_cpu(handle) {
                error!("DMA-BUF sync_for_cpu failed: {e:?}");
            }
        }

        let output_start = Instant::now();
        let res = {
            let _span = info_span!("decode_outputs").entered();
            decode_outputs(&interpreter, &decoder, &mut output_boxes, &mut output_masks)
        };

        if res.is_ok() && args.track {
            let _span = info_span!("tracker_update").entered();
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

        if !args.classes.is_empty() {
            let keep: Vec<bool> = output_boxes
                .iter()
                .map(|b| {
                    model_labels
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

        let output_duration = output_start.elapsed();

        if first_run {
            info!("First run complete. Found {} boxes", output_boxes.len());
            first_run = false;
        }

        let _pub_span = info_span!("zenoh_publish").entered();
        if has_seg && let Some(mask_tx) = mask_tx.as_ref() {
            let masks = build_segmentation_msg_(dma_buf.header.stamp.clone(), &output_masks);
            match mask_tx.send(masks).await {
                Ok(_) => {}
                Err(e) => {
                    error!("Cannot send to mask publishing thread {e:?}");
                }
            }
        }

        if has_box && let Some(publ_detect) = publ_detect.as_ref() {
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

        {
            let model_output = build_model_output_msg(
                &output_boxes,
                &output_tracks,
                &model_labels,
                &output_masks,
                dma_buf.header.clone(),
                input_duration,
                model_duration,
                output_duration.as_nanos(),
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
