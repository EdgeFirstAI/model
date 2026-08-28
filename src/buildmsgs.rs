// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use edgefirst_schemas::{
    builtin_interfaces::{Duration, Time},
    edgefirst_msgs::{
        Detect, DetectBoxView, Mask, MaskView, Model as ModelMsg, ModelInfo, model_info,
    },
    foxglove_msgs::{
        FoxgloveColor, FoxgloveImageAnnotation, FoxglovePoint2, FoxglovePointAnnotationView,
        FoxgloveTextAnnotationView,
        point_annotation_type::{LINE_LOOP, UNKNOWN},
    },
};
use log::debug;
use std::path::Path;
use tracing::instrument;
use zenoh::bytes::{Encoding, ZBytes};

use crate::{args::LabelSetting, model::ModelContext};
use edgefirst_hal::tensor::DType;

const WHITE: FoxgloveColor = FoxgloveColor {
    r: 1.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};

const TRANSPARENT: FoxgloveColor = FoxgloveColor {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 0.0,
};

const EMPTY_ENCODING: &str = "";

fn u128_to_foxglove_color(hexcode: u128) -> FoxgloveColor {
    const BYTES_PER_CHANNEL: u8 = 8;
    const FACTOR: u32 = (1 << BYTES_PER_CHANNEL) - 1;

    let hexcode = (hexcode >> (128 - (4 * BYTES_PER_CHANNEL))) as u32;
    FoxgloveColor {
        r: ((hexcode >> (BYTES_PER_CHANNEL * 3)) & FACTOR) as f64 / FACTOR as f64,
        g: ((hexcode >> (BYTES_PER_CHANNEL * 2)) & FACTOR) as f64 / FACTOR as f64,
        b: ((hexcode >> BYTES_PER_CHANNEL) & FACTOR) as f64 / FACTOR as f64,
        a: 1.0,
    }
}

pub fn build_image_annotations_msg_and_encode_(
    boxes: &[edgefirst_hal::decoder::DetectBox],
    tracks: &[edgefirst_tracker::TrackInfo],
    labels: &[String],
    timestamp: Time,
    stream_dims: (f64, f64),
    text: &str,
    labels_setting: LabelSetting,
) -> (ZBytes, Encoding) {
    let (stream_width, stream_height) = stream_dims;
    let mut point_views = Vec::with_capacity(boxes.len() + 1);
    let mut text_entries: Vec<(String, FoxglovePoint2, FoxgloveColor, f64)> =
        Vec::with_capacity(boxes.len() + 1);

    point_views.push(FoxglovePointAnnotationView {
        timestamp,
        type_: UNKNOWN,
        points: Vec::new(),
        outline_color: WHITE,
        outline_colors: Vec::new(),
        fill_color: TRANSPARENT,
        thickness: 2.0,
    });
    text_entries.push((
        text.to_owned(),
        FoxglovePoint2 {
            x: stream_width * 0.025,
            y: stream_height * 0.95,
        },
        WHITE,
        0.015 * stream_width.max(stream_height),
    ));

    for (i, b) in boxes.iter().enumerate() {
        let color = match tracks.get(i) {
            None => WHITE,
            Some(track) => u128_to_foxglove_color(track.uuid.as_u128()),
        };
        let outline_colors = vec![color, color, color, color];
        let points = vec![
            FoxglovePoint2 {
                x: b.bbox.xmin as f64 * stream_width,
                y: b.bbox.ymin as f64 * stream_height,
            },
            FoxglovePoint2 {
                x: b.bbox.xmax as f64 * stream_width,
                y: b.bbox.ymin as f64 * stream_height,
            },
            FoxglovePoint2 {
                x: b.bbox.xmax as f64 * stream_width,
                y: b.bbox.ymax as f64 * stream_height,
            },
            FoxglovePoint2 {
                x: b.bbox.xmin as f64 * stream_width,
                y: b.bbox.ymax as f64 * stream_height,
            },
        ];
        point_views.push(FoxglovePointAnnotationView {
            timestamp,
            type_: LINE_LOOP,
            points,
            outline_color: color,
            outline_colors,
            fill_color: TRANSPARENT,
            thickness: 2.0,
        });

        let label_text = match labels_setting {
            LabelSetting::Index => format!("{:.2}", b.label),
            LabelSetting::Score => format!("{:.2}", b.score),
            LabelSetting::Label => labels
                .get(b.label)
                .cloned()
                .unwrap_or_else(|| b.label.to_string()),
            LabelSetting::LabelScore => {
                let name = labels
                    .get(b.label)
                    .cloned()
                    .unwrap_or_else(|| b.label.to_string());
                format!("{name} {:.2}", b.score)
            }
            LabelSetting::Track => match tracks.get(i) {
                Some(track) => track.uuid.to_string()[..8].to_owned(),
                None => format!("{:.2}", b.score),
            },
        };
        text_entries.push((
            label_text,
            FoxglovePoint2 {
                x: b.bbox.xmin as f64 * stream_width,
                y: b.bbox.ymin as f64 * stream_height,
            },
            color,
            0.02 * stream_width.max(stream_height),
        ));
    }

    let text_views: Vec<FoxgloveTextAnnotationView<'_>> = text_entries
        .iter()
        .map(
            |(text, position, text_color, font_size)| FoxgloveTextAnnotationView {
                timestamp,
                text: text.as_str(),
                position: *position,
                font_size: *font_size,
                text_color: *text_color,
                background_color: TRANSPARENT,
            },
        )
        .collect();

    let annotations = FoxgloveImageAnnotation::builder()
        .circles(&[])
        .points(&point_views)
        .texts(&text_views)
        .build()
        .expect("valid annotations");
    let msg = ZBytes::from(annotations.into_cdr());
    let enc = Encoding::APPLICATION_CDR.with_schema("foxglove_msgs/msg/ImageAnnotations");

    (msg, enc)
}

#[instrument(skip_all)]
pub fn build_segmentation_msg(
    _in_time: Time,
    model_ctx: Option<&ModelContext>,
    output_index: usize,
    output_data: Option<&[u8]>,
) -> Mask<Vec<u8>> {
    let output_shape = model_ctx
        .and_then(|ctx| ctx.output_shapes.get(output_index).cloned())
        .unwrap_or_else(|| vec![0, 0, 0, 0]);

    let mask = output_data.map(|d| d.to_vec()).unwrap_or_default();

    Mask::builder()
        .height(output_shape.get(1).copied().unwrap_or(0) as u32)
        .width(output_shape.get(2).copied().unwrap_or(0) as u32)
        .length(1)
        .encoding(EMPTY_ENCODING)
        .mask(&mask)
        .boxed(false)
        .build()
        .expect("valid mask message")
}

#[instrument(skip_all)]
pub fn build_segmentation_msg_(
    _in_time: Time,
    output_masks: &[edgefirst_hal::decoder::Segmentation],
) -> Mask<Vec<u8>> {
    let (shape, mask) = if !output_masks.is_empty() {
        let output_mask = &output_masks[0];
        let shape = output_mask.segmentation.shape();
        (
            (shape[0], shape[1]),
            output_mask.segmentation.flatten().to_vec(),
        )
    } else {
        ((0, 0), Vec::new())
    };

    Mask::builder()
        .height(shape.0 as u32)
        .width(shape.1 as u32)
        .length(1)
        .encoding(EMPTY_ENCODING)
        .mask(&mask)
        .boxed(false)
        .build()
        .expect("valid mask message")
}

pub fn time_from_ns<T: Into<u128>>(ts: T) -> Time {
    let ts: u128 = ts.into();
    Time {
        sec: (ts / 1_000_000_000) as i32,
        nanosec: (ts % 1_000_000_000) as u32,
    }
}

pub fn duration_from_ns<T: Into<u128>>(ts: T) -> Duration {
    let ts: u128 = ts.into();
    Duration {
        sec: (ts / 1_000_000_000) as i32,
        nanosec: (ts % 1_000_000_000) as u32,
    }
}

fn build_detect_box_views<'a>(
    boxes: &[edgefirst_hal::decoder::DetectBox],
    tracks: &[edgefirst_tracker::TrackInfo],
    labels: &[String],
    ts: Time,
    label_strings: &'a mut Vec<String>,
    track_ids: &'a mut Vec<String>,
) -> Vec<DetectBoxView<'a>> {
    label_strings.clear();
    track_ids.clear();
    label_strings.reserve(boxes.len());
    track_ids.reserve(boxes.len());

    for (i, b) in boxes.iter().enumerate() {
        label_strings.push(
            labels
                .get(b.label)
                .cloned()
                .unwrap_or_else(|| b.label.to_string()),
        );
        track_ids.push(match tracks.get(i) {
            Some(v) => v.uuid.to_string(),
            None => String::new(),
        });
    }

    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let (track_lifetime, track_created) = match tracks.get(i) {
                Some(v) => (v.count, time_from_ns(v.created)),
                None => (1, ts),
            };
            DetectBoxView {
                center_x: (b.bbox.xmax + b.bbox.xmin) / 2.0,
                center_y: (b.bbox.ymax + b.bbox.ymin) / 2.0,
                width: b.bbox.xmax - b.bbox.xmin,
                height: b.bbox.ymax - b.bbox.ymin,
                label: &label_strings[i],
                score: b.score,
                distance: 0.0,
                speed: 0.0,
                track_id: &track_ids[i],
                track_lifetime,
                track_created,
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
pub fn build_detect_msg_and_encode_(
    boxes: &[edgefirst_hal::decoder::DetectBox],
    tracks: &[edgefirst_tracker::TrackInfo],
    labels: &[String],
    stamp: Time,
    frame_id: &str,
    in_time: Time,
    model_time: Time,
    curr_time: Time,
) -> (ZBytes, Encoding) {
    let mut label_strings = Vec::new();
    let mut track_ids = Vec::new();
    let box_views = build_detect_box_views(
        boxes,
        tracks,
        labels,
        curr_time,
        &mut label_strings,
        &mut track_ids,
    );

    let detect = Detect::builder()
        .stamp(stamp)
        .frame_id(frame_id)
        .input_timestamp(in_time)
        .model_time(model_time)
        .output_time(curr_time)
        .boxes(&box_views)
        .build()
        .expect("valid detect message");

    let msg = ZBytes::from(detect.into_cdr());
    let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Detect");

    (msg, enc)
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
pub fn build_model_output_msg(
    boxes: &[edgefirst_hal::decoder::DetectBox],
    tracks: &[edgefirst_tracker::TrackInfo],
    labels: &[String],
    output_masks: &[edgefirst_hal::decoder::Segmentation],
    stamp: Time,
    frame_id: &str,
    input_duration: u128,
    model_duration: u128,
    output_duration: u128,
    decode_duration: u128,
    has_instance_seg: bool,
) -> ModelMsg<Vec<u8>> {
    let mut label_strings = Vec::new();
    let mut track_ids = Vec::new();
    let box_views = build_detect_box_views(
        boxes,
        tracks,
        labels,
        stamp,
        &mut label_strings,
        &mut track_ids,
    );

    let mut mask_data: Vec<Vec<u8>> = Vec::new();
    let mut mask_views: Vec<MaskView<'_>> = Vec::new();
    if has_instance_seg {
        for seg in output_masks {
            let _shape = seg.segmentation.shape();
            let data: Vec<u8> = seg.segmentation.iter().copied().collect();
            mask_data.push(data);
        }
        for (seg, data) in output_masks.iter().zip(mask_data.iter()) {
            let shape = seg.segmentation.shape();
            mask_views.push(MaskView {
                height: shape[0] as u32,
                width: shape[1] as u32,
                length: 1,
                encoding: EMPTY_ENCODING,
                mask: data,
                boxed: true,
            });
        }
    } else if !output_masks.is_empty() {
        let seg = &output_masks[0];
        let shape = seg.segmentation.shape();
        mask_data.push(seg.segmentation.iter().copied().collect());
        mask_views.push(MaskView {
            height: shape[0] as u32,
            width: shape[1] as u32,
            length: 1,
            encoding: EMPTY_ENCODING,
            mask: &mask_data[0],
            boxed: false,
        });
    }

    ModelMsg::builder()
        .stamp(stamp)
        .frame_id(frame_id)
        .input_time(duration_from_ns(input_duration))
        .model_time(duration_from_ns(model_duration))
        .output_time(duration_from_ns(output_duration))
        .decode_time(duration_from_ns(decode_duration))
        .boxes(&box_views)
        .masks(&mask_views)
        .build()
        .expect("valid model message")
}

fn tensor_type_to_model_info_datatype(t: DType) -> u8 {
    match t {
        DType::I8 => model_info::INT8,
        DType::U8 => model_info::UINT8,
        DType::I16 => model_info::INT16,
        DType::U16 => model_info::UINT16,
        DType::F16 => model_info::FLOAT16,
        DType::I32 => model_info::INT32,
        DType::U32 => model_info::UINT32,
        DType::F32 => model_info::FLOAT32,
        DType::I64 => model_info::INT64,
        DType::U64 => model_info::UINT64,
        DType::F64 => model_info::FLOAT64,
        _ => model_info::RAW,
    }
}

fn get_input_info(model_ctx: Option<&ModelContext>) -> (Vec<u32>, u8) {
    let mut input_shape = vec![0, 0, 0, 0];
    let mut input_type = model_info::RAW;

    if let Some(ctx) = model_ctx {
        if let Some(shape) = ctx.input_shapes.first() {
            input_shape = shape.iter().map(|f| *f as u32).collect();
        }
        if let Some(dt) = ctx.input_types.first() {
            input_type = tensor_type_to_model_info_datatype(*dt);
        }
    }
    (input_shape, input_type)
}

pub fn build_model_info_msg(
    in_time: Time,
    model_ctx: Option<&ModelContext>,
    path: &Path,
    has_det: bool,
    has_seg: bool,
) -> ModelInfo<Vec<u8>> {
    let mut output_shape = vec![0, 0, 0, 0];
    let mut output_type = model_info::RAW;
    if let Some(ctx) = model_ctx {
        if let Some(shape) = ctx.output_shapes.first() {
            output_shape = shape.iter().map(|f| *f as u32).collect();
        }
        if let Some(dt) = ctx.output_types.first() {
            output_type = tensor_type_to_model_info_datatype(*dt);
        }
    }

    let model_format = match path.extension() {
        Some(v) => match v.to_string_lossy().to_ascii_lowercase().as_str() {
            "tflite" => String::from("TFLite"),
            _ => v.to_string_lossy().into_owned(),
        },
        None => String::from("unknown"),
    };

    let model_name = match model_ctx {
        Some(ctx) if !ctx.name.is_empty() => ctx.name.clone(),
        Some(_) => path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        None => String::from("Loading Model..."),
    };
    debug!("Model name = {model_name}");
    let mut model_types = Vec::new();
    if has_seg {
        model_types.push("Segmentation");
    }
    if has_det {
        model_types.push("Detection");
    }
    let (input_shape, input_type) = get_input_info(model_ctx);
    // Borrow labels from ModelContext — avoid cloning the Vec<String> on every call.
    let label_refs: Vec<&str> = model_ctx
        .map(|ctx| ctx.labels.iter().map(String::as_str).collect())
        .unwrap_or_default();

    let model_type = model_types.join(";");
    ModelInfo::builder()
        .stamp(in_time)
        .frame_id("")
        .input_shape(&input_shape)
        .input_type(input_type)
        .output_shape(&output_shape)
        .output_type(output_type)
        .labels(&label_refs)
        .model_type(&model_type)
        .model_format(&model_format)
        .model_name(&model_name)
        .build()
        .expect("valid model info message")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::LabelSetting;
    use edgefirst_hal::decoder::{BoundingBox, DetectBox};
    use edgefirst_schemas::edgefirst_msgs::{CameraFrame, TensorFields, TensorPlaneView};
    use std::path::PathBuf;

    fn sample_box(label: usize, score: f32) -> DetectBox {
        DetectBox {
            bbox: BoundingBox::new(0.1, 0.2, 0.5, 0.6),
            score,
            label,
        }
    }

    #[test]
    fn time_and_duration_from_ns() {
        let t = time_from_ns(1_500_000_000u128);
        assert_eq!(t.sec, 1);
        assert_eq!(t.nanosec, 500_000_000);
        let d = duration_from_ns(2_250_000_000u128);
        assert_eq!(d.sec, 2);
        assert_eq!(d.nanosec, 250_000_000);
    }

    #[test]
    fn build_segmentation_empty_and_with_data() {
        let empty = build_segmentation_msg(time_from_ns(0u32), None, 0, None);
        assert_eq!(empty.height(), 0);
        assert_eq!(empty.width(), 0);
        assert!(empty.mask_data().is_empty());

        let ctx = ModelContext {
            input_shapes: vec![vec![1, 3, 64, 64]],
            input_types: vec![DType::U8],
            output_shapes: vec![vec![1, 32, 48, 1]],
            output_types: vec![DType::U8],
            labels: vec!["a".into()],
            name: "t".into(),
        };
        let data = vec![1u8, 2, 3, 4];
        let mask = build_segmentation_msg(time_from_ns(0u32), Some(&ctx), 0, Some(&data));
        assert_eq!(mask.height(), 32);
        assert_eq!(mask.width(), 48);
        assert_eq!(mask.mask_data(), data.as_slice());
    }

    #[test]
    fn build_detect_and_model_output_roundtrip() {
        let boxes = [sample_box(0, 0.9)];
        let labels = vec!["person".to_string()];
        let stamp = time_from_ns(1_000_000_000u32);
        let (bytes, enc) = build_detect_msg_and_encode_(
            &boxes,
            &[],
            &labels,
            stamp,
            "cam0",
            time_from_ns(10u32),
            time_from_ns(20u32),
            time_from_ns(30u32),
        );
        assert!(enc.to_string().contains("Detect") || format!("{enc:?}").contains("Detect"));
        let detect = Detect::from_cdr(bytes.to_bytes().to_vec()).unwrap();
        assert_eq!(detect.frame_id(), "cam0");
        assert_eq!(detect.boxes_len(), 1);

        let model = build_model_output_msg(
            &boxes,
            &[],
            &labels,
            &[],
            stamp,
            "cam0",
            100,
            200,
            300,
            400,
            false,
        );
        assert_eq!(model.frame_id(), "cam0");
        assert_eq!(model.boxes_len(), 1);
        assert_eq!(model.masks_len(), 0);
    }

    #[test]
    fn build_model_info_uses_context_and_path() {
        let ctx = ModelContext {
            input_shapes: vec![vec![1, 640, 640, 3]],
            input_types: vec![DType::U8],
            output_shapes: vec![vec![1, 84, 8400]],
            output_types: vec![DType::F32],
            labels: vec!["person".into(), "car".into()],
            name: "yolo".into(),
        };
        let path = PathBuf::from("/models/yolo.tflite");
        let info = build_model_info_msg(time_from_ns(5u32), Some(&ctx), &path, true, false);
        assert_eq!(info.model_name(), "yolo");
        assert_eq!(info.model_format(), "TFLite");
        assert!(info.model_type().contains("Detection"));
        assert_eq!(info.labels_len(), 2);

        let loading = build_model_info_msg(time_from_ns(0u32), None, &path, false, true);
        assert_eq!(loading.model_name(), "Loading Model...");
        assert!(loading.model_type().contains("Segmentation"));
    }

    #[test]
    fn build_image_annotations_with_boxes() {
        let boxes = [sample_box(0, 0.8), sample_box(1, 0.7)];
        let labels = vec!["person".to_string(), "car".to_string()];
        let (bytes, _) = build_image_annotations_msg_and_encode_(
            &boxes,
            &[],
            &labels,
            time_from_ns(0u32),
            (1920.0, 1080.0),
            "status",
            LabelSetting::LabelScore,
        );
        assert!(!bytes.is_empty());
        let (bytes2, _) = build_image_annotations_msg_and_encode_(
            &boxes,
            &[],
            &labels,
            time_from_ns(0u32),
            (1.0, 1.0),
            "x",
            LabelSetting::Index,
        );
        assert!(!bytes2.is_empty());
    }

    #[test]
    fn camera_frame_cdr_roundtrip_planes() {
        const TENSOR_STORAGE_KIND_DMA_BUF: u32 = 2;
        const TENSOR_DTYPE_U8: u32 = 0;
        let shape = [1080u64, 1920];
        let strides = [3840i64, 2];
        let plane = TensorPlaneView {
            handle: 7,
            offset: 0,
            stride: 3840,
            size: 1920 * 1080 * 2,
            used: 1920 * 1080 * 2,
            modifier: 0,
            handle_bytes: &[],
            data: &[],
        };
        let tensor = TensorFields {
            storage_kind: TENSOR_STORAGE_KIND_DMA_BUF,
            pid: 1234,
            fence_fd: -1,
            dtype: TENSOR_DTYPE_U8,
            quant_axis: -2,
            shape: &shape,
            strides: &strides,
            quant_scales: &[],
            quant_zero_points: &[],
            format: "YUYV".into(),
            color_space: "".into(),
            color_transfer: "".into(),
            color_encoding: "".into(),
            color_range: "".into(),
            planes: std::slice::from_ref(&plane),
        };
        let frame = CameraFrame::builder()
            .stamp(time_from_ns(42u32))
            .frame_id("camera")
            .seq(3)
            .tensor(&tensor)
            .build()
            .expect("camera frame");
        let decoded = CameraFrame::from_cdr(frame.to_cdr()).unwrap();
        assert_eq!(decoded.frame_id(), "camera");
        let t = decoded.tensor();
        assert_eq!(t.pid(), 1234);
        assert_eq!(t.shape().collect::<Vec<_>>(), vec![1080, 1920]);
        assert_eq!(t.format(), "YUYV");
        assert_eq!(t.num_planes(), 1);
        let plane = t.plane_at(0).unwrap();
        assert_eq!(plane.handle, 7);
        assert_eq!(plane.stride, 3840);
    }

    #[test]
    fn tensor_type_mapping_covers_common_dtypes() {
        assert_eq!(
            tensor_type_to_model_info_datatype(DType::U8),
            model_info::UINT8
        );
        assert_eq!(
            tensor_type_to_model_info_datatype(DType::F32),
            model_info::FLOAT32
        );
        assert_eq!(
            tensor_type_to_model_info_datatype(DType::I8),
            model_info::INT8
        );
    }
}
