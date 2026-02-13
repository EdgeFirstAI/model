// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use cdr::{CdrLe, Infinite};
use edgefirst_decoder::DetectBox;
use edgefirst_schemas::{
    builtin_interfaces::Time,
    edgefirst_msgs::{Detect, DetectBox2D, DetectTrack, Mask, ModelInfo, model_info},
    foxglove_msgs::{
        FoxgloveColor, FoxgloveImageAnnotations, FoxglovePoint2, FoxglovePointAnnotations,
        FoxgloveTextAnnotations,
        point_annotation_type::{LINE_LOOP, UNKNOWN},
    },
    std_msgs::Header,
};
use log::{debug, error};
use std::path::Path;
use tracing::instrument;
use zenoh::bytes::{Encoding, ZBytes};

use crate::{
    args::LabelSetting,
    model::{Model, SupportedModel},
};
use edgefirst_decoder::configs::DataType;

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

fn u128_to_foxglove_color(hexcode: u128) -> FoxgloveColor {
    const BYTES_PER_CHANNEL: u8 = 8;
    const FACTOR: u32 = (1 << BYTES_PER_CHANNEL) - 1;

    // only use the first 32 bits
    let hexcode = (hexcode >> (128 - (4 * BYTES_PER_CHANNEL))) as u32;
    FoxgloveColor {
        r: ((hexcode >> (BYTES_PER_CHANNEL * 3)) & FACTOR) as f64 / FACTOR as f64,
        g: ((hexcode >> (BYTES_PER_CHANNEL * 2)) & FACTOR) as f64 / FACTOR as f64,
        b: ((hexcode >> BYTES_PER_CHANNEL) & FACTOR) as f64 / FACTOR as f64,
        a: 1.0,
    }
}

pub fn build_image_annotations_msg_and_encode_(
    boxes: &[edgefirst_decoder::DetectBox],
    tracks: &[edgefirst_tracker::TrackInfo<DetectBox>],
    labels: &[String],
    timestamp: Time,
    stream_dims: (f64, f64),
    text: &str,
    labels_setting: LabelSetting,
) -> (ZBytes, Encoding) {
    let (stream_width, stream_height) = stream_dims;
    let mut annotations = FoxgloveImageAnnotations {
        circles: Vec::new(),
        points: Vec::new(),
        texts: Vec::new(),
    };

    let empty_points = FoxglovePointAnnotations {
        timestamp: timestamp.clone(),
        type_: UNKNOWN,
        points: Vec::new(),
        outline_color: WHITE.clone(),
        outline_colors: Vec::new(),
        fill_color: TRANSPARENT.clone(),
        thickness: 2.0,
    };

    let empty_text = FoxgloveTextAnnotations {
        timestamp: timestamp.clone(),
        text: text.to_owned(),
        position: FoxglovePoint2 {
            x: stream_width * 0.025,
            y: stream_height * 0.95,
        },
        font_size: 0.015 * stream_width.max(stream_height),
        text_color: WHITE.clone(),
        background_color: TRANSPARENT.clone(),
    };

    annotations.points.push(empty_points);
    annotations.texts.push(empty_text);

    for (i, b) in boxes.iter().enumerate() {
        let color = match tracks.get(i) {
            None => WHITE.clone(),
            Some(track) => u128_to_foxglove_color(track.uuid.as_u128()),
        };
        let outline_colors = vec![color.clone(), color.clone(), color.clone(), color.clone()];
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
        let points = FoxglovePointAnnotations {
            timestamp: timestamp.clone(),
            type_: LINE_LOOP,
            points,
            outline_color: color.clone(),
            outline_colors,
            fill_color: TRANSPARENT.clone(),
            thickness: 2.0,
        };

        let text = match labels_setting {
            LabelSetting::Index => format!("{:.2}", b.label),
            LabelSetting::Score => format!("{:.2}", b.score),
            LabelSetting::Label => labels[b.label].clone(),
            LabelSetting::LabelScore => {
                format!("{} {:.2}", b.label, b.score)
            }
            LabelSetting::Track => {
                //     match &b.track {
                //     None => format!("{:.2}", b.score),
                //     // only shows first 8 characters of the UUID
                //     Some(v) => v.uuid.to_string().split_at(8).0.to_owned(),
                // },
                format!("{:.2}", b.score)
            }
        };

        let text = FoxgloveTextAnnotations {
            timestamp: timestamp.clone(),
            text,
            position: FoxglovePoint2 {
                x: b.bbox.xmin as f64 * stream_width,
                y: b.bbox.ymin as f64 * stream_height,
            },
            font_size: 0.02 * stream_width.max(stream_height),
            text_color: color.clone(),
            background_color: TRANSPARENT.clone(),
        };
        annotations.points.push(points);
        annotations.texts.push(text);
    }

    let msg = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&annotations, Infinite).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("foxglove_msgs/msg/ImageAnnotations");

    (msg, enc)
}

#[instrument(skip_all)]
pub fn build_segmentation_msg(
    _in_time: Time,
    model_ctx: Option<&SupportedModel>,
    output_index: usize,
) -> Mask {
    let mut output_shape = vec![0, 0, 0, 0];
    let mask = if let Some(model) = model_ctx {
        match model.output_shape(output_index) {
            Ok(v) => output_shape = v,
            Err(e) => error!("Could not get output shape: {e:?}"),
        }
        let len = output_shape.iter().product();

        let output_type = match model.output_type(output_index) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get output type: {e:?}");
                DataType::UInt8
            }
        };

        match output_type {
            DataType::Int8 => {
                let mut buffer = vec![0i8; len];
                if let Err(e) = model.output_data(output_index, &mut buffer) {
                    error!("Could not get output data from segmentation tensor: {e:?}");
                }
                buffer.into_iter().map(|x| (x as i32 + 128) as u8).collect()
            }
            DataType::UInt8 => {
                let mut buffer = vec![0u8; len];
                if let Err(e) = model.output_data(output_index, &mut buffer) {
                    error!("Could not get output data from segmentation tensor: {e:?}");
                }
                buffer
            }
            _ => todo!(),
        }
    } else {
        Vec::new()
    };

    Mask {
        height: output_shape[1] as u32,
        width: output_shape[2] as u32,
        length: 1,
        encoding: "".to_string(),
        mask,
        boxed: false,
    }
}

#[instrument(skip_all)]
pub fn build_segmentation_msg_(
    _in_time: Time,
    output_masks: &[edgefirst_decoder::Segmentation],
) -> Mask {
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

    Mask {
        height: shape.0 as u32,
        width: shape.1 as u32,
        length: 1,
        encoding: "".to_string(),
        mask,
        boxed: false,
    }
}

#[instrument(skip_all)]
pub fn build_instance_segmentation_msg(
    _in_time: Time,
    model_ctx: Option<&SupportedModel>,
    output_index: usize,
) -> Mask {
    let mut output_shape = vec![0, 0, 0, 0];
    let mask = if let Some(model) = model_ctx {
        match model.output_shape(output_index) {
            Ok(v) => output_shape = v,
            Err(e) => error!("Could not get output shape: {e:?}"),
        }
        let len = output_shape.iter().product();

        let output_type = match model.output_type(output_index) {
            Ok(v) => v,
            Err(e) => {
                error!("Could not get output type: {e:?}");
                DataType::UInt8
            }
        };

        match output_type {
            DataType::Int8 => {
                let mut buffer = vec![0i8; len];
                if let Err(e) = model.output_data(output_index, &mut buffer) {
                    error!("Could not get output data from segmentation tensor: {e:?}");
                }
                buffer.into_iter().map(|x| (x as i32 + 128) as u8).collect()
            }
            DataType::UInt8 => {
                let mut buffer = vec![0u8; len];
                if let Err(e) = model.output_data(output_index, &mut buffer) {
                    error!("Could not get output data from segmentation tensor: {e:?}");
                }
                buffer
            }
            _ => todo!(),
        }
    } else {
        Vec::new()
    };

    Mask {
        height: output_shape[1] as u32,
        width: output_shape[2] as u32,
        length: 1,
        encoding: "".to_string(),
        mask,
        boxed: false,
    }
}

pub fn time_from_ns<T: Into<u128>>(ts: T) -> Time {
    let ts: u128 = ts.into();
    Time {
        sec: (ts / 1_000_000_000) as i32,
        nanosec: (ts % 1_000_000_000) as u32,
    }
}

pub fn convert_boxes(
    box_: &edgefirst_decoder::DetectBox,
    track: Option<&edgefirst_tracker::TrackInfo<DetectBox>>,
    labels: &[String],
    ts: Time,
) -> DetectBox2D {
    let track = match track {
        Some(v) => DetectTrack {
            id: v.uuid.to_string(),
            lifetime: v.count,
            created: time_from_ns(v.created),
        },
        None => DetectTrack {
            id: String::from(""),
            lifetime: 1,
            created: ts.clone(),
        },
    };
    DetectBox2D {
        center_x: (box_.bbox.xmax + box_.bbox.xmin) / 2.0,
        center_y: (box_.bbox.ymax + box_.bbox.ymin) / 2.0,
        width: box_.bbox.xmax - box_.bbox.xmin,
        height: box_.bbox.ymax - box_.bbox.ymin,
        label: labels
            .get(box_.label)
            .cloned()
            .unwrap_or_else(|| box_.label.to_string()),
        score: box_.score,
        distance: 0.0,
        speed: 0.0,
        track,
    }
}

#[instrument(skip_all)]
pub fn build_detect_msg_and_encode_(
    boxes: &[edgefirst_decoder::DetectBox],
    tracks: &[edgefirst_tracker::TrackInfo<DetectBox>],
    labels: &[String],
    header: Header,
    in_time: Time,
    model_time: Time,
    curr_time: Time,
) -> (ZBytes, Encoding) {
    let detect = Detect {
        header,
        input_timestamp: in_time,
        model_time,
        output_time: curr_time.clone(),
        boxes: boxes
            .iter()
            .enumerate()
            .map(|(ind, b)| convert_boxes(b, tracks.get(ind), labels, curr_time.clone()))
            .collect(),
    };
    let msg = ZBytes::from(cdr::serialize::<_, _, CdrLe>(&detect, Infinite).unwrap());
    let enc = Encoding::APPLICATION_CDR.with_schema("edgefirst_msgs/msg/Detect");

    (msg, enc)
}

fn tensor_type_to_model_info_datatype(t: DataType) -> u8 {
    match t {
        DataType::Raw => model_info::RAW,
        DataType::Int8 => model_info::INT8,
        DataType::UInt8 => model_info::UINT8,
        DataType::Int16 => model_info::INT16,
        DataType::UInt16 => model_info::UINT16,
        DataType::Float16 => model_info::FLOAT16,
        DataType::Int32 => model_info::INT32,
        DataType::UInt32 => model_info::UINT32,
        DataType::Float32 => model_info::FLOAT32,
        DataType::Int64 => model_info::INT64,
        DataType::UInt64 => model_info::UINT64,
        DataType::Float64 => model_info::FLOAT64,
        DataType::String => model_info::STRING,
    }
}

fn get_input_info(model_ctx: Option<&SupportedModel>) -> (Vec<u32>, u8) {
    let mut input_shape = vec![0, 0, 0, 0];
    let mut input_type = model_info::RAW;

    if let Some(ctx) = model_ctx {
        match ctx.input_shape(0) {
            Ok(v) => input_shape = v.iter().map(|f| *f as u32).collect(),
            Err(e) => error!("Cannot get input shape: {e:?}"),
        }
        match ctx.input_type(0) {
            Ok(v) => input_type = tensor_type_to_model_info_datatype(v),
            Err(e) => error!("Cannot get input datatype: {e:?}"),
        };
    }
    (input_shape, input_type)
}

pub fn build_model_info_msg(
    in_time: Time,
    model_ctx: Option<&SupportedModel>,
    path: &Path,
    has_det: bool,
    has_seg: bool,
) -> ModelInfo {
    let mut output_shape = vec![0, 0, 0, 0];
    let mut output_type = model_info::RAW;
    let mut labels = Vec::new();
    if let Some(model_ctx) = model_ctx {
        match model_ctx.output_shape(0) {
            Ok(v) => output_shape = v.iter().map(|f| *f as u32).collect(),
            Err(e) => {
                error!("Cannot get output shape of model: {e:?}");
            }
        };
        let model_output_type = match model_ctx.output_type(0) {
            Ok(v) => v,
            Err(e) => {
                error!("Cannot get output data type of model: {e:?}");
                DataType::Raw
            }
        };
        output_type = tensor_type_to_model_info_datatype(model_output_type);

        match model_ctx.labels() {
            Ok(v) => labels = v,
            Err(e) => {
                error!("Cannot get labels of model: {e:?}");
            }
        };
    }

    let model_format = match path.extension() {
        // , HailoRT, RKNN, TensorRT, TFLite
        Some(v) => match v.to_string_lossy().to_ascii_lowercase().as_str() {
            "rtm" => String::from("DeepViewRT"),
            "rknn" => String::from("RKNN"),
            "tflite" => String::from("TFLite"),
            "hef" => String::from("HailoRT"),
            _ => v.to_string_lossy().into_owned(),
        },
        None => String::from("unknown"),
    };

    let model_name = match model_ctx {
        Some(ctx) if ctx.model_name().is_ok_and(|n| !n.is_empty()) => ctx.model_name().unwrap(),
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
        model_types.push("Segmentation".to_string());
    }
    if has_det {
        model_types.push("Detection".to_string());
    }
    let (input_shape, input_type) = get_input_info(model_ctx);

    ModelInfo {
        header: Header {
            stamp: in_time.clone(),
            frame_id: String::new(),
        },
        labels, // input_shape = model_ctx.
        input_shape,
        input_type,
        output_shape,
        output_type,
        model_format,
        model_name,
        model_type: model_types.join(";"),
    }
}
