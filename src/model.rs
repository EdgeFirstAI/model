// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use std::{
    error::Error,
    fmt,
    os::fd::{FromRawFd, OwnedFd},
};

use edgefirst_hal::decoder::{
    ConfigOutput, ConfigOutputs,
    configs::{
        Boxes, DataType, DecoderType, Detection, DimName, Mask, MaskCoefficients, Protos, Scores,
        Segmentation,
    },
};
use edgefirst_hal::image::TensorImage;
use edgefirst_hal::tensor::{DmaTensor, Tensor, TensorTrait};
use edgefirst_schemas::edgefirst_msgs::DmaBuffer;
use four_char_code::FourCharCode;
use tracing::instrument;

// ── ModelError ───────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ModelError {
    kind: ModelErrorKind,
    source: Box<dyn std::error::Error>,
}

#[derive(Debug)]
pub enum ModelErrorKind {
    Io,
    TFLite,
    Tensor,
    Image,
    Decoding,
    Other,
}

impl fmt::Display for ModelErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "kind: {:?}, source: {:?}", self.kind, self.source)
    }
}

impl From<std::io::Error> for ModelError {
    fn from(value: std::io::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::Io,
            source: Box::from(value),
        }
    }
}

impl From<edgefirst_tflite::Error> for ModelError {
    fn from(value: edgefirst_tflite::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::TFLite,
            source: Box::from(value),
        }
    }
}

impl From<Box<dyn Error>> for ModelError {
    fn from(value: Box<dyn Error>) -> Self {
        ModelError {
            kind: ModelErrorKind::Other,
            source: value,
        }
    }
}

impl From<edgefirst_hal::tensor::Error> for ModelError {
    fn from(value: edgefirst_hal::tensor::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::Tensor,
            source: Box::from(value),
        }
    }
}

impl From<edgefirst_hal::image::Error> for ModelError {
    fn from(value: edgefirst_hal::image::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::Image,
            source: Box::from(value),
        }
    }
}

impl From<edgefirst_hal::decoder::DecoderError> for ModelError {
    fn from(value: edgefirst_hal::decoder::DecoderError) -> Self {
        ModelError {
            kind: ModelErrorKind::Decoding,
            source: Box::from(value),
        }
    }
}

impl Error for ModelError {}

impl ModelError {
    pub fn new(kind: ModelErrorKind, msg: String) -> Self {
        ModelError {
            kind,
            source: msg.into(),
        }
    }
}

// ── ModelContext ──────────────────────────────────────────────────────────────

/// Static model metadata used for building info and segmentation messages.
/// Populated once at startup from interpreter tensor introspection.
#[derive(Debug, Clone)]
pub struct ModelContext {
    pub input_shapes: Vec<Vec<usize>>,
    pub input_types: Vec<DataType>,
    pub output_shapes: Vec<Vec<usize>>,
    pub output_types: Vec<DataType>,
    pub labels: Vec<String>,
    pub name: String,
}

// ── Metadata ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Metadata {
    pub name: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub config_yaml: Option<String>,
}

// ── dmabuf_to_tensor_image ───────────────────────────────────────────────────

#[instrument(skip_all)]
pub fn dmabuf_to_tensor_image(dma: &DmaBuffer) -> Result<TensorImage, ModelError> {
    // Force DMA tensor type — the camera fd is always a DMA-BUF but
    // Tensor::from_fd auto-detection may misclassify it as SHM based on
    // anon_inode minor numbers, which prevents G2D hardware acceleration.
    let raw_fd = unsafe { nix::libc::dup(dma.fd) };
    if raw_fd < 0 {
        return Err(ModelError::new(
            ModelErrorKind::Io,
            format!(
                "Failed to dup DMA-BUF fd: {}",
                std::io::Error::last_os_error()
            ),
        ));
    }
    let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
    let dma_tensor = DmaTensor::<u8>::from_fd(
        fd,
        &[
            dma.height as usize,
            dma.width as usize,
            (dma.length / dma.width / dma.height) as usize,
        ],
        None,
    )?;
    let tensor = Tensor::Dma(dma_tensor);

    // DmaBuffer.fourcc uses V4L2/DRM convention where characters are packed
    // little-endian (first char in lowest byte), while FourCharCode uses
    // big-endian (first char in highest byte). Swap bytes to convert.
    let fourcc = FourCharCode::new(dma.fourcc.swap_bytes())
        .map_err(|e| ModelError::new(ModelErrorKind::Image, format!("Invalid FourCC code: {e}")))?;
    let img = TensorImage::from_tensor(tensor, fourcc)?;
    Ok(img)
}

// ── decode_outputs ───────────────────────────────────────────────────────────

/// Decode edgefirst-tflite output tensors through the HAL decoder.
///
/// Classifies outputs as float or quantized, builds ndarray views using
/// typed slices, and dispatches to `decoder.decode_float()` or
/// `decoder.decode_quantized()`.
#[instrument(skip_all)]
pub fn decode_outputs(
    interpreter: &edgefirst_tflite::Interpreter<'_>,
    decoder: &edgefirst_hal::decoder::Decoder,
    output_boxes: &mut Vec<edgefirst_hal::decoder::DetectBox>,
    output_masks: &mut Vec<edgefirst_hal::decoder::Segmentation>,
) -> Result<(), ModelError> {
    let outputs = interpreter
        .outputs()
        .map_err(|e| ModelError::new(ModelErrorKind::TFLite, format!("Cannot get outputs: {e}")))?;

    let mut float_views = Vec::new();
    let mut quant_views = Vec::new();

    for tensor in &outputs {
        let shape = tensor.shape().map_err(|e| {
            ModelError::new(ModelErrorKind::TFLite, format!("Cannot get shape: {e}"))
        })?;
        match tensor.tensor_type() {
            edgefirst_tflite::TensorType::Float32 => {
                let data = tensor.as_slice::<f32>().map_err(|e| {
                    ModelError::new(ModelErrorKind::TFLite, format!("Cannot read f32: {e}"))
                })?;
                let arr = ndarray::ArrayView::from_shape(shape, data).map_err(|e| {
                    ModelError::new(
                        ModelErrorKind::TFLite,
                        format!("f32 shape/data mismatch: {e}"),
                    )
                })?;
                float_views.push(arr);
            }
            edgefirst_tflite::TensorType::UInt8 => {
                let data = tensor.as_slice::<u8>().map_err(|e| {
                    ModelError::new(ModelErrorKind::TFLite, format!("Cannot read u8: {e}"))
                })?;
                let arr = ndarray::ArrayView::from_shape(shape, data).map_err(|e| {
                    ModelError::new(
                        ModelErrorKind::TFLite,
                        format!("u8 shape/data mismatch: {e}"),
                    )
                })?;
                let arr: edgefirst_hal::decoder::ArrayViewDQuantized = arr.into();
                quant_views.push(arr);
            }
            edgefirst_tflite::TensorType::Int8 => {
                let data = tensor.as_slice::<i8>().map_err(|e| {
                    ModelError::new(ModelErrorKind::TFLite, format!("Cannot read i8: {e}"))
                })?;
                let arr = ndarray::ArrayView::from_shape(shape, data).map_err(|e| {
                    ModelError::new(
                        ModelErrorKind::TFLite,
                        format!("i8 shape/data mismatch: {e}"),
                    )
                })?;
                let arr: edgefirst_hal::decoder::ArrayViewDQuantized = arr.into();
                quant_views.push(arr);
            }
            other => {
                log::warn!("Ignoring output tensor with type {other:?}");
            }
        }
    }

    match (float_views.is_empty(), quant_views.is_empty()) {
        (false, true) => decoder.decode_float(&float_views, output_boxes, output_masks)?,
        (true, false) => decoder.decode_quantized(&quant_views, output_boxes, output_masks)?,
        (true, true) => log::error!("No outputs for decoder"),
        (false, false) => log::error!("Mixed float and quantized outputs"),
    }
    log::trace!("Decoded boxes: {:?}", output_boxes);
    Ok(())
}

// ── guess_model_config ───────────────────────────────────────────────────────

const MIN_NUM_BOXES: usize = 256;
#[instrument(skip_all)]
pub fn guess_model_config(
    output_shapes: &[Vec<usize>],
    output_quantization: &[Option<(f32, i32)>],
) -> Option<ConfigOutputs> {
    // We assume this cannot be ModelPack split detection because we will not be
    // able to get the anchors without metadata.

    if output_shapes.is_empty() {
        return None;
    }

    if output_shapes.len() == 1 {
        // Only one output, could be Modelpack segmentation or Yolo detection
        if let Some(cfg) = guess_yolo_detection([&output_shapes[0]], [&output_quantization[0]]) {
            return Some(cfg);
        }

        if let Some(cfg) =
            guess_modelpack_segmentation([&output_shapes[0]], [&output_quantization[0]])
        {
            return Some(cfg);
        }

        return None;
    }

    if output_shapes.len() == 2 {
        // Two outputs, could be Modelpack detection or Yolo segmentation detection

        let shape0 = &output_shapes[0];
        let shape1 = &output_shapes[1];
        let quant0 = &output_quantization[0];
        let quant1 = &output_quantization[1];

        if let Some(cfg) = guess_modelpack_detection([shape0, shape1], [quant0, quant1]) {
            log::debug!("Guessed modelpack det with 2 outputs");
            return Some(cfg);
        }

        if let Some(cfg) = guess_yolo_split_det([shape0, shape1], [quant0, quant1]) {
            log::debug!("Guessed yolo split det with 2 outputs");
            return Some(cfg);
        }

        if let Some(cfg) = guess_yolo_segdet([shape0, shape1], [quant0, quant1]) {
            log::debug!("Guessed yolo segdet with 2 outputs");
            return Some(cfg);
        }

        if let Some(cfg) = guess_modelpack_segmentation_2([shape0, shape1], [quant0, quant1]) {
            log::debug!("Guessed modelpack seg with 2 outputs");
            return Some(cfg);
        }
    }

    if output_shapes.len() == 3 {
        let shape0 = &output_shapes[0];
        let shape1 = &output_shapes[1];
        let shape2 = &output_shapes[2];
        let quant0 = &output_quantization[0];
        let quant1 = &output_quantization[1];
        let quant2 = &output_quantization[2];

        if let Some(cfg) =
            guess_modelpack_segdet_3([shape0, shape1, shape2], [quant0, quant1, quant2])
        {
            return Some(cfg);
        }
    }

    if output_shapes.len() == 4 {
        let shape0 = &output_shapes[0];
        let shape1 = &output_shapes[1];
        let shape2 = &output_shapes[2];
        let shape3 = &output_shapes[3];
        let quant0 = &output_quantization[0];
        let quant1 = &output_quantization[1];
        let quant2 = &output_quantization[2];
        let quant3 = &output_quantization[3];

        if let Some(cfg) = guess_yolo_split_segdet(
            [shape0, shape1, shape2, shape3],
            [quant0, quant1, quant2, quant3],
        ) {
            return Some(cfg);
        }

        if let Some(cfg) = guess_modelpack_segdet_4(
            [shape0, shape1, shape2, shape3],
            [quant0, quant1, quant2, quant3],
        ) {
            return Some(cfg);
        }
    }

    None
}

fn guess_modelpack_segmentation(
    shape: [&[usize]; 1],
    quant: [&Option<(f32, i32)>; 1],
) -> Option<ConfigOutputs> {
    // Modelpack segmentation has one output with NC (num classes) , and
    // the width and height are greater than 160
    let [shape] = shape;
    let quantization = quant[0].map(|x| x.into());
    if shape.len() != 4 {
        return None;
    }
    // Check if 1/H/W/NC or 1/NC/H/W
    let mut dshape = None;
    if shape[1] >= 160 && shape[2] >= 160 {
        dshape = Some(vec![
            (DimName::Batch, shape[0]),
            (DimName::Height, shape[1]),
            (DimName::Width, shape[2]),
            (DimName::NumClasses, shape[3]),
        ]);
    } else if shape[2] >= 160 && shape[3] >= 160 {
        dshape = Some(vec![
            (DimName::Batch, shape[0]),
            (DimName::NumClasses, shape[1]),
            (DimName::Height, shape[2]),
            (DimName::Width, shape[3]),
        ]);
    }
    if let Some(dshape) = dshape {
        return Some(ConfigOutputs {
            outputs: vec![ConfigOutput::Segmentation(Segmentation {
                decoder: DecoderType::ModelPack,
                quantization,
                shape: shape.to_vec(),
                dshape,
            })],
            nms: None,
            decoder_version: None,
        });
    }
    None
}

fn guess_yolo_detection(
    shape: [&[usize]; 1],
    quant: [&Option<(f32, i32)>; 1],
) -> Option<ConfigOutputs> {
    // Modelpack segmentation has one output with NC (num classes) , and
    // the width and height are greater than 160
    let [shape] = shape;
    let quantization = quant[0].map(|x| x.into());
    if shape.len() != 3 {
        return None;
    }
    // Check if 1/NF/NB or 1/NB/NF
    let dshape;

    // we assume NB > NF, and NF >= 5
    if shape[1] > shape[2] && shape[2] >= 5 && shape[1] >= MIN_NUM_BOXES {
        dshape = vec![
            (DimName::Batch, shape[0]),
            (DimName::NumBoxes, shape[1]),
            (DimName::NumFeatures, shape[2]),
        ];
    } else if shape[2] > shape[1] && shape[1] >= 5 && shape[2] >= MIN_NUM_BOXES {
        dshape = vec![
            (DimName::Batch, shape[0]),
            (DimName::NumFeatures, shape[1]),
            (DimName::NumBoxes, shape[2]),
        ];
    } else {
        return None;
    }

    Some(ConfigOutputs {
        outputs: vec![ConfigOutput::Detection(Detection {
            anchors: None,
            decoder: DecoderType::Ultralytics,
            quantization,
            shape: shape.to_vec(),
            dshape,
            normalized: None,
        })],
        nms: None,
        decoder_version: None,
    })
}

fn guess_modelpack_detection(
    shape: [&[usize]; 2],
    quant: [&Option<(f32, i32)>; 2],
) -> Option<ConfigOutputs> {
    // Modelpack detection has one output with 1/NB/1/4, where NB is number of boxes
    // Another output with 1/NB/NC, where NC is number of classes
    let [mut shape0, mut shape1] = shape;
    let [mut quant0, mut quant1] = quant.map(|x| x.map(|x| x.into()));

    // put the longer output first
    if shape0.len() < shape1.len() {
        std::mem::swap(&mut shape0, &mut shape1);
        std::mem::swap(&mut quant0, &mut quant1);
    }

    let dshape;

    if shape0.len() == 4 && shape1.len() == 3 {
        // shape0 is boxes, shape1 is scores
        let mut dshape_boxes: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
        let mut num_boxes = 0;
        for &dim in &shape0[1..] {
            match dim {
                1 => dshape_boxes.push((DimName::Padding, 1)),
                4 => dshape_boxes.push((DimName::BoxCoords, 4)),
                d if d > MIN_NUM_BOXES && num_boxes == 0 => {
                    dshape_boxes.push((DimName::NumBoxes, d));
                    num_boxes = d;
                }
                _ => return None,
            }
        }

        if num_boxes == 0 {
            return None;
        }

        let mut dshape_scores: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
        let mut found_num_boxes = false;
        for &dim in &shape1[1..] {
            match dim {
                d if d == num_boxes => {
                    found_num_boxes = true;
                    dshape_scores.push((DimName::NumBoxes, d))
                }
                d => dshape_scores.push((DimName::NumClasses, d)),
            }
        }
        if !found_num_boxes {
            return None;
        }

        dshape = (dshape_boxes, dshape_scores);
    } else {
        return None;
    }

    Some(ConfigOutputs {
        outputs: vec![
            ConfigOutput::Boxes(Boxes {
                decoder: DecoderType::ModelPack,
                quantization: quant0,
                shape: shape0.to_vec(),
                dshape: dshape.0,
                normalized: None,
            }),
            ConfigOutput::Scores(Scores {
                decoder: DecoderType::ModelPack,
                quantization: quant1,
                shape: shape1.to_vec(),
                dshape: dshape.1,
            }),
        ],
        nms: None,
        decoder_version: None,
    })
}

fn guess_modelpack_segmentation_2(
    shape: [&[usize]; 2],
    quant: [&Option<(f32, i32)>; 2],
) -> Option<ConfigOutputs> {
    // 1/H/W/NC+1
    // 1/H/W
    let [mut shape0, mut shape1] = shape;
    let [mut quant0, mut quant1] = quant.map(|x| x.map(|x| x.into()));

    if shape0.len() < shape1.len() {
        std::mem::swap(&mut shape0, &mut shape1);
        std::mem::swap(&mut quant0, &mut quant1);
    }

    if shape0.len() != 4 || shape1.len() != 3 {
        return None;
    }

    let mut dshape_mask: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    let mut height = None;
    let mut width = None;
    for &dim in &shape1[1..] {
        match dim {
            d if d >= 80 && height.is_none() => {
                dshape_mask.push((DimName::Height, dim));
                height = Some(dim);
            }
            d if d >= 80 && height.is_some() => {
                dshape_mask.push((DimName::Width, dim));
                width = Some(dim);
            }
            _ => return None,
        }
    }

    let height = height?;
    let width = width?;

    let mut dshape_segmentation: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    let mut classes = 0;
    for &dim in &shape0[1..] {
        match dim {
            _ if dim == height => {
                dshape_segmentation.push((DimName::Height, dim));
            }
            _ if dim == width => {
                dshape_segmentation.push((DimName::Width, dim));
            }
            _ if classes == 0 => {
                dshape_segmentation.push((DimName::NumClasses, dim));
                classes = dim;
            }
            _ => return None,
        }
    }
    let dshape = (dshape_segmentation, dshape_mask);
    Some(ConfigOutputs {
        outputs: vec![
            ConfigOutput::Segmentation(Segmentation {
                decoder: DecoderType::ModelPack,
                quantization: quant0,
                shape: shape0.to_vec(),
                dshape: dshape.0,
            }),
            ConfigOutput::Mask(Mask {
                decoder: DecoderType::ModelPack,
                quantization: quant1,
                shape: shape1.to_vec(),
                dshape: dshape.1,
            }),
        ],
        nms: None,
        decoder_version: None,
    })
}

fn guess_yolo_segdet(
    shape: [&[usize]; 2],
    quant: [&Option<(f32, i32)>; 2],
) -> Option<ConfigOutputs> {
    // Yolo segmentation detection has one output with 1/NF/NB, where NB is number
    // of boxes Another output with 1/H/W/NP, where NC is number of protos
    let [mut shape0, mut shape1] = shape;
    let [mut quant0, mut quant1] = quant.map(|x| x.map(|x| x.into()));

    // put the longer output first
    if shape0.len() < shape1.len() {
        std::mem::swap(&mut shape0, &mut shape1);
        std::mem::swap(&mut quant0, &mut quant1);
    }

    let dshape;

    if shape0.len() == 4 && shape1.len() == 3 {
        // shape0 is protos, shape1 is detection
        let mut dshape_protos: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
        let mut num_protos = 0;
        let mut found_height = false;
        for &dim in &shape0[1..] {
            match dim {
                160.. if !found_height => {
                    dshape_protos.push((DimName::Height, dim));
                    found_height = true;
                }
                160.. if found_height => {
                    dshape_protos.push((DimName::Width, dim));
                }
                d if num_protos == 0 => {
                    dshape_protos.push((DimName::NumProtos, d));
                    num_protos = d;
                }
                _ => return None,
            }
        }

        if num_protos == 0 {
            return None;
        }

        let mut dshape_det: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
        for &dim in &shape1[1..] {
            match dim {
                d if d > MIN_NUM_BOXES => dshape_det.push((DimName::NumBoxes, d)),
                d if d > 4 + num_protos => dshape_det.push((DimName::NumFeatures, d)),
                _ => return None,
            }
        }

        dshape = (dshape_protos, dshape_det);
    } else {
        return None;
    }

    Some(ConfigOutputs {
        outputs: vec![
            ConfigOutput::Protos(Protos {
                decoder: DecoderType::Ultralytics,
                quantization: quant0,
                shape: shape0.to_vec(),
                dshape: dshape.0,
            }),
            ConfigOutput::Detection(Detection {
                decoder: DecoderType::Ultralytics,
                anchors: None,
                quantization: quant1,
                shape: shape1.to_vec(),
                dshape: dshape.1,
                normalized: None,
            }),
        ],
        nms: None,
        decoder_version: None,
    })
}

fn guess_yolo_split_det(
    shape: [&[usize]; 2],
    quant: [&Option<(f32, i32)>; 2],
) -> Option<ConfigOutputs> {
    // one output is 1/NC/NB
    // one output is 1/4/NB
    let [mut shape0, mut shape1] = shape;
    let [mut quant0, mut quant1] = quant.map(|x| x.map(|x| x.into()));

    let dshape;

    if shape0.len() == 3 && shape1.len() == 3 {
        // look for which one has 4 as a dimension, if both have it then make a guess
        // and print a warning.
        let has_boxes0 = shape0.contains(&4);
        let has_boxes1 = shape1.contains(&4);
        let mut warning = None;

        match (has_boxes0, has_boxes1) {
            (true, false) => {}
            (false, true) => {
                std::mem::swap(&mut shape0, &mut shape1);
                std::mem::swap(&mut quant0, &mut quant1);
            }
            (true, true) => {
                warning = Some(
                    "Both outputs have 4 as a dimension, guessing output 0 is boxes for yolo split detection",
                );
            }
            (false, false) => {
                return None;
            }
        }

        // shape0 is boxes, shape1 is scores
        let mut num_boxes = 0;

        let mut dshape_protos: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
        for &dim in &shape0[1..] {
            match dim {
                4 => {
                    dshape_protos.push((DimName::BoxCoords, 4));
                }
                d if d > MIN_NUM_BOXES && num_boxes == 0 => {
                    dshape_protos.push((DimName::NumBoxes, d));
                    num_boxes = d;
                }
                _ => return None,
            }
        }

        if num_boxes == 0 {
            return None;
        }

        let mut dshape_det: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
        for &dim in &shape1[1..] {
            match dim {
                d if d == num_boxes => dshape_det.push((DimName::NumBoxes, d)),
                d => dshape_det.push((DimName::NumClasses, d)),
            }
        }

        if let Some(warning) = warning {
            log::warn!("{}", warning);
        }
        dshape = (dshape_protos, dshape_det);
    } else {
        return None;
    }

    Some(ConfigOutputs {
        outputs: vec![
            ConfigOutput::Boxes(Boxes {
                decoder: DecoderType::Ultralytics,
                quantization: quant0,
                shape: shape0.to_vec(),
                dshape: dshape.0,
                normalized: None,
            }),
            ConfigOutput::Scores(Scores {
                decoder: DecoderType::Ultralytics,
                quantization: quant1,
                shape: shape1.to_vec(),
                dshape: dshape.1,
            }),
        ],
        nms: None,
        decoder_version: None,
    })
}

fn guess_modelpack_segdet_3(
    shape: [&[usize]; 3],
    quant: [&Option<(f32, i32)>; 3],
) -> Option<ConfigOutputs> {
    // one output is 1/NB/NC
    // one output is 1/NB/1/4
    // one output is 1/H/W/NC+1

    let [mut shape0, mut shape1, mut shape2] = shape;
    let [mut quant0, mut quant1, mut quant2] = quant.map(|x| x.map(|x| x.into()));

    // find shape with a 1 and a 4
    if get_count_in_iter(shape0[1..].iter(), &1) == 2 && shape0.contains(&4) && shape0.len() == 4 {
        // shape0 is boxes
    } else if get_count_in_iter(shape1[1..].iter(), &1) == 2
        && shape1.contains(&4)
        && shape1.len() == 4
    {
        std::mem::swap(&mut shape0, &mut shape1);
        std::mem::swap(&mut quant0, &mut quant1);
    } else if get_count_in_iter(shape2[1..].iter(), &1) == 2
        && shape2.contains(&4)
        && shape2.len() == 4
    {
        std::mem::swap(&mut shape0, &mut shape2);
        std::mem::swap(&mut quant0, &mut quant2);
    } else {
        return None;
    }

    let num_boxes = shape0[1..].iter().find(|&&d| d > MIN_NUM_BOXES).cloned()?;

    // find shape with 1/NB/NC
    if shape1.len() == 3 && shape1[1..].contains(&num_boxes) {
        // shape1 is scores
    } else if shape2.len() == 3 && shape2[1..].contains(&num_boxes) {
        std::mem::swap(&mut shape1, &mut shape2);
        std::mem::swap(&mut quant1, &mut quant2);
    } else {
        return None;
    }

    let num_classes = shape1[1..].iter().find(|&&d| d != num_boxes).cloned()?;

    // shape0 is boxes, shape1 is scores, shape3 is segmentation
    let mut dshape_boxes: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    for &dim in &shape0[1..] {
        match dim {
            1 => dshape_boxes.push((DimName::Padding, 1)),
            4 => dshape_boxes.push((DimName::BoxCoords, 4)),
            d if d == num_boxes => {
                dshape_boxes.push((DimName::NumBoxes, d));
            }
            _ => return None,
        }
    }

    let mut dshape_scores: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    for &dim in &shape1[1..] {
        match dim {
            d if d == num_boxes => dshape_scores.push((DimName::NumBoxes, d)),
            d => dshape_scores.push((DimName::NumClasses, d)),
        }
    }

    let mut dshape_segmentation: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    let mut found_height = false;
    for &dim in &shape2[1..] {
        match dim {
            d if d == num_classes + 1 => {
                dshape_segmentation.push((DimName::NumClasses, d));
            }
            _ if !found_height => {
                dshape_segmentation.push((DimName::Height, dim));
                found_height = true;
            }
            _ if found_height => {
                dshape_segmentation.push((DimName::Width, dim));
            }
            _ => return None,
        }
    }

    let dshape = (dshape_boxes, dshape_scores, dshape_segmentation);

    Some(ConfigOutputs {
        outputs: vec![
            ConfigOutput::Boxes(Boxes {
                decoder: DecoderType::ModelPack,
                quantization: quant0,
                shape: shape0.to_vec(),
                dshape: dshape.0,
                normalized: None,
            }),
            ConfigOutput::Scores(Scores {
                decoder: DecoderType::ModelPack,
                quantization: quant1,
                shape: shape1.to_vec(),
                dshape: dshape.1,
            }),
            ConfigOutput::Segmentation(Segmentation {
                decoder: DecoderType::ModelPack,
                quantization: quant2,
                shape: shape2.to_vec(),
                dshape: dshape.2,
            }),
        ],
        nms: None,
        decoder_version: None,
    })
}

fn guess_modelpack_segdet_4(
    shape: [&[usize]; 4],
    quant: [&Option<(f32, i32)>; 4],
) -> Option<ConfigOutputs> {
    // one output is 1/NB/NC
    // one output is 1/NB/1/4
    // one output is 1/H/W/NC+1
    // one output is 1/H/W

    let [mut shape0, mut shape1, mut shape2, mut shape3] = shape;
    let [mut quant0, mut quant1, mut quant2, mut quant3] = quant.map(|x| x.map(|x| x.into()));

    // find shape with a 1 and a 4
    if get_count_in_iter(shape0[1..].iter(), &1) == 2 && shape0.contains(&4) {
        // shape0 is boxes
    } else if get_count_in_iter(shape1[1..].iter(), &1) == 2 && shape1.contains(&4) {
        std::mem::swap(&mut shape0, &mut shape1);
        std::mem::swap(&mut quant0, &mut quant1);
    } else if get_count_in_iter(shape2[1..].iter(), &1) == 2 && shape2.contains(&4) {
        std::mem::swap(&mut shape0, &mut shape2);
        std::mem::swap(&mut quant0, &mut quant2);
    } else if get_count_in_iter(shape3[1..].iter(), &1) == 2 && shape3.contains(&4) {
        std::mem::swap(&mut shape0, &mut shape3);
        std::mem::swap(&mut quant0, &mut quant3);
    } else {
        return None;
    }

    let num_boxes = shape0[1..].iter().find(|&&d| d > MIN_NUM_BOXES).cloned()?;

    // find shape with 1/NB/NC
    if shape1.len() == 3 && shape1[1..].contains(&num_boxes) {
        // shape1 is scores
    } else if shape2.len() == 3 && shape2[1..].contains(&num_boxes) {
        std::mem::swap(&mut shape1, &mut shape2);
        std::mem::swap(&mut quant1, &mut quant2);
    } else if shape3.len() == 3 && shape3[1..].contains(&num_boxes) {
        std::mem::swap(&mut shape1, &mut shape3);
        std::mem::swap(&mut quant1, &mut quant3);
    } else {
        return None;
    }

    let num_classes = shape1[1..].iter().find(|&&d| d != num_boxes).cloned()?;

    // find shape with 1/H/W/NC+1
    if shape2.len() == 4 {
        // shape2 is segmentation
    } else if shape3.len() == 4 {
        std::mem::swap(&mut shape2, &mut shape3);
        std::mem::swap(&mut quant2, &mut quant3);
    } else {
        return None;
    }

    // shape0 is boxes, shape1 is scores, shape2 is segmentation, shape3 is mask
    let mut dshape_boxes: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    for &dim in &shape0[1..] {
        match dim {
            1 => dshape_boxes.push((DimName::Padding, 1)),
            4 => dshape_boxes.push((DimName::BoxCoords, 4)),
            d if d == num_boxes => {
                dshape_boxes.push((DimName::NumBoxes, d));
            }
            _ => return None,
        }
    }

    let mut dshape_scores: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    for &dim in &shape1[1..] {
        match dim {
            d if d == num_boxes => dshape_scores.push((DimName::NumBoxes, d)),
            d => dshape_scores.push((DimName::NumClasses, d)),
        }
    }

    let mut dshape_segmentation: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    let mut found_height = false;
    for &dim in &shape2[1..] {
        match dim {
            d if d == num_classes + 1 => {
                dshape_segmentation.push((DimName::NumClasses, d));
            }
            _ if !found_height => {
                dshape_segmentation.push((DimName::Height, dim));
                found_height = true;
            }
            _ if found_height => {
                dshape_segmentation.push((DimName::Width, dim));
            }
            _ => return None,
        }
    }

    let dshape_mask: Vec<(DimName, usize)> = vec![
        (DimName::Batch, 1),
        (DimName::Height, shape3[1]),
        (DimName::Width, shape3[2]),
    ];

    let dshape = (
        dshape_boxes,
        dshape_scores,
        dshape_segmentation,
        dshape_mask,
    );

    Some(ConfigOutputs {
        outputs: vec![
            ConfigOutput::Boxes(Boxes {
                decoder: DecoderType::ModelPack,
                quantization: quant0,
                shape: shape0.to_vec(),
                dshape: dshape.0,
                normalized: None,
            }),
            ConfigOutput::Scores(Scores {
                decoder: DecoderType::ModelPack,
                quantization: quant1,
                shape: shape1.to_vec(),
                dshape: dshape.1,
            }),
            ConfigOutput::Segmentation(Segmentation {
                decoder: DecoderType::ModelPack,
                quantization: quant2,
                shape: shape2.to_vec(),
                dshape: dshape.2,
            }),
            ConfigOutput::Mask(Mask {
                decoder: DecoderType::ModelPack,
                quantization: quant3,
                shape: shape3.to_vec(),
                dshape: dshape.3,
            }),
        ],
        nms: None,
        decoder_version: None,
    })
}

fn guess_yolo_split_segdet(
    shape: [&[usize]; 4],
    quant: [&Option<(f32, i32)>; 4],
) -> Option<ConfigOutputs> {
    // one output is 1/4/NB
    // one output is 1/H/W/NP
    // one output is 1/NP/NB
    // one output is 1/NC/NB

    let [mut shape0, mut shape1, mut shape2, mut shape3] = shape;
    let [mut quant0, mut quant1, mut quant2, mut quant3] = quant.map(|x| x.map(|x| x.into()));

    // find shape with 4
    if shape0.contains(&4) {
        // shape0 is boxes
    } else if shape1.contains(&4) {
        std::mem::swap(&mut shape0, &mut shape1);
        std::mem::swap(&mut quant0, &mut quant1);
    } else if shape2.contains(&4) {
        std::mem::swap(&mut shape0, &mut shape2);
        std::mem::swap(&mut quant0, &mut quant2);
    } else if shape3.contains(&4) {
        std::mem::swap(&mut shape0, &mut shape3);
        std::mem::swap(&mut quant0, &mut quant3);
    } else {
        return None;
    }

    let num_boxes = shape0[1..].iter().find(|&&d| d > MIN_NUM_BOXES).cloned()?;

    // find shape with 1/H/W/NP
    if shape1.len() == 4 {
        // shape1 is protos
    } else if shape2.len() == 4 {
        std::mem::swap(&mut shape1, &mut shape2);
        std::mem::swap(&mut quant1, &mut quant2);
    } else if shape3.len() == 4 {
        std::mem::swap(&mut shape1, &mut shape3);
        std::mem::swap(&mut quant1, &mut quant3);
    } else {
        return None;
    }

    let num_protos = shape1[1..].iter().min().copied()?;

    // find shape with 1/NP/NB
    if shape2.len() == 3 && shape2[1..].contains(&num_protos) && shape2[1..].contains(&num_boxes) {
        // shape2 is proto coeffs
    } else if shape3.len() == 3
        && shape3[1..].contains(&num_protos)
        && shape3[1..].contains(&num_boxes)
    {
        std::mem::swap(&mut shape2, &mut shape3);
        std::mem::swap(&mut quant2, &mut quant3);
    } else {
        return None;
    }

    // confirm last shape is 1/NC/NB
    if shape3.len() != 3 || !shape3[1..].contains(&num_boxes) {
        return None;
    }

    // shape0 is boxes, shape1 is protos, shape2 is proto coeffs, shape3 is scores
    let mut dshape_boxes: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    for &dim in &shape0[1..] {
        match dim {
            4 => dshape_boxes.push((DimName::BoxCoords, 4)),
            d if d == num_boxes => {
                dshape_boxes.push((DimName::NumBoxes, d));
            }
            _ => return None,
        }
    }

    let mut dshape_protos: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    let mut found_height = false;
    for &dim in &shape1[1..] {
        match dim {
            n if n == num_protos => dshape_protos.push((DimName::NumProtos, num_protos)),
            _ if !found_height => {
                dshape_protos.push((DimName::Height, dim));
                found_height = true;
            }
            _ if found_height => {
                dshape_protos.push((DimName::Width, dim));
            }
            _ => return None,
        }
    }

    let mut dshape_mask_coeffs: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    for &dim in &shape2[1..] {
        match dim {
            d if d == num_protos => dshape_mask_coeffs.push((DimName::NumProtos, d)),
            d if d == num_boxes => dshape_mask_coeffs.push((DimName::NumBoxes, d)),
            _ => return None,
        }
    }

    let mut dshape_scores: Vec<(DimName, usize)> = vec![(DimName::Batch, 1)];
    for &dim in &shape3[1..] {
        match dim {
            d if d == num_boxes => dshape_scores.push((DimName::NumBoxes, d)),
            d => dshape_scores.push((DimName::NumClasses, d)),
        }
    }

    let dshape = (
        dshape_boxes,
        dshape_protos,
        dshape_mask_coeffs,
        dshape_scores,
    );

    Some(ConfigOutputs {
        outputs: vec![
            ConfigOutput::Boxes(Boxes {
                decoder: DecoderType::Ultralytics,
                quantization: quant0,
                shape: shape0.to_vec(),
                dshape: dshape.0,
                normalized: None,
            }),
            ConfigOutput::Protos(Protos {
                decoder: DecoderType::Ultralytics,
                quantization: quant1,
                shape: shape1.to_vec(),
                dshape: dshape.1,
            }),
            ConfigOutput::MaskCoefficients(MaskCoefficients {
                decoder: DecoderType::Ultralytics,
                quantization: quant2,
                shape: shape2.to_vec(),
                dshape: dshape.2,
            }),
            ConfigOutput::Scores(Scores {
                decoder: DecoderType::Ultralytics,
                quantization: quant3,
                shape: shape3.to_vec(),
                dshape: dshape.3,
            }),
        ],
        nms: None,
        decoder_version: None,
    })
}

fn get_count_in_iter<T: PartialEq>(iter: impl Iterator<Item = T>, val: T) -> usize {
    iter.filter(|x| x == &val).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guess_yolo_det_shape() {
        // YOLOv8n detection: 1x84x8400 (84 = 4 bbox + 80 classes)
        let shapes = vec![vec![1, 84, 8400]];
        let quants = vec![None];
        let config = guess_model_config(&shapes, &quants);
        assert!(config.is_some(), "Should detect YOLOv8 detection shape");
    }

    #[test]
    fn guess_empty_shapes() {
        let config = guess_model_config(&[], &[]);
        assert!(config.is_none(), "Empty shapes should return None");
    }

    #[test]
    fn model_error_display() {
        let err = ModelError::new(ModelErrorKind::TFLite, "test error".to_string());
        assert_eq!(format!("{err}"), "kind: TFLite, source: \"test error\"");
    }
}
