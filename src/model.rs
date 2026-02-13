// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use std::{
    error::Error,
    fmt,
    os::fd::{FromRawFd, OwnedFd},
};

use edgefirst_decoder::{
    ConfigOutput, ConfigOutputs,
    configs::{
        Boxes, DataType, DecoderType, Detection, DimName, Mask, MaskCoefficients, Protos, Scores,
        Segmentation,
    },
};
use edgefirst_image::{ImageProcessor, TensorImage};
use edgefirst_schemas::edgefirst_msgs::DmaBuffer;
use edgefirst_tensor::{Tensor, TensorTrait};
use enum_dispatch::enum_dispatch;
use tflitec_sys::TfLiteError;
use tracing::instrument;

#[cfg(feature = "rtm")]
use crate::rtm_model::RtmModel;

use crate::tflite_model::TFLiteModel;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preprocessing {
    Raw = 0x0,
    UnsignedNorm = 0x1,
    SignedNorm = 0x2,
    ImageNet = 0x8,
}

pub static RGB_MEANS_IMAGENET: [f32; 4] = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0, 128.0]; // last value is for Alpha channel when needed
pub static RGB_STDS_IMAGENET: [f32; 4] = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0, 64.0]; // last value is for Alpha channel when needed

#[enum_dispatch(Model)]
pub enum SupportedModel<'a> {
    TfLiteModel(TFLiteModel<'a>),
    #[cfg(feature = "rtm")]
    RtmModel(RtmModel),
}

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Metadata {
    pub name: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub config_yaml: Option<String>,
}

impl From<tflitec_sys::metadata::Metadata> for Metadata {
    fn from(value: tflitec_sys::metadata::Metadata) -> Self {
        Self {
            name: value.name,
            version: value.version,
            description: value.description,
            author: value.author,
            license: value.license,
            config_yaml: value.config_yaml,
        }
    }
}

#[derive(Debug)]
pub struct ModelError {
    kind: ModelErrorKind,
    source: Box<dyn std::error::Error>,
}

#[derive(Debug)]
pub enum ModelErrorKind {
    Io,
    TFLite,
    #[cfg(feature = "rtm")]
    Rtm,
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

impl From<TfLiteError> for ModelError {
    fn from(value: TfLiteError) -> Self {
        ModelError {
            kind: ModelErrorKind::TFLite,
            source: Box::from(value),
        }
    }
}

#[cfg(feature = "rtm")]
impl From<vaal::error::Error> for ModelError {
    fn from(value: vaal::error::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::Rtm,
            source: Box::from(value),
        }
    }
}

#[cfg(feature = "rtm")]
impl From<vaal::deepviewrt::error::Error> for ModelError {
    fn from(value: vaal::deepviewrt::error::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::Rtm,
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

impl From<edgefirst_tensor::Error> for ModelError {
    fn from(value: edgefirst_tensor::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::Tensor,
            source: Box::from(value),
        }
    }
}

impl From<edgefirst_image::Error> for ModelError {
    fn from(value: edgefirst_image::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::Image,
            source: Box::from(value),
        }
    }
}

impl From<edgefirst_decoder::DecoderError> for ModelError {
    fn from(value: edgefirst_decoder::DecoderError) -> Self {
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

#[enum_dispatch]
pub trait Model {
    fn model_name(&self) -> Result<String, ModelError>;

    fn load_frame_dmabuf_(
        &mut self,
        dmabuf: &DmaBuffer,
        img_mgr: &mut ImageProcessor,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError>;

    fn run_model(&mut self) -> Result<(), ModelError>;

    fn input_count(&self) -> Result<usize, ModelError>;
    fn input_shape(&self, index: usize) -> Result<Vec<usize>, ModelError>;
    fn input_type(&self, index: usize) -> Result<DataType, ModelError>;
    fn load_input(
        &mut self,
        index: usize,
        data: &[u8],
        data_channels: usize,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError>;

    fn output_count(&self) -> Result<usize, ModelError>;
    fn output_shape(&self, index: usize) -> Result<Vec<usize>, ModelError>;
    fn output_type(&self, index: usize) -> Result<DataType, ModelError>;
    fn output_data<T: Copy>(&self, index: usize, data: &mut [T]) -> Result<(), ModelError>;
    fn output_quantization(&self, index: usize) -> Result<Option<(f32, i32)>, ModelError>;

    fn labels(&self) -> Result<Vec<String>, ModelError>;

    fn decode_outputs(
        &self,
        decoder: &edgefirst_decoder::Decoder,
        output_boxes: &mut Vec<edgefirst_decoder::DetectBox>,
        output_masks: &mut Vec<edgefirst_decoder::Segmentation>,
    ) -> Result<(), ModelError>;

    fn decode_outputs_tracked(
        &self,
        decoder: &mut edgefirst_decoder::Decoder,
        output_boxes: &mut Vec<edgefirst_decoder::DetectBox>,
        output_masks: &mut Vec<edgefirst_decoder::Segmentation>,
        output_tracks: &mut Vec<edgefirst_tracker::TrackInfo<edgefirst_decoder::DetectBox>>,
        timestamp: u64,
    ) -> Result<(), ModelError>;

    fn get_model_metadata(&self) -> Result<Metadata, ModelError>;
}

use nix::libc::dup;
#[instrument(skip_all)]
pub(crate) fn dmabuf_to_tensor_image(
    dma: &DmaBuffer,
) -> Result<edgefirst_image::TensorImage, ModelError> {
    let tensor = Tensor::from_fd(
        unsafe { OwnedFd::from_raw_fd(dup(dma.fd)) },
        &[
            dma.height as usize,
            dma.width as usize,
            (dma.length / dma.width / dma.height) as usize,
        ],
        None,
    )?;

    let fourcc = edgefirst_image::FourCharCode::from_array(dma.fourcc.to_le_bytes())
        .map_err(|e| ModelError::new(ModelErrorKind::Image, format!("Invalid FourCC code: {e}")))?;
    let img = TensorImage::from_tensor(tensor, fourcc)?;
    Ok(img)
}

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
        })],
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
            }),
            ConfigOutput::Scores(Scores {
                decoder: DecoderType::ModelPack,
                quantization: quant1,
                shape: shape1.to_vec(),
                dshape: dshape.1,
            }),
        ],
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
            }),
        ],
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
            }),
            ConfigOutput::Scores(Scores {
                decoder: DecoderType::Ultralytics,
                quantization: quant1,
                shape: shape1.to_vec(),
                dshape: dshape.1,
            }),
        ],
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
    })
}

fn get_count_in_iter<T: PartialEq>(iter: impl Iterator<Item = T>, val: T) -> usize {
    iter.filter(|x| x == &val).count()
}
