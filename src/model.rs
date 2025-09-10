use std::{error::Error, fmt};

use crate::{BoxWithTrack, image::ImageManager, tflite_model::TFLiteModel};
use edgefirst_schemas::edgefirst_msgs::DmaBuf;
use log::error;
use ndarray::{
    Array1, Array2, Array3, ArrayView1, ArrayView3,
    parallel::prelude::{IntoParallelIterator, ParallelIterator},
    s,
};
use ndarray_stats::QuantileExt;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use tflitec_sys::TfLiteError;

use enum_dispatch::enum_dispatch;
use tracing::instrument;

#[cfg(feature = "rtm")]
use crate::rtm_model::RtmModel;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct DetectBox {
    #[doc = " left-most normalized coordinate of the bounding box."]
    pub xmin: f32,
    #[doc = " top-most normalized coordinate of the bounding box."]
    pub ymin: f32,
    #[doc = " right-most normalized coordinate of the bounding box."]
    pub xmax: f32,
    #[doc = " bottom-most normalized coordinate of the bounding box."]
    pub ymax: f32,
    #[doc = " model-specific score for this detection, higher implies more confidence."]
    pub score: f32,
    #[doc = " label index for this detection, text representation can be retrived using\n @ref VAALContext::vaal_label()"]
    pub label: usize,
    #[doc = " Optional mask coefficients for computing instanced masks"]
    pub mask_coeff: Option<Array1<f32>>,
}

pub struct SegmentationMask {
    #[doc = " left-most normalized coordinate of the mask bounding box."]
    pub xmin: f32,
    #[doc = " top-most normalized coordinate of the mask bounding box."]
    pub ymin: f32,
    #[doc = " right-most normalized coordinate of the mask bounding box."]
    pub xmax: f32,
    #[doc = " bottom-most normalized coordinate of the mask bounding box."]
    pub ymax: f32,
    #[doc = " Mask data"]
    pub mask: Array2<u8>,
}

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
    pub config: Option<ConfigOutputs>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct ConfigOutputs {
    pub outputs: Vec<ConfigOutput>,
}
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum ConfigOutput {
    #[serde(rename = "detection")]
    Detection(Detection),
    #[serde(rename = "masks")]
    Mask(Mask),
    #[serde(rename = "segmentation")]
    Segmentation(Segmentation),
    #[serde(rename = "scores")]
    Scores(Scores),
    #[serde(rename = "boxes")]
    Boxes(Boxes),
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Segmentation {
    pub decode: bool,
    pub decoder: Decoder,
    pub dtype: DataType,
    pub name: String,
    pub quantization: Option<[f32; 2]>,
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Mask {
    pub decode: bool,
    pub decoder: Decoder,
    pub dtype: DataType,
    pub name: String,
    pub quantization: Option<[f32; 2]>,
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Detection {
    pub anchors: Option<Vec<[f32; 2]>>,
    pub decode: bool,
    pub decoder: Decoder,
    pub dtype: DataType,
    pub quantization: Option<[f32; 2]>, // this quantization isn't used for dequant
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Scores {
    pub decoder: Decoder,
    pub dtype: DataType,
    pub quantization: Option<[f32; 2]>, // this quantization isn't used for dequant
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Boxes {
    pub decoder: Decoder,
    pub dtype: DataType,
    pub name: String,
    pub quantization: Option<[f32; 2]>, // this quantization isn't used for dequant
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub enum Decoder {
    #[serde(rename = "modelpack")]
    ModelPack,
    #[serde(rename = "yolov8")]
    Yolov8,
}

impl From<tflitec_sys::metadata::Metadata> for Metadata {
    fn from(value: tflitec_sys::metadata::Metadata) -> Self {
        Self {
            name: value.name,
            version: value.version,
            description: value.description,
            author: value.author,
            license: value.license,
            config: match value.config_yaml {
                Some(yaml) => match serde_yaml::from_str::<ConfigOutputs>(&yaml) {
                    Ok(parsed) => Some(parsed),
                    Err(err) => {
                        error!("Yaml Error {err}");
                        None
                    }
                },
                None => None,
            },
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

impl Error for ModelError {}

impl ModelError {
    pub fn new(kind: ModelErrorKind, msg: String) -> Self {
        ModelError {
            kind,
            source: msg.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Raw = 0,
    Int8 = 1,
    UInt8 = 2,
    Int16 = 3,
    UInt16 = 4,
    Float16 = 5,
    Int32 = 6,
    UInt32 = 7,
    Float32 = 8,
    Int64 = 9,
    UInt64 = 10,
    Float64 = 11,
    String = 12,
}

#[enum_dispatch]
pub trait Model {
    fn model_name(&self) -> Result<String, ModelError>;

    fn load_frame_dmabuf(
        &mut self,
        dmabuf: &DmaBuf,
        img_mgr: &ImageManager,
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

    fn labels(&self) -> Result<Vec<String>, ModelError>;

    fn decode_outputs(
        &mut self,
        boxes: &mut Vec<DetectBox>,
        protos: &mut Option<Array3<f32>>,
    ) -> Result<(), ModelError>;

    fn get_model_metadata(&self) -> Result<Metadata, ModelError>;
}

// Decodes each tensors into box coordinates and scores. Multiple tensors will
// have their ouput box coordinates/scores appended together.
// Output is (box coordinates, scores, number of classes)
pub fn decode_model_pack_detection_outputs(
    outputs: Vec<Vec<f32>>,
    details: &[&Detection],
) -> (Array2<f32>, Array2<f32>) {
    let mut total_capacity = 0;
    let mut nc = 0;
    for detail in details {
        let shape = &detail.shape;
        let na = detail.anchors.as_ref().unwrap().len();
        nc = *shape.last().unwrap() / na - 5;
        total_capacity += shape[1] * shape[2] * na;
    }
    let mut bboxes = Vec::with_capacity(total_capacity * 4);
    let mut bscores = Vec::with_capacity(total_capacity * nc);

    for (mut p, detail) in outputs.into_iter().zip(details) {
        p.iter_mut().for_each(|x| *x = fast_sigmoid(*x));

        let anchors = detail.anchors.as_ref().unwrap();
        let na = anchors.len();
        let shape = &detail.shape;
        assert_eq!(
            shape.iter().product::<usize>(),
            p.len(),
            "Shape product doesn't match tensor length"
        );
        let height = shape[1];
        let width = shape[2];

        let mut grid = Vec::with_capacity(height * width * na * 2);
        for y in 0..height {
            for x in 0..width {
                for _ in 0..na {
                    grid.push(x as f32);
                    grid.push(y as f32);
                }
            }
        }

        let div_width = 1.0 / width as f32;
        let div_height = 1.0 / height as f32;
        for ((p, g), anchor) in p
            .chunks_exact(nc + 5)
            .zip(grid.chunks_exact(2))
            .zip(anchors.iter().cycle())
        {
            let (x, y) = (p[0], p[1]);
            let x = (x * 2.0 + g[0] - 0.5) * div_width;
            let y = (y * 2.0 + g[1] - 0.5) * div_height;
            let (w, h) = (p[2], p[3]);
            let w_half = w * w * 2.0 * anchor[0];
            let h_half = h * h * 2.0 * anchor[1];

            bboxes.push(x - w_half);
            bboxes.push(y - h_half);
            bboxes.push(x + w_half);
            bboxes.push(y + h_half);

            let obj = p[4];
            let probs = p[5..].iter().map(|x| *x * obj);
            bscores.extend(probs);
        }
    }

    let bboxes = Array2::from_shape_vec((bboxes.len() / 4, 4), bboxes).unwrap();
    let bscores = Array2::from_shape_vec((bscores.len() / nc, nc), bscores).unwrap();

    (bboxes, bscores)
}

pub fn decode_yolo_outputs_seg(
    mut outputs: Vec<Vec<f32>>,
    details: &[&Segmentation],
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array3<f32>) {
    let mut proto_detail = details[0];
    let mut boxes_detail = details[1];
    if proto_detail.shape.len() != 4 {
        std::mem::swap(&mut boxes_detail, &mut proto_detail);
    }
    assert_eq!(boxes_detail.shape.len(), 3);
    assert_eq!(proto_detail.shape.len(), 4);

    let mut proto_output = outputs.pop().unwrap();
    let mut boxes_output = outputs.pop().unwrap();

    if proto_output.len() != proto_detail.shape.iter().product::<usize>() {
        std::mem::swap(&mut boxes_output, &mut proto_output);
    }

    assert_eq!(
        proto_output.len(),
        proto_detail.shape.iter().product::<usize>()
    );
    assert_eq!(
        boxes_output.len(),
        boxes_detail.shape.iter().product::<usize>()
    );

    let nc = boxes_detail.shape[1] - 4 - 32;
    let boxes_output =
        Array2::from_shape_vec((boxes_detail.shape[1], boxes_detail.shape[2]), boxes_output)
            .unwrap();

    let bboxes = boxes_output.slice(s![..4, ..]).to_owned().reversed_axes();
    let bscores = boxes_output
        .slice(s![4..(4 + nc), ..])
        .to_owned()
        .reversed_axes();
    let bmasks = boxes_output
        .slice(s![(4 + nc).., ..])
        .to_owned()
        .reversed_axes();

    let mut protos = Array3::from_shape_vec(
        (
            proto_detail.shape[1],
            proto_detail.shape[2],
            proto_detail.shape[3],
        ),
        proto_output,
    )
    .unwrap();
    // swap axis to become protos x height x width
    protos.swap_axes(0, 2);
    protos.swap_axes(1, 2);
    (bboxes, bscores, bmasks, protos)
}

pub fn decode_yolo_outputs_det(
    boxes_output: Vec<f32>,
    boxes_detail: &Detection,
) -> (Array2<f32>, Array2<f32>) {
    assert_eq!(boxes_detail.shape.len(), 3);

    assert_eq!(
        boxes_output.len(),
        boxes_detail.shape.iter().product::<usize>()
    );

    let boxes_output =
        Array2::from_shape_vec((boxes_detail.shape[1], boxes_detail.shape[2]), boxes_output)
            .unwrap();

    let bboxes = boxes_output.slice(s![..4, ..]).to_owned().reversed_axes();
    let bscores = boxes_output.slice(s![4.., ..]).to_owned().reversed_axes();

    (bboxes, bscores)
}

#[inline(always)]
#[allow(dead_code)]
pub fn sigmoid(f: f32) -> f32 {
    use std::f32::consts::E;
    1.0 / (1.0 + E.powf(-f))
}

#[inline(always)]
pub fn fast_sigmoid(f: f32) -> f32 {
    if f.abs() > 80.0 {
        f.signum() * 0.5 + 0.5
    } else {
        1.0 / (1.0 + fast_math::exp_raw(-f))
    }
}
#[inline]
#[allow(dead_code)]
/// A fast polynomial sigmoid approximation. Roughly 7x faster than the sigmoid
/// function. See https://www.desmos.com/calculator/g4e3vbju6l for a visual comparison
/// Not suitable for SIMD usage on the Arm Cortex-a53 based on benchmark results
pub fn fast_sigmoid2(f: f32) -> f32 {
    if f.abs() > 8.5 {
        f.signum() * 0.5 + 0.5
    } else if f.abs() > 4.95716 {
        0.5 + 0.278857_f32 * f.signum() + 0.108554 * f - 0.020359 * f * f.abs()
            + 0.00172085 * f.powi(3)
            - 0.0000550985 * f.powi(3) * f.abs()
    } else if f.abs() > 1.48487 {
        0.5 - 0.0525242_f32 * f.signum() + 0.3658526 * f - 0.09530283 * f * f.abs()
            + 0.01137951 * f.powi(3)
            - 0.0005171742 * f.powi(3) * f.abs()
    } else {
        -0.016123232 * f.powi(3) + 0.24758115 * f + 0.5
    }
}

#[inline]
pub fn dequant<T: AsPrimitive<f32>>(data: &[T], output: &mut [f32], scale: f32, zero_point: f32) {
    let scaled_zp = -scale * zero_point;
    data.iter()
        .zip(output.iter_mut())
        .for_each(|(d, out)| *out = scale * (*d).as_() + scaled_zp);
}

#[instrument(skip_all)]
pub fn decode_masks(
    boxes: &[BoxWithTrack],
    protos: ArrayView3<f32>,
) -> Vec<Option<(Array2<u8>, [usize; 4])>> {
    if boxes.is_empty() {
        return Vec::new();
    }

    boxes
        .into_par_iter()
        // .iter()
        .map(|b| {
            if let Some(mask) = &b.mask_coeff {
                let (protos, shape, roi) = protobox(&protos, &[b.xmin, b.ymin, b.xmax, b.ymax]);
                Some((make_mask(mask.view(), protos.view(), shape), roi))
            } else {
                None
            }
        })
        .collect()
}

#[instrument(skip_all)]
pub fn protobox<'a>(
    protos: &'a ArrayView3<f32>,
    roi: &[f64; 4],
) -> (ArrayView3<'a, f32>, [usize; 3], [usize; 4]) {
    let width = protos.dim().2 as f64;
    let height = protos.dim().1 as f64;
    let roi = [
        (roi[0] * width - 0.5).clamp(0.0, width) as usize,
        (roi[1] * height - 0.5).clamp(0.0, height) as usize,
        (roi[2] * width + 0.5).clamp(0.0, width).ceil() as usize,
        (roi[3] * height + 0.5).clamp(0.0, height).ceil() as usize,
    ];

    let shape = [protos.dim().0, (roi[3] - roi[1]), (roi[2] - roi[0])];
    if shape[0] * shape[1] * shape[2] == 0 {
        return (protos.slice(s![.., 0..0, 0..0]), shape, roi);
    }

    let cropped = protos.slice(s![.., roi[1]..roi[3], roi[0]..roi[2]]);
    let shape = [cropped.shape()[0], cropped.shape()[1], cropped.shape()[2]];

    (cropped, shape, roi)
}

#[instrument(skip_all)]
#[allow(dead_code)]
pub fn protobox_safe(protos: ArrayView3<f32>, roi: &[f32; 4]) -> (Array2<f32>, [usize; 3]) {
    let roi = [
        (roi[0] * 0.25).clamp(0.0, 160.0) as usize,
        (roi[1] * 0.25).clamp(0.0, 160.0) as usize,
        (roi[2] * 0.25).clamp(0.0, 160.0) as usize,
        (roi[3] * 0.25).clamp(0.0, 160.0) as usize,
    ];
    let protos = protos
        .slice(s![.., roi[1]..roi[3], roi[0]..roi[2]])
        .as_standard_layout()
        .to_owned();
    let shape = protos.dim();
    (
        protos
            .into_shape_with_order((shape.0, shape.1 * shape.2))
            .unwrap(),
        [shape.0, shape.1, shape.2],
    )
}

#[instrument(skip_all)]
pub fn make_mask(mask: ArrayView1<f32>, protos: ArrayView3<f32>, shape: [usize; 3]) -> Array2<u8> {
    let mask = mask.into_shape_with_order((1, mask.len())).unwrap();
    let protos = protos
        .into_shape_with_order([shape[0], shape[1] * shape[2]])
        .unwrap();

    let mask = mask
        .dot(&protos)
        .into_shape_with_order((shape[1], shape[2]))
        .unwrap();

    let min = *mask.min_skipnan();
    let max = *mask.max_skipnan();

    mask.map(|x| ((x - min) / max * 255.0) as u8)
}
