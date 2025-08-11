use std::{error::Error, fmt};

use crate::{image::ImageManager, tflite_model::TFLiteModel};
use edgefirst_schemas::edgefirst_msgs::DmaBuf;
use log::error;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use tflitec_sys::TfLiteError;

use enum_dispatch::enum_dispatch;

#[cfg(feature = "rtm")]
use crate::rtm_model::RtmModel;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
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
    pub index: usize,
    pub name: String,
    pub output_index: usize,
    pub quantization: Option<[f32; 2]>,
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Mask {
    pub decode: bool,
    pub decoder: Decoder,
    pub dtype: DataType,
    pub index: usize,
    pub name: String,
    pub output_index: usize,
    pub quantization: Option<[f32; 2]>,
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Detection {
    pub anchors: Vec<[f32; 2]>,
    pub decode: bool,
    pub decoder: Decoder,
    pub dtype: DataType,
    pub index: usize,
    pub name: String,
    pub output_index: usize,
    pub quantization: Option<[f32; 2]>, // this quantization isn't used for dequant
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Scores {
    pub decoder: Decoder,
    pub dtype: DataType,
    pub index: usize,
    pub name: String,
    pub output_index: usize,
    pub quantization: Option<[f32; 2]>, // this quantization isn't used for dequant
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Boxes {
    pub decoder: Decoder,
    pub dtype: DataType,
    pub index: usize,
    pub name: String,
    pub output_index: usize,
    pub quantization: Option<[f32; 2]>, // this quantization isn't used for dequant
    pub shape: Vec<usize>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub enum Decoder {
    #[serde(rename = "modelpack")]
    ModelPack,
    #[serde(rename = "yolo")]
    Yolo,
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
                        error!("Yaml Error {err:?}");
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

    fn boxes(&self, boxes: &mut [DetectBox]) -> Result<usize, ModelError>;

    fn get_model_metadata(&self) -> Result<Metadata, ModelError>;
}

// Decodes each tensors into box coordinates and scores. Multiple tensors will
// have their ouput box coordinates/scores appended together.
// Output is (box coordinates, scores, number of classes)
pub fn decode_detection_outputs(
    outputs: Vec<Vec<f32>>,
    details: &[Detection],
) -> (Vec<f32>, Vec<f32>, usize) {
    let mut total_capacity = 0;
    let mut nc = 0;
    for detail in details {
        let shape = &detail.shape;
        let na = detail.anchors.len();
        nc = *shape.last().unwrap() / na - 5;
        total_capacity += shape[1] * shape[2] * na;
    }
    let mut bboxes = Vec::with_capacity(total_capacity * 4);
    let mut bscores = Vec::with_capacity(total_capacity * nc);
    // bboxes, bscores = [], []

    for (mut p, detail) in outputs.into_iter().zip(details) {
        p.iter_mut().for_each(|x| *x = sigmoid(*x));

        let anchors = &detail.anchors;
        let na = detail.anchors.len();
        let shape = &detail.shape;
        assert_eq!(
            shape.iter().product::<usize>(),
            p.len(),
            "Shape product doesn't match tensor length"
        );
        let height = shape[1];
        let width = shape[2];

        let mut grid = Vec::new();
        for y in 0..height {
            for x in 0..width {
                for _ in 0..na {
                    grid.push(x as f32);
                    grid.push(y as f32);
                }
            }
        }
        // let grid = Array::from_shape_vec((h, w, na, nc + 5), grid).unwrap();
        for (p, g) in p.chunks_exact(na * (nc + 5)).zip(grid.chunks_exact(na * 2)) {
            for (anchor_ind, (p, g)) in p.chunks_exact(nc + 5).zip(g.chunks_exact(2)).enumerate() {
                let (x, y) = (p[0], p[1]);
                let x = (x * 2.0 + g[0] - 0.5) / width as f32;
                let y = (y * 2.0 + g[1] - 0.5) / height as f32;
                let (w, h) = (p[2], p[3]);
                let w_half = w * w * 2.0 * anchors[anchor_ind][0];
                let h_half = h * h * 2.0 * anchors[anchor_ind][1];

                let obj = p[4];
                let probs = p[5..(nc + 5)].iter().map(|x| *x * obj);
                bboxes.push(x - w_half);
                bboxes.push(y - h_half);
                bboxes.push(x + w_half);
                bboxes.push(y + h_half);
                bscores.extend(probs);
            }
        }
    }

    (bboxes, bscores, nc)
}

#[inline]
pub fn sigmoid(f: f32) -> f32 {
    use std::f32::consts::E;
    1.0 / (1.0 + E.powf(-f))
}

#[inline]
pub fn dequant<T: AsPrimitive<f32>>(data: &[T], output: &mut [f32], scale: f32, zero_point: f32) {
    let scaled_zp = -scale * zero_point;
    data.iter()
        .zip(output.iter_mut())
        .for_each(|(d, out)| *out = scale * (*d).as_() + scaled_zp);
}
