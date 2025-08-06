use std::{error::Error, fmt};

use edgefirst_schemas::edgefirst_msgs::DmaBuf;
use tflitec_sys::TfLiteError;
use yaml_rust2::Yaml;

use crate::{image::ImageManager, tflite_model::TFLiteModel};

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

#[derive(Debug, Default, Eq, PartialEq)]
pub struct Metadata {
    pub name: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub config: Option<Yaml>,
}

impl From<tflitec_sys::metadata::Metadata> for Metadata {
    fn from(value: tflitec_sys::metadata::Metadata) -> Self {
        Self {
            name: value.name,
            version: value.version,
            description: value.description,
            author: value.author,
            license: value.license,
            config: value.config,
        }
    }
}

#[derive(Debug)]
pub struct ModelError {
    kind: ModelErrorKind,
    source: Box<dyn std::error::Error>,
}

#[derive(Debug)]
enum ModelErrorKind {
    Io,
    TFLite,
    #[cfg(feature = "rtm")]
    Rtm,
    Other,
}

impl fmt::Display for ModelErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
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

#[derive(Debug, Clone, PartialEq, Eq)]
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

    fn get_model_metadata(&self) -> Metadata;
}
