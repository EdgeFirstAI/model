use std::{error::Error, fmt, path::Path};

use edgefirst_schemas::edgefirst_msgs::DmaBuf;
use tflitec_sys::TfLiteError;

use crate::{image::ImageManager, rtm_model::RtmModel, tflite_model::TFLiteModel};

// #[derive(Debug)]
// struct ModelError {
//     kind: ModelErrorKind,
//     source: Box<dyn std::error::Error>,
// }

// enum ModelErrorKind {
//     IoError,
//     Other,
// }

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

pub enum SupportedModel<'a> {
    RtmModel(RtmModel),
    TfLiteModel(TFLiteModel<'a>),
}

impl<'a> SupportedModel<'a> {
    pub fn from_rtm_model(model: RtmModel) -> Self {
        Self::RtmModel(model)
    }

    pub fn from_tflite_model(model: TFLiteModel<'a>) -> Self {
        Self::TfLiteModel(model)
    }
}

impl<'a> Model for SupportedModel<'a> {
    fn load_frame_dmabuf(
        &mut self,
        dmabuf: &DmaBuf,
        img_mgr: &ImageManager,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        match self {
            Self::RtmModel(m) => m.load_frame_dmabuf(dmabuf, img_mgr, preprocessing),
            Self::TfLiteModel(m) => m.load_frame_dmabuf(dmabuf, img_mgr, preprocessing),
        }
    }

    fn run_model(&mut self) -> Result<(), ModelError> {
        match self {
            Self::RtmModel(m) => m.run_model(),
            Self::TfLiteModel(m) => m.run_model(),
        }
    }

    fn input_count(&self) -> Result<usize, ModelError> {
        match self {
            Self::RtmModel(m) => m.input_count(),
            Self::TfLiteModel(m) => m.input_count(),
        }
    }

    fn input_shape(&self, index: usize) -> Result<Vec<usize>, ModelError> {
        match self {
            Self::RtmModel(m) => m.input_shape(index),
            Self::TfLiteModel(m) => m.input_shape(index),
        }
    }

    fn load_input(
        &mut self,
        index: usize,
        data: &[u8],
        data_channels: usize,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        match self {
            Self::RtmModel(m) => m.load_input(index, data, data_channels, preprocessing),
            Self::TfLiteModel(m) => m.load_input(index, data, data_channels, preprocessing),
        }
    }

    fn output_count(&self) -> Result<usize, ModelError> {
        match self {
            Self::RtmModel(m) => m.output_count(),
            Self::TfLiteModel(m) => m.output_count(),
        }
    }

    fn output_shape(&self, index: usize) -> Result<Vec<usize>, ModelError> {
        match self {
            Self::RtmModel(m) => m.output_shape(index),
            Self::TfLiteModel(m) => m.output_shape(index),
        }
    }

    fn output_data<T: Copy>(&self, index: usize, data: &mut [T]) -> Result<(), ModelError> {
        match self {
            Self::RtmModel(m) => m.output_data(index, data),
            Self::TfLiteModel(m) => m.output_data(index, data),
        }
    }

    fn boxes(&self, boxes: &mut [DetectBox]) -> Result<usize, ModelError> {
        match self {
            Self::RtmModel(m) => m.boxes(boxes),
            Self::TfLiteModel(m) => m.boxes(boxes),
        }
    }

    fn input_type(&self, index: usize) -> Result<DataType, ModelError> {
        match self {
            Self::RtmModel(m) => m.input_type(index),
            Self::TfLiteModel(m) => m.input_type(index),
        }
    }

    fn output_type(&self, index: usize) -> Result<DataType, ModelError> {
        match self {
            Self::RtmModel(m) => m.output_type(index),
            Self::TfLiteModel(m) => m.output_type(index),
        }
    }

    fn labels(&self) -> Result<Vec<String>, ModelError> {
        match self {
            Self::RtmModel(m) => m.labels(),
            Self::TfLiteModel(m) => m.labels(),
        }
    }

    fn model_name(&self) -> Result<String, ModelError> {
        match self {
            Self::RtmModel(m) => m.model_name(),
            Self::TfLiteModel(m) => m.model_name(),
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
    IoError,
    TFLiteError,
    RtmError,
    Other,
}

impl fmt::Display for ModelErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<std::io::Error> for ModelError {
    fn from(value: std::io::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::IoError,
            source: Box::from(value),
        }
    }
}

impl From<TfLiteError> for ModelError {
    fn from(value: TfLiteError) -> Self {
        ModelError {
            kind: ModelErrorKind::TFLiteError,
            source: Box::from(value),
        }
    }
}

impl From<deepviewrt::error::Error> for ModelError {
    fn from(value: deepviewrt::error::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::RtmError,
            source: Box::from(value),
        }
    }
}

impl From<vaal::error::Error> for ModelError {
    fn from(value: vaal::error::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::RtmError,
            source: Box::from(value),
        }
    }
}

impl From<vaal::deepviewrt::error::Error> for ModelError {
    fn from(value: vaal::deepviewrt::error::Error) -> Self {
        ModelError {
            kind: ModelErrorKind::RtmError,
            source: Box::from(value),
        }
    }
}

impl Error for ModelError {}

pub enum DataType {
    RAW = 0,
    INT8 = 1,
    UINT8 = 2,
    INT16 = 3,
    UINT16 = 4,
    FLOAT16 = 5,
    INT32 = 6,
    UINT32 = 7,
    FLOAT32 = 8,
    INT64 = 9,
    UINT64 = 10,
    FLOAT64 = 11,
    STRING = 12,
}

pub trait Model {
    // fn load_model(&mut self, model: &[u8]) -> Result<(), Self::ModelError>;

    // fn load_model_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(),
    // Self::ModelError> {     let data = std::fs::read(path)?;
    //     self.load_model(&data)
    // }

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
}
