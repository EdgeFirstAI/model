use std::{error::Error, path::Path};

use edgefirst_schemas::edgefirst_msgs::DmaBuf;

use crate::image::ImageManager;

// #[derive(Debug)]
// struct ModelError {
//     kind: ModelErrorKind,
//     source: Box<dyn std::error::Error>,
// }

// enum ModelErrorKind {
//     IoError,
//     Other,
// }

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
    pub label: i32,
}

#[allow(dead_code)]
pub enum Preprocessing {
    Raw = 0x0,
    UnsignedNorm = 0x1,
    SignedNorm = 0x2,
    ImageNet = 0x8,
}

pub static RGB_MEANS_IMAGENET: [f32; 4] = [0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0, 128.0]; // last value is for Alpha channel when needed
pub static RGB_STDS_IMAGENET: [f32; 4] = [0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0, 64.0]; // last value is for Alpha channel when needed

pub trait Model {
    type ModelError;

    // fn load_model(&mut self, model: &[u8]) -> Result<(), Self::ModelError>;

    // fn load_model_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(),
    // Self::ModelError> {     let data = std::fs::read(path)?;
    //     self.load_model(&data)
    // }

    fn load_frame_dmabuf(
        &mut self,
        dmabuf: &DmaBuf,
        img_mgr: &ImageManager,
        preprocessing: Preprocessing,
    ) -> Result<(), Self::ModelError>;

    fn run_model(&mut self) -> Result<(), Self::ModelError>;

    fn input_count(&self) -> Result<usize, Self::ModelError>;
    fn input_shape(&self, index: usize) -> Result<Vec<usize>, Self::ModelError>;
    fn load_input(
        &mut self,
        index: usize,
        data: &[u8],
        preprocessing: Preprocessing,
    ) -> Result<(), Self::ModelError>;

    fn output_count(&self) -> Result<usize, Self::ModelError>;
    fn output_shape(&self, index: usize) -> Result<Vec<usize>, Self::ModelError>;
    fn output_data<T: Copy>(&self, index: usize, data: &mut [T]) -> Result<(), Self::ModelError>;

    fn boxes(&self, boxes: &mut [DetectBox]) -> Result<usize, Self::ModelError>;
}
