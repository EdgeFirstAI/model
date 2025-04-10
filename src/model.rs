use std::{error::Error, fmt, iter::Zip, path::Path};

use edgefirst_schemas::edgefirst_msgs::DmaBuf;
use ndarray::{Array2, ArrayView2};
use tflitec_sys::TfLiteError;

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

pub trait Model {
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
    ) -> Result<(), ModelError>;

    fn run_model(&mut self) -> Result<(), ModelError>;

    fn input_count(&self) -> Result<usize, ModelError>;
    fn input_shape(&self, index: usize) -> Result<Vec<usize>, ModelError>;
    fn load_input(
        &mut self,
        index: usize,
        data: &[u8],
        data_channels: usize,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError>;

    fn output_count(&self) -> Result<usize, ModelError>;
    fn output_shape(&self, index: usize) -> Result<Vec<usize>, ModelError>;
    fn output_data<T: Copy>(&self, index: usize, data: &mut [T]) -> Result<(), ModelError>;

    fn boxes(&self, boxes: &mut [DetectBox]) -> Result<usize, ModelError>;
}

// pub fn decode_boxes(
//     threshold: f32,
//     scores: &[f32],
//     boxes: &[f32],
//     num_classes: usize,
// ) -> Vec<(usize, f32, [f32; 4])> {
//     let scores = ArrayView2::from_shape([scores.len() / num_classes,
// num_classes], scores).unwrap();     let boxes =
// ArrayView2::from_shape([boxes.len() / 4, 4], boxes).unwrap();
//     Zip::from(scores.columns())
//         .and(boxes.columns())
//         .into_par_iter()
//         .filter(|(score, _)| *score.max().unwrap() > threshold)
//         .map(|(score, bbox)| {
//             let label = score.argmax().unwrap();
//             (
//                 label,
//                 score[label],
//                 [
//                     (grid[0] - bbox[0]) * grid[2],
//                     (grid[1] - bbox[1]) * grid[2],
//                     (grid[0] + bbox[2]) * grid[2],
//                     (grid[1] + bbox[3]) * grid[2],
//                 ],
//             )
//         })
//         .collect()
// }

// pub fn nms(
//     iou: f32,
//     mut boxes: Vec<(usize, f32, &[f32], [f32; 4])>,
// ) -> Vec<(usize, f32, [f32; 4])> {
//     // Boxes get sorted by score in descending order so we know based on the
//     // index the scoring of the boxes and can skip parts of the loop.
//     // let mut boxes = boxes.to_vec();
//     boxes.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

//     // Outer loop over all boxes.
//     for i in 0..boxes.len() {
//         // Inner loop over boxes with higher score (earlier in the list).
//         for j in 0..i {
//             // If the boxes have the same class and the IoU is higher than
// the             // threshold, the boxes are merged and the outer box is
// removed.             if boxes[i].0 == boxes[j].0 && jaccard(boxes[i].3,
// boxes[j].3) > iou {                 let maxbox = [
//                     boxes[i].3[0].min(boxes[j].3[0]),
//                     boxes[i].3[1].min(boxes[j].3[1]),
//                     boxes[i].3[2].max(boxes[j].3[2]),
//                     boxes[i].3[3].max(boxes[j].3[3]),
//                 ];
//                 boxes[i].1 = 0.0;
//                 boxes[j].3 = maxbox;
//             }
//         }
//     }
//     // Filter out boxes with a score of 0.0.
//     boxes
//         .into_iter()
//         .filter(|(_, score, _, _)| *score > 0.0)
//         .collect()
// }

// fn jaccard(a: [f32; 4], b: [f32; 4]) -> f32 {
//     let left = a[0].max(b[0]);
//     let top = a[1].max(b[1]);
//     let right = a[2].min(b[2]);
//     let bottom = a[3].min(b[3]);

//     let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
//     let area_a = (a[2] - a[0]) * (a[3] - a[1]);
//     let area_b = (b[2] - b[0]) * (b[3] - b[1]);
//     let union = area_a + area_b - intersection;

//     intersection / union
// }
