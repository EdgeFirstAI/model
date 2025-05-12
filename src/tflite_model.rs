use crate::{
    args::Args,
    image::{Image, ImageManager, Rotation, RGBX},
    model::{
        DataType, DetectBox, Model, ModelError, Preprocessing, RGB_MEANS_IMAGENET,
        RGB_STDS_IMAGENET,
    },
    nms::decode_boxes_and_nms,
};
use edgefirst_schemas::edgefirst_msgs::DmaBuf;
use log::debug;
use std::{error::Error, io, path::Path};
use tflitec_sys::{
    delegate::Delegate,
    tensor::{Tensor, TensorMut, TensorType},
    Interpreter, LibloadingError, TFLiteLib as TFLiteLib_,
};

pub static DEFAULT_NPU_DELEGATE_PATH: &str = "libvx_delegate.so";
pub static DEFAULT_TFLITEC_PATH: &str = "libtensorflowlite_c.so";
pub static DEFAULT_TFLITECPP_PATH: &str = "libtensorflow-lite.so";

pub struct TFLiteLib {
    tflite_lib: TFLiteLib_,
}

impl TFLiteLib {
    pub fn new() -> Result<Self, LibloadingError> {
        let tflite_lib = TFLiteLib_::new()?;
        Ok(TFLiteLib { tflite_lib })
    }

    #[allow(dead_code)]
    pub fn new_with_path<P: AsRef<Path>>(path: P) -> Result<Self, LibloadingError> {
        let tflite_lib = TFLiteLib_::new_with_path(path)?;
        Ok(TFLiteLib { tflite_lib })
    }

    #[allow(dead_code)]
    pub fn load_model_from_mem(&self, mem: Vec<u8>) -> Result<TFLiteModel, Box<dyn Error>> {
        self.load_model_from_mem_with_delegate(mem, None::<String>)
    }

    pub fn load_model_from_mem_with_delegate<P: AsRef<Path>>(
        &self,
        mem: Vec<u8>,
        delegate: Option<P>,
    ) -> Result<TFLiteModel, Box<dyn Error>> {
        let model = self.tflite_lib.new_model_from_mem(mem)?;
        let mut builder = self.tflite_lib.new_interpreter_builder()?;

        if let Some(delegate) = delegate {
            let delegate = Delegate::load_external(delegate)?;
            builder.add_owned_delegate(delegate);
        }
        let runner = builder.build(model)?;
        TFLiteModel::new(runner)
    }
}

impl From<TensorType> for DataType {
    fn from(value: TensorType) -> Self {
        match value {
            TensorType::UnknownType => DataType::Raw,
            TensorType::NoType => DataType::Raw,
            TensorType::Float32 => DataType::Float32,
            TensorType::Int32 => DataType::Int32,
            TensorType::UInt8 => DataType::UInt8,
            TensorType::Int64 => DataType::Int64,
            TensorType::String => DataType::String,
            TensorType::Bool => DataType::Raw,
            TensorType::Int16 => DataType::Int16,
            TensorType::Complex64 => DataType::Raw,
            TensorType::Int8 => DataType::Int8,
            TensorType::Float16 => DataType::Float16,
            TensorType::Float64 => DataType::Float64,
            TensorType::Complex128 => DataType::Raw,
            TensorType::UInt64 => DataType::UInt64,
            TensorType::Resource => DataType::Raw,
            TensorType::Variant => DataType::Raw,
            TensorType::UInt32 => DataType::UInt32,
            TensorType::UInt16 => DataType::UInt16,
            TensorType::Int4 => DataType::Raw,
            TensorType::BFloat16 => DataType::Raw,
        }
    }
}

pub struct TFLiteModel<'a> {
    model: Interpreter<'a>,
    img: Image,
    inputs: Vec<TensorMut<'a>>,
    outputs: Vec<Tensor<'a>>,
    score_threshold: f32,
    iou_threshold: f32,
}

impl<'a> TFLiteModel<'a> {
    fn input_shape(model: &Interpreter, index: usize) -> Result<Vec<usize>, ModelError> {
        let inputs = model.inputs()?;
        let tensor = match inputs.get(index) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access input tensor {index} of {}",
                    inputs.len()
                ))
                .into();
                return Err(e);
            }
        };
        Ok(tensor.shape()?)
    }

    pub fn new(model: Interpreter<'a>) -> Result<Self, Box<dyn Error>> {
        let inp_shape = match Self::input_shape(&model, 0) {
            Ok(v) => v,
            Err(e) => return Err(Box::from(e)),
        };
        let img = Image::new(inp_shape[2] as u32, inp_shape[1] as u32, RGBX)?;
        let mut m = TFLiteModel {
            model,
            img,
            inputs: Vec::new(),
            outputs: Vec::new(),
            score_threshold: 0.5,
            iou_threshold: 0.5,
        };
        m.init_tensors()?;
        m.run_model()?;
        Ok(m)
    }

    pub fn setup_context(&mut self, args: &Args) {
        self.score_threshold = args.threshold;
        self.iou_threshold = args.iou;
    }

    fn output_data_ref<T>(&self, index: usize) -> Result<&[T], ModelError> {
        let tensor = match self.outputs.get(index) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access output tensor {index} of {}",
                    self.outputs.len()
                ))
                .into();
                return Err(e);
            }
        };
        let data = tensor.mapro()?;
        Ok(data)
    }

    fn dequant_output(&self, index: usize) -> Result<Vec<f32>, ModelError> {
        match self.output_type(index)? {
            DataType::Raw => todo!(),
            DataType::Int8 => {
                let data: &[i8] = self.output_data_ref(index)?;
                let quant = self.outputs[index].get_quantization_params();
                Ok(data
                    .iter()
                    .map(|d| quant.scale * (*d as i32 - quant.zero_point) as f32)
                    .collect())
            }
            DataType::UInt8 => {
                let data: &[u8] = self.output_data_ref(index)?;
                let quant = self.outputs[index].get_quantization_params();
                Ok(data
                    .iter()
                    .map(|d| quant.scale * (*d as i32 - quant.zero_point) as f32)
                    .collect())
            }
            DataType::Int16 => {
                let data: &[i16] = self.output_data_ref(index)?;
                let quant = self.outputs[index].get_quantization_params();
                Ok(data
                    .iter()
                    .map(|d| quant.scale * (*d as i32 - quant.zero_point) as f32)
                    .collect())
            }
            DataType::UInt16 => {
                let data: &[u16] = self.output_data_ref(index)?;
                let quant = self.outputs[index].get_quantization_params();
                Ok(data
                    .iter()
                    .map(|d| quant.scale * (*d as i32 - quant.zero_point) as f32)
                    .collect())
            }
            DataType::Float16 => todo!(),
            DataType::Int32 => {
                let data: &[i32] = self.output_data_ref(index)?;
                let quant = self.outputs[index].get_quantization_params();
                Ok(data
                    .iter()
                    .map(|d| quant.scale * (*d as i64 - quant.zero_point as i64) as f32)
                    .collect())
            }
            DataType::UInt32 => {
                let data: &[u32] = self.output_data_ref(index)?;
                let quant = self.outputs[index].get_quantization_params();
                Ok(data
                    .iter()
                    .map(|d| quant.scale * (*d as i64 - quant.zero_point as i64) as f32)
                    .collect())
            }
            DataType::Float32 => todo!(),
            DataType::Int64 => todo!(),
            DataType::UInt64 => todo!(),
            DataType::Float64 => todo!(),
            DataType::String => todo!(),
        }
    }

    fn init_tensors(&mut self) -> Result<(), Box<dyn Error>> {
        let mut outputs = self.model.outputs()?;
        self.outputs.append(&mut outputs);
        let mut inputs = self.model.inputs_mut()?;
        self.inputs.append(&mut inputs);
        Ok(())
    }
}

impl Model for TFLiteModel<'_> {
    fn load_frame_dmabuf(
        &mut self,
        dmabuf: &DmaBuf,
        img_mgr: &ImageManager,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        let image = dmabuf.try_into()?;
        img_mgr.convert(&image, &self.img, None, Rotation::Rotation0)?;
        let mut dest_mapped = self.img.mmap();
        let data = dest_mapped.as_slice_mut();
        self.load_input(0, data, 4, preprocessing)?;
        Ok(())
    }

    fn run_model(&mut self) -> Result<(), ModelError> {
        Ok(self.model.invoke()?)
    }

    fn input_count(&self) -> Result<usize, ModelError> {
        Ok(self.inputs.len())
    }

    fn input_shape(&self, index: usize) -> Result<Vec<usize>, ModelError> {
        TFLiteModel::input_shape(&self.model, index)
    }

    fn load_input(
        &mut self,
        index: usize,
        data: &[u8],
        data_channels: usize,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        let tensor = match self.inputs.get_mut(index) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access input tensor {index} of {}",
                    self.inputs.len()
                ))
                .into();
                return Err(e);
            }
        };
        let tensor_vol = tensor.volume()?;
        let tensor_shape = tensor.shape()?;
        let tensor_channels = { *tensor_shape.last().unwrap_or(&3) };
        match tensor.tensor_type() {
            TensorType::UInt8 => {
                let tensor_mapped = tensor.maprw()?;
                for i in 0..tensor_vol / tensor_channels {
                    for j in 0..tensor_channels {
                        tensor_mapped[i * tensor_channels + j] = data[i * data_channels + j];
                    }
                }
            }
            TensorType::Int16 => todo!(),
            TensorType::Int32 => todo!(),
            TensorType::NoType => todo!(),
            TensorType::String => todo!(),
            TensorType::Int8 => {
                let tensor_mapped = tensor.maprw()?;
                for i in 0..tensor_vol / tensor_channels {
                    for j in 0..tensor_channels {
                        tensor_mapped[i * tensor_channels + j] =
                            (data[i * data_channels + j] as i16 - 128) as i8;
                    }
                }
            }
            TensorType::Int64 => todo!(),
            TensorType::Float16 => todo!(),
            TensorType::Float32 => {
                let tensor_mapped = tensor.maprw()?;
                match preprocessing {
                    Preprocessing::Raw => {
                        for i in 0..tensor_vol / tensor_channels {
                            for j in 0..tensor_channels {
                                tensor_mapped[i * tensor_channels + j] =
                                    data[i * data_channels + j] as f32;
                            }
                        }
                    }
                    Preprocessing::UnsignedNorm => {
                        for i in 0..tensor_vol / tensor_channels {
                            for j in 0..tensor_channels {
                                tensor_mapped[i * tensor_channels + j] =
                                    data[i * data_channels + j] as f32 / 255.0;
                            }
                        }
                    }
                    Preprocessing::SignedNorm => {
                        for i in 0..tensor_vol / tensor_channels {
                            for j in 0..tensor_channels {
                                tensor_mapped[i * tensor_channels + j] =
                                    data[i * data_channels + j] as f32 / 127.5 - 1.0;
                            }
                        }
                    }
                    Preprocessing::ImageNet => {
                        for i in 0..tensor_vol / tensor_channels {
                            for j in 0..tensor_channels {
                                tensor_mapped[i * tensor_channels + j] =
                                    (data[i * data_channels + j] as f32 - RGB_MEANS_IMAGENET[j])
                                        / RGB_STDS_IMAGENET[j];
                            }
                        }
                    }
                }
            }
            TensorType::Complex64 => todo!(),
            TensorType::Bool => todo!(),
            TensorType::UnknownType => todo!(),
            TensorType::Float64 => todo!(),
            TensorType::Complex128 => todo!(),
            TensorType::UInt64 => todo!(),
            TensorType::Resource => todo!(),
            TensorType::Variant => todo!(),
            TensorType::UInt32 => todo!(),
            TensorType::UInt16 => todo!(),
            TensorType::Int4 => todo!(),
            TensorType::BFloat16 => todo!(),
        };

        Ok(())
    }

    fn output_shape(&self, index: usize) -> Result<Vec<usize>, ModelError> {
        let tensor = match self.outputs.get(index) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access output tensor {index} of {}",
                    self.outputs.len()
                ))
                .into();
                return Err(e);
            }
        };
        Ok(tensor.shape()?)
    }

    fn output_data<T: Copy>(&self, index: usize, buffer: &mut [T]) -> Result<(), ModelError> {
        let tensor = match self.outputs.get(index) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access output tensor {index} of {}",
                    self.outputs.len()
                ))
                .into();
                return Err(e);
            }
        };
        let data = tensor.mapro()?;
        let len = data.len();
        if buffer.len() != len {
            return Err(io::Error::other(format!(
                "buffer has length {} but needs to be length {}",
                buffer.len(),
                len,
            ))
            .into());
        }
        buffer.copy_from_slice(data);

        Ok(())
    }

    fn output_count(&self) -> Result<usize, ModelError> {
        Ok(self.outputs.len())
    }

    fn boxes(&self, boxes: &mut [DetectBox]) -> Result<usize, ModelError> {
        let mut box_ind = None;
        let mut score_ind = None;
        let mut num_classes = None;
        for i in 0..self.output_count()? {
            let shape = self.output_shape(i)?;
            if shape.len() == 3 {
                score_ind = Some(i);
                num_classes = Some(*shape.last().unwrap());
            } else if shape.len() == 4 && shape[2] == 1 && shape[3] == 4 {
                box_ind = Some(i);
            }
        }
        if box_ind.is_none() || score_ind.is_none() {
            return Ok(0);
        }

        let box_data = self.dequant_output(box_ind.unwrap())?;
        let score_data = self.dequant_output(score_ind.unwrap())?;
        let n_boxes = decode_boxes_and_nms(
            self.score_threshold,
            self.iou_threshold,
            &score_data,
            &box_data,
            num_classes.unwrap(),
            boxes,
        );
        Ok(n_boxes)
    }

    fn input_type(&self, index: usize) -> Result<crate::model::DataType, ModelError> {
        let tensor = match self.inputs.get(index) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access input tensor {index} of {}",
                    self.inputs.len()
                ))
                .into();
                return Err(e);
            }
        };
        Ok(tensor.tensor_type().into())
    }

    fn output_type(&self, index: usize) -> Result<crate::model::DataType, ModelError> {
        let tensor = match self.outputs.get(index) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access input tensor {index} of {}",
                    self.inputs.len()
                ))
                .into();
                return Err(e);
            }
        };
        Ok(tensor.tensor_type().into())
    }

    fn labels(&self) -> Result<Vec<String>, ModelError> {
        Ok(Vec::new())
    }

    fn model_name(&self) -> Result<String, ModelError> {
        Ok("".to_string())
    }
}
