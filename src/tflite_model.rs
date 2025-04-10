use crate::{
    image::{Image, ImageManager, Rotation, RGBA},
    model::{DetectBox, Model, ModelError, Preprocessing, RGB_MEANS_IMAGENET, RGB_STDS_IMAGENET},
};
use cdr::{CdrLe, Infinite};
use edgefirst_schemas::edgefirst_msgs::{DmaBuf, Mask};
use log::{debug, error, info, trace};
use std::{
    error::Error,
    ffi::c_void,
    fmt,
    fs::read,
    io,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::io::OwnedFd,
    },
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use tflitec_sys::{
    delegate::Delegate,
    tensor::{Tensor, TensorMut, TensorType},
    Interpreter, Model as TFLiteModel_, TFLiteLib as TFLiteLib_, TfLiteError,
};

pub static DEFAULT_NPU_DELEGATE_PATH: &str = "libvx_delegate.so";
pub static DEFAULT_TFLITEC_PATH: &str = "libtensorflowlite_c.so";

struct TFLiteLib {
    tflite_lib: TFLiteLib_,
}

impl TFLiteLib {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, libloading::Error> {
        let tflite_lib = TFLiteLib_::new(path)?;
        Ok(TFLiteLib { tflite_lib })
    }

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

struct TFLiteModel<'a> {
    model: Interpreter<'a>,
    img: Image,
    inputs: Vec<TensorMut<'a>>,
    outputs: Vec<Tensor<'a>>,
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
        let img = Image::new(inp_shape[2] as u32, inp_shape[1] as u32, RGBA)?;
        let mut m = TFLiteModel {
            model,
            img,
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        m.init_tensors()?;
        Ok(m)
    }

    fn init_tensors(&mut self) -> Result<(), Box<dyn Error>> {
        let mut outputs = self.model.outputs()?;
        self.outputs.append(&mut outputs);
        let mut inputs = self.model.inputs_mut()?;
        self.inputs.append(&mut inputs);
        Ok(())
    }
}

impl<'a> Model for TFLiteModel<'a> {
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
        let tensor_channels = { *tensor.shape()?.last().unwrap_or(&3) };
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
        todo!()
    }
}
