use edgefirst_schemas::edgefirst_msgs::DmaBuf as DmaBufMsg;
use log::trace;
use std::{error::Error, io};
use vaal::{
    deepviewrt::{
        model,
        tensor::{Tensor, TensorType},
    },
    Context, VAALBox,
};

use crate::{
    args::Args,
    image::{Image, ImageManager, Rotation, RGBX},
    model::{
        DataType, DetectBox, Model, ModelError, Preprocessing, RGB_MEANS_IMAGENET,
        RGB_STDS_IMAGENET,
    },
};

impl From<TensorType> for DataType {
    fn from(value: TensorType) -> Self {
        match value {
            TensorType::RAW => DataType::Raw,
            TensorType::STR => DataType::String,
            TensorType::I8 => DataType::Int8,
            TensorType::U8 => DataType::UInt8,
            TensorType::I16 => DataType::Int16,
            TensorType::U16 => DataType::UInt16,
            TensorType::I32 => DataType::Int32,
            TensorType::U32 => DataType::UInt32,
            TensorType::I64 => DataType::Int64,
            TensorType::U64 => DataType::UInt64,
            TensorType::F16 => DataType::Float16,
            TensorType::F32 => DataType::Float32,
            TensorType::F64 => DataType::Float64,
        }
    }
}

pub struct RtmModel {
    ctx: Context,
    img: Image,
}

impl RtmModel {
    pub fn load_model_from_mem_with_engine(
        mem: Vec<u8>,
        engine: &str,
    ) -> Result<RtmModel, Box<dyn Error>> {
        let mut ctx = vaal::Context::new(engine)?;
        ctx.load_model(mem)?;
        let drvt = ctx.dvrt_context_const()?;
        let inps = deepviewrt::model::inputs(ctx.model()?)?;
        let inp_shape = drvt.tensor_index(inps[0] as usize)?.shape();
        let img = Image::new(inp_shape[2] as u32, inp_shape[1] as u32, RGBX)?;
        let rtm_model = RtmModel { ctx, img };
        Ok(rtm_model)
    }

    pub fn setup_context(&mut self, args: &Args) {
        self.ctx
            .parameter_seti("max_detection", &[args.max_boxes as i32])
            .unwrap();
        self.ctx
            .parameter_setf("score_threshold", &[args.threshold])
            .unwrap();
        self.ctx
            .parameter_setf("iou_threshold", &[args.iou])
            .unwrap();
        self.ctx.parameter_sets("nms_type", "standard").unwrap();
    }

    pub fn get_input_tensor(&self, index: usize) -> Result<&Tensor, ModelError> {
        let inps = deepviewrt::model::inputs(self.ctx.model()?)?;
        Ok(self
            .ctx
            .dvrt_context_const()?
            .tensor_index(inps[index] as usize)?)
    }

    pub fn get_input_tensor_mut(&mut self, index: usize) -> Result<&mut Tensor, ModelError> {
        let inps = deepviewrt::model::inputs(self.ctx.model()?)?;
        Ok(self
            .ctx
            .dvrt_context()?
            .tensor_index_mut(inps[index] as usize)?)
    }
}

impl Model for RtmModel {
    fn load_frame_dmabuf(
        &mut self,
        dmabuf: &DmaBufMsg,
        img_mgr: &ImageManager,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        trace!("load_frame_dmabuf");
        let image: Image = dmabuf.try_into()?;
        img_mgr.convert(&image, &self.img, None, Rotation::Rotation0)?;
        let mut dest_mapped = self.img.mmap();
        let data = dest_mapped.as_slice_mut();
        self.load_input(0, data, 4, preprocessing)?;
        Ok(())
    }

    fn run_model(&mut self) -> Result<(), ModelError> {
        trace!("run_model");
        Ok(self.ctx.run_model()?)
    }

    fn input_count(&self) -> Result<usize, ModelError> {
        trace!("input_count");
        Ok(model::inputs(self.ctx.model()?)?.len())
    }

    fn input_shape(&self, index: usize) -> Result<Vec<usize>, ModelError> {
        trace!("input_shape");
        let tensor = self.get_input_tensor(index)?;
        let inp_shape = tensor.shape();
        Ok(inp_shape.iter().map(|f| *f as usize).collect())
    }

    fn load_input(
        &mut self,
        index: usize,
        data: &[u8],
        data_channels: usize,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        trace!("load_input");
        let tensor = self.get_input_tensor_mut(index)?;
        let tensor_shape = tensor.shape();
        let tensor_vol = tensor.volume() as usize;
        let tensor_channels = *tensor_shape.last().unwrap_or(&3) as usize;
        match tensor.tensor_type() {
            TensorType::U8 => {
                let mut tensor_mapped = tensor.maprw()?;
                for i in 0..tensor_vol / tensor_channels {
                    for j in 0..tensor_channels {
                        tensor_mapped[i * tensor_channels + j] = data[i * data_channels + j];
                    }
                }
            }
            TensorType::I16 => todo!(),
            TensorType::U16 => todo!(),
            TensorType::I32 => todo!(),
            TensorType::RAW => todo!(),
            TensorType::STR => todo!(),
            TensorType::I8 => {
                let mut tensor_mapped = tensor.maprw()?;
                for i in 0..tensor_vol / tensor_channels {
                    for j in 0..tensor_channels {
                        tensor_mapped[i * tensor_channels + j] =
                            (data[i * data_channels + j] as i16 - 128) as i8;
                    }
                }
            }
            TensorType::U32 => todo!(),
            TensorType::I64 => todo!(),
            TensorType::U64 => todo!(),
            TensorType::F16 => todo!(),
            TensorType::F32 => {
                let mut tensor_mapped = tensor.maprw()?;
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
            TensorType::F64 => todo!(),
        };
        Ok(())
    }

    fn output_count(&self) -> Result<usize, ModelError> {
        trace!("output_count");
        Ok(model::outputs(self.ctx.model()?)?.len())
    }

    fn output_shape(&self, index: usize) -> Result<Vec<usize>, ModelError> {
        trace!("output_shape");
        let output = match self.ctx.output_tensor(index as i32) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access output tensor {index} of {}",
                    model::outputs(self.ctx.model()?)?.len()
                ))
                .into();
                return Err(e);
            }
        };
        let shape = output.shape();
        Ok(shape.iter().map(|f| *f as usize).collect())
    }

    fn output_data<T: Copy>(&self, index: usize, buffer: &mut [T]) -> Result<(), ModelError> {
        trace!("output_data");
        let tensor = match self.ctx.output_tensor(index as i32) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access output tensor {index} of {}",
                    model::outputs(self.ctx.model()?)?.len()
                ))
                .into();
                return Err(e);
            }
        };
        let t = tensor.mapro()?;
        let len = tensor.volume() as usize;
        if len != buffer.len() {
            return Err(io::Error::other(format!(
                "buffer has length {} but needs to be length {}",
                buffer.len(),
                len,
            ))
            .into());
        }
        buffer.copy_from_slice(&t);
        Ok(())
    }

    fn boxes(&self, boxes: &mut [DetectBox]) -> Result<usize, ModelError> {
        trace!("boxes");
        let mut vaal_boxes = Vec::new();
        let len = boxes.len();
        for _ in 0..boxes.len() {
            vaal_boxes.push(VAALBox {
                xmin: 0.0,
                ymin: 0.0,
                xmax: 0.0,
                ymax: 0.0,
                score: 0.0,
                label: 0,
            });
        }

        let box_count = self.ctx.boxes(&mut vaal_boxes, len)?;
        for i in 0..box_count {
            boxes[i] = vaal_boxes[i].into();
        }

        Ok(box_count)
    }

    fn input_type(&self, index: usize) -> Result<crate::model::DataType, ModelError> {
        trace!("input_type");
        let tensor = self.get_input_tensor(index)?;
        Ok(tensor.tensor_type().into())
    }

    fn output_type(&self, index: usize) -> Result<crate::model::DataType, ModelError> {
        trace!("output_type");
        let tensor = match self.ctx.output_tensor(index as i32) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access output tensor {index} of {}",
                    model::outputs(self.ctx.model()?)?.len()
                ))
                .into();
                return Err(e);
            }
        };
        Ok(tensor.tensor_type().into())
    }

    fn labels(&self) -> Result<Vec<String>, ModelError> {
        trace!("labels");
        Ok(self.ctx.labels().iter().map(|s| s.to_string()).collect())
    }

    fn model_name(&self) -> Result<String, ModelError> {
        trace!("model_name");
        Ok(model::name(self.ctx.model()?)?.to_string())
    }
}

impl From<VAALBox> for DetectBox {
    fn from(value: VAALBox) -> Self {
        DetectBox {
            xmin: value.xmin,
            ymin: value.ymin,
            xmax: value.xmax,
            ymax: value.ymax,
            score: value.score,
            label: value.label as usize,
        }
    }
}
