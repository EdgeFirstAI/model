use edgefirst_schemas::edgefirst_msgs::DmaBuf as DmaBufMsg;
use std::{error::Error, io, path::Path};
use vaal::{
    deepviewrt::{
        engine::Engine,
        model,
        tensor::{Tensor, TensorType},
    },
    Context, VAALBox,
};

use crate::{
    image::{Image, ImageManager, Rotation, RGBA},
    model::{DetectBox, Model, ModelError, Preprocessing, RGB_MEANS_IMAGENET, RGB_STDS_IMAGENET},
};

pub struct RtmModel {
    ctx: Context,
    img: Image,
}

impl RtmModel {
    pub fn load_model_from_mem_with_engine<P: AsRef<Path> + Into<Vec<u8>>>(
        mem: Vec<u8>,
        engine: &str,
    ) -> Result<RtmModel, Box<dyn Error>> {
        // let engine = if let Some(p) = engine {
        //     Some(Engine::new(p)?)
        // } else {
        //     None
        // };
        // let mut ctx = Context::new(engine, model::memory_size(&mem), 4096 *
        // 1024)?; let inp_shape = ctx.input(0)?.shape();
        // let img = Image::new(inp_shape[2] as u32, inp_shape[1] as u32,
        // RGBA)?; ctx.load_model(mem)?;

        // let rtm_model = RtmModel { ctx, img };
        // Ok(rtm_model)
        let mut ctx = vaal::Context::new(engine)?;
        ctx.load_model(mem)?;
        let inp_shape = ctx.dvrt_context()?.input(0)?.shape();
        let img = Image::new(inp_shape[2] as u32, inp_shape[1] as u32, RGBA)?;
        let rtm_model = RtmModel { ctx, img };
        Ok(rtm_model)
    }
}

impl Model for RtmModel {
    fn load_frame_dmabuf(
        &mut self,
        dmabuf: &DmaBufMsg,
        img_mgr: &ImageManager,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        let image = dmabuf.try_into()?;
        img_mgr.convert(&image, &self.img, None, Rotation::Rotation0)?;
        let mut dest_mapped = self.img.mmap();
        let data = dest_mapped.as_slice_mut();
        self.load_input(0, data, 4, preprocessing)
    }

    fn run_model(&mut self) -> Result<(), ModelError> {
        Ok(self.ctx.run_model()?)
    }

    fn input_count(&self) -> Result<usize, ModelError> {
        Ok(model::inputs(self.ctx.model()?)?.len())
    }

    fn input_shape(&self, index: usize) -> Result<Vec<usize>, ModelError> {
        Ok(self
            .ctx
            .dvrt_context_const()?
            .input(index)?
            .shape()
            .iter()
            .map(|f| *f as usize)
            .collect())
    }

    fn load_input(
        &mut self,
        index: usize,
        data: &[u8],
        data_channels: usize,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        let tensor = self.ctx.dvrt_context()?.input_mut(index)?;
        let tensor_vol = tensor.volume() as usize;
        let tensor_channels = *tensor.shape().last().unwrap_or(&3) as usize;
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
        Ok(model::outputs(self.ctx.model()?)?.len())
    }

    fn output_shape(&self, index: usize) -> Result<Vec<usize>, ModelError> {
        Ok(self
            .ctx
            .dvrt_context_const()?
            .output(index)?
            .shape()
            .iter()
            .map(|f| *f as usize)
            .collect())
    }

    fn output_data<T: Copy>(&self, index: usize, buffer: &mut [T]) -> Result<(), ModelError> {
        let tensor = self.ctx.dvrt_context_const()?.output(index)?;
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
}

impl From<VAALBox> for DetectBox {
    fn from(value: VAALBox) -> Self {
        DetectBox {
            xmin: value.xmin,
            ymin: value.ymin,
            xmax: value.xmax,
            ymax: value.ymax,
            score: value.score,
            label: value.label,
        }
    }
}
