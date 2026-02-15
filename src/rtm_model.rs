// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use edgefirst_hal::decoder::configs::DataType;
use edgefirst_hal::image::{Crop, ImageProcessor, ImageProcessorTrait, TensorImage};
use edgefirst_schemas::edgefirst_msgs::DmaBuffer as DmaBufMsg;
use edgefirst_hal::tensor::{TensorMapTrait, TensorTrait};
use log::trace;
use ndarray::{ArrayView, ArrayViewD};
use std::{error::Error, io};
use tracing::instrument;
use vaal::{
    Context,
    deepviewrt::{
        model,
        tensor::{Tensor, TensorData, TensorType},
    },
};

use crate::{
    args::Args,
    model::{
        Metadata, Model, ModelError, Preprocessing, RGB_MEANS_IMAGENET, RGB_STDS_IMAGENET,
        dmabuf_to_tensor_image,
    },
};
fn tensor_type_to_datatype(value: TensorType) -> DataType {
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

pub struct RtmModel {
    ctx: Context,
    img_: TensorImage,
}

impl RtmModel {
    pub fn load_model_from_mem_with_engine(
        mem: Vec<u8>,
        engine: &str,
    ) -> Result<RtmModel, Box<dyn Error>> {
        let mut ctx = vaal::Context::new(engine)?;
        ctx.load_model(mem)?;
        let drvt = ctx.dvrt_context_const()?;
        let inps = model::inputs(ctx.model()?)?;
        let inp_shape = drvt.tensor_index(inps[0] as usize)?.shape();
        let img_ = TensorImage::new(
            inp_shape[2] as usize,
            inp_shape[1] as usize,
            edgefirst_hal::image::RGBA,
            None,
        )?;
        let rtm_model = RtmModel { ctx, img_ };
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
        let inps = model::inputs(self.ctx.model()?)?;
        Ok(self
            .ctx
            .dvrt_context_const()?
            .tensor_index(inps[index] as usize)?)
    }

    pub fn get_input_tensor_mut(&mut self, index: usize) -> Result<&mut Tensor, ModelError> {
        let inps = model::inputs(self.ctx.model()?)?;
        Ok(self
            .ctx
            .dvrt_context()?
            .tensor_index_mut(inps[index] as usize)?)
    }
}

impl Model for RtmModel {
    #[instrument(skip_all)]
    fn load_frame_dmabuf_(
        &mut self,
        dmabuf: &DmaBufMsg,
        img_proc: &mut ImageProcessor,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        let image = dmabuf_to_tensor_image(dmabuf)?;
        img_proc.convert(
            &image,
            &mut self.img_,
            edgefirst_hal::image::Rotation::None,
            edgefirst_hal::image::Flip::None,
            Crop::no_crop(),
        )?;
        let dest_mapped = self.img_.tensor().map()?;
        let data = dest_mapped.as_slice();
        self.load_input(0, data, 4, preprocessing)?;
        Ok(())
    }

    #[instrument(skip_all)]
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

    #[instrument(skip_all)]
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
                if tensor_channels == data_channels {
                    tensor_mapped.copy_from_slice(&data[0..tensor_vol]);
                    return Ok(());
                }
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

    #[instrument(skip_all)]
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

    fn decode_outputs(
        &self,
        decoder: &edgefirst_hal::decoder::Decoder,
        output_boxes: &mut Vec<edgefirst_hal::decoder::DetectBox>,
        output_masks: &mut Vec<edgefirst_hal::decoder::Segmentation>,
    ) -> Result<(), ModelError> {
        let mut outputs_quant = Vec::new();
        let mut outputs_float = Vec::new();
        for ind in 0..self.output_count()? {
            let o = match self.ctx.output_tensor(ind as i32) {
                Some(v) => v,
                None => {
                    let e = io::Error::other(format!(
                        "Tried to access output tensor {ind} of {}",
                        model::outputs(self.ctx.model()?)?.len()
                    ))
                    .into();
                    return Err(e);
                }
            };
            match o.tensor_type() {
                TensorType::F32 => {
                    // let shape = o.shape().iter().map(|f| *f as usize).collect::<Vec<_>>();
                    // let arr = ndarray::ArrayView::from_shape(shape, &data).unwrap();
                    outputs_float.push(o);
                }
                TensorType::U8 => {
                    // let shape = o.shape();
                    // let arr = ndarray::ArrayView::from_shape(shape, data).unwrap();
                    // let arr: ArrayViewDQuantized = arr.into();
                    outputs_quant.push(o);
                }
                TensorType::I8 => {
                    // let shape = o.shape();
                    // let arr = ndarray::ArrayView::from_shape(shape, data).unwrap();
                    // let arr: ArrayViewDQuantized = arr.into();
                    outputs_quant.push(o);
                }
                _ => {
                    log::warn!("Output of other type");
                }
            }
        }

        match (outputs_float.is_empty(), outputs_quant.is_empty()) {
            (false, true) => {
                let tensor_maps = outputs_float
                    .iter()
                    .map(|x| -> Result<_, ModelError> { Ok(x.mapro_f32()?) })
                    .collect::<Result<Vec<TensorData<'_, f32>>, _>>()?;
                let array_views = outputs_float
                    .iter()
                    .zip(tensor_maps.iter())
                    .map(|(tensor, map)| {
                        let shape = tensor
                            .shape()
                            .iter()
                            .map(|x| *x as usize)
                            .collect::<Vec<_>>();
                        ArrayView::<f32, _>::from_shape(shape, map).unwrap()
                    })
                    .collect::<Vec<_>>();
                decoder
                    .decode_float(&array_views, output_boxes, output_masks)
                    .unwrap()
            }
            (true, false) => {
                let tensor_maps = outputs_quant
                    .iter()
                    .map(|x| -> Result<_, ModelError> {
                        match x.tensor_type() {
                            TensorType::U8 => Ok(x.mapro_u8()?.into()),
                            TensorType::I8 => Ok(x.mapro_i8()?.into()),
                            _ => Err(io::Error::other("Unexpected tensor type").into()),
                        }
                    })
                    .collect::<Result<Vec<TensorDataHolderQuant<'_>>, _>>()?;
                let array_views = outputs_quant
                    .iter()
                    .zip(tensor_maps.iter())
                    .map(|(tensor, map)| {
                        let shape = tensor
                            .shape()
                            .iter()
                            .map(|x| *x as usize)
                            .collect::<Vec<_>>();
                        map.as_arrayview(&shape)
                    })
                    .collect::<Vec<_>>();
                decoder
                    .decode_quantized(&array_views, output_boxes, output_masks)
                    .unwrap()
            }
            (true, true) => {
                log::error!("No outputs for decoder");
            }
            (false, false) => {
                log::error!("Mixed floating point and quantized outputs for decoder");
            }
        }
        log::trace!("Decoded boxes: {:?}", output_boxes);
        Ok(())
    }

    fn decode_outputs_tracked(
        &self,
        decoder: &edgefirst_hal::decoder::Decoder,
        output_boxes: &mut Vec<edgefirst_hal::decoder::DetectBox>,
        output_masks: &mut Vec<edgefirst_hal::decoder::Segmentation>,
        _output_tracks: &mut Vec<edgefirst_tracker::TrackInfo>,
        _timestamp: u64,
    ) -> Result<(), ModelError> {
        // Tracking is handled separately by the caller; delegate to decode_outputs
        self.decode_outputs(decoder, output_boxes, output_masks)
    }

    fn input_type(&self, index: usize) -> Result<DataType, ModelError> {
        trace!("input_type");
        let tensor = self.get_input_tensor(index)?;
        Ok(tensor_type_to_datatype(tensor.tensor_type()))
    }

    fn output_type(&self, index: usize) -> Result<DataType, ModelError> {
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
        Ok(tensor_type_to_datatype(tensor.tensor_type()))
    }

    fn labels(&self) -> Result<Vec<String>, ModelError> {
        trace!("labels");
        Ok(self.ctx.labels().iter().map(|s| s.to_string()).collect())
    }

    fn model_name(&self) -> Result<String, ModelError> {
        trace!("model_name");
        Ok(model::name(self.ctx.model()?)?.to_string())
    }

    fn get_model_metadata(&self) -> Result<Metadata, ModelError> {
        let name = self.model_name().ok();
        Ok(Metadata {
            name,
            version: None,
            description: None,
            author: None,
            license: None,
            config_yaml: None,
        })
    }

    fn output_quantization(&self, index: usize) -> Result<Option<(f32, i32)>, ModelError> {
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
        let scale = Some(1.0); // TODO: Adjust to get actual scales?
        let zero = tensor.zeros()?.first().copied();

        if let Some(scale) = scale
            && let Some(zero) = zero
        {
            Ok(Some((scale, zero)))
        } else {
            Ok(None)
        }
    }
}

enum TensorDataHolderQuant<'a> {
    TensorU8(TensorData<'a, u8>),
    TensorI8(TensorData<'a, i8>),
}

impl<'a> TensorDataHolderQuant<'a> {
    fn as_arrayview(&'a self, shape: &[usize]) -> edgefirst_hal::decoder::ArrayViewDQuantized<'a> {
        match self {
            TensorDataHolderQuant::TensorU8(arr) => {
                ArrayViewD::<'a, u8>::from_shape(shape, arr).unwrap().into()
            }
            TensorDataHolderQuant::TensorI8(arr) => {
                ArrayViewD::<'a, i8>::from_shape(shape, arr).unwrap().into()
            }
        }
    }
}

impl<'a> From<TensorData<'a, u8>> for TensorDataHolderQuant<'a> {
    fn from(value: TensorData<'a, u8>) -> Self {
        TensorDataHolderQuant::TensorU8(value)
    }
}

impl<'a> From<TensorData<'a, i8>> for TensorDataHolderQuant<'a> {
    fn from(value: TensorData<'a, i8>) -> Self {
        TensorDataHolderQuant::TensorI8(value)
    }
}
