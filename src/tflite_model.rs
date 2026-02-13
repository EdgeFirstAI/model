// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use crate::{
    args::Args,
    model::{
        Metadata, Model, ModelError, Preprocessing, RGB_MEANS_IMAGENET, RGB_STDS_IMAGENET,
        dmabuf_to_tensor_image,
    },
};
use edgefirst_decoder::{ArrayViewDQuantized, BoundingBox, DetectBox, configs::DataType};
use edgefirst_image::{Crop, ImageProcessor, ImageProcessorTrait, RGBA, TensorImage};
use edgefirst_schemas::edgefirst_msgs::DmaBuffer;
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
use log::error;
use std::{
    error::Error,
    io::{self, Read},
    path::Path,
};
use tflitec_sys::{
    Interpreter, LibloadingError, TFLiteLib as TFLiteLib_,
    delegate::Delegate,
    metadata::get_model_metadata,
    tensor::{Tensor, TensorMut, TensorType},
};
use tracing::instrument;

pub static DEFAULT_NPU_DELEGATE_PATH: &str = "libvx_delegate.so";

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
    pub fn load_model_from_mem(&'_ self, mem: Vec<u8>) -> Result<TFLiteModel<'_>, Box<dyn Error>> {
        self.load_model_from_mem_with_delegate(mem, None::<String>)
    }

    pub fn load_model_from_mem_with_delegate<P: AsRef<Path>>(
        &'_ self,
        mem: Vec<u8>,
        delegate: Option<P>,
    ) -> Result<TFLiteModel<'_>, Box<dyn Error>> {
        let model = self.tflite_lib.new_model_from_mem(mem)?;
        let mut builder = self.tflite_lib.new_interpreter_builder()?;
        if let Some(delegate) = delegate {
            let delegate = Delegate::load_external(delegate)?;
            builder.add_owned_delegate(delegate);
        }
        let runner = builder.build(model)?;
        let model = TFLiteModel::new(runner)?;

        Ok(model)
    }
}

fn tensor_type_to_datatype(value: TensorType) -> DataType {
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

pub struct TFLiteModel<'a> {
    model: Interpreter<'a>,
    img_: TensorImage,
    inputs: Vec<TensorMut<'a>>,
    outputs: Vec<Tensor<'a>>,
    metadata: Metadata,
    labels: Vec<String>,
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

        let config_filenames = [
            "edgefirst.yaml",
            "edgefirst.yml",
            "config.yaml",
            "config.yml",
        ];
        let mut labels = Vec::new();
        let mut metadata = get_model_metadata(&model.model_mem);
        if let Ok(mut z) = zip::ZipArchive::new(std::io::Cursor::new(&model.model_mem)) {
            for name in config_filenames {
                if let Ok(mut f) = z.by_name(name)
                    && f.is_file()
                {
                    let mut yaml = String::new();
                    if let Err(e) = f.read_to_string(&mut yaml) {
                        error!("Error while reading {name} {e:?}");
                    }
                    metadata.config_yaml = Some(yaml);
                    break;
                }
            }

            if let Ok(mut f) = z.by_name("labels.txt")
                && f.is_file()
            {
                let mut labels_txt = String::new();
                if let Err(e) = f.read_to_string(&mut labels_txt) {
                    error!("Error while reading config.yaml {e:?}");
                }
                labels = labels_txt.lines().map(|l| l.to_string()).collect();
            }
        }

        let img_ = TensorImage::new(inp_shape[2], inp_shape[1], RGBA, None)?;
        let mut m = TFLiteModel {
            model,
            img_,
            inputs: Vec::new(),
            outputs: Vec::new(),
            metadata: metadata.into(),
            labels,
        };
        m.init_tensors()?;
        m.run_model()?;
        Ok(m)
    }

    pub fn setup_context(&mut self, _args: &Args) {}

    // Must have 4 output tensors, in order of boxes, classes, scores, num_det
    fn ssd_decode_boxes(&self, boxes: &mut Vec<DetectBox>) -> Result<(), ModelError> {
        let box_loc = self.outputs[0].mapro::<f32>()?;
        let classes = self.outputs[1].mapro::<f32>()?;
        let scores = self.outputs[2].mapro::<f32>()?;
        let num_det = self.outputs[3].mapro::<f32>()?;
        assert_eq!(
            scores.len(),
            classes.len(),
            "classes and scores tensor don't have the same size"
        );
        assert_eq!(
            box_loc.len(),
            classes.len() * 4,
            "boxes tensor isn't 4x the size of the classes tensor"
        );
        assert_eq!(num_det.len(), 1, "num_det tensor isn't size of 1");
        let num_det = num_det[0].round() as usize;
        let b = Vec::from_iter((0..num_det).map(|i| DetectBox {
            bbox: BoundingBox {
                ymin: box_loc[i * 4],
                xmin: box_loc[i * 4 + 1],
                ymax: box_loc[i * 4 + 2],
                xmax: box_loc[i * 4 + 3],
            },
            score: scores[i],
            label: classes[i].round() as usize,
        }));

        for b in b {
            boxes.push(b);
        }

        Ok(())
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
    #[instrument(skip_all)]
    fn load_frame_dmabuf_(
        &mut self,
        dmabuf: &DmaBuffer,
        img_proc: &mut ImageProcessor,
        preprocessing: Preprocessing,
    ) -> Result<(), ModelError> {
        let image = dmabuf_to_tensor_image(dmabuf)?;
        img_proc.convert(
            &image,
            &mut self.img_,
            edgefirst_image::Rotation::None,
            edgefirst_image::Flip::None,
            Crop::no_crop(),
        )?;
        let dest_mapped = self.img_.tensor().map()?;
        let data = dest_mapped.as_slice();
        self.load_input(0, data, 4, preprocessing)?;
        Ok(())
    }

    #[instrument(skip_all)]
    fn run_model(&mut self) -> Result<(), ModelError> {
        Ok(self.model.invoke()?)
    }

    fn input_count(&self) -> Result<usize, ModelError> {
        Ok(self.inputs.len())
    }

    fn input_shape(&self, index: usize) -> Result<Vec<usize>, ModelError> {
        TFLiteModel::input_shape(&self.model, index)
    }

    #[instrument(skip_all)]
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

    #[instrument(skip_all)]
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

    #[instrument(skip_all)]
    fn decode_outputs(
        &self,
        decoder: &edgefirst_decoder::Decoder,
        output_boxes: &mut Vec<edgefirst_decoder::DetectBox>,
        output_masks: &mut Vec<edgefirst_decoder::Segmentation>,
    ) -> Result<(), ModelError> {
        let mut outputs_quant = Vec::new();
        let mut outputs_float = Vec::new();
        for o in &self.outputs {
            match o.tensor_type() {
                TensorType::Float32 => {
                    let data: &[f32] = o.mapro()?;
                    let shape = o.shape()?;
                    let arr = ndarray::ArrayView::from_shape(shape, data).unwrap();
                    outputs_float.push(arr);
                }
                TensorType::UInt8 => {
                    let data: &[u8] = o.mapro()?;
                    let shape = o.shape()?;
                    let arr = ndarray::ArrayView::from_shape(shape, data).unwrap();
                    let arr: ArrayViewDQuantized = arr.into();
                    outputs_quant.push(arr);
                }
                TensorType::Int8 => {
                    let data: &[i8] = o.mapro()?;
                    let shape = o.shape()?;
                    let arr = ndarray::ArrayView::from_shape(shape, data).unwrap();
                    let arr: ArrayViewDQuantized = arr.into();
                    outputs_quant.push(arr);
                }
                _ => {
                    log::warn!("Output of other type");
                }
            }
        }

        if matches!(
            decoder.model_type(),
            edgefirst_decoder::configs::ModelType::Custom {}
        ) {
            // attempt SSD decode
            let mut boxes = Vec::with_capacity(output_boxes.capacity());
            self.ssd_decode_boxes(&mut boxes)?;

            return Ok(decoder.decode_custom(boxes, output_boxes)?);
        }

        match (outputs_float.is_empty(), outputs_quant.is_empty()) {
            (false, true) => decoder
                .decode_float(&outputs_float, output_boxes, output_masks)
                .unwrap(),
            (true, false) => decoder
                .decode_quantized(&outputs_quant, output_boxes, output_masks)
                .unwrap(),
            (false, false) => {
                log::error!("No outputs for decoder");
            }
            (true, true) => {
                log::error!("Fixed floating point and quantized outputs for decoder");
            }
        }
        log::trace!("Decoded boxes: {:?}", output_boxes);
        Ok(())
    }

    #[instrument(skip_all)]
    fn decode_outputs_tracked(
        &self,
        decoder: &mut edgefirst_decoder::Decoder,
        output_boxes: &mut Vec<edgefirst_decoder::DetectBox>,
        output_masks: &mut Vec<edgefirst_decoder::Segmentation>,
        output_tracks: &mut Vec<edgefirst_tracker::TrackInfo<edgefirst_decoder::DetectBox>>,
        timestamp: u64,
    ) -> Result<(), ModelError> {
        let mut outputs_quant = Vec::new();
        let mut outputs_float = Vec::new();
        for o in &self.outputs {
            match o.tensor_type() {
                TensorType::Float32 => {
                    let data: &[f32] = o.mapro()?;
                    let shape = o.shape()?;
                    let arr = ndarray::ArrayView::from_shape(shape, data).unwrap();
                    outputs_float.push(arr);
                }
                TensorType::UInt8 => {
                    let data: &[u8] = o.mapro()?;
                    let shape = o.shape()?;
                    let arr = ndarray::ArrayView::from_shape(shape, data).unwrap();
                    let arr: ArrayViewDQuantized = arr.into();
                    outputs_quant.push(arr);
                }
                TensorType::Int8 => {
                    let data: &[i8] = o.mapro()?;
                    let shape = o.shape()?;
                    let arr = ndarray::ArrayView::from_shape(shape, data).unwrap();
                    let arr: ArrayViewDQuantized = arr.into();
                    outputs_quant.push(arr);
                }
                _ => {
                    log::warn!("Output of other type");
                }
            }
        }

        if matches!(
            decoder.model_type(),
            edgefirst_decoder::configs::ModelType::Custom {}
        ) {
            // attempt SSD decode
            let mut boxes = Vec::with_capacity(output_boxes.capacity());
            self.ssd_decode_boxes(&mut boxes)?;

            return Ok(decoder.decode_custom_tracked(
                boxes,
                output_boxes,
                output_tracks,
                timestamp,
            )?);
        }

        match (outputs_float.is_empty(), outputs_quant.is_empty()) {
            (false, true) => decoder.decode_float_tracked(
                &outputs_float,
                output_boxes,
                output_masks,
                output_tracks,
                timestamp,
            )?,
            (true, false) => decoder.decode_quantized_tracked(
                &outputs_quant,
                output_boxes,
                output_masks,
                output_tracks,
                timestamp,
            )?,
            (false, false) => {
                log::error!("No outputs for decoder");
            }
            (true, true) => {
                log::error!("Fixed floating point and quantized outputs for decoder");
            }
        }
        log::trace!("Decoded boxes tracked: {:?}", output_boxes);
        Ok(())
    }

    fn input_type(&self, index: usize) -> Result<DataType, ModelError> {
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
        Ok(tensor_type_to_datatype(tensor.tensor_type()))
    }

    fn output_type(&self, index: usize) -> Result<DataType, ModelError> {
        let tensor = match self.outputs.get(index) {
            Some(v) => v,
            None => {
                let e = io::Error::other(format!(
                    "Tried to access output tensor {index} of {}",
                    self.inputs.len()
                ))
                .into();
                return Err(e);
            }
        };
        Ok(tensor_type_to_datatype(tensor.tensor_type()))
    }

    fn labels(&self) -> Result<Vec<String>, ModelError> {
        Ok(self.labels.clone())
    }

    fn model_name(&self) -> Result<String, ModelError> {
        match &self.metadata.name {
            Some(v) => Ok(v.clone()),
            None => Ok("".to_string()),
        }
    }

    fn get_model_metadata(&self) -> Result<Metadata, ModelError> {
        Ok(self.metadata.clone())
    }

    fn output_quantization(&self, index: usize) -> Result<Option<(f32, i32)>, ModelError> {
        let quant = self
            .outputs
            .get(index)
            .ok_or_else(|| {
                io::Error::other(format!(
                    "Tried to access output tensor {index} of {}",
                    self.outputs.len()
                ))
            })?
            .get_quantization_params();
        Ok(Some((quant.scale, quant.zero_point)))
    }
}
