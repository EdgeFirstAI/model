// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! TFLite runtime implementation wrapping `edgefirst-tflite` Interpreter.
//!
//! Provides [`TfLiteRuntime`] which implements the [`Runtime`](super::Runtime) trait
//! for TensorFlow Lite model inference with optional DMA-BUF zero-copy and
//! CameraAdaptor NPU preprocessing.

use std::os::fd::FromRawFd;
use std::path::Path;
use std::time::Instant;

use edgefirst_hal::decoder::Quantization;
use edgefirst_hal::decoder::configs::DataType;
use edgefirst_hal::image::{RGB, RGB_INT8, RGBA};
use edgefirst_hal::tensor::{DmaTensor, Tensor, TensorMapTrait, TensorMemory, TensorTrait};
use edgefirst_tflite::dmabuf::BufferHandle;
use edgefirst_tflite::{Delegate, Interpreter, Library, TensorType};
use four_char_code::FourCharCode;
use log::{error, info, warn};

use super::{InferenceTiming, ModelFormat, ModelInfo, Runtime, extract_metadata};
use crate::model::{ModelError, ModelErrorKind};

/// Map a TFLite `TensorType` to an `edgefirst_hal` `DataType`.
fn tflite_type_to_datatype(tt: TensorType) -> DataType {
    match tt {
        TensorType::Float32 => DataType::Float32,
        TensorType::Float16 => DataType::Float16,
        TensorType::Float64 => DataType::Float64,
        TensorType::Int8 => DataType::Int8,
        TensorType::UInt8 => DataType::UInt8,
        TensorType::Int16 => DataType::Int16,
        TensorType::UInt16 => DataType::UInt16,
        TensorType::Int32 => DataType::Int32,
        TensorType::UInt32 => DataType::UInt32,
        TensorType::Int64 => DataType::Int64,
        TensorType::UInt64 => DataType::UInt64,
        _ => DataType::Raw,
    }
}

/// TFLite-based inference runtime.
///
/// Wraps an `edgefirst-tflite` [`Interpreter`] and implements the [`Runtime`]
/// trait. Supports optional DMA-BUF zero-copy and CameraAdaptor NPU
/// preprocessing when a compatible delegate is loaded.
pub struct TfLiteRuntime {
    interpreter: Interpreter<'static>,

    // Cached tensor metadata (populated during load)
    input_shapes: Vec<Vec<usize>>,
    #[allow(dead_code)]
    input_types: Vec<TensorType>,
    input_quants: Vec<Option<Quantization>>,
    output_shapes: Vec<Vec<usize>>,
    output_types: Vec<TensorType>,
    output_quants: Vec<Option<Quantization>>,

    // DMA-BUF state
    dmabuf_handle: Option<BufferHandle>,

    // Input tensor (writable by main loop)
    input_tensor: Tensor<u8>,
    input_fourcc_val: FourCharCode,

    // Pre-allocated output tensors (reused across frames, no per-frame alloc)
    output_tensors: Vec<Tensor<u8>>,

    // Model metadata
    info: ModelInfo,
}

impl TfLiteRuntime {
    /// Load a TFLite model and build an interpreter, returning a ready-to-use
    /// [`TfLiteRuntime`].
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the `.tflite` model file.
    /// * `delegate_path` - Path to a delegate shared library (e.g.,
    ///   `libvx_delegate.so`). Pass an empty string to skip delegate loading.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if any step in the load sequence fails (I/O,
    /// library loading, interpreter creation, DMA-BUF setup, etc.).
    pub fn load(model_path: &Path, delegate_path: &str) -> Result<Self, ModelError> {
        // 1. Read model file
        let model_data = std::fs::read(model_path)?;

        // 2. Create Library (leak for 'static lifetime required by Interpreter)
        let lib: &'static Library = Box::leak(Box::new(Library::new()?));

        // 3. Create Model from bytes (leak so model_mem stays available for
        //    metadata extraction via model.data())
        let model: &'static edgefirst_tflite::Model<'static> = Box::leak(Box::new(
            edgefirst_tflite::Model::from_bytes(lib, model_data)?,
        ));

        // 4. Load delegate (if path is not empty)
        let mut use_camera_adaptor = false;
        let use_dmabuf;

        let delegate = if !delegate_path.is_empty() {
            match Delegate::load(delegate_path) {
                Ok(d) => {
                    info!("Delegate loaded: {delegate_path}");

                    if d.has_camera_adaptor()
                        && let Some(adaptor) = d.camera_adaptor()
                    {
                        if let Err(e) = adaptor.set_format(0, "rgba") {
                            warn!("CameraAdaptor set_format failed: {e:?}");
                        } else {
                            use_camera_adaptor = true;
                            info!("CameraAdaptor: enabled (RGBA -> RGB on NPU)");
                        }
                    }

                    use_dmabuf = d.has_dmabuf();
                    if use_dmabuf {
                        info!("DMA-BUF: available");
                    }

                    Some(d)
                }
                Err(e) => {
                    return Err(ModelError::new(
                        ModelErrorKind::TFLite,
                        format!("Could not load delegate {delegate_path}: {e:?}"),
                    ));
                }
            }
        } else {
            info!("No delegate specified, using CPU inference");
            use_dmabuf = false;
            None
        };

        // 5. Build interpreter
        let mut builder = Interpreter::builder(lib)?;
        if let Some(d) = delegate {
            builder = builder.delegate(d);
        }
        let interpreter = builder.build(model)?;
        info!("Loaded model");

        // 6. Inspect input tensors
        let (input_shapes, input_types, input_quants) = {
            let inputs = interpreter.inputs()?;
            let shapes: Vec<Vec<usize>> = inputs
                .iter()
                .map(|t| t.shape().unwrap_or_default())
                .collect();
            let types: Vec<TensorType> = inputs.iter().map(|t| t.tensor_type()).collect();
            let quants: Vec<Option<Quantization>> = inputs
                .iter()
                .map(|t| {
                    let qp = t.quantization_params();
                    Some(Quantization::new(qp.scale, qp.zero_point))
                })
                .collect();

            for (i, t) in inputs.iter().enumerate() {
                let qp = t.quantization_params();
                info!("Input {i}: {t} (scale={}, zp={})", qp.scale, qp.zero_point);
            }

            (shapes, types, quants)
        };

        // 7. Inspect output tensors
        let (output_shapes, output_types, output_quants) = {
            let outputs = interpreter.outputs()?;
            let shapes: Vec<Vec<usize>> = outputs
                .iter()
                .map(|t| t.shape().unwrap_or_default())
                .collect();
            let types: Vec<TensorType> = outputs.iter().map(|t| t.tensor_type()).collect();
            let quants: Vec<Option<Quantization>> = outputs
                .iter()
                .map(|t| {
                    let qp = t.quantization_params();
                    Some(Quantization::new(qp.scale, qp.zero_point))
                })
                .collect();

            for (i, t) in outputs.iter().enumerate() {
                let qp = t.quantization_params();
                info!("Output {i}: {t} (scale={}, zp={})", qp.scale, qp.zero_point);
            }

            (shapes, types, quants)
        };

        // Primary input shape: [batch, height, width, channels]
        let in_h = input_shapes[0].get(1).copied().unwrap_or(0);
        let in_w = input_shapes[0].get(2).copied().unwrap_or(0);
        let input_type = input_types[0];

        // 8. Set up DMA-BUF and create input tensor
        let dmabuf_handle = if use_dmabuf {
            let delegate_ref = interpreter
                .delegate(0)
                .expect("delegate not found after load");
            let dmabuf = delegate_ref
                .dmabuf()
                .expect("DMA-BUF probed but not available");
            let buf_size = if use_camera_adaptor {
                in_h * in_w * 4 // RGBA
            } else {
                let inputs = interpreter.inputs()?;
                inputs[0].byte_size()
            };
            match dmabuf.request(0, edgefirst_tflite::dmabuf::Ownership::Delegate, buf_size) {
                Ok((handle, _desc)) => {
                    if let Err(e) = dmabuf.bind_to_tensor(handle, 0) {
                        error!("Could not bind DMA-BUF to input tensor: {e:?}");
                        return Err(ModelError::new(
                            ModelErrorKind::TFLite,
                            format!("Could not bind DMA-BUF to input tensor: {e:?}"),
                        ));
                    }
                    info!("DMA-BUF bound to input tensor (size={buf_size})");
                    Some(handle)
                }
                Err(e) => {
                    warn!("DMA-BUF request failed, falling back to CPU: {e:?}");
                    None
                }
            }
        } else {
            None
        };

        // 9. Create HAL input tensor
        let (input_tensor, input_fourcc_val) = if let Some(handle) = dmabuf_handle {
            // DMA-BUF mode: wrap the delegate-allocated buffer as a HAL DMA tensor
            let delegate_ref = interpreter.delegate(0).unwrap();
            let dmabuf = delegate_ref.dmabuf().unwrap();
            let fd = dmabuf.fd(handle)?;

            // dup() the fd so the HAL tensor owns its own copy
            let dup_fd = unsafe { nix::libc::dup(fd) };
            if dup_fd < 0 {
                return Err(ModelError::new(
                    ModelErrorKind::Io,
                    format!(
                        "Failed to dup DMA-BUF fd: {}",
                        std::io::Error::last_os_error()
                    ),
                ));
            }
            let owned_fd = unsafe { std::os::fd::OwnedFd::from_raw_fd(dup_fd) };

            if use_camera_adaptor {
                // CameraAdaptor: input is RGBA (4 channels)
                let shape = [in_h, in_w, 4];
                let dma = DmaTensor::<u8>::from_fd(owned_fd, &shape, None)?;
                (Tensor::Dma(dma), RGBA)
            } else if input_type == TensorType::Int8 {
                let shape = [in_h, in_w, 3];
                let dma = DmaTensor::<u8>::from_fd(owned_fd, &shape, None)?;
                (Tensor::Dma(dma), RGB_INT8)
            } else {
                let shape = [in_h, in_w, 3];
                let dma = DmaTensor::<u8>::from_fd(owned_fd, &shape, None)?;
                (Tensor::Dma(dma), RGB)
            }
        } else {
            // CPU fallback: create a regular memory tensor
            let fourcc = if use_camera_adaptor {
                RGBA
            } else if input_type == TensorType::Int8 {
                RGB_INT8
            } else {
                RGB
            };

            let channels: usize = if use_camera_adaptor { 4 } else { 3 };
            let shape = [in_h, in_w, channels];
            let tensor = Tensor::<u8>::new(&shape, None, None)?;
            (tensor, fourcc)
        };

        // 10. Extract metadata from model bytes
        let mut info = extract_metadata(model.data(), ModelFormat::TfLite);

        // Also extract the TFLite-specific name from FlatBuffer metadata
        let tflite_meta = edgefirst_tflite::metadata::Metadata::from_model_bytes(model.data());
        if info.name.is_none() {
            info.name = tflite_meta.name;
        }

        info!("Model metadata: {info:?}");

        // Pre-allocate output tensors with correct shapes.
        // These are reused every frame — only the data is copied, not the
        // allocation. For non-u8 types (e.g. float32) the byte_size differs
        // from the shape volume, so we use a flat [byte_size] shape.
        let mut output_tensors = Vec::with_capacity(output_shapes.len());
        for (i, shape) in output_shapes.iter().enumerate() {
            let volume: usize = shape.iter().product();
            let bytes_per_element = match output_types[i] {
                TensorType::Float32 => 4,
                TensorType::Float16 => 2,
                TensorType::Float64 => 8,
                TensorType::Int16 | TensorType::UInt16 => 2,
                TensorType::Int32 | TensorType::UInt32 => 4,
                TensorType::Int64 | TensorType::UInt64 => 8,
                _ => 1,
            };
            let byte_size = volume * bytes_per_element;
            let tensor_shape = if bytes_per_element == 1 {
                shape.clone()
            } else {
                vec![byte_size]
            };
            output_tensors.push(Tensor::<u8>::new(
                &tensor_shape,
                Some(TensorMemory::Mem),
                None,
            )?);
        }

        Ok(TfLiteRuntime {
            interpreter,
            input_shapes,
            input_types,
            input_quants,
            output_shapes,
            output_types,
            output_quants,
            dmabuf_handle,
            input_tensor,
            input_fourcc_val,
            output_tensors,
            info,
        })
    }

    /// Copy interpreter output data into pre-allocated HAL tensors.
    ///
    /// TFLite output tensor borrows are temporary, so we copy the raw bytes
    /// into the pre-allocated `output_tensors`. No allocations occur here.
    fn cache_output_tensors(&mut self) -> Result<(), ModelError> {
        let outputs = self.interpreter.outputs()?;

        for (i, tensor) in outputs.iter().enumerate() {
            let data: &[u8] = tensor.as_slice::<u8>()?;
            let mut map = self.output_tensors[i].map()?;
            map.as_mut_slice().copy_from_slice(data);
        }

        Ok(())
    }
}

impl Runtime for TfLiteRuntime {
    fn invoke(&mut self) -> Result<InferenceTiming, ModelError> {
        let input_start = Instant::now();

        if let Some(handle) = self.dmabuf_handle {
            // DMA-BUF mode: input_tensor IS the DMA buffer bound to the interpreter.
            // Sync to device before invoke.
            if let Some(delegate_ref) = self.interpreter.delegate(0)
                && let Some(dmabuf) = delegate_ref.dmabuf()
                && let Err(e) = dmabuf.sync_for_device(handle)
            {
                error!("DMA-BUF sync_for_device failed: {e:?}");
            }
        } else {
            // CPU mode: write directly into the interpreter's typed input slice.
            // No intermediate buffer — as_mut_slice gives us a mutable view of
            // the interpreter's own memory.
            let map = self.input_tensor.map()?;
            let pixels = map.as_slice();
            let mut inputs = self.interpreter.inputs_mut()?;
            let input = &mut inputs[0];
            match self.input_types[0] {
                TensorType::Float32 => {
                    let dst = input.as_mut_slice::<f32>().map_err(|e| {
                        ModelError::new(ModelErrorKind::TFLite, format!("map f32 input: {e:?}"))
                    })?;
                    for (d, &src) in dst.iter_mut().zip(pixels.iter()) {
                        *d = f32::from(src) / 255.0;
                    }
                }
                TensorType::Int8 => {
                    let dst = input.as_mut_slice::<i8>().map_err(|e| {
                        ModelError::new(ModelErrorKind::TFLite, format!("map i8 input: {e:?}"))
                    })?;
                    #[expect(clippy::cast_possible_wrap, reason = "intentional u8→i8 quantization")]
                    for (d, &src) in dst.iter_mut().zip(pixels.iter()) {
                        *d = src.wrapping_sub(128) as i8;
                    }
                }
                _ => {
                    input.copy_from_slice(pixels).map_err(|e| {
                        ModelError::new(ModelErrorKind::TFLite, format!("copy u8 input: {e:?}"))
                    })?;
                }
            }
        }

        let input_time = input_start.elapsed();

        // Run inference
        let model_start = Instant::now();
        self.interpreter.invoke()?;
        let model_time = model_start.elapsed();

        let output_start = Instant::now();

        // Sync DMA-BUF output back to CPU (if DMA-BUF mode)
        if let Some(handle) = self.dmabuf_handle
            && let Some(delegate_ref) = self.interpreter.delegate(0)
            && let Some(dmabuf) = delegate_ref.dmabuf()
            && let Err(e) = dmabuf.sync_for_cpu(handle)
        {
            error!("DMA-BUF sync_for_cpu failed: {e:?}");
        }

        // Cache output tensors
        self.cache_output_tensors()?;

        let output_time = output_start.elapsed();

        Ok(InferenceTiming {
            input_time,
            model_time,
            output_time,
        })
    }

    fn input_count(&self) -> usize {
        self.input_shapes.len()
    }

    fn input_tensor(&mut self, _idx: usize) -> &mut Tensor<u8> {
        &mut self.input_tensor
    }

    fn input_shape(&self, idx: usize) -> &[usize] {
        &self.input_shapes[idx]
    }

    fn input_quantization(&self, idx: usize) -> Option<Quantization> {
        self.input_quants[idx]
    }

    fn input_dtype(&self, idx: usize) -> DataType {
        tflite_type_to_datatype(self.input_types[idx])
    }

    fn input_fourcc(&self, _idx: usize) -> FourCharCode {
        self.input_fourcc_val
    }

    fn output_count(&self) -> usize {
        self.output_shapes.len()
    }

    fn output_tensor(&self, idx: usize) -> &Tensor<u8> {
        &self.output_tensors[idx]
    }

    fn output_dtype(&self, idx: usize) -> DataType {
        tflite_type_to_datatype(self.output_types[idx])
    }

    fn output_shape(&self, idx: usize) -> &[usize] {
        &self.output_shapes[idx]
    }

    fn output_quantization(&self, idx: usize) -> Option<Quantization> {
        self.output_quants[idx]
    }

    fn metadata(&self) -> &ModelInfo {
        &self.info
    }
}
