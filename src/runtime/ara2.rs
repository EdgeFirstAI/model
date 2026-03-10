// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! ARA-2 NPU runtime implementation wrapping the `ara2` crate.
//!
//! Provides [`Ara2Runtime`] which implements the [`Runtime`](super::Runtime) trait
//! for Kinara ARA-2 neural processing unit inference via the ara2 proxy service.

use std::path::Path;

use edgefirst_hal::decoder::Quantization;
use edgefirst_hal::decoder::configs::DataType;
use edgefirst_hal::image::{PLANAR_RGB, PLANAR_RGB_INT8};
use edgefirst_hal::tensor::{Tensor, TensorMemory};
use four_char_code::FourCharCode;
use log::info;

use super::{InferenceTiming, ModelFormat, ModelInfo, Runtime, extract_metadata};
use crate::model::{ModelError, ModelErrorKind};

/// Map bpp (bytes per pixel/element) and signedness to a `DataType`.
fn bpp_to_datatype(bpp: usize, is_signed: bool) -> DataType {
    match (bpp, is_signed) {
        (1, true) => DataType::Int8,
        (1, false) => DataType::UInt8,
        (2, true) => DataType::Int16,
        (2, false) => DataType::UInt16,
        (4, _) => DataType::Float32,
        _ => DataType::Raw,
    }
}

/// ARA-2 NPU inference runtime.
///
/// Wraps an `ara2::Model` loaded onto a Kinara ARA-2 endpoint and implements
/// the [`Runtime`] trait. Uses DMA tensors for zero-copy data transfer between
/// the host and NPU.
pub struct Ara2Runtime {
    model: ara2::Model,

    // Cached tensor metadata (populated during load)
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
    input_dtypes: Vec<DataType>,
    output_dtypes: Vec<DataType>,
    input_quants: Vec<Option<Quantization>>,
    output_quants: Vec<Option<Quantization>>,

    // Input format for the image pipeline
    input_fourcc_val: FourCharCode,

    // Model metadata
    info: ModelInfo,
}

impl Ara2Runtime {
    /// Load a DVM model onto the ARA-2 NPU and return a ready-to-use
    /// [`Ara2Runtime`].
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the `.dvm` model file.
    /// * `delegate_path` - Path to the ARA-2 proxy UNIX socket. Pass an empty
    ///   string to use the default socket (`/var/run/ara2.sock`).
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if any step in the load sequence fails (session
    /// creation, model loading, tensor allocation, etc.).
    pub fn load(model_path: &Path, delegate_path: &str) -> Result<Self, ModelError> {
        // 1. Read model file
        let model_data = std::fs::read(model_path)?;

        // 2. Connect to ARA-2 proxy
        let socket_path = if delegate_path.is_empty() {
            ara2::DEFAULT_SOCKET
        } else {
            delegate_path
        };

        let session = ara2::Session::create_via_unix_socket(socket_path).map_err(|e| {
            ModelError::new(
                ModelErrorKind::Other,
                format!("Failed to connect to ARA-2 proxy at {socket_path}: {e}"),
            )
        })?;
        info!("Connected to ARA-2 proxy at {socket_path}");

        // 3. List endpoints and load model on the first one
        let endpoints = session.list_endpoints().map_err(|e| {
            ModelError::new(
                ModelErrorKind::Other,
                format!("Failed to list ARA-2 endpoints: {e}"),
            )
        })?;

        if endpoints.is_empty() {
            return Err(ModelError::new(
                ModelErrorKind::Other,
                "No ARA-2 endpoints available".to_string(),
            ));
        }

        let mut model = endpoints[0].load_model_from_file(model_path).map_err(|e| {
            ModelError::new(
                ModelErrorKind::Other,
                format!("Failed to load model {}: {e}", model_path.display()),
            )
        })?;

        model
            .allocate_tensors(Some(TensorMemory::Dma))
            .map_err(|e| {
                ModelError::new(
                    ModelErrorKind::Other,
                    format!("Failed to allocate ARA-2 tensors: {e}"),
                )
            })?;
        info!("Loaded model on ARA-2 endpoint");

        // 4. Cache input tensor metadata
        let mut input_shapes = Vec::with_capacity(model.n_inputs());
        let mut input_dtypes = Vec::with_capacity(model.n_inputs());
        let mut input_quants = Vec::with_capacity(model.n_inputs());

        for i in 0..model.n_inputs() {
            let shape = model.input_shape(i);
            input_shapes.push(shape.to_vec());

            let input_info = model.input_info(i);
            let dtype = bpp_to_datatype(input_info.bpp, input_info.quant.is_signed);
            input_dtypes.push(dtype);

            let quant = Quantization::new(input_info.quant.qn, input_info.quant.mean as i32);
            input_quants.push(Some(quant));

            info!(
                "Input {i}: shape={shape:?} bpp={} signed={} qn={} mean={} (scale={})",
                input_info.bpp, input_info.quant.is_signed, input_info.quant.qn,
                input_info.quant.mean, quant.scale,
            );
        }

        // 5. Cache output tensor metadata
        let mut output_shapes = Vec::with_capacity(model.n_outputs());
        let mut output_dtypes = Vec::with_capacity(model.n_outputs());
        let mut output_quants = Vec::with_capacity(model.n_outputs());

        for i in 0..model.n_outputs() {
            let raw_shape = model.output_shape(i);
            // ARA-2 returns CHW shapes (e.g. [80, 8400, 1]). Normalize to
            // batch-prefixed format [1, C, H] by stripping trailing 1s and
            // prepending batch=1. This matches the NHWC-like convention the
            // decoder/guesser expects without changing memory layout.
            let mut shape: Vec<usize> = raw_shape.to_vec();
            while shape.len() > 1 && shape.last() == Some(&1) {
                shape.pop();
            }
            shape.insert(0, 1);
            let output_info = model.output_info(i).map_err(|e| {
                ModelError::new(
                    ModelErrorKind::Other,
                    format!("Failed to get output {i} info: {e}"),
                )
            })?;
            let dtype = bpp_to_datatype(output_info.bpp, output_info.quant.is_signed);
            output_dtypes.push(dtype);

            let quant = Quantization::new(output_info.quant.qn, output_info.quant.offset);
            output_quants.push(Some(quant));

            info!(
                "Output {i}: raw_shape={raw_shape:?} shape={shape:?} bpp={} signed={} qn={} offset={}",
                output_info.bpp,
                output_info.quant.is_signed,
                output_info.quant.qn,
                output_info.quant.offset,
            );

            output_shapes.push(shape);
        }

        // 6. Determine input fourcc based on quantization type
        let input_fourcc_val = if !input_dtypes.is_empty() && input_dtypes[0] == DataType::Int8 {
            PLANAR_RGB_INT8
        } else {
            PLANAR_RGB
        };

        // 7. Extract metadata from model bytes (ZIP-based)
        let mut info = extract_metadata(&model_data, ModelFormat::Dvm);

        // Additionally try DVM-specific metadata for model name
        if let Ok(Some(dvm_meta)) = ara2::read_metadata(&model_data)
            && info.name.is_none()
            && let Some(ref deployment) = dvm_meta.deployment
        {
            info.name = deployment
                .model_name
                .clone()
                .or_else(|| deployment.name.clone());
        }

        info!("Model metadata: {info:?}");

        Ok(Ara2Runtime {
            model,
            input_shapes,
            output_shapes,
            input_dtypes,
            output_dtypes,
            input_quants,
            output_quants,
            input_fourcc_val,
            info,
        })
    }
}

impl Runtime for Ara2Runtime {
    fn invoke(&mut self) -> Result<InferenceTiming, ModelError> {
        let timing = self.model.run().map_err(|e| {
            ModelError::new(
                ModelErrorKind::Other,
                format!("ARA-2 inference failed: {e}"),
            )
        })?;

        Ok(InferenceTiming {
            input_time: timing.input_time,
            model_time: timing.run_time,
            output_time: timing.output_time,
        })
    }

    fn input_count(&self) -> usize {
        self.input_shapes.len()
    }

    fn input_tensor(&mut self, idx: usize) -> &mut Tensor<u8> {
        self.model.input_tensor(idx)
    }

    fn input_shape(&self, idx: usize) -> &[usize] {
        &self.input_shapes[idx]
    }

    fn input_quantization(&self, idx: usize) -> Option<Quantization> {
        self.input_quants[idx]
    }

    fn input_dtype(&self, idx: usize) -> DataType {
        self.input_dtypes[idx].clone()
    }

    fn input_fourcc(&self, _idx: usize) -> FourCharCode {
        self.input_fourcc_val
    }

    fn output_count(&self) -> usize {
        self.output_shapes.len()
    }

    fn output_tensor(&self, idx: usize) -> &Tensor<u8> {
        self.model.output_tensor(idx)
    }

    fn output_dtype(&self, idx: usize) -> DataType {
        self.output_dtypes[idx].clone()
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
