// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! ARA-2 NPU runtime implementation wrapping the `ara2` crate.
//!
//! Provides [`Ara2Runtime`] which implements the [`Runtime`](super::Runtime) trait
//! for Kinara ARA-2 neural processing unit inference via the ara2 proxy service.
//!
//! ARA-2 expects planar `CHW` input and the HAL `ImageProcessor` only emits
//! packed/interleaved pixels, so this runtime always uses the CPU staging path
//! (`supports_dmabuf` returns `false`): the main loop deinterleaves the
//! preprocessed RGB image into the ARA-2 input tensor. ARA-2 tensors are still
//! DMA-backed internally for the proxy transfer — that is unaffected.

use std::path::Path;

use edgefirst_hal::decoder::Quantization;
use edgefirst_hal::tensor::{
    DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};
use log::{info, warn};

use super::{
    InferenceTiming, ModelFormat, ModelInfo, Runtime, extract_metadata, fill_output_bytes,
};
use crate::model::{ModelError, ModelErrorKind};

/// Map bytes-per-element and signedness to a HAL [`DType`].
fn bpp_to_dtype(bpp: usize, is_signed: bool) -> DType {
    match (bpp, is_signed) {
        (1, true) => DType::I8,
        (1, false) => DType::U8,
        (2, true) => DType::I16,
        (2, false) => DType::U16,
        (4, _) => DType::F32,
        _ => DType::U8,
    }
}

/// ARA-2 NPU inference runtime.
///
/// Wraps an `ara2::Model` loaded onto a Kinara ARA-2 endpoint and implements
/// the [`Runtime`] trait. The `ara2::Model` borrows into the session, so the
/// [`ara2::Session`] and [`ara2::Endpoint`] are retained for the lifetime of
/// this struct.
pub struct Ara2Runtime {
    model: ara2::Model,

    // The model borrows into the session/endpoint — keep them alive.
    _session: ara2::Session,
    _endpoint: ara2::Endpoint,

    // Cached tensor metadata (populated during load)
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
    input_dtypes: Vec<DType>,
    output_dtypes: Vec<DType>,
    input_quants: Vec<Option<Quantization>>,
    output_quants: Vec<Option<Quantization>>,

    // Pixel format for the preprocessing pipeline (planar for ARA-2).
    input_format: PixelFormat,

    // Owned, correctly-typed output tensors fed to the HAL decoder.
    // `sync_outputs_for_cpu` copies the proxy's typed bytes into them
    // per frame.
    output_tensors: Vec<TensorDyn>,

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

        let Some(endpoint) = endpoints.into_iter().next() else {
            return Err(ModelError::new(
                ModelErrorKind::Other,
                "No ARA-2 endpoints available".to_string(),
            ));
        };

        let mut model = endpoint.load_model_from_file(model_path).map_err(|e| {
            ModelError::new(
                ModelErrorKind::Other,
                format!("Failed to load model {}: {e}", model_path.display()),
            )
        })?;

        // Prefer DMA-backed tensors for the fastest proxy transfer; fall back
        // to heap allocation if DMA memory is unavailable.
        match model.allocate_tensors(Some(TensorMemory::Dma)) {
            Ok(()) => info!("ARA-2 tensors allocated with DMA memory"),
            Err(e) => {
                warn!("ARA-2 DMA allocation failed ({e}), falling back to heap");
                model.allocate_tensors(None).map_err(|e| {
                    ModelError::new(
                        ModelErrorKind::Other,
                        format!("Failed to allocate ARA-2 tensors: {e}"),
                    )
                })?;
            }
        }
        info!("Loaded model on ARA-2 endpoint");

        // 4. Cache input tensor metadata
        let mut input_shapes = Vec::with_capacity(model.n_inputs());
        let mut input_dtypes = Vec::with_capacity(model.n_inputs());
        let mut input_quants = Vec::with_capacity(model.n_inputs());

        for i in 0..model.n_inputs() {
            let shape = model.input_shape(i);
            input_shapes.push(shape.to_vec());

            let input_info = model.input_info(i);
            let dtype = bpp_to_dtype(input_info.bpp, input_info.quant.is_signed);
            input_dtypes.push(dtype);

            // `qn` IS the quantization scale for ARA-2 (see project notes —
            // do not invert to 1/qn).
            let quant = Quantization::new(input_info.quant.qn, input_info.quant.offset);
            input_quants.push(Some(quant));

            info!(
                "Input {i}: shape={shape:?} bpp={} signed={} qn={} offset={}",
                input_info.bpp,
                input_info.quant.is_signed,
                input_info.quant.qn,
                input_info.quant.offset,
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
            let dtype = bpp_to_dtype(output_info.bpp, output_info.quant.is_signed);
            output_dtypes.push(dtype);

            // `qn` IS the quantization scale for ARA-2 (see project notes).
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

        // 6. Pre-allocate owned output tensors. `sync_outputs_for_cpu`
        // copies the proxy's typed bytes into them per frame — same
        // pattern as the profiler. (True zero-copy aliasing would need
        // either dedicated per-output fds from the proxy or a HAL
        // `Tensor::from_fd_with_offset`; tracked as a follow-up.) Attach
        // quantization on integer outputs so per-scale decoders that read
        // it off the TensorDyn work correctly.
        let mut output_tensors = Vec::with_capacity(output_shapes.len());
        for (i, (shape, &dtype)) in output_shapes.iter().zip(output_dtypes.iter()).enumerate() {
            let mut t =
                TensorDyn::new(shape, dtype, Some(TensorMemory::Mem), None).map_err(|e| {
                    ModelError::new(
                        ModelErrorKind::Tensor,
                        format!("Failed to allocate output tensor {i}: {e}"),
                    )
                })?;
            if let Some(q) = output_quants[i].as_ref()
                && !matches!(dtype, DType::F16 | DType::F32 | DType::F64)
            {
                let _ = t.set_quantization(edgefirst_hal::tensor::Quantization::per_tensor(
                    q.scale,
                    q.zero_point,
                ));
            }
            output_tensors.push(t);
        }

        // 7. ARA-2 always wants planar RGB input; the int8/uint8 distinction is
        //    carried by `input_dtype`, not the pixel format.
        let input_format = PixelFormat::PlanarRgb;

        // 8. Extract metadata from model bytes (ZIP-based)
        let mut info = extract_metadata(&model_data, ModelFormat::Dvm);

        // Additionally try DVM-specific metadata for model name and labels.
        if let Ok(Some(dvm_meta)) = ara2::read_metadata(&model_data)
            && info.name.is_none()
            && let Some(ref deployment) = dvm_meta.deployment
        {
            info.name = deployment
                .model_name
                .clone()
                .or_else(|| deployment.name.clone());
        }
        if info.labels.is_empty()
            && let Ok(labels) = ara2::dvm_metadata::read_labels(&model_data)
            && !labels.is_empty()
        {
            info.labels = labels;
        }

        info!("Model metadata: {info:?}");

        Ok(Ara2Runtime {
            model,
            _session: session,
            _endpoint: endpoint,
            input_shapes,
            output_shapes,
            input_dtypes,
            output_dtypes,
            input_quants,
            output_quants,
            input_format,
            output_tensors,
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

        // Output staging is performed in `sync_outputs_for_cpu` — symmetric
        // with TfLiteRuntime, where the DMA-BUF cache flush must precede the
        // CPU read.
        Ok(InferenceTiming {
            input_time: timing.input_time,
            model_time: timing.run_time,
            output_time: timing.output_time,
        })
    }

    fn sync_outputs_for_cpu(&mut self) -> Result<(), ModelError> {
        // The ARA-2 proxy returns coherent output buffers after
        // `model.run()`; no explicit sync is needed. Stage the typed bytes
        // into the decoder-ready TensorDyns so the trait contract holds —
        // outputs are safe to read after this call.
        for i in 0..self.output_tensors.len() {
            let map = self.model.output_tensor(i).map()?;
            fill_output_bytes(&mut self.output_tensors[i], map.as_slice())?;
        }
        Ok(())
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

    fn input_dtype(&self, idx: usize) -> DType {
        self.input_dtypes[idx]
    }

    fn input_pixel_format(&self, _idx: usize) -> PixelFormat {
        self.input_format
    }

    fn output_count(&self) -> usize {
        self.output_shapes.len()
    }

    fn output_tensor(&self, idx: usize) -> &TensorDyn {
        &self.output_tensors[idx]
    }

    fn output_dtype(&self, idx: usize) -> DType {
        self.output_dtypes[idx]
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
