// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! TFLite runtime implementation wrapping `edgefirst-tflite` Interpreter.
//!
//! Provides [`TfLiteRuntime`] which implements the [`Runtime`](super::Runtime)
//! trait for TensorFlow Lite inference on both the i.MX 8M Plus **VX delegate**
//! and the i.MX 95 **Neutron delegate**, with a full DMA-BUF zero-copy input
//! pipeline.
//!
//! Delegate selection is automatic: an empty/`auto` delegate string scans the
//! model for the Neutron `NeutronGraph` custom op, then probes `/usr/lib` for
//! `libneutron_delegate.so` / `libvx_delegate.so`, falling back to XNNPACK.
//! Explicit `.so` paths are honored; `none`/`cpu` forces plain CPU inference.

use std::os::fd::BorrowedFd;
use std::path::Path;
use std::time::Instant;

use edgefirst_hal::decoder::Quantization;
use edgefirst_hal::image::ImageProcessor;
use edgefirst_hal::tensor::{
    DType, PixelFormat, PlaneDescriptor, Tensor, TensorDyn, TensorMapTrait, TensorMemory,
    TensorTrait,
};
use edgefirst_tflite::{Delegate, DelegateOptions, Interpreter, Library, Model, TensorType};
use log::{info, warn};

use super::{
    InferenceTiming, ModelFormat, ModelInfo, Runtime, extract_metadata, fill_output_bytes,
};
use crate::model::{ModelError, ModelErrorKind};

/// Default i.MX 8M Plus Vivante VX delegate path.
const VX_DELEGATE_PATH: &str = "/usr/lib/libvx_delegate.so";
/// Default i.MX 95 Neutron delegate path.
const NEUTRON_DELEGATE_PATH: &str = "/usr/lib/libneutron_delegate.so";
/// Custom-op marker present in models compiled for the Neutron delegate.
const NEUTRON_CUSTOM_OP: &str = "NeutronGraph";

/// Selected delegate after auto-resolution.
enum DelegateChoice {
    /// Plain CPU inference, no delegate.
    None,
    /// Built-in XNNPACK CPU delegate.
    XnnPack,
    /// External delegate shared library (VX or Neutron).
    ExternalSo(String),
}

/// Scan a TFLite flatbuffer for a custom-op marker (`\0<name>\0`).
///
/// Neutron-compiled models embed a `NeutronGraph` custom op; detecting it lets
/// the runtime demand the Neutron delegate on i.MX 95 without explicit config.
fn model_requires_delegate(model_bytes: &[u8], custom_op_name: &str) -> bool {
    let mut marker = Vec::with_capacity(custom_op_name.len() + 2);
    marker.push(0u8);
    marker.extend_from_slice(custom_op_name.as_bytes());
    marker.push(0u8);
    model_bytes.windows(marker.len()).any(|w| w == marker)
}

/// Resolve the requested delegate string into a concrete [`DelegateChoice`].
///
/// `""`/`"auto"` probes the model for the Neutron marker, then the filesystem
/// for the Neutron / VX delegate libraries, falling back to XNNPACK.
fn resolve_delegate_choice(
    input: &str,
    model_bytes: &[u8],
) -> Result<(DelegateChoice, String), ModelError> {
    let needs_neutron = model_requires_delegate(model_bytes, NEUTRON_CUSTOM_OP);

    match input.trim() {
        "" | "auto" => {
            if needs_neutron {
                if Path::new(NEUTRON_DELEGATE_PATH).is_file() {
                    Ok((
                        DelegateChoice::ExternalSo(NEUTRON_DELEGATE_PATH.to_string()),
                        "auto:neutron".to_string(),
                    ))
                } else {
                    Err(ModelError::new(
                        ModelErrorKind::TFLite,
                        format!(
                            "model contains a `{NEUTRON_CUSTOM_OP}` custom op but \
                             {NEUTRON_DELEGATE_PATH} is not present"
                        ),
                    ))
                }
            } else if Path::new(VX_DELEGATE_PATH).is_file() {
                Ok((
                    DelegateChoice::ExternalSo(VX_DELEGATE_PATH.to_string()),
                    "auto:vx".to_string(),
                ))
            } else {
                Ok((DelegateChoice::XnnPack, "auto:xnnpack".to_string()))
            }
        }
        "none" | "cpu" => Ok((DelegateChoice::None, "none".to_string())),
        "xnnpack" => Ok((DelegateChoice::XnnPack, "xnnpack".to_string())),
        path => {
            if needs_neutron && !path.contains("neutron") {
                warn!(
                    "model needs the Neutron delegate but an unrelated delegate \
                     '{path}' was requested"
                );
            }
            Ok((
                DelegateChoice::ExternalSo(path.to_string()),
                "explicit".to_string(),
            ))
        }
    }
}

/// Map a TFLite `TensorType` to a HAL [`DType`].
fn tensor_type_to_dtype(tt: TensorType) -> DType {
    match tt {
        TensorType::Float32 => DType::F32,
        TensorType::Float16 => DType::F16,
        TensorType::Float64 => DType::F64,
        TensorType::Int8 => DType::I8,
        TensorType::UInt8 => DType::U8,
        TensorType::Int16 => DType::I16,
        TensorType::UInt16 => DType::U16,
        TensorType::Int32 => DType::I32,
        TensorType::UInt32 => DType::U32,
        TensorType::Int64 => DType::I64,
        TensorType::UInt64 => DType::U64,
        _ => DType::U8,
    }
}

/// Convert TFLite quantization params to a HAL [`Quantization`], treating an
/// all-zero scale (non-quantized tensors) as absent.
fn tflite_quant(scale: f32, zero_point: i32) -> Option<Quantization> {
    if scale == 0.0 {
        None
    } else {
        Some(Quantization::new(scale, zero_point))
    }
}

/// TFLite-based inference runtime (VX delegate, Neutron delegate, or CPU).
pub struct TfLiteRuntime {
    interpreter: Interpreter<'static>,

    has_dmabuf: bool,
    /// `Some("rgba")` when a CameraAdaptor performs on-NPU RGBA→RGB.
    camera_format: Option<&'static str>,

    // Cached tensor metadata (populated during load)
    input_shapes: Vec<Vec<usize>>,
    input_dtypes: Vec<DType>,
    input_quants: Vec<Option<Quantization>>,
    output_shapes: Vec<Vec<usize>>,
    output_dtypes: Vec<DType>,
    output_quants: Vec<Option<Quantization>>,

    /// CPU staging input buffer — only used when `has_dmabuf` is false.
    input_tensor: Tensor<u8>,

    /// Owned, correctly-typed output tensors fed to the HAL decoder.
    /// `sync_outputs_for_cpu` copies the interpreter's typed output bytes
    /// into these Mem-backed buffers per frame (see the load-time comment
    /// on the allocation loop for the rationale).
    output_tensors: Vec<TensorDyn>,

    info: ModelInfo,
}

impl TfLiteRuntime {
    /// Load a TFLite model and build an interpreter.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the `.tflite` model file.
    /// * `delegate_path` - Delegate selector: empty/`auto` for auto-detection,
    ///   `none`/`cpu` for CPU, `xnnpack`, or an explicit `.so` path.
    pub fn load(model_path: &Path, delegate_path: &str) -> Result<Self, ModelError> {
        // Library and Model are leaked for the 'static lifetime the Interpreter
        // requires; the process owns a single model for its whole lifetime.
        let lib: &'static Library = Box::leak(Box::new(Library::new()?));
        let model: &'static Model<'static> =
            Box::leak(Box::new(Model::from_file(lib, model_path)?));

        // Auto-resolve the delegate (Neutron-aware).
        let (choice, label) = resolve_delegate_choice(delegate_path, model.data())?;

        let mut builder = Interpreter::builder(lib)?.num_threads(4);
        let mut has_dmabuf = false;
        let mut camera_format: Option<&'static str> = None;

        match choice {
            DelegateChoice::None => {
                info!("Delegate: none ({label}) — CPU inference");
            }
            DelegateChoice::XnnPack => {
                info!("Delegate: XNNPACK ({label})");
                builder = builder.delegate(Delegate::xnnpack(lib, 4)?);
            }
            DelegateChoice::ExternalSo(path) => {
                let opts = DelegateOptions::new().option("camera_adaptor", "rgba");
                let delegate = Delegate::load_with_options(&path, &opts).map_err(|e| {
                    ModelError::new(
                        ModelErrorKind::TFLite,
                        format!("Could not load delegate '{path}': {e:?}"),
                    )
                })?;
                has_dmabuf = delegate.has_dmabuf();
                // The CameraAdaptor's RGBA path only makes sense when paired
                // with DMA-BUF zero-copy input — otherwise the CPU staging
                // path would have to write RGBA into an RGB-shaped tensor.
                // Gate accordingly so `input_pixel_format` and the staging
                // buffer agree.
                if has_dmabuf
                    && delegate.has_camera_adaptor()
                    && let Some(adaptor) = delegate.camera_adaptor()
                    && adaptor.is_format_supported("rgba")
                {
                    camera_format = Some("rgba");
                }
                info!(
                    "Delegate: {path} ({label}) dmabuf={has_dmabuf} camera_adaptor={}",
                    camera_format.is_some()
                );
                builder = builder.delegate(delegate);
            }
        }

        let interpreter = builder.build(model)?;
        info!("Loaded TFLite model");

        // Inspect input tensors.
        let (input_shapes, input_dtypes, input_quants) = {
            let inputs = interpreter.inputs()?;
            let mut shapes = Vec::with_capacity(inputs.len());
            let mut dtypes = Vec::with_capacity(inputs.len());
            let mut quants = Vec::with_capacity(inputs.len());
            for (i, t) in inputs.iter().enumerate() {
                let shape = t.shape()?;
                let qp = t.quantization_params();
                info!(
                    "Input {i}: shape={shape:?} type={:?} (scale={}, zp={})",
                    t.tensor_type(),
                    qp.scale,
                    qp.zero_point
                );
                shapes.push(shape);
                dtypes.push(tensor_type_to_dtype(t.tensor_type()));
                quants.push(tflite_quant(qp.scale, qp.zero_point));
            }
            (shapes, dtypes, quants)
        };

        // Inspect output tensors.
        let (output_shapes, output_dtypes, output_quants) = {
            let outputs = interpreter.outputs()?;
            let mut shapes = Vec::with_capacity(outputs.len());
            let mut dtypes = Vec::with_capacity(outputs.len());
            let mut quants = Vec::with_capacity(outputs.len());
            for (i, t) in outputs.iter().enumerate() {
                let shape = t.shape()?;
                let qp = t.quantization_params();
                info!(
                    "Output {i}: shape={shape:?} type={:?} (scale={}, zp={})",
                    t.tensor_type(),
                    qp.scale,
                    qp.zero_point
                );
                shapes.push(shape);
                dtypes.push(tensor_type_to_dtype(t.tensor_type()));
                quants.push(tflite_quant(qp.scale, qp.zero_point));
            }
            (shapes, dtypes, quants)
        };

        // Primary input geometry: [batch, height, width, channels].
        let in_h = input_shapes[0].get(1).copied().unwrap_or(0);
        let in_w = input_shapes[0].get(2).copied().unwrap_or(0);

        // CPU staging input buffer (interleaved RGB u8). Unused on the
        // DMA-BUF path but always allocated to satisfy the trait.
        let input_tensor = Tensor::<u8>::new(&[in_h, in_w, 3], Some(TensorMemory::Mem), None)?;

        // Pre-allocate owned, decoder-ready output tensors. Each frame
        // `sync_outputs_for_cpu` stages the interpreter's typed bytes into
        // them via `fill_output_bytes` — the same pattern the profiler
        // follows and the path the HAL recommends for CPU-side
        // post-processing. (`hal_dmabuf_get_tensor_info` for outputs is
        // documented as optional and intended for GPU consumers; the
        // Neutron delegate packs all tensors into a single shared fd at
        // per-tensor offsets, which `Tensor::from_fd` does not currently
        // support — tracked as a HAL follow-up for true zero-copy CPU
        // decode.) Per-scale decoders (e.g. the Neutron-compiled YOLO
        // split) read quantization off each TensorDyn directly, so attach
        // it here for integer outputs.
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

        let info = extract_metadata(model.data(), ModelFormat::TfLite);
        info!("Model metadata: {info:?}");

        Ok(TfLiteRuntime {
            interpreter,
            has_dmabuf,
            camera_format,
            input_shapes,
            input_dtypes,
            input_quants,
            output_shapes,
            output_dtypes,
            output_quants,
            input_tensor,
            output_tensors,
            info,
        })
    }

    /// Resolve the `(PixelFormat, DType)` the HAL should write into the
    /// imported DMA-BUF input tensor.
    ///
    /// `ImageProcessor::convert` performs the dtype conversion when the
    /// destination tensor's dtype differs from the source's (u8 camera
    /// pixels → i8, f32, etc.), so the import dtype must match the model's
    /// actual input dtype — not always u8.
    fn import_format(&self) -> (PixelFormat, DType) {
        if self.camera_format == Some("rgba") {
            // CameraAdaptor: NPU does the RGBA → model-input conversion
            // itself; HAL only has to deliver RGBA u8.
            (PixelFormat::Rgba, DType::U8)
        } else {
            // No adaptor: HAL writes interleaved RGB at the model's input
            // dtype. Default to u8 only when the model declares no input.
            let dtype = self.input_dtypes.first().copied().unwrap_or(DType::U8);
            (PixelFormat::Rgb, dtype)
        }
    }
}

impl Runtime for TfLiteRuntime {
    fn invoke(&mut self) -> Result<InferenceTiming, ModelError> {
        let input_start = Instant::now();

        // On the DMA-BUF path the HAL has already written preprocessed pixels
        // into the delegate's input buffer (and `sync_input_for_device` ran),
        // so no input copy is needed. On the CPU path, convert the staged
        // RGB u8 buffer into the interpreter's typed input.
        if !self.has_dmabuf {
            let map = self.input_tensor.map()?;
            let pixels = map.as_slice();
            let mut inputs = self.interpreter.inputs_mut()?;
            let input = &mut inputs[0];
            match self.input_dtypes.first().copied().unwrap_or(DType::U8) {
                DType::F32 => {
                    let dst = input.as_mut_slice::<f32>().map_err(|e| {
                        ModelError::new(ModelErrorKind::TFLite, format!("map f32 input: {e:?}"))
                    })?;
                    for (d, &src) in dst.iter_mut().zip(pixels.iter()) {
                        *d = f32::from(src) / 255.0;
                    }
                }
                DType::I8 => {
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

        let model_start = Instant::now();
        self.interpreter.invoke()?;
        let model_time = model_start.elapsed();

        // The output copy into `self.output_tensors` moves to
        // `sync_outputs_for_cpu`; on the DMA-BUF path the dmabuf must be
        // flushed for the CPU before reading is safe.
        Ok(InferenceTiming {
            input_time,
            model_time,
            output_time: std::time::Duration::ZERO,
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

    fn input_dtype(&self, idx: usize) -> DType {
        self.input_dtypes[idx]
    }

    fn input_pixel_format(&self, _idx: usize) -> PixelFormat {
        if self.camera_format == Some("rgba") {
            PixelFormat::Rgba
        } else {
            PixelFormat::Rgb
        }
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

    fn supports_dmabuf(&self) -> bool {
        self.has_dmabuf
    }

    fn import_input_image(
        &mut self,
        processor: &ImageProcessor,
        width: usize,
        height: usize,
    ) -> Option<Result<TensorDyn, ModelError>> {
        if !self.has_dmabuf {
            return None;
        }

        // `import_format` selects the destination dtype from the model's
        // input dtype, so `ImageProcessor::convert` writes the correct
        // representation directly into the delegate's input dmabuf —
        // including u8 → i8 / f32 conversion when the model demands it.
        let (fmt, dtype) = self.import_format();

        let Some(dmabuf) = self.interpreter.delegate(0).and_then(Delegate::dmabuf) else {
            return Some(Err(ModelError::new(
                ModelErrorKind::TFLite,
                "delegate reports DMA-BUF support but the DmaBuf API is unavailable".into(),
            )));
        };

        // Modern path (Neutron / HAL delegate API): tensor_info yields the
        // input DMA-BUF fd + offset directly.
        let (input_fd, input_offset) = match dmabuf.tensor_info(0) {
            Ok(info) => (info.fd, info.offset),
            Err(_) => {
                // Legacy VX delegate fallback: request a delegate-owned buffer
                // and bind it to the input tensor.
                if self.camera_format == Some("rgba")
                    && let Some(adaptor) = self
                        .interpreter
                        .delegate(0)
                        .and_then(Delegate::camera_adaptor)
                {
                    #[allow(deprecated)]
                    if let Err(e) = adaptor.set_format(0, "rgba") {
                        warn!("legacy CameraAdaptor set_format failed: {e:?}");
                    }
                }
                let bpp = if self.camera_format == Some("rgba") {
                    4
                } else {
                    3
                };
                let buf_size = width * height * bpp;
                #[allow(deprecated)]
                let request =
                    dmabuf.request(0, edgefirst_tflite::dmabuf::Ownership::Delegate, buf_size);
                let (handle, desc) = match request {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(ModelError::new(
                            ModelErrorKind::TFLite,
                            format!("legacy DMA-BUF request failed: {e:?}"),
                        )));
                    }
                };
                #[allow(deprecated)]
                if let Err(e) = dmabuf.bind_to_tensor(handle, 0) {
                    return Some(Err(ModelError::new(
                        ModelErrorKind::TFLite,
                        format!("legacy DMA-BUF bind_to_tensor failed: {e:?}"),
                    )));
                }
                #[allow(deprecated)]
                if let Err(e) = dmabuf.invalidate_graph() {
                    warn!("legacy DMA-BUF invalidate_graph failed: {e:?}");
                }
                #[allow(deprecated)]
                let legacy_fd = desc.fd;
                (legacy_fd, 0usize)
            }
        };

        // Validate the fd before the unsafe borrow — `BorrowedFd::borrow_raw`
        // is UB on a negative/closed descriptor.
        if input_fd < 0 {
            return Some(Err(ModelError::new(
                ModelErrorKind::TFLite,
                format!("delegate returned invalid input DMA-BUF fd {input_fd}"),
            )));
        }

        // Wrap the borrowed fd in a PlaneDescriptor (which dups it) and import
        // it as a HAL tensor the ImageProcessor can write into.
        // SAFETY: `input_fd` is a live fd owned by the delegate (validated
        // non-negative above); PlaneDescriptor duplicates it and the delegate
        // retains ownership of the original.
        let plane = match PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(input_fd) }) {
            Ok(p) => p.with_offset(input_offset),
            Err(e) => {
                return Some(Err(ModelError::new(
                    ModelErrorKind::Tensor,
                    format!("PlaneDescriptor from delegate fd failed: {e}"),
                )));
            }
        };

        let result = processor
            .import_image(plane, None, width, height, fmt, dtype)
            .map_err(|e| {
                ModelError::new(
                    ModelErrorKind::Image,
                    format!("import_image {width}x{height} {fmt:?}/{dtype:?} failed: {e}"),
                )
            });
        Some(result)
    }

    fn sync_input_for_device(&mut self) -> Result<(), ModelError> {
        if !self.has_dmabuf {
            return Ok(());
        }
        let dmabuf = self
            .interpreter
            .delegate(0)
            .and_then(Delegate::dmabuf)
            .ok_or_else(|| {
                ModelError::new(
                    ModelErrorKind::TFLite,
                    "DmaBuf API unavailable for sync".into(),
                )
            })?;
        dmabuf
            .sync_for_device(0)
            .map_err(|e| ModelError::new(ModelErrorKind::TFLite, format!("sync_for_device: {e:?}")))
    }

    fn sync_outputs_for_cpu(&mut self) -> Result<(), ModelError> {
        // On the DMA-BUF path the delegate-owned output buffer must be
        // flushed to the CPU before the runtime reads it; otherwise stale
        // cache lines can corrupt the decoded outputs. The sync API takes
        // a buffer index — vendor delegates pack all tensors into buffer 0,
        // so sync 0 and never iterate per output.
        if self.has_dmabuf {
            let dmabuf = self
                .interpreter
                .delegate(0)
                .and_then(Delegate::dmabuf)
                .ok_or_else(|| {
                    ModelError::new(
                        ModelErrorKind::TFLite,
                        "DmaBuf API unavailable for sync".into(),
                    )
                })?;
            dmabuf.sync_for_cpu(0).map_err(|e| {
                ModelError::new(ModelErrorKind::TFLite, format!("sync_for_cpu: {e:?}"))
            })?;
        }

        // Stage the interpreter's typed output bytes into the owned
        // Mem-backed TensorDyns so the decoder reads coherent data on
        // both the DMA-BUF and CPU paths. The interpreter's
        // `as_slice::<u8>()` is already a borrow into the delegate's
        // output buffer (zero-copy from the HAL boundary's perspective);
        // the explicit copy here is what eliminating would require a
        // HAL/trait change to let the decoder consume those slices
        // directly (tracked separately).
        let outputs = self.interpreter.outputs()?;
        for (i, t) in outputs.iter().enumerate() {
            let bytes: &[u8] = t.as_slice::<u8>()?;
            fill_output_bytes(&mut self.output_tensors[i], bytes)?;
        }
        Ok(())
    }
}
