// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

//! Inference runtime abstraction.
//!
//! Defines the [`Runtime`] trait and its implementors ([`tflite::TfLiteRuntime`]
//! and [`ara2::Ara2Runtime`]). The trait carries the full DMA-BUF preprocessing
//! pipeline surface (`import_input_image`, `sync_*`) so the main loop can run a
//! zero-copy path for the i.MX 8M Plus VX delegate and the i.MX 95 Neutron
//! delegate, while falling back to CPU staging for the Ara-2 NPU (planar input)
//! and CPU-only TFLite.

pub mod ara2;
pub mod tflite;

use edgefirst_hal::decoder::Quantization;
use edgefirst_hal::image::ImageProcessor;
use edgefirst_hal::tensor::{DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorTrait};
use std::path::Path;
use std::time::Duration;

use crate::model::{ModelError, ModelErrorKind};

/// Timing breakdown for a single inference invocation.
#[derive(Debug, Clone)]
pub struct InferenceTiming {
    pub input_time: Duration,
    pub model_time: Duration,
    pub output_time: Duration,
}

/// Decoder configuration text embedded in the model archive.
///
/// EdgeFirst models may carry their decoder config as either `edgefirst.yaml`
/// / `config.yaml` or `edgefirst.json` / `config.json`. The variant carries
/// the corresponding `DecoderBuilder` setter (`with_config_yaml_str` or
/// `with_config_json_str`).
#[derive(Debug, Clone)]
pub enum EmbeddedConfig {
    /// YAML configuration text.
    Yaml(String),
    /// JSON configuration text.
    Json(String),
}

/// Model metadata extracted from the model file.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub labels: Vec<String>,
    pub config: Option<EmbeddedConfig>,
    pub name: Option<String>,
    pub format: ModelFormat,
}

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    TfLite,
    Dvm,
}

/// Abstraction over inference runtimes (TFLite VX, TFLite Neutron, Ara-2).
///
/// Implementors expose model I/O metadata plus an optional DMA-BUF zero-copy
/// pipeline. When [`Runtime::supports_dmabuf`] is true the main loop imports
/// the runtime's input tensor via [`Runtime::import_input_image`] and lets the
/// HAL `ImageProcessor` write preprocessed pixels directly into device memory.
/// Otherwise the main loop allocates a CPU staging buffer and writes the input
/// via [`Runtime::input_tensor`].
pub trait Runtime {
    /// Run a single inference pass and return the timing breakdown.
    fn invoke(&mut self) -> Result<InferenceTiming, ModelError>;

    fn input_count(&self) -> usize;

    /// CPU staging input buffer. Only meaningful (and only written) when the
    /// runtime does not support DMA-BUF zero-copy input.
    fn input_tensor(&mut self, idx: usize) -> &mut Tensor<u8>;

    fn input_shape(&self, idx: usize) -> &[usize];
    fn input_quantization(&self, idx: usize) -> Option<Quantization>;
    fn input_dtype(&self, idx: usize) -> DType;

    /// Pixel format the runtime expects for image inputs. Drives the HAL
    /// `ImageProcessor` conversion target (`Rgb`/`Rgba` for TFLite, `PlanarRgb`
    /// for Ara-2).
    fn input_pixel_format(&self, idx: usize) -> PixelFormat;

    fn output_count(&self) -> usize;

    /// Decoded output tensor with its true element type. Fed directly to the
    /// HAL `Decoder::decode()`, which dispatches on the `TensorDyn` variant.
    fn output_tensor(&self, idx: usize) -> &TensorDyn;

    fn output_dtype(&self, idx: usize) -> DType;
    fn output_shape(&self, idx: usize) -> &[usize];
    fn output_quantization(&self, idx: usize) -> Option<Quantization>;
    fn metadata(&self) -> &ModelInfo;

    // ── DMA-BUF zero-copy pipeline ──────────────────────────────────────────

    /// Whether the runtime supports DMA-BUF zero-copy input. When true,
    /// [`Runtime::import_input_image`] yields the convert destination and
    /// [`Runtime::input_tensor`] is not used.
    fn supports_dmabuf(&self) -> bool {
        false
    }

    /// Import the runtime's input tensor as a HAL [`TensorDyn`] so the
    /// `ImageProcessor` can write preprocessed pixels directly into it.
    ///
    /// * `Some(Ok(t))` — zero-copy active, `t` aliases the engine input.
    /// * `Some(Err(e))` — backend claims DMA-BUF but the import failed; fatal.
    /// * `None` — no DMA-BUF input; the caller allocates a CPU staging buffer.
    fn import_input_image(
        &mut self,
        _processor: &ImageProcessor,
        _width: usize,
        _height: usize,
    ) -> Option<Result<TensorDyn, ModelError>> {
        None
    }

    /// Sync the input DMA-BUF for device access — called after `convert()`
    /// writes pixels and before [`Runtime::invoke`]. No-op by default.
    fn sync_input_for_device(&mut self) -> Result<(), ModelError> {
        Ok(())
    }

    /// Sync output DMA-BUFs for CPU access — called after [`Runtime::invoke`]
    /// and before reading [`Runtime::output_tensor`]. No-op by default.
    fn sync_outputs_for_cpu(&mut self) -> Result<(), ModelError> {
        Ok(())
    }
}

/// Copy raw little-endian bytes into a pre-allocated [`TensorDyn`] output
/// buffer, regardless of element type.
///
/// Used by the runtime implementations to stage inference outputs into
/// owned, correctly-typed tensors that the HAL decoder can consume.
pub(crate) fn fill_output_bytes(dst: &mut TensorDyn, src: &[u8]) -> Result<(), ModelError> {
    fn copy<T: Copy>(slice: &mut [T], src: &[u8]) -> Result<(), ModelError> {
        // SAFETY: `slice` is a contiguous, properly-aligned `[T]` buffer; we
        // reinterpret it as the equivalent `[u8]` byte view for a bulk copy.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(slice),
            )
        };
        if bytes.len() != src.len() {
            return Err(ModelError::new(
                ModelErrorKind::Tensor,
                format!(
                    "output byte size mismatch: dst {} src {}",
                    bytes.len(),
                    src.len()
                ),
            ));
        }
        bytes.copy_from_slice(src);
        Ok(())
    }

    match dst {
        TensorDyn::U8(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::I8(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::U16(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::I16(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::U32(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::I32(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::U64(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::I64(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::F16(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::F32(t) => copy(t.map()?.as_mut_slice(), src),
        TensorDyn::F64(t) => copy(t.map()?.as_mut_slice(), src),
        other => Err(ModelError::new(
            ModelErrorKind::Tensor,
            format!("unsupported output tensor dtype {:?}", other.dtype()),
        )),
    }
}

/// Extract metadata from any model file by treating it as a zip archive.
pub fn extract_metadata(data: &[u8], format: ModelFormat) -> ModelInfo {
    let mut info = ModelInfo {
        labels: Vec::new(),
        config: None,
        name: None,
        format,
    };

    let Ok(mut archive) = zip::ZipArchive::new(std::io::Cursor::new(data)) else {
        return info;
    };

    // Extract labels
    if let Ok(mut f) = archive.by_name("labels.txt")
        && f.is_file()
    {
        let mut txt = String::new();
        if std::io::Read::read_to_string(&mut f, &mut txt).is_ok() {
            info.labels = txt.lines().map(|l| l.to_string()).collect();
        }
    }

    // Re-open archive for config (ZipArchive borrows mutably). Probe both
    // YAML and JSON archive members — the Neutron-compiled per-scale models
    // ship `edgefirst.json` rather than `edgefirst.yaml`.
    let Ok(mut archive) = zip::ZipArchive::new(std::io::Cursor::new(data)) else {
        return info;
    };
    for (name, is_json) in [
        ("edgefirst.yaml", false),
        ("edgefirst.yml", false),
        ("edgefirst.json", true),
        ("config.yaml", false),
        ("config.yml", false),
        ("config.json", true),
    ] {
        if let Ok(mut f) = archive.by_name(name)
            && f.is_file()
        {
            let mut text = String::new();
            if std::io::Read::read_to_string(&mut f, &mut text).is_ok() {
                info.config = Some(if is_json {
                    EmbeddedConfig::Json(text)
                } else {
                    EmbeddedConfig::Yaml(text)
                });
                break;
            }
        }
    }

    info
}

/// Create a runtime from model file extension.
pub fn create_runtime(
    model_path: &Path,
    delegate_path: &str,
) -> Result<Box<dyn Runtime>, ModelError> {
    let ext = model_path.extension().and_then(|e| e.to_str());
    match ext {
        Some("tflite") => Ok(Box::new(tflite::TfLiteRuntime::load(
            model_path,
            delegate_path,
        )?)),
        Some("dvm") => Ok(Box::new(ara2::Ara2Runtime::load(
            model_path,
            delegate_path,
        )?)),
        _ => Err(ModelError::new(
            crate::model::ModelErrorKind::Other,
            format!("Unsupported model format: {ext:?}"),
        )),
    }
}
