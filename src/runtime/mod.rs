// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

pub mod ara2;
pub mod tflite;

use edgefirst_hal::decoder::Quantization;
use edgefirst_hal::decoder::configs::DataType;
use edgefirst_hal::tensor::Tensor;
use four_char_code::FourCharCode;
use std::path::Path;
use std::time::Duration;

use crate::model::ModelError;

/// Timing breakdown for a single inference invocation.
#[derive(Debug, Clone)]
pub struct InferenceTiming {
    pub input_time: Duration,
    pub model_time: Duration,
    pub output_time: Duration,
}

/// Model metadata extracted from the model file.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub labels: Vec<String>,
    pub config_yaml: Option<String>,
    pub name: Option<String>,
    pub format: ModelFormat,
}

/// Supported model file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    TfLite,
    Dvm,
}

/// Abstraction over inference runtimes (TFLite, Ara2, etc.).
pub trait Runtime {
    fn invoke(&mut self) -> Result<InferenceTiming, ModelError>;
    fn input_count(&self) -> usize;
    fn input_tensor(&mut self, idx: usize) -> &mut Tensor<u8>;
    fn input_shape(&self, idx: usize) -> &[usize];
    fn input_quantization(&self, idx: usize) -> Option<Quantization>;
    fn input_dtype(&self, idx: usize) -> DataType;
    fn input_fourcc(&self, idx: usize) -> FourCharCode;
    fn output_count(&self) -> usize;
    fn output_tensor(&self, idx: usize) -> &Tensor<u8>;
    fn output_dtype(&self, idx: usize) -> DataType;
    fn output_shape(&self, idx: usize) -> &[usize];
    fn output_quantization(&self, idx: usize) -> Option<Quantization>;
    fn metadata(&self) -> &ModelInfo;
}

/// Extract metadata from any model file by treating it as a zip archive.
pub fn extract_metadata(data: &[u8], format: ModelFormat) -> ModelInfo {
    let mut info = ModelInfo {
        labels: Vec::new(),
        config_yaml: None,
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

    // Re-open archive for config (ZipArchive borrows mutably)
    let Ok(mut archive) = zip::ZipArchive::new(std::io::Cursor::new(data)) else {
        return info;
    };
    for name in [
        "edgefirst.yaml",
        "edgefirst.yml",
        "config.yaml",
        "config.yml",
    ] {
        if let Ok(mut f) = archive.by_name(name)
            && f.is_file()
        {
            let mut yaml = String::new();
            if std::io::Read::read_to_string(&mut f, &mut yaml).is_ok() {
                info.config_yaml = Some(yaml);
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
