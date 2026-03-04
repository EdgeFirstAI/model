// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use clap::Parser;
use serde_json::json;
use std::path::{Path, PathBuf};
use zenoh::config::{Config, WhatAmI};

/// Bounding-box label annotation options.
///
/// Controls what text is drawn next to each detected bounding box.
#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Copy)]
pub enum LabelSetting {
    /// Show class index only
    Index,
    /// Show class label name
    Label,
    /// Show confidence score only
    Score,
    /// Show label and score
    LabelScore,
    /// Show tracking ID
    Track,
}

/// Command-line arguments for EdgeFirst Model Node.
///
/// This structure defines all configuration options for the model inference
/// node, including model selection, detection parameters, tracking, mask
/// processing, Zenoh configuration, and debugging options. Arguments can be
/// specified via command line or environment variables.
///
/// Empty-string environment variables (e.g. `EDGEFIRST_CONFIG=""`) are treated
/// as unset so that systemd EnvironmentFile defaults work without commenting
/// out optional parameters.
///
/// # Example
///
/// ```bash
/// # Via command line
/// edgefirst-model --model /path/to/model.tflite --delegate /usr/lib/libvx_delegate.so
///
/// # Via environment variables
/// export MODEL=/path/to/model.tflite
/// export DELEGATE=/usr/lib/libvx_delegate.so
/// export THRESHOLD=0.5
/// edgefirst-model
/// ```
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Zenoh key expression for camera DMA buffers
    #[arg(long, env = "CAMERA_TOPIC", default_value = "rt/camera/dma")]
    pub camera_topic: String,

    /// Legacy: use --output-topic instead. Empty string disables publishing.
    #[arg(long, env = "DETECT_TOPIC", default_value = "")]
    pub detect_topic: String,

    /// Zenoh key expression for publishing model info
    #[arg(long, env = "INFO_TOPIC", default_value = "rt/model/info")]
    pub info_topic: String,

    /// Legacy: use --output-topic instead. Empty string disables publishing.
    #[arg(long, env = "MASK_TOPIC", default_value = "")]
    pub mask_topic: String,

    /// Zenoh key expression for publishing unified model output
    #[arg(long, env = "OUTPUT_TOPIC", default_value = "rt/model/output")]
    pub output_topic: String,

    /// Path to the inference model file (e.g., .tflite)
    #[arg(short, long, env = "MODEL", required = true)]
    pub model: PathBuf,

    /// EdgeFirst config file to override config in model, or supply one
    /// when the model does not include a config. Can be YAML or JSON.
    /// An empty string is treated as unset.
    #[arg(long, env = "EDGEFIRST_CONFIG", default_value = "", value_parser = parse_optional_path)]
    edgefirst_config: PathBuf,

    /// Text annotation style for detected bounding boxes
    #[arg(long, env = "LABELS", default_value = "label", value_enum)]
    pub labels: LabelSetting,

    /// Path to TFLite delegate shared library (empty = CPU only)
    #[arg(long, env = "DELEGATE", default_value = "")]
    pub delegate: String,

    /// Score threshold for detections
    #[arg(short, long, env = "THRESHOLD", default_value = "0.45")]
    pub threshold: f32,

    /// IOU threshold for non-maximum suppression
    #[arg(short, long, env = "IOU", default_value = "0.45")]
    pub iou: f32,

    /// Maximum number of detection boxes to output
    #[arg(long, env = "MAX_BOXES", default_value = "100")]
    pub max_boxes: usize,

    /// Label index offset for detections
    #[arg(long, env = "LABEL_OFFSET", default_value = "0")]
    pub label_offset: i32,

    /// Enable multi-object tracking (required for other --track-* flags)
    #[arg(
        long,
        env = "TRACK",
        default_value = "false",
        default_missing_value = "true",
        num_args(0..=1),
        value_parser = parse_bool
    )]
    pub track: bool,

    /// Seconds a tracked object can be missing before removal
    #[arg(long, env = "TRACK_EXTRA_LIFESPAN", default_value = "0.5")]
    pub track_extra_lifespan: f32,

    /// High score threshold for ByteTrack algorithm
    #[arg(long, env = "TRACK_HIGH_CONF", default_value = "0.7")]
    pub track_high_conf: f32,

    /// Tracking IOU threshold for box association (higher = stricter)
    #[arg(long, env = "TRACK_IOU", default_value = "0.25")]
    pub track_iou: f32,

    /// Tracking update factor — higher means less smoothing (0.0 to 1.0)
    #[arg(long, env = "TRACK_UPDATE", default_value = "0.25")]
    pub track_update: f32,

    /// Enable publishing visualization message
    #[arg(
        long,
        env = "VISUALIZATION",
        default_value = "false",
        default_missing_value = "true",
        num_args(0..=1),
        value_parser = parse_bool
    )]
    pub visualization: bool,

    /// Zenoh key expression for publishing Foxglove visualization topic
    #[arg(long, env = "VISUAL_TOPIC", default_value = "rt/model/visualization")]
    pub visual_topic: String,

    /// Zenoh key expression for camera info (needed for visualization)
    #[arg(long, env = "CAMERA_INFO_TOPIC", default_value = "rt/camera/info")]
    pub camera_info_topic: String,

    /// Filter output to only include these class labels (space-separated; empty = all)
    #[arg(long, env = "CLASSES", hide_short_help = true, value_parser = parse_class_names, default_value = "")]
    pub classes: std::vec::Vec<String>, /* we use std::vec::Vec to bypass clap automatic
                                         * processing on Vec. This allows us to parse "" as
                                         * Vec::new(). */

    /// Enable SSD model mode when a different model config is not found
    #[arg(long, env = "SSD_MODEL", hide_short_help = true, default_value = "false", value_parser = parse_bool)]
    pub ssd_model: bool,

    /// Enable Tracy profiler broadcast
    #[arg(long, env = "TRACY", default_value = "false", value_parser = parse_bool)]
    pub tracy: bool,

    /// Zenoh participant mode (peer, client, or router)
    #[arg(long, env = "MODE", default_value = "peer")]
    mode: WhatAmI,

    /// Zenoh endpoints to connect to (can specify multiple)
    #[arg(long, env = "CONNECT")]
    connect: Vec<String>,

    /// Zenoh endpoints to listen on (can specify multiple)
    #[arg(long, env = "LISTEN")]
    listen: Vec<String>,

    /// Disable Zenoh multicast peer discovery
    #[arg(long, env = "NO_MULTICAST_SCOUTING", default_value = "false", value_parser = parse_bool)]
    no_multicast_scouting: bool,
}

impl Args {
    /// Returns the EdgeFirst config path, or `None` if empty / unset.
    pub fn edgefirst_config(&self) -> Option<&Path> {
        if self.edgefirst_config.as_os_str().is_empty() {
            None
        } else {
            Some(&self.edgefirst_config)
        }
    }
}

/// Parse a boolean from a string value. Accepts "true"/"false" (case-insensitive)
/// and "1"/"0". An empty string is treated as false.
fn parse_bool(arg: &str) -> Result<bool, String> {
    match arg.to_ascii_lowercase().as_str() {
        "" | "false" | "0" | "no" => Ok(false),
        "true" | "1" | "yes" => Ok(true),
        other => Err(format!("invalid boolean value '{other}'")),
    }
}

/// Parse a path that may be empty. An empty string produces an empty PathBuf
/// which `Args::edgefirst_config()` maps to `None`.
fn parse_optional_path(arg: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(arg))
}

fn parse_class_names(arg: &str) -> Result<Vec<String>, String> {
    if arg.is_empty() {
        return Ok(Vec::new());
    }
    Ok(arg.split_whitespace().map(String::from).collect())
}

impl From<Args> for Config {
    fn from(args: Args) -> Self {
        let mut config = Config::default();

        config
            .insert_json5("mode", &json!(args.mode).to_string())
            .unwrap();

        let connect: Vec<_> = args.connect.into_iter().filter(|s| !s.is_empty()).collect();
        if !connect.is_empty() {
            config
                .insert_json5("connect/endpoints", &json!(connect).to_string())
                .unwrap();
        }

        let listen: Vec<_> = args.listen.into_iter().filter(|s| !s.is_empty()).collect();
        if !listen.is_empty() {
            config
                .insert_json5("listen/endpoints", &json!(listen).to_string())
                .unwrap();
        }

        if args.no_multicast_scouting {
            config
                .insert_json5("scouting/multicast/enabled", &json!(false).to_string())
                .unwrap();
        }

        config
            .insert_json5("scouting/multicast/interface", &json!("lo").to_string())
            .unwrap();

        config
    }
}
