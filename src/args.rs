// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use clap::Parser;
use serde_json::json;
use std::path::PathBuf;
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
/// # Example
///
/// ```bash
/// # Via command line
/// edgefirst-model --model /path/to/model.tflite --engine npu
///
/// # Via environment variables
/// export MODEL=/path/to/model.tflite
/// export ENGINE=npu
/// export THRESHOLD=0.5
/// edgefirst-model
/// ```
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Zenoh key expression for camera DMA buffers
    #[arg(long, default_value = "rt/camera/dma")]
    pub camera_topic: String,

    /// Zenoh key expression for publishing detection results
    #[arg(long, default_value = "rt/model/boxes2d")]
    pub detect_topic: String,

    /// Zenoh key expression for publishing model info
    #[arg(long, default_value = "rt/model/info")]
    pub info_topic: String,

    /// Zenoh key expression for publishing mask results
    #[arg(long, default_value = "rt/model/mask")]
    pub mask_topic: String,

    /// Zenoh key expression for publishing compressed mask results
    #[arg(long, default_value = "rt/model/mask_compressed")]
    pub mask_compressed_topic: String,

    /// Path to the inference model file (e.g., .tflite)
    #[arg(short, long, env = "MODEL", required = true)]
    pub model: PathBuf,

    /// EdgeFirst config file to override config in model, or supply one
    /// when the model does not include a config. Can be YAML or JSON.
    #[arg(long, env = "EDGEFIRST_CONFIG")]
    pub edgefirst_config: Option<PathBuf>,

    /// Text annotation style for detected bounding boxes
    #[arg(long, env = "LABELS", default_value = "label", value_enum)]
    pub labels: LabelSetting,

    /// Inference engine / delegate for model execution
    #[arg(long, env = "ENGINE", default_value = "npu")]
    pub engine: String,

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

    /// Optional decoder model that always runs on CPU
    #[arg(long, env = "DECODER_MODEL")]
    pub decoder_model: Option<PathBuf>,

    /// Enable multi-object tracking (required for other --track-* flags)
    #[arg(long, env = "TRACK", action)]
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
    #[arg(long, env = "VISUALIZATION", action)]
    pub visualization: bool,

    /// Zenoh key expression for publishing Foxglove visualization topic
    #[arg(long, default_value = "rt/model/visualization")]
    pub visual_topic: String,

    /// Zenoh key expression for camera info (needed for visualization)
    #[arg(long, default_value = "rt/camera/info")]
    pub camera_info_topic: String,

    /// Enable publishing zstd-compressed segmentation masks
    #[arg(long, env = "MASK_COMPRESSION")]
    pub mask_compression: bool,

    /// Mask zstd compression level (-7 to 22; 0 behaves as 3)
    #[arg(long, env = "MASK_COMPRESSION_LEVEL", default_value = "1")]
    pub mask_compression_level: i32,

    /// Class indices to include in mask output (space-separated; empty = all)
    #[arg(long, env = "MASK_CLASSES", hide_short_help = true, value_parser=parse_classes, default_value="")]
    pub mask_classes: std::vec::Vec<usize>, /* we use std::vec::Vec to bypass clap automatic
                                             * processing on Vec. This allows us to parse "" as
                                             * Vec::new(). */

    /// Enable SSD model mode when a different model config is not found
    #[arg(long, env = "SSD_MODEL", hide_short_help = true)]
    pub ssd_model: bool,

    /// Enable Tracy profiler broadcast
    #[arg(long, env = "TRACY")]
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
    #[arg(long, env = "NO_MULTICAST_SCOUTING")]
    no_multicast_scouting: bool,
}

fn parse_classes(arg: &str) -> Result<Vec<usize>, std::num::ParseIntError> {
    if arg.is_empty() {
        return Ok(Vec::new());
    }
    let args = arg.split(" ");
    let mut ret = Vec::new();
    for a in args {
        ret.push(a.parse()?);
    }
    Ok(ret)
}

impl From<Args> for Config {
    fn from(args: Args) -> Self {
        let mut config = Config::default();

        config
            .insert_json5("mode", &json!(args.mode).to_string())
            .unwrap();

        if !args.connect.is_empty() {
            config
                .insert_json5("connect/endpoints", &json!(args.connect).to_string())
                .unwrap();
        }

        if !args.listen.is_empty() {
            config
                .insert_json5("listen/endpoints", &json!(args.listen).to_string())
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
