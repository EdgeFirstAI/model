use clap::Parser;
use serde_json::json;
use std::path::PathBuf;
use tracing::level_filters::LevelFilter;
use zenoh::config::{Config, WhatAmI};

#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Copy)]
pub enum LabelSetting {
    Index,
    Label,
    Score,
    LabelScore,
    Track,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// zenoh key expression for camera DMA buffers
    #[arg(long, default_value = "rt/camera/dma")]
    pub camera_topic: String,

    /// zenoh key expression for publishing detection results
    #[arg(long, default_value = "rt/model/boxes2d")]
    pub detect_topic: String,

    /// zenoh key expression for publishing model info
    #[arg(long, default_value = "rt/model/info")]
    pub info_topic: String,

    /// zenoh key expression for publishing mask results
    #[arg(long, default_value = "rt/model/mask")]
    pub mask_topic: String,

    /// zenoh key expression for publishing compressed mask results
    #[arg(long, default_value = "rt/model/mask_compressed")]
    pub mask_compressed_topic: String,

    /// model
    #[arg(short, long, env, required = true)]
    pub model: PathBuf,

    /// configure the text annotation of the detected bounding boxes
    #[arg(long, env, default_value = "label", value_enum)]
    pub labels: LabelSetting,

    /// engine for model context
    #[arg(long, env, default_value = "npu")]
    pub engine: String,

    /// score threshold for detections
    #[arg(short, long, env, default_value = "0.45")]
    pub threshold: f32,

    /// IOU for detections
    #[arg(short, long, env, default_value = "0.45")]
    pub iou: f32,

    /// max boxes for detections
    #[arg(long, env, default_value = "100")]
    pub max_boxes: i32,

    /// Label offset for detections
    #[arg(long, env, default_value = "0")]
    pub label_offset: i32,

    /// optional decoder model that always runs on CPU
    #[arg(long, env)]
    pub decoder_model: Option<PathBuf>,

    /// enable tracking objects. Must be enabled for other --track_[...] flags
    /// to work
    #[arg(long, env, action)]
    pub track: bool,

    /// number of seconds the tracked object can be missing for before being
    /// removed.
    #[arg(long, env, default_value = "0.5")]
    pub track_extra_lifespan: f32,

    /// high score threshold for ByteTrack algorithm.
    #[arg(long, env, default_value = "0.7")]
    pub track_high_conf: f32,

    /// tracking iou threshold for box association. Higher values will require
    /// boxes to have higher IOU to the predicted track to be associated.
    #[arg(long, env, default_value = "0.25")]
    pub track_iou: f32,

    /// tracking update factor. Higher update factor will also mean
    /// less smoothing but more rapid response to change (0.0 to 1.0)
    #[arg(long, env, default_value = "0.25")]
    pub track_update: f32,

    /// enable publising visualization message
    #[arg(long, env, action)]
    pub visualization: bool,

    /// zenoh key expression for publishing foxglove visualization topic
    #[arg(long, default_value = "rt/model/visualization")]
    pub visual_topic: String,

    /// resolution info topic, needed for visualization message type
    #[arg(long, default_value = "rt/camera/info")]
    pub camera_info_topic: String,

    /// Enables publishing compressed mask
    #[arg(long, env)]
    pub mask_compression: bool,

    /// Set the mask zstd compression level. Valid from -7 to 22. A value of 0
    /// will be the same as a value of 3. Lower values are faster but less
    /// compressed.
    #[arg(long, env, default_value = "1")]
    pub mask_compression_level: i32,

    /// Application log level
    #[arg(long, env, default_value = "info")]
    pub rust_log: LevelFilter,

    /// Enable Tracy profiler broadcast
    #[arg(long, env)]
    pub tracy: bool,

    /// zenoh connection mode
    #[arg(long, env, default_value = "peer")]
    mode: WhatAmI,

    /// connect to zenoh endpoints
    #[arg(long, env)]
    connect: Vec<String>,

    /// listen to zenoh endpoints
    #[arg(long, env)]
    listen: Vec<String>,

    /// disable zenoh multicast scouting
    #[arg(long, env)]
    no_multicast_scouting: bool,
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
