use clap::Parser;
use log::warn;
use std::path::PathBuf;

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
pub enum LabelSetting {
    Index,
    Label,
    Score,
    LabelScore,
    Track,
}

#[derive(Debug, Clone, Parser)]
#[command(author, version, about, long_about = None)]
pub struct Settings {
    /// zenoh key expression for camera DMA buffers
    #[arg(short, long, default_value = "rt/camera/dma")]
    pub camera_topic: String,

    /// zenoh key expression for publishing detection results
    #[arg(short, long, default_value = "rt/detect/boxes2d")]
    pub detect_topic: String,

    /// resolution info topic
    #[arg(long, default_value = "rt/camera/info")]
    pub info_topic: String,

    /// connect to zenoh endpoints
    #[arg(long, default_value = "tcp/127.0.0.1:7447")]
    pub connect: Vec<String>,

    /// listen to zenoh endpoints
    #[arg(long)]
    pub listen: Vec<String>,

    /// zenoh connection mode
    #[arg(long, default_value = "client")]
    pub mode: String,

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
    #[arg(long, env, default_value = "2.0")]
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
}

pub fn validate_settings(s: &mut Settings) {
    if !s.track && s.labels == LabelSetting::Track {
        warn!("Tracking was not enabled, label setting will be changed from `track` to `score`");
        s.labels = LabelSetting::Score;
    }
}
