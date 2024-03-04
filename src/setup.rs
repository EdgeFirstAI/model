use clap::Parser;
use std::path::PathBuf;

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
pub enum LabelSetting {
    Index,
    Label,
    Score,
    LabelScore,
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

    /// camera mirror
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
    #[arg(long, env, default_value = "50")]
    pub max_boxes: i32,

    /// Label offset for detections
    #[arg(long, env, default_value = "0")]
    pub label_offset: i32,

    /// optional decoder model that always runs on CPU
    #[arg(long, env)]
    pub decoder_model: Option<PathBuf>,
}
