use clap::Parser;
use std::path::PathBuf;

#[derive(clap::ValueEnum, Clone, Debug, PartialEq, Copy)]
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
    #[arg(short, long, default_value = "rt/detect/mask")]
    pub detect_topic: String,

    /// zenoh key expression for publishing model info
    #[arg(long, default_value = "rt/detect/info")]
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

    /// engine for model context
    #[arg(long, env, default_value = "npu")]
    pub engine: String,

    /// To use compression or not
    #[arg(long, env)]
    pub compression: bool,
}
