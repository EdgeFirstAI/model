// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Au-Zone Technologies. All Rights Reserved.

use clap::{CommandFactory, Parser};
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
    /// Zenoh key expression for camera frames (`CameraFrame`)
    #[arg(long, env = "CAMERA_TOPIC", default_value = "camera/frame")]
    pub camera_topic: String,

    /// Legacy: use --output-topic instead. Empty string disables publishing.
    #[arg(long, env = "DETECT_TOPIC", default_value = "")]
    pub detect_topic: String,

    /// Zenoh key expression for publishing model info
    #[arg(long, env = "INFO_TOPIC", default_value = "model/info")]
    pub info_topic: String,

    /// Legacy: use --output-topic instead. Empty string disables publishing.
    #[arg(long, env = "MASK_TOPIC", default_value = "")]
    pub mask_topic: String,

    /// Zenoh key expression for publishing unified model output
    #[arg(long, env = "OUTPUT_TOPIC", default_value = "model/output")]
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

    /// Score threshold for the decoder when tracking is enabled. Lower than
    /// --threshold to allow tracker-assisted recovery of low-confidence
    /// detections. Only used when --track is true.
    #[arg(long, env = "TRACK_SCORE", default_value = "0.1")]
    pub track_score: f32,

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
    #[arg(long, env = "VISUAL_TOPIC", default_value = "model/visualization")]
    pub visual_topic: String,

    /// Zenoh key expression for camera info (needed for visualization)
    #[arg(long, env = "CAMERA_INFO_TOPIC", default_value = "camera/info")]
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

/// Environment variables where an empty value is meaningful and must be preserved
/// (i.e. the argument has a non-empty default but "" is a documented "disable" sentinel).
///
/// The model service has no such variables: every argument that accepts an
/// empty value already declares `default_value = ""`, so scrubbing yields the
/// same result as keeping it.
pub const KEEP: &[&str] = &[];

/// Treat an empty environment variable as unset, so clap's declared
/// `default_value` applies instead of failing to parse.
///
/// Only variables bound to this program's own arguments are considered;
/// unrelated process environment is left alone. `keep` names variables
/// where an empty value is meaningful and must be preserved.
///
/// # Safety
/// Must be called before any thread is spawned — that is, before the tokio
/// runtime is built. Mutating the process environment is not thread-safe.
pub unsafe fn scrub_empty_env<C: CommandFactory>(keep: &[&str]) {
    for arg in C::command().get_arguments() {
        let Some(env) = arg.get_env() else { continue };
        let name = env.to_string_lossy().into_owned();
        if keep.contains(&name.as_str()) {
            continue;
        }
        if matches!(std::env::var(&name), Ok(v) if v.is_empty()) {
            unsafe { std::env::remove_var(&name) };
        }
    }
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

/// System hostname used as the Zenoh session namespace.
///
/// Empty or `/`-containing hostnames would create unintended sub-keys, so we
/// fall back to `"localhost"` and warn. Two devices both falling back would
/// silently share a namespace; that is a deployment defect.
fn zenoh_namespace() -> String {
    let raw = gethostname::gethostname().to_string_lossy().into_owned();
    if raw.is_empty() || raw.contains('/') {
        tracing::warn!(
            hostname = %raw,
            "system hostname is empty or contains '/' — falling back to \"localhost\""
        );
        "localhost".into()
    } else {
        raw
    }
}

impl From<Args> for Config {
    fn from(args: Args) -> Self {
        let mut config = Config::default();

        // Session namespace = hostname: application keys are bare
        // (`model/output`) and the wire form is `{hostname}/model/output`.
        config
            .insert_json5("namespace", &json!(zenoh_namespace()).to_string())
            .unwrap();

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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::sync::{Mutex, MutexGuard};

    /// Serialises tests that read or mutate the process environment. clap
    /// consults env vars on every parse, so parsing tests must hold this too.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Env-bound arguments with a non-empty default where we have consciously decided
    /// that an empty value is NOT meaningful (so scrubbing to the default is correct).
    const SCRUB_REVIEWED: &[&str] = &[
        "CAMERA_TOPIC",
        "INFO_TOPIC",
        "OUTPUT_TOPIC",
        "LABELS",
        "THRESHOLD",
        "IOU",
        "MAX_BOXES",
        "LABEL_OFFSET",
        "TRACK",
        "TRACK_EXTRA_LIFESPAN",
        "TRACK_SCORE",
        "TRACK_IOU",
        "TRACK_UPDATE",
        "VISUALIZATION",
        "VISUAL_TOPIC",
        "CAMERA_INFO_TOPIC",
        "SSD_MODEL",
        "TRACY",
        "MODE",
        "NO_MULTICAST_SCOUTING",
    ];

    #[test]
    fn every_env_arg_is_either_scrubbable_or_explicitly_kept() {
        for arg in Args::command().get_arguments() {
            let Some(env) = arg.get_env() else { continue };
            let name = env.to_string_lossy().into_owned();
            let has_nonempty_default = arg
                .get_default_values()
                .first()
                .is_some_and(|d| !d.is_empty());
            if has_nonempty_default && !KEEP.contains(&name.as_str()) {
                assert!(
                    SCRUB_REVIEWED.contains(&name.as_str()),
                    "{name} has a non-empty default; decide whether empty is meaningful \
                     and add it to KEEP or SCRUB_REVIEWED"
                );
            }
        }
    }

    /// Restores the named environment variables to their prior values on drop,
    /// so a failing assertion cannot leak state into other tests.
    struct EnvRestore(Vec<(&'static str, Option<std::ffi::OsString>)>);

    impl EnvRestore {
        fn capture(names: &[&'static str]) -> Self {
            Self(names.iter().map(|n| (*n, std::env::var_os(n))).collect())
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            for (name, value) in self.0.drain(..) {
                // SAFETY: caller holds ENV_LOCK; no other thread mutates env.
                unsafe {
                    match value {
                        Some(v) => std::env::set_var(name, v),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    #[test]
    fn empty_env_vars_are_treated_as_unset() {
        const VARS: &[&str] = &[
            "MODEL",
            "EDGEFIRST_CONFIG",
            "DELEGATE",
            "THRESHOLD",
            "TRACK",
        ];
        let _guard = env_lock();
        let _restore = EnvRestore::capture(VARS);

        // MODEL="" must still fail: scrubbing makes it absent and --model is
        // required, so clap reports the missing argument rather than parsing "".
        // SAFETY: ENV_LOCK is held and no runtime threads exist in this test.
        unsafe {
            for name in VARS {
                std::env::set_var(name, "");
            }
            scrub_empty_env::<Args>(KEEP);
        }
        for name in VARS {
            assert!(
                std::env::var_os(name).is_none(),
                "{name} should be scrubbed"
            );
        }
        let err = Args::try_parse_from(["prog"]).expect_err("MODEL=\"\" must not parse");
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);

        // With a real model path the remaining "" vars fall back to defaults.
        // SAFETY: as above.
        unsafe {
            std::env::set_var("MODEL", "/dev/null");
            std::env::set_var("EDGEFIRST_CONFIG", "");
            std::env::set_var("DELEGATE", "");
            std::env::set_var("THRESHOLD", "");
            std::env::set_var("TRACK", "");
            scrub_empty_env::<Args>(KEEP);
        }
        assert_eq!(std::env::var("MODEL").as_deref(), Ok("/dev/null"));
        let args = Args::try_parse_from(["prog"]).expect("defaults should apply");
        assert_eq!(args.model, PathBuf::from("/dev/null"));
        assert_eq!(args.edgefirst_config(), None);
        assert_eq!(args.delegate, "");
        assert_eq!(args.threshold, 0.45);
        assert!(!args.track);
    }

    fn parse_defaults() -> Args {
        let _guard = env_lock();
        Args::parse_from([
            "edgefirst-model",
            "--model",
            "/dev/null",
            "--output-topic",
            "model/output",
            "--info-topic",
            "model/info",
            "--visual-topic",
            "model/visualization",
            "--camera-info-topic",
            "camera/info",
            "--camera-topic",
            "camera/frame",
        ])
    }

    #[test]
    fn zenoh_config_sets_namespace() {
        let ns = zenoh_namespace();
        assert!(!ns.is_empty(), "namespace should be non-empty");
        assert!(!ns.contains('/'), "namespace must not contain '/'");
        let rendered = Config::from(parse_defaults()).to_string();
        assert!(
            rendered.contains(&ns),
            "config should include namespace {ns}: {rendered}"
        );
    }

    #[test]
    fn cli_topics_have_no_rt_prefix() {
        let args = parse_defaults();
        assert_eq!(args.output_topic, "model/output");
        assert_eq!(args.info_topic, "model/info");
        assert_eq!(args.visual_topic, "model/visualization");
        assert_eq!(args.camera_info_topic, "camera/info");
        assert_eq!(args.camera_topic, "camera/frame");
    }
}
