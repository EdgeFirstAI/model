# Changelog

All notable changes to EdgeFirst Model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Set the Zenoh session namespace to the system hostname and drop the `rt/`
  prefix from default key expressions. Wire keys are `{hostname}/model/…`
  (EDGEAI-1396).
- Upgrade `edgefirst-schemas` 3.5 → 4.0.0. CameraFrame ingest now reads the
  embedded `Tensor` (`shape` `[h, w]`, `pid`, `TensorPlane.handle`) instead of
  the removed frame-level `width`/`planes`/`pid` fields.

## [2.9.0] - 2026-05-23

### Added

- **Neutron NPU support for i.MX 95**: `TfLiteRuntime` scans loaded models for
  the Neutron custom op and auto-selects between `libneutron_delegate.so` and
  `libvx_delegate.so` (i.MX 8M Plus). Verified at 8 detections @ threshold 0.1
  with zero-copy enabled on i.MX 95 hardware.
- **Aspect-preserving letterboxing**: preprocessing now letterboxes camera
  frames into the model input instead of stretching to fit. Detection boxes
  and instance masks are back-projected through the letterbox transform so
  published coordinates remain aligned to the original camera frame.
- Support for the `edgefirst.json` embedded model config used by
  Neutron-compiled models, in addition to the existing `edgefirst.yaml`.
- Full DMA-BUF zero-copy input pipeline across all three runtimes — the HAL
  `ImageProcessor` writes preprocessed pixels directly into the delegate input
  buffer, with explicit `sync_input_for_device` / `sync_outputs_for_cpu`
  cache flushes around inference.

### Changed

- Upgraded core dependencies: `edgefirst-hal` 0.9 → 0.23,
  `edgefirst-tflite` 0.1 → 0.7, `edgefirst-tracker` 0.9 → 0.23,
  and `ara2` to 0.10.
- Output decoding migrated to the dtype-dispatching `Decoder::decode` over
  `TensorDyn`. Box coordinate normalization is now handled by
  `DecoderBuilder::with_input_dims`, replacing the legacy
  `1/input_dim` scale-fold hack. Per-tensor quantization is attached to each
  integer output so the Neutron per-scale decoder consumes the right scales.
- Forced the G2D image backend for preprocessing. The HAL 0.23 threaded
  OpenGL backend uses `tokio::mpsc::blocking_send`, which panics under the
  service's tokio runtime; G2D is synchronous and avoids the issue.
- **Cross-compilation now uses `cargo zigbuild`** (with `cargo-zigbuild` and
  `zig` installed) instead of `cargo build --target ...`. The previous
  `.cargo/config.toml` cross-linker stanza has been removed; plain
  `cargo build --target aarch64-unknown-linux-gnu` will no longer link.
  See `.github/copilot-instructions.md` for the updated setup.

### Fixed

- Validate delegate-returned DMA-BUF file descriptors before
  `BorrowedFd::borrow_raw` and fail loudly on staging/input size mismatches,
  preventing silent corruption when the delegate hands back an unexpected
  buffer.
- The `CameraAdaptor` RGBA fast path is now gated on `has_dmabuf`, so it
  is not advertised when the CPU staging buffer cannot honor it.
- Repository clones no longer fail on missing Git LFS objects: the
  benchmark data files referenced LFS blobs that were never pushed to the
  remote.

### Removed

- Benchmark data files in `benches/benchmark_data/` and the associated
  `.gitattributes` LFS tracking rules.

## [2.8.0] - 2026-03-10

### Added

- **Tracker-assisted instance segmentation recovery**: when tracking is enabled,
  the decoder runs at a lower score threshold (`--track-score`, default 0.1) to
  produce more candidate detections. ByteTrack promotes low-confidence detections
  that match existing tracks, enabling recovery of temporarily occluded objects
  while preserving mask-box alignment.
- New `--track-score` / `TRACK_SCORE` parameter for decoder threshold when
  tracking is enabled.
- ARA-2 NPU runtime backend via `Runtime` trait abstraction, supporting
  `.dvm` model files over `/var/run/ara2.sock`.
- Unified model output topic (`rt/model/output`) combining boxes, masks, tracks,
  and timing in a single CDR message.
- General `--classes` / `CLASSES` label filter replacing the old `MASK_CLASSES`,
  filtering both detection boxes and instance segmentation masks.
- Tracy tracing spans for main loop pipeline stages (preprocess, invoke, decode,
  tracker_update, zenoh_publish).
- On-target integration tests with LLVM coverage instrumentation.
- Three-phase CI architecture: unit tests on GitHub runners, hardware integration
  tests on i.MX8M Plus EVK, coverage aggregation and SonarCloud reporting.

### Changed

- **Breaking**: replaced `--track-high-conf` / `TRACK_HIGH_CONF` with
  `--track-score` / `TRACK_SCORE`. The existing `--threshold` now serves as
  ByteTrack's new-track-creation threshold when tracking is enabled.
- **Breaking**: replaced `--engine` / `ENGINE` with `--delegate` / `DELEGATE`
  for TFLite delegate library path.
- **Breaking**: replaced `MASK_CLASSES` with `CLASSES` for label filtering.
- **Breaking**: legacy detection and mask topics are now disabled by default
  (empty `DETECT_TOPIC` / `MASK_TOPIC`). Set to `rt/model/boxes2d` /
  `rt/model/mask` to re-enable.
- Rewrote inference pipeline using `edgefirst-tflite` with 3-tier preprocessing
  (G2D DMA → planar deinterleave → input tensor copy).
- Replaced `SupportedModel` with `ModelContext` for decoder configuration,
  supporting auto-detection of model config from output tensor shapes.
- Folded box coordinate normalization into quantization scale for ARA-2 DVM
  models.
- Updated edgefirst-hal 0.9.0 → 0.9.1 (backend selection and mask decoding
  fixes).
- Updated edgefirst-tracker 0.9.0 → 0.9.1.
- Pinned Rust toolchain to 1.94.0 in all CI workflows with version-aware
  cache keys to prevent proc-macro ABI mismatches.

### Fixed

- Post-tracker filtering now correctly aligns boxes, masks, and track IDs using
  a shared `keep` mask before publishing. Previously, `.flatten()` on tracker
  results silently dropped untracked entries, causing index misalignment.
- G2D acceleration restored by forcing DMA tensor allocation for camera frames.
- Clippy warnings in main.rs resolved.

### Removed

- `tflite_model.rs` and `rtm_model.rs` (replaced by unified `model.rs` with
  `ModelContext` and `decode_outputs`).
- `--track-high-conf` / `TRACK_HIGH_CONF` parameter (replaced by two-threshold
  approach using `--track-score` and `--threshold`).
- `MASK_CLASSES` parameter (replaced by `CLASSES`).

## [2.7.0] - 2026-02-26

### Changed

- Replaced long environment variable names with short names (e.g. EDGEFIRST_MODEL → MODEL)
- Added complete model.default configuration file with all supported options
- Hardcoded multicast interface to loopback, removed --multicast-interface CLI flag

### Removed

- Configurable --multicast-interface CLI option (multicast now always uses loopback)

## [2.6.1] - 2026-02-15

### Fixed

- Contributor Covenant version reference corrected from v2.1 to v3.0 in CHANGELOG
- Release date corrected to 2026-02-15
- Release workflow now fails when build artifacts are missing
- Removed redundant `workflow_call` trigger from build workflow

## [2.6.0] - 2026-02-15

### Changed

- Migrated repository from Bitbucket to GitHub (EdgeFirstAI/model)
- Refactored from 4 individual git-pinned HAL crates to `edgefirst-hal 0.6.2`
  and `edgefirst-tracker 0.6.2` from crates.io
- Decoupled object tracking from decoder: separate `ByteTrack::update()` calls
  instead of `decode_outputs_tracked()`
- Added `TrackerBox` newtype wrapper to bridge `DetectBox` and `DetectionBox` traits
- Replaced local g2d-sys with upstream v1.2.0 from crates.io
- Added four-char-code 2.3.0 dependency for FourCharCode type
- Updated all documentation to reference GitHub URLs
- Renamed project to EdgeFirst Model
- Updated SonarCloud project metadata
- Updated edgefirst-schemas to 1.5.3
- Configurable multicast interface via `--multicast-interface` CLI flag

### Added

- Complete GitHub Actions CI/CD workflows (test, build, SBOM, release)
- GitHub issue templates (bug report, feature request, hardware compatibility)
- Pull request template with comprehensive checklist
- SBOM generation and license compliance automation
- Comprehensive open-source documentation (README, CONTRIBUTING, ARCHITECTURE,
  SECURITY, TESTING, NOTICE, CHANGELOG)
- SPDX license headers on all source files
- CODE_OF_CONDUCT.md (Contributor Covenant v3.0)
- Apache-2.0 LICENSE file
- Support for YoloEndToEndDet and YoloEndToEndSegDet model types

### Fixed

- Swapped error messages in tflite_model.rs and rtm_model.rs decode functions
- Index out-of-bounds panic in label lookup for annotations
- Division by zero in FPS calculation when frame time is zero
- `assert_eq` in production code replaced with proper error returns
- Wrong tensor count check using input length instead of output length
- LabelSetting::Track now shows track UUID instead of score
- Typos: "Recieved" to "Received", "publising" to "publishing",
  "seperated" to "separated"

### Removed

- Bitbucket Pipelines configuration
- Local g2d-sys crate (replaced by crates.io upstream)
- SSD/Custom model support (`ModelType::Custom` variant removed)
- `ssd_decode_boxes` function (dead code after Custom removal)
- `build_instance_segmentation_msg` dead code from buildmsgs.rs
- Local tracker, kalman, nms, and image modules (now in edgefirst-hal/tracker)
