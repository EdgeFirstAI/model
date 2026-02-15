# Changelog

All notable changes to EdgeFirst Model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
