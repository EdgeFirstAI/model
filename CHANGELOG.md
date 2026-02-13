# Changelog

All notable changes to EdgeFirst Model will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.6.0] - 2026-02-13

### Changed

- Migrated repository from Bitbucket to GitHub (EdgeFirstAI/model)
- Replaced local g2d-sys with upstream v1.2.0 from crates.io
- Added four-char-code 2.3.0 dependency for FourCharCode type
- Updated all documentation to reference GitHub URLs
- Renamed project to EdgeFirst Model
- Updated SonarCloud project metadata

### Added

- Complete GitHub Actions CI/CD workflows (test, build, SBOM, release)
- GitHub issue templates (bug report, feature request, hardware compatibility)
- Pull request template with comprehensive checklist
- SBOM generation and license compliance automation
- Comprehensive open-source documentation (README, CONTRIBUTING, ARCHITECTURE,
  SECURITY, NOTICE, CHANGELOG)
- SPDX license headers on all source files
- CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
- Apache-2.0 LICENSE file

### Removed

- Bitbucket Pipelines configuration
- Local g2d-sys crate (replaced by crates.io upstream)
