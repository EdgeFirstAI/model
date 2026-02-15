# AI Assistant Development Guidelines

This document provides instructions for AI coding assistants (GitHub Copilot, Cursor, Claude Code, etc.) working on Au-Zone Technologies projects. These guidelines ensure consistent code quality, proper workflow adherence, and maintainable contributions.

**Version:** 1.1
**Last Updated:** February 2026
**Applies To:** All Au-Zone Technologies software repositories

---

## Table of Contents

1. [Overview](#overview)
2. [Git Workflow](#git-workflow)
3. [Code Quality Standards](#code-quality-standards)
4. [Testing Requirements](#testing-requirements)
5. [Documentation Expectations](#documentation-expectations)
6. [License Policy](#license-policy)
7. [Security Practices](#security-practices)
8. [Project-Specific Guidelines](#project-specific-guidelines)

---

## Overview

Au-Zone Technologies develops edge AI and computer vision solutions for resource-constrained embedded devices. Our software spans:
- Edge AI inference engines and model optimization tools
- Computer vision processing pipelines
- Embedded Linux device drivers and system software
- MLOps platform (EdgeFirst Studio) for model deployment and management
- Open source libraries and tools (Apache-2.0 licensed)

When contributing to Au-Zone projects, AI assistants should prioritize:
- **Resource efficiency**: Memory, CPU, and power consumption matter on embedded devices
- **Code quality**: Maintainability, readability, and adherence to established patterns
- **Testing**: Comprehensive coverage with unit, integration, and edge case tests
- **Documentation**: Clear explanations for complex logic and public APIs
- **License compliance**: Strict adherence to approved open source licenses

---

## Git Workflow

### Branch Naming Convention

**REQUIRED FORMAT**: `<type>/<PROJECTKEY-###>[-optional-description]`

**Branch Types:**
- `feature/` - New features and enhancements
- `bugfix/` - Non-critical bug fixes
- `hotfix/` - Critical production issues requiring immediate fix

**Examples:**
```bash
feature/EDGEAI-123-add-authentication
bugfix/STUDIO-456-fix-memory-leak
hotfix/MAIVIN-789-security-patch

# Minimal format (JIRA key only)
feature/EDGEAI-123
bugfix/STUDIO-456
```

**Rules:**
- JIRA key is REQUIRED (format: `PROJECTKEY-###`)
- Description is OPTIONAL but recommended for clarity
- Use kebab-case for descriptions (lowercase with hyphens)
- Branch from `develop` for features/bugfixes, from `main` for hotfixes

### Commit Message Format

**REQUIRED FORMAT**: `PROJECTKEY-###: Brief description of what was done`

**Rules:**
- Subject line: 50-72 characters ideal
- Focus on WHAT changed, not HOW (implementation details belong in code)
- No type prefixes (`feat:`, `fix:`, etc.) - JIRA provides context
- Optional body: Use bullet points for additional detail

**Examples of Good Commits:**
```bash
EDGEAI-123: Add JWT authentication to user API

STUDIO-456: Fix memory leak in CUDA kernel allocation

MAIVIN-789: Optimize tensor operations for inference
- Implemented tiled memory access pattern
- Reduced memory bandwidth by 40%
- Added benchmarks to verify improvements
```

**Examples of Bad Commits:**
```bash
fix bug                           # Missing JIRA key, too vague
feat(auth): add OAuth2           # Has type prefix (not our convention)
EDGEAI-123                       # Missing description
edgeai-123: update code          # Lowercase key, vague description
```

### Pull Request Process

**Requirements:**
- **2 approvals required** for merging to `main`
- **1 approval required** for merging to `develop`
- All CI/CD checks must pass
- PR title: `PROJECTKEY-### Brief description of changes`
- PR description must link to JIRA ticket

**PR Description Template:**
```markdown
## JIRA Ticket
Link: [PROJECTKEY-###](https://au-zone.atlassian.net/browse/PROJECTKEY-###)

## Changes
Brief summary of what changed and why

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project conventions
- [ ] Documentation updated
- [ ] No secrets or credentials committed
- [ ] License policy compliance verified
```

**Process:**
1. Create PR via GitHub web interface
2. Link to JIRA ticket in description
3. Wait for CI/CD to complete successfully
4. Address reviewer feedback through additional commits
5. Obtain required approvals
6. Merge using squash or rebase to keep history clean

### JIRA Integration

While full JIRA details are internal, contributors should know:
- **Branch naming triggers automation**: Creating a branch with format `<type>/PROJECTKEY-###` automatically updates the linked JIRA ticket
- **PR creation triggers status updates**: Opening a PR moves tickets to review status
- **Merge triggers closure**: Merging a PR to main/develop closes the associated ticket
- **Commit messages link to JIRA**: Format `PROJECTKEY-###: Description` creates automatic linkage

**Note**: External contributors without JIRA access can use branch naming like `feature/issue-123-description` referencing GitHub issue numbers instead.

---

## Code Quality Standards

### General Principles

- **Consistency**: Follow existing codebase patterns and conventions
- **Readability**: Code is read more often than written - optimize for comprehension
- **Simplicity**: Prefer simple, straightforward solutions over clever ones
- **Error Handling**: Validate inputs, sanitize outputs, provide actionable error messages
- **Performance**: Consider time/space complexity, especially for edge deployment

### Language-Specific Standards

Follow established conventions for each language:
- **Rust**: Use `cargo fmt` and `cargo clippy`; follow Rust API guidelines
- **Python**: Follow PEP 8; use autopep8 formatter (or project-specified tool); type hints preferred
- **C/C++**: Follow project's .clang-format; use RAII patterns
- **Go**: Use `go fmt`; follow Effective Go guidelines
- **JavaScript/TypeScript**: Use ESLint; Prettier formatter; prefer TypeScript

### Code Quality Tools

**SonarQube Integration:**
- Projects with `sonar-project.properties` must follow SonarQube guidelines
- Verify code quality using:
  - MCP integration for automated checks
  - VSCode SonarLint plugin for real-time feedback
  - SonarCloud reports in CI/CD pipeline
- Address critical and high-severity issues before submitting PR
- Maintain or improve project quality gate scores

### Code Review Checklist

Before submitting code, verify:
- [ ] Code follows project style guidelines (check `.editorconfig`, `CONTRIBUTING.md`)
- [ ] No commented-out code or debug statements
- [ ] Error handling is comprehensive and provides useful messages
- [ ] Complex logic has explanatory comments
- [ ] Public APIs have documentation
- [ ] No hardcoded values that should be configuration
- [ ] Resource cleanup (memory, file handles, connections) is proper
- [ ] No obvious security vulnerabilities (SQL injection, XSS, etc.)
- [ ] SonarQube quality checks pass (if applicable)

### Performance Considerations

For edge AI applications, always consider:
- **Memory footprint**: Minimize allocations; reuse buffers where possible
- **CPU efficiency**: Profile critical paths; optimize hot loops
- **Power consumption**: Reduce wake-ups; batch operations
- **Latency**: Consider real-time requirements for vision processing
- **Hardware acceleration**: Leverage NPU/GPU/DSP when available

---

## Testing Requirements

### Coverage Standards

- **Minimum coverage**: 70% (project-specific thresholds may vary)
- **Critical paths**: 90%+ coverage for core functionality
- **Edge cases**: Explicit tests for boundary conditions
- **Error paths**: Validate error handling and recovery

### Test Types

**Unit Tests:**
- Test individual functions/methods in isolation
- Mock external dependencies
- Fast execution (< 1 second per test suite)
- Use property-based testing where applicable

**Integration Tests:**
- Test component interactions
- Use real dependencies when feasible
- Validate API contracts and data flows
- Test configuration and initialization

**Edge Case Tests:**
- Null/empty inputs
- Boundary values (min, max, overflow)
- Concurrent access and race conditions
- Resource exhaustion scenarios
- Platform-specific behaviors

### Test Organization

**Test layout follows language/framework conventions. Each project should define specific practices.**

**Rust (common pattern):**
```rust
// Unit tests at end of implementation file
// src/module/component.rs
pub fn process_data(input: &[u8]) -> Result<Vec<u8>, Error> {
    // implementation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_data_valid_input() {
        // test implementation
    }
}
```

```
# Integration tests in separate directory
tests/
├── integration_test.rs
└── common/
    └── mod.rs
```

**Python (depends on pytest vs unittest):**
```
# Common patterns - follow project conventions
project/
├── src/
│   └── mypackage/
│       └── module.py
└── tests/
    ├── unit/
    │   └── test_module.py
    └── integration/
        └── test_api_workflow.py
```

**General guidance:**
- Follow common patterns for your language and testing framework
- Consult project's `CONTRIBUTING.md` for specific conventions
- Keep test organization consistent within the project
- Co-locate unit tests or separate - project decides

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Language-specific examples
cargo test --workspace              # Rust
pytest tests/                       # Python with pytest
python -m unittest discover tests/  # Python with unittest
go test ./...                       # Go
```

---

## Documentation Expectations

### Code Documentation

**When to document:**
- Public APIs, functions, and classes (ALWAYS)
- Complex algorithms or non-obvious logic
- Performance considerations or optimization rationale
- Edge cases and error conditions
- Thread safety and concurrency requirements
- Hardware-specific code or platform dependencies

**Documentation style:**
```python
def preprocess_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize and normalize image for model inference.

    Args:
        image: Input image as HWC numpy array (uint8)
        target_size: Target dimensions as (width, height)

    Returns:
        Preprocessed image as CHW float32 array normalized to [0, 1]

    Raises:
        ValueError: If image dimensions are invalid or target_size is negative

    Performance:
        Uses bilinear interpolation. For better quality with 2x cost,
        use bicubic interpolation via config.interpolation = 'bicubic'
    """
```

### Project Documentation

**Essential files for public repositories:**
- `README.md` - Project overview, quick start, documentation links
- `CONTRIBUTING.md` - Development setup, contribution process, coding standards
- `CODE_OF_CONDUCT.md` - Community standards (Contributor Covenant)
- `SECURITY.md` - Vulnerability reporting process
- `LICENSE` - Complete license text (Apache-2.0 for open source)

**Additional documentation:**
- User guides for features and workflows
- API reference documentation
- Migration guides for breaking changes

### Documentation Updates

When modifying code, update corresponding documentation:
- README if user-facing behavior changes
- API docs if function signatures or semantics change
- CHANGELOG for all user-visible changes
- Configuration guides if new options added

---

## License Policy

**CRITICAL**: Au-Zone has strict license policy for all dependencies.

### Allowed Licenses

✅ **Permissive licenses (APPROVED)**:
- MIT
- Apache-2.0
- BSD-2-Clause, BSD-3-Clause
- ISC
- 0BSD
- Unlicense

### Review Required

⚠️ **Weak copyleft (REQUIRES LEGAL REVIEW)**:
- MPL-2.0 (Mozilla Public License)
- LGPL-2.1-or-later, LGPL-3.0-or-later (if dynamically linked)

### Strictly Disallowed

❌ **NEVER USE THESE LICENSES**:
- GPL (any version)
- AGPL (any version)
- Creative Commons with NC (Non-Commercial) or ND (No Derivatives)
- SSPL (Server Side Public License)
- BSL (Business Source License, before conversion)
- OSL-3.0 (Open Software License)

### Verification Process

**Before adding dependencies:**
1. Check license compatibility with project license (typically Apache-2.0)
2. Verify no GPL/AGPL in dependency tree
3. Review project's SBOM (Software Bill of Materials) if available
4. Document third-party licenses in NOTICE file

**CI/CD will automatically:**
- Generate SBOM using scancode-toolkit
- Validate CycloneDX SBOM schema
- Check for disallowed licenses
- Block PR merges if violations detected

**If you need a library with incompatible license:**
- Search for alternatives with permissive licenses
- Consider implementing functionality yourself
- Escalate to technical leadership for approval (rare exceptions)

---

## Security Practices

### Vulnerability Reporting

**For security issues**, use project's SECURITY.md process:
- Email: `support@au-zone.com` with subject "Security Vulnerability"
- Expected acknowledgment: 48 hours
- Expected assessment: 7 days
- Fix timeline based on severity

### Secure Coding Guidelines

**Input Validation:**
- Validate all external inputs (API requests, file uploads, user input)
- Use allowlists rather than blocklists
- Enforce size/length limits
- Sanitize for appropriate context (HTML, SQL, shell)

**Authentication & Authorization:**
- Never hardcode credentials or API keys
- Use environment variables or secure vaults for secrets
- Implement proper session management
- Follow principle of least privilege

**Data Protection:**
- Encrypt sensitive data at rest and in transit
- Use secure protocols (HTTPS, TLS 1.2+)
- Implement proper key management
- Sanitize logs (no passwords, tokens, PII)

**Common Vulnerabilities to Avoid:**
- SQL Injection: Use parameterized queries
- XSS (Cross-Site Scripting): Escape output, use CSP headers
- CSRF (Cross-Site Request Forgery): Use tokens
- Path Traversal: Validate and sanitize file paths
- Command Injection: Avoid shell execution; use safe APIs
- Buffer Overflows: Use safe string functions; bounds checking

### Dependencies

- Keep dependencies up to date
- Monitor for security advisories
- Use dependency scanning tools (Dependabot, Snyk)
- Audit new dependencies before adding

---

## Project-Specific Guidelines

This section is customized for the **EdgeFirst Model Node** project.

### Technology Stack

- **Language**: Rust 1.90.0+ (edition 2024)
- **Build system**: Cargo workspace with 2 crates:
  - `edgefirst-model`: Main inference service binary
  - `tflitec-sys`: TensorFlow Lite C bindings (internal)
- **Key dependencies**:
  - `edgefirst-hal 0.6.2`: Hardware abstraction (decoder, image processing, tensor)
  - `edgefirst-tracker 0.6.2`: ByteTrack multi-object tracking
  - `edgefirst-schemas 1.5.3`: Message schemas for EdgeFirst Perception
  - `zenoh 1.5.0`: Pub/sub communication layer
  - `tokio`: Async runtime for Zenoh and concurrent operations
  - `four-char-code 2.3.0`: FourCharCode type for pixel format identification
  - `tflitec-sys`: TensorFlow Lite inference engine (internal crate)
  - `vaal` (optional): RTM/Ara-2 runtime support (feature-gated)
  - `ndarray`: Numerical computing
  - `tracing`, `tracing-tracy`: Logging and profiling
- **Target platforms**: Linux on x86_64 and aarch64 (primary: NXP i.MX8)
- **Hardware acceleration**: NPU via VX delegate, G2D for image operations

### Architecture

- **Pattern**: Event-driven async architecture with Zenoh pub/sub
- **Component Type**: Microservice node in EdgeFirst Perception middleware
- **Data flow**:
  - Subscribe to camera frames via Zenoh (`{namespace}/camera/frame`)
  - Perform inference using TFLite or RTM models
  - Publish results: detections, masks, model_info, visualization
  - Zero-copy DMA buffer passing via pidfd
- **Module organization**:
  - `main.rs`: Application entry, Zenoh session, main inference loop
  - `lib.rs`: Public library interface, TrackerBox wrapper, DmaBuf handling
  - `model.rs`: Model trait, enum_dispatch, model config guessing
  - `tflite_model.rs` / `rtm_model.rs`: Model loading and inference
  - `buildmsgs.rs`: Zenoh message construction (CDR serialization)
  - `masks.rs`: Segmentation mask processing and compression
  - `args.rs`: CLI argument parsing, `fps.rs`: FPS monitoring
  - External: `edgefirst-hal` (decoder, image, tensor), `edgefirst-tracker` (ByteTrack)
- **Error handling**: Result types with ModelError for error propagation

### Build and Deployment

```bash
# Format code (REQUIRED before commit)
cargo fmt

# Lint code (must pass with zero warnings)
cargo clippy -- -D warnings

# Build release binary
cargo build --release

# Build with profiling support (Tracy)
cargo build --profile profiling --features profiling

# Run tests
cargo test --all-features

# Run tests with coverage
cargo llvm-cov --html
cargo llvm-cov --fail-under-lines 70

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --no-deps --open

# Cross-compile for ARM64
cargo build --target aarch64-unknown-linux-gnu --release

# Build with RTM support (optional)
cargo build --release --features rtm
```

### Performance Targets

Critical performance characteristics for edge AI inference:

- **Latency**: < 50ms for typical models (e.g., YOLOv8n on i.MX8)
- **Throughput**: 15-30 FPS depending on model complexity and NPU availability
- **Memory**: < 512MB RAM for typical workload (model-dependent)
- **Startup time**: < 2 seconds to first inference
- **CPU usage**: Minimal when using NPU acceleration
- **Power**: < 5W average for vision processing on target hardware

**Optimization priorities:**
1. Zero-copy data paths (DMA buffers, G2D operations)
2. NPU offloading for inference (VX delegate)
3. Efficient post-processing (NMS, tracking)
4. Memory reuse (pre-allocate buffers)

### Hardware Specifics

- **Primary platform**: NXP i.MX8M Plus with neural processing unit (NPU)
- **NPU support**: VX delegate for TensorFlow Lite (`libvx_delegate.so`)
- **GPU/2D acceleration**: G2D engine for image preprocessing
- **DMA buffer handling**: Zero-copy via `dma-buf` and `dma-heap`
- **CPU fallback**: All operations work on CPU when NPU unavailable
- **Memory alignment**: Be aware of DMA buffer alignment requirements
- **Platform quirks**:
  - VX delegate path may vary by platform (`/usr/lib/libvx_delegate.so`)
  - G2D operations require specific pixel format conversions
  - NPU tensor format may differ from CPU format

**Secondary platforms** (future):
- Maivin and Raivin developer kits
- Other NXP i.MX8 variants
- x86_64 for development and testing (CPU-only)

### Testing Conventions

- **Unit tests**: Co-located in `#[cfg(test)] mod tests` at end of implementation files
  - Currently exist in: `masks.rs`
  - Target: All modules should have unit tests
- **Integration tests**: To be created in `tests/` directory
  - Will include mock Zenoh nodes for end-to-end testing
  - Test complete inference pipeline
- **Test naming**: `test_<function>_<scenario>` format
  - Example: `test_decode_boxes_with_valid_input`, `test_nms_removes_overlapping_boxes`
- **Fixtures**: Test data in `benches/benchmark_data/` (reuse for tests)
- **Benchmarks**: Using `divan` framework in `benches/benchmark.rs`
- **Coverage requirement**: Minimum 70% overall, 90% for critical paths

**Running tests:**
```bash
# All tests
cargo test --all-features

# Specific module
cargo test --test integration_test

# With output
cargo test -- --nocapture

# Coverage
cargo llvm-cov --html --open
```

### Zenoh Communication Patterns

**Critical for AI assistants working on this project:**

- **Topic structure**: `{namespace}/component/topic`
  - Namespace typically: `edgefirst` or user-defined
- **Subscriptions**:
  - Camera frames: `{namespace}/camera/frame` (with DMA buffer FD via pidfd)
- **Publications**:
  - Detections: `{namespace}/model/detect` (CDR serialized, edgefirst-schemas)
  - Segmentation masks: `{namespace}/model/mask` (zstd compressed)
  - Model metadata: `{namespace}/model/model_info`
  - Visualization: `{namespace}/model/visualization` (annotated images)
- **Message format**: CDR serialization via `edgefirst-schemas` crate
- **DMA buffer passing**: Use `pidfd_getfd` to pass file descriptors between processes

**When modifying Zenoh code:**
- Maintain backward compatibility with schema versions
- Test with mock publishers/subscribers
- Verify CDR serialization/deserialization
- Document topic schema changes in ARCHITECTURE.md

### Tracy Profiling

This project uses Tracy for performance profiling:

- **Features**: `tracy` (default), `profiling` (extended tracing)
- **Build**: `cargo build --profile profiling --features profiling`
- **Usage**: Run binary, connect Tracy profiler GUI
- **Instrumentation**: Spans already added to critical paths
- **Zones**: Use `#[tracing::instrument]` for new functions

### SonarCloud Integration

- **Organization**: `edgefirstai`
- **Project**: `EdgeFirstAI_model`
- **Quality checks**: Run on every PR via CI/CD
- **Local analysis**: Use `cargo sonar` after running clippy/audit/outdated
- **Requirements**: Zero critical/high severity issues before merge

### Code Style Specifics

- **Formatting**: `rustfmt.toml` configuration is project-specific
- **Clippy**: Zero warnings policy (enforced in CI)
- **Comments**: Explain "why" not "what" (code is self-documenting)
- **Documentation**: All public APIs must have rustdoc comments
- **Error messages**: User-facing errors should be actionable
  - Bad: "Model loading failed"
  - Good: "Failed to load model: file not found at /path/to/model.tflite. Ensure the model path is correct."

### Common Pitfalls

- **DMA buffers**: Always check buffer validity before G2D operations
- **NPU delegate**: Gracefully fall back to CPU if delegate fails to load
- **Zenoh sessions**: Handle disconnections and reconnections
- **Model formats**: Validate model format before attempting to load
- **Memory leaks**: Ensure proper cleanup of TFLite tensors and allocators
- **Thread safety**: Be careful with shared state in async context

---

## Working with AI Assistants

### For GitHub Copilot / Cursor

These tools provide inline suggestions. Ensure:
- Suggestions match project conventions (run linters after accepting)
- Complex logic has explanatory comments
- Generated tests have meaningful assertions
- Security best practices are followed

### For Claude Code / Chat-Based Assistants

When working with conversational AI:
1. **Provide context**: Share relevant files, error messages, and requirements
2. **Verify outputs**: Review generated code critically before committing
3. **Iterate**: Refine solutions through follow-up questions
4. **Document decisions**: Capture architectural choices and tradeoffs
5. **Test thoroughly**: AI-generated code needs human verification

### Common AI Assistant Pitfalls

- **Hallucinated APIs**: Verify library functions exist before using
- **Outdated patterns**: Check if suggestions match current best practices
- **Over-engineering**: Prefer simple solutions over complex ones
- **Missing edge cases**: Explicitly test boundary conditions
- **License violations**: AI may suggest code with incompatible licenses

---

## Workflow Example

**Implementing a new feature:**

```bash
# 1. Create branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/EDGEAI-123-add-image-preprocessing

# 2. Implement feature with tests
# - Write unit tests first (TDD)
# - Implement functionality
# - Add integration tests
# - Update documentation

# 3. Verify quality
make format    # Auto-format code
make lint      # Run linters
make test      # Run all tests
make coverage  # Check coverage meets threshold

# 4. Commit with proper message
git add .
git commit -m "EDGEAI-123: Add image preprocessing pipeline

- Implemented resize, normalize, and augment functions
- Added comprehensive unit and integration tests
- Documented API with usage examples
- Achieved 85% test coverage"

# 5. Push and create PR
git push -u origin feature/EDGEAI-123-add-image-preprocessing
# Create PR via GitHub UI with template

# 6. Address review feedback
# - Make requested changes
# - Push additional commits
# - Respond to comments

# 7. Merge after approvals
# Maintainer merges via GitHub PR interface (squash or rebase)
```

---

## Getting Help

**For development questions:**
- Check project's `CONTRIBUTING.md` for setup instructions
- Review existing code for patterns and conventions
- Search GitHub Issues for similar problems
- Ask in GitHub Discussions (for public repos)

**For security concerns:**
- Email `support@au-zone.com` with subject "Security Vulnerability"
- Do not disclose vulnerabilities publicly

**For license questions:**
- Review license policy section above
- Check project's `LICENSE` file
- Contact technical leadership if unclear

**For contribution guidelines:**
- Read project's `CONTRIBUTING.md`
- Review recent merged PRs for examples
- Follow PR template and checklist

---

## Document Maintenance

**Project maintainers should:**
- Update [Project-Specific Guidelines](#project-specific-guidelines) with repository details
- Add technology stack, architecture patterns, and performance targets
- Document build/test/deployment procedures specific to the project
- Specify testing conventions (unit test location, framework choice, etc.)
- Keep examples and code snippets current
- Review and update annually or when major changes occur

**This template version**: 1.1 (February 2026)
**Organization**: Au-Zone Technologies
**License**: Apache-2.0 (for open source projects)

---

*This document helps AI assistants contribute effectively to Au-Zone projects while maintaining quality, security, and consistency. For questions or suggestions, contact `support@au-zone.com`.*
