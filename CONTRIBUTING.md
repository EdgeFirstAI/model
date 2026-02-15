# Contributing to EdgeFirst Model Node

Thank you for your interest in contributing to the EdgeFirst Model Node! This document provides guidelines for contributing to the project.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Profiling and Performance Analysis](#profiling-and-performance-analysis)
5. [Coding Standards](#coding-standards)
6. [Testing Requirements](#testing-requirements)
7. [Pull Request Process](#pull-request-process)
8. [License Policy](#license-policy)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to support@au-zone.com.

---

## Getting Started

### Prerequisites

**Development Environment:**

- Linux (Ubuntu 20.04+ or Debian 11+ recommended)
- Rust 1.90.0 or later (Edition 2024)
- For hardware testing: NXP i.MX8M Plus with NPU, camera node running, model file

**Install Rust:**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update
```

**Install Development Tools:**

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install build-essential

# For cross-compilation
sudo apt-get install gcc-aarch64-linux-gnu
rustup target add aarch64-unknown-linux-gnu
```

**Note:** No system dependencies are required for building. The `g2d-sys` and `tflitec-sys` crates use `dlopen` to dynamically load hardware libraries at runtime.

### Clone Repository

```bash
git clone https://github.com/EdgeFirstAI/model.git
cd model
```

### Build and Run

```bash
# Build debug binary
cargo build

# Run tests (no hardware required)
cargo test --lib

# Build release binary
cargo build --release

# Run (requires model file)
cargo run -- --model <path-to-model>

# Run on hardware (requires NXP i.MX8M Plus, camera running, model file)
./target/release/edgefirst-model --model <path-to-model>
```

---

## Development Workflow

### Branch Strategy

**Main Branches:**

- `main` - Stable release branch (protected)
- `develop` - Integration branch for features (protected)

**Feature Branches:**
Create branches from `develop` using the pattern:

```
feature/<PROJECTKEY-###>[-description]
bugfix/<PROJECTKEY-###>[-description]
hotfix/<PROJECTKEY-###>[-description]
```

External contributors without JIRA access may use `feature/issue-<N>-description` referencing GitHub issue numbers.

**Example:**

```bash
git checkout develop
git pull origin develop
git checkout -b feature/EDGEAI-123-add-yolo-decoder
```

### Commit Messages

Use clear, descriptive commit messages with the JIRA key:

```
PROJECTKEY-###: Brief description of what was done

Optional detailed description with bullet points
```

**Rules:**
- Subject line: 50-72 characters ideal
- Focus on WHAT changed, not HOW
- No type prefixes (`feat:`, `fix:`, etc.) - JIRA provides context

**Examples:**

```
EDGEAI-123: Add YOLO v8 decoder support

- Implemented decoder for YOLO v8 models with configurable NMS
- Added box format conversion from YOLO to normalized coordinates

EDGEAI-456: Fix bounding box coordinates for rotated frames
```

External contributors without JIRA access may use `#<issue>:` referencing GitHub issue numbers.

### Development Loop

1. **Create feature branch** from `develop`
2. **Make changes** with atomic commits
3. **Write tests** for new functionality
4. **Run linters** and formatters
5. **Test locally** on hardware if applicable
6. **Push branch** to GitHub
7. **Open pull request** to `develop`
8. **Address review** feedback
9. **Merge** after approval

---

## Profiling and Performance Analysis

### Tracy Profiler Setup

The model node includes Tracy profiler integration for detailed performance analysis. Tracy is a real-time profiler that provides frame timing, CPU profiling, memory tracking, and GPU activity visualization.

#### Installing Tracy Profiler

**Download Tracy:**

Tracy profiler GUI is available from the official repository:

```bash
# Clone Tracy repository
git clone https://github.com/wolfpld/tracy.git
cd tracy

# Build profiler GUI (requires X11, OpenGL)
cd profiler/build/unix
make release

# Binary will be in: tracy/profiler/build/unix/Tracy-release
```

**Platform Requirements:**
- Linux: X11, OpenGL 3.2+
- macOS: Supported via XQuartz
- Windows: Native support

**Pre-built Binaries:**

Pre-built Tracy releases are available at: https://github.com/wolfpld/tracy/releases

#### Using Tracy with Model Node

**1. Build with Tracy Support:**

Tracy is enabled by default. For standard profiling:

```bash
# Build with basic Tracy support (default)
cargo build --release

# Build with advanced profiling (memory, sampling, system-tracing)
cargo build --release --profile=profiling --features=profiling
```

**2. Run Model Node with Tracy:**

```bash
# Start model node with Tracy enabled
./target/release/edgefirst-model --tracy --model <path-to-model>
```

**3. Connect Tracy Profiler:**

```bash
# In separate terminal, start Tracy GUI
./Tracy-release

# Tracy will automatically discover and connect to the model node
```

**4. Alternative: Manual Connection:**

If automatic discovery doesn't work:
- In Tracy GUI, click "Connect" button
- Enter IP address of device running model node
- Default port: 8086

#### Tracy Features Available

**Frame Markers:**

The model node emits frame markers for inference processing:
- **Main frames**: Inference rate
- **Pre-processing**: Input preprocessing stages
- **Post-processing**: NMS, tracking, decoder stages

In Tracy, switch between frame views using the frame dropdown menu.

**Performance Plots:**

Real-time plots visible in Tracy:
- `inference_fps`: Inference frames per second
- `inference_latency_ms`: End-to-end inference latency

**Zones and Spans:**

All instrumented functions appear as zones in Tracy timeline:
- Model loading and initialization
- TFLite/RTM inference operations
- Preprocessing (G2D conversions, resizing)
- Post-processing (NMS, tracking, decoders)
- Zenoh publishing

Zones are automatically named from Rust function names via the `#[instrument]` attribute.

**Memory Profiling:**

When built with `profiling` feature:
- Memory allocations tracked
- Call stacks recorded
- Memory leaks detected
- Allocation statistics available

#### Profiling Workflow

**1. Identify Performance Issues:**

```bash
# Run with Tracy enabled
./edgefirst-model --tracy --model <path-to-model>

# In Tracy GUI:
# - Check frame timing consistency
# - Look for dropped frames (gaps in timeline)
# - Identify long-running zones
# - Monitor inference FPS plot
```

**2. Investigate Specific Functions:**

- Click on zones in timeline to see call stacks
- Use statistics view to find slowest functions
- Compare frame timing between good/bad frames
- Check memory allocations in hot paths

**3. Validate Optimizations:**

After making changes:
- Capture new Tracy trace
- Compare zone timings before/after
- Verify inference rate improvements
- Check for regressions in other areas

#### Common Profiling Scenarios

**Scenario 1: Slow Inference**

If inference is slower than expected:

1. Check inference zone duration
2. Verify NPU is being used (not CPU fallback)
3. Look for preprocessing bottlenecks (G2D operations)
4. Check if post-processing (NMS, tracking) is dominating

**Scenario 2: High CPU Usage**

1. Use Tracy's CPU sampling (profiling feature)
2. Identify functions consuming most CPU time
3. Check for unexpected software fallbacks
4. Verify tensor operations use hardware acceleration

**Scenario 3: Memory Growth**

1. Enable memory profiling (profiling feature)
2. Capture trace over several minutes
3. Check memory plot for growth trend
4. Use allocation list to find leaks

#### Performance Best Practices

**During Development:**
- Profile early and often
- Establish baseline performance
- Use Tracy to validate assumptions
- Check both average and worst-case timings

**Before Release:**
- Capture Tracy trace of typical workload
- Verify inference timing meets requirements
- Check for memory leaks (long-running test)
- Profile on target hardware (i.MX8), not development machine

**Optimization Guidelines:**
- Profile before optimizing (measure, don't guess)
- Focus on hot paths (zones that appear frequently)
- Prefer hardware acceleration (NPU, G2D)
- Avoid allocations in inference pipeline

---

## Coding Standards

### Rust Style Guidelines

**Follow Official Style:**

- Use `rustfmt` for formatting (CI enforces this)
- Use `clippy` for linting (CI enforces this)
- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

**Format Code:**

```bash
cargo fmt --all
```

**Lint Code:**

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Code Quality Principles

**Readability:**

- Prefer clear, self-documenting code over clever code
- Use descriptive variable names (`detection_confidence` not `dc`)
- Keep functions small and focused (< 50 lines ideal)
- Add comments for complex algorithms or hardware-specific quirks

**Error Handling:**

- Use `Result<T, Box<dyn Error>>` for fallible operations
- Provide context with error messages:

  ```rust
  .with_context(|| format!("Failed to load model from {}", model_path))?
  ```

- Log errors before returning:

  ```rust
  error!("Model initialization failed: {}", e);
  ```

**Performance:**

- Avoid unnecessary allocations in hot paths
- Use zero-copy patterns where possible (DMA buffers)
- Profile before optimizing (use `--tracy` flag)
- Document performance-critical sections

**Documentation:**

- Public APIs must have rustdoc comments
- Include examples for complex functions
- Document panic conditions
- Explain hardware-specific behavior

**Example:**

```rust
/// Performs non-maximum suppression on detection boxes.
///
/// # Arguments
/// * `boxes` - Array of bounding boxes with confidence scores
/// * `iou_threshold` - IoU threshold for suppression (typically 0.45)
///
/// # Returns
/// Indices of boxes to keep after NMS
///
/// # Example
/// ```no_run
/// let kept_indices = nms(&detections, 0.45);
/// let filtered = detections.select(Axis(0), &kept_indices);
/// ```
pub fn nms(boxes: &ArrayView2<f32>, iou_threshold: f32) -> Vec<usize> {
    // Implementation
}
```

### Hardware-Specific Code

**Platform Conditionals:**

```rust
#[cfg(target_arch = "aarch64")]
fn use_npu_acceleration() -> bool {
    // Check for NXP NPU hardware
    std::path::Path::new("/dev/galcore").exists()
}

#[cfg(not(target_arch = "aarch64"))]
fn use_npu_acceleration() -> bool {
    false
}
```

**Feature Gates:**

```rust
#[cfg(feature = "rtm")]
use vaal::VaalModel;

#[cfg(feature = "rtm")]
pub fn load_rtm_model(path: &Path) -> Result<VaalModel> {
    // RTM-specific implementation
}
```

**Fallback Implementations:**

- Always provide software fallback for hardware features
- Gracefully degrade when hardware unavailable
- Log when using fallback: `warn!("NPU unavailable, using CPU inference")`

---

## Testing Requirements

### Test Categories

**Unit Tests:**

- Test individual functions in isolation
- Located in same file as implementation
- Use `#[cfg(test)]` module
- Present in: `tracker.rs`, `kalman.rs`, `masks.rs`, `nms.rs`

**Integration Tests:**

- Test component interactions
- Located in `tests/` directory
- May require hardware (use `#[ignore]` for hardware-dependent tests)

**Benchmarks:**

- Performance tests in `benches/` directory
- Use `divan` framework
- Run with `cargo bench`

### Running Tests

**All Unit Tests (no hardware):**

```bash
cargo test --lib
```

**All tests pass on both x86_64 and aarch64 architectures.**

**Hardware-Dependent Tests (on device):**

```bash
cargo test -- --ignored --test-threads=1
```

**With Coverage:**

```bash
# Install coverage tools
cargo install cargo-llvm-cov cargo-nextest

# Run coverage with nextest
cargo llvm-cov nextest --all-features --workspace --html
open target/llvm-cov/html/index.html
```

**Benchmarks:**

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench benchmark
```

### Writing Tests

**Unit Test Example:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_calculation() {
        let box1 = array![0.0, 0.0, 10.0, 10.0];
        let box2 = array![5.0, 5.0, 15.0, 15.0];
        let iou = calculate_iou(&box1.view(), &box2.view());
        assert!((iou - 0.142857).abs() < 0.001);
    }

    #[test]
    #[ignore]  // Requires hardware
    fn test_npu_inference() {
        let model = TFLiteModel::new("model.tflite", true).expect("NPU required");
        // Hardware test
    }
}
```

**Benchmark Example:**

```rust
// benches/benchmark.rs
use divan::Bencher;

#[divan::bench]
fn bench_nms(bencher: Bencher) {
    let boxes = create_test_boxes(100);
    bencher.bench_local(|| {
        nms(&boxes.view(), 0.45)
    });
}
```

### Coverage Requirements

- **Minimum overall**: 70%
- **Core modules** (`model.rs`, `nms.rs`, `tracker.rs`): 80%+
- **Public APIs**: 100%
- **Hardware paths**: Best effort (requires device)

**Coverage is enforced in CI pipeline**

---

## Pull Request Process

### Before Submitting

**Checklist:**

- [ ] Code follows Rust style guidelines (`cargo fmt`)
- [ ] All lints pass (`cargo clippy -- -D warnings`)
- [ ] Tests pass (`cargo test --lib`)
- [ ] New functionality has tests
- [ ] Documentation is updated (README, rustdoc)
- [ ] Commit messages are clear and descriptive
- [ ] No secrets or credentials committed
- [ ] SBOM license policy compliance verified

### Creating Pull Request

1. **Push your branch** to GitHub:

   ```bash
   git push origin feature/your-feature
   ```

2. **Open PR** via GitHub web interface:
   - Base: `develop` (or `main` for hotfixes)
   - Provide clear title and description
   - Link related issues: "Fixes #123" or "Related to #456"

3. **Fill out PR template:**

   ```markdown
   ## Summary
   Brief description of changes

   ## Changes
   - Added feature X
   - Fixed bug Y
   - Updated documentation Z

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Tested on hardware (if applicable)

   ## Checklist
   - [ ] Code formatted with `rustfmt`
   - [ ] No `clippy` warnings
   - [ ] Tests pass
   - [ ] Documentation updated
   ```

### Review Process

**Reviewers will check:**

- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Hardware compatibility
- License compliance

**Address feedback:**

- Make requested changes
- Push additional commits to same branch
- Respond to review comments
- Re-request review when ready

**Approval requirements:**

- **1 approval** for merges to `develop`
- **2 approvals** for merges to `main`
- All CI checks must pass

### Merge

Once approved, a maintainer will merge using **squash and merge** to keep history clean.

---

## License Policy

### CRITICAL: License Compliance

This project is licensed under **Apache-2.0**. All dependencies must comply with Au-Zone's license policy.

**Allowed Licenses:**

- MIT, Apache-2.0, BSD-2/3-Clause, ISC, 0BSD, Unlicense
- EPL-2.0, MPL-2.0 (for dependencies only)

**Disallowed Licenses:**

- GPL, AGPL (all versions)
- LGPL (requires review)
- Creative Commons with NC/ND/SA restrictions

**Before adding dependencies:**

1. Check license in `Cargo.toml` or repository
2. Verify compatibility with Apache-2.0
3. Run SBOM generation to detect issues:

   ```bash
   .github/scripts/generate_sbom.sh
   python3 .github/scripts/check_license_policy.py sbom.json
   ```

**If license checker fails, DO NOT merge**

See [SBOM_PROCESS.md](SBOM_PROCESS.md) for details.

---

## Development Tips

### Profiling with Tracy

```bash
# Build with profiling
cargo build --release --profile=profiling --features=profiling

# Run with Tracy enabled
./target/release/edgefirst-model --tracy --model <path-to-model>

# Connect Tracy profiler GUI
# Download from: https://github.com/wolfpld/tracy
```

### Cross-Compilation for ARM64

```bash
# Set linker in .cargo/config.toml
mkdir -p .cargo
cat > .cargo/config.toml <<EOF
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
EOF

# Build for ARM64
cargo build --release --target aarch64-unknown-linux-gnu

# Binary at: target/aarch64-unknown-linux-gnu/release/edgefirst-model
```

### Testing on Hardware

**Copy to Device:**

```bash
# Build ARM64 binary
cargo build --release --target aarch64-unknown-linux-gnu

# Copy to device
scp target/aarch64-unknown-linux-gnu/release/edgefirst-model user@device:/tmp/

# SSH and test (requires camera running and model file)
ssh user@device
cd /tmp
./edgefirst-model --model /path/to/model.tflite
```

### Building with RTM Support

For VAAL RTM models:

```bash
# Build with RTM feature
cargo build --release --features rtm

# RTM-specific code is gated with #[cfg(feature = "rtm")]
```

---

## Community

### Getting Help

**Questions:**

- [GitHub Discussions](https://github.com/EdgeFirstAI/model/discussions) - Q&A, ideas
- [Documentation](https://doc.edgefirst.ai/) - Guides and tutorials

**Issues:**

- [Bug Reports](https://github.com/EdgeFirstAI/model/issues/new?template=bug_report.md)
- [Feature Requests](https://github.com/EdgeFirstAI/model/issues/new?template=feature_request.md)

**Chat:**

- EdgeFirst Community Discord (coming soon)

### Recognition

Contributors will be acknowledged in:

- [CONTRIBUTORS.md](CONTRIBUTORS.md) - Hall of fame
- Release notes for significant contributions
- GitHub contribution graph

---

## Thank You

Every contribution, no matter how small, helps make EdgeFirst Model Node better for everyone. We appreciate your time and effort!

**Questions about contributing?** Open a [discussion](https://github.com/EdgeFirstAI/model/discussions) or email support@au-zone.com

---

_Last updated: 2026-02-13_
