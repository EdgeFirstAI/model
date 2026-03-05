# Testing the EdgeFirst Model Node

This document describes how to test the EdgeFirst Model Node service, covering unit tests, integration testing, on-target testing, model loading, benchmarks, coverage, profiling, and Zenoh communication verification.

---

## 1. Unit Tests

Run unit tests with:

```bash
cargo test --lib
```

Unit tests are co-located with their implementation files inside `#[cfg(test)] mod tests` blocks. The following modules currently have unit tests:

**Test naming convention:** `test_<function>_<scenario>`

All unit tests run on both **x86_64** and **aarch64** architectures without hardware dependencies. Hardware-dependent tests are annotated with `#[ignore]` and excluded from default runs.

---

## 2. Integration Testing

Integration testing requires the full **Zenoh** middleware running. The model node operates as a microservice in the EdgeFirst Perception pipeline: it subscribes to camera frames and publishes inference results over Zenoh pub/sub topics.

All topic key expressions are configurable via CLI arguments.

### Subscribed Topics

| Topic (default)    | CLI flag                | Message Type                | Description                                                        |
| ------------------ | ----------------------- | --------------------------- | ------------------------------------------------------------------ |
| `rt/camera/dma`    | `--camera-topic`        | `edgefirst_msgs/DmaBuf`    | Camera DMA buffer metadata                                        |
| `rt/camera/info`   | `--camera-info-topic`   | `sensor_msgs/CameraInfo`   | Camera resolution info (only when `--visualization` is enabled)    |

### Published Topics

| Topic (default)              | CLI flag / Env var                     | Message Type                       | Description                                                  |
| ---------------------------- | -------------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| `rt/model/output`            | `--output-topic` / `OUTPUT_TOPIC`      | `edgefirst_msgs/Model`             | Unified model output (boxes, masks, timing)                  |
| `rt/model/info`              | `--info-topic` / `INFO_TOPIC`          | `edgefirst_msgs/ModelInfo`         | Model metadata                                               |
| `rt/model/visualization`     | `--visual-topic` / `VISUAL_TOPIC`      | `foxglove_msgs/ImageAnnotations`   | Visualization overlays (when `--visualization` enabled)      |
| *(disabled by default)*      | `--detect-topic` / `DETECT_TOPIC`      | `edgefirst_msgs/Detect`            | Legacy detection boxes (set to `rt/model/boxes2d` to enable) |
| *(disabled by default)*      | `--mask-topic` / `MASK_TOPIC`          | `edgefirst_msgs/Mask`              | Legacy segmentation masks (set to `rt/model/mask` to enable) |

---

## 3. On-Target Testing

On-target testing requires:

- **NXP i.MX8M Plus** hardware with NPU
- Camera node running (provides DMA buffers over Zenoh)
- A TFLite model file

### Basic inference

```bash
./edgefirst-model --model model.tflite
```

### Test with object tracking

```bash
./edgefirst-model --model model.tflite --track
```

### Test with visualization overlays

```bash
./edgefirst-model --model model.tflite --visualization
```

### Test with legacy topics re-enabled

```bash
./edgefirst-model --model model.tflite \
  --detect-topic rt/model/boxes2d \
  --mask-topic rt/model/mask
```

### Test unified model output (`rt/model/output`)

The unified `Model` message is published on every frame for all model types. Verify it with:

```bash
# Subscribe to the unified output topic
z_sub -k "rt/model/output"
```

**Detection model** — verify `boxes` is populated and `masks` is empty:

```bash
./edgefirst-model --model yolov8n.tflite
```

**Instance segmentation model** — verify `masks` has one entry per box with `boxed: true`:

```bash
./edgefirst-model --model yolov8n-seg.tflite
```

**Semantic segmentation model** — verify `masks` has a single entry with `boxed: false`:

```bash
./edgefirst-model --model deeplab.tflite
```

For all model types, verify that the `input_time`, `model_time`, and `decode_time` fields contain non-zero durations.

Hardware-dependent unit tests (marked `#[ignore]`) can be run on the device with:

```bash
cargo test -- --ignored --test-threads=1
```

### Automated Integration Tests

Integration tests in `tests/integration_test.rs` exercise the full inference pipeline on real hardware. Both tests are marked `#[ignore]` and require:

- A running camera node (provides DMA buffers over Zenoh)
- A TFLite model file (set via `MODEL` environment variable)
- Optionally a VX delegate (set via `DELEGATE` environment variable)

| Test | Description |
|------|-------------|
| `test_model_inference` | Starts the model service, subscribes to `rt/model/output`, collects messages for 10s, validates `Model` messages have non-zero timing fields and rate >= 10 Hz |
| `test_graceful_shutdown` | Starts the model service, sends SIGTERM, verifies clean exit within 5s |

**Run manually on device:**

```bash
# Build the integration tests
cargo test --test integration_test --no-run

# Run with required environment variables
MODEL=/path/to/yolov8n_640x640.tflite \
DELEGATE=/usr/lib/libvx_delegate.so \
  cargo test --test integration_test -- --include-ignored --test-threads=1
```

---

## 4. CI Architecture (Three-Phase)

The CI pipeline uses a three-phase architecture to collect coverage from on-target hardware tests:

```
Phase 1: BUILD (GitHub ARM Runner)
├── Run unit tests with coverage
├── Build instrumented binaries (cargo llvm-cov + profiling profile)
├── Build instrumented integration test binaries
└── Upload artifacts: coverage.lcov, instrumented binaries, llvm-cov objects

Phase 2: RUN (imx8mpevk Self-Hosted Runner)
├── Download instrumented binaries
├── Download model file from repo.edgefirst.ai
├── Run integration tests with LLVM_PROFILE_FILE set
├── Collect profraw files from service + test processes
└── Upload artifacts: profraw files, test output

Phase 3: PROCESS (GitHub ARM Runner)
├── Download Phase 1 instrumented objects
├── Download Phase 2 profraw files
├── Merge profraw → profdata with llvm-profdata
├── Generate LCOV with llvm-cov export
└── Upload coverage-hardware.lcov for SonarCloud
```

Phase 2 is triggered on `main` push or when a PR has the `test-hardware` label. The `imx8mpevk` runner is used because it has guaranteed tflite-vx-delegate/tim-vx patches for DMA-BUF and CameraAdaptor support.

### Graceful Shutdown

The model service installs SIGTERM/SIGINT signal handlers that set a `SHUTDOWN` flag, causing the main inference loop to exit cleanly. This is critical because LLVM coverage instrumentation uses `atexit()` handlers to flush `.profraw` files — without graceful shutdown, no coverage data is generated.

---

## 5. Model Loading Verification

The model node auto-detects model configuration through the following resolution order:

1. **External config file** -- Provide a YAML or JSON file via `--edgefirst-config`. This overrides any embedded config.
2. **Embedded metadata** -- The model file may contain an `edgefirst.yaml` entry with model configuration.
3. **Shape-based heuristic guessing** -- When no config is found, the node inspects input/output tensor shapes to guess the model type (fallback).

Once loaded, model metadata is published on the info topic (`rt/model/info` by default). Verify correct loading by subscribing to this topic and inspecting the `ModelInfo` message.

---

## 6. Benchmarks

Benchmarks use the **divan** framework. Run all benchmarks with:

```bash
cargo bench
```

Benchmark test data is stored in `benches/benchmark_data/` and includes sample model outputs and configuration files used by the benchmark harness.

---

## 7. Coverage

Generate an HTML coverage report:

```bash
cargo llvm-cov --html
```

The report is written to `target/llvm-cov/html/index.html`.

To enforce the minimum coverage threshold:

```bash
cargo llvm-cov --fail-under-lines 70
```

**Coverage targets:**
- Minimum overall: **70%**
- Core modules (`model.rs`, `masks.rs`): **80%+**
- Public APIs: **100%**
- Hardware-specific paths: best effort (requires device)

Install coverage tools if not already available:

```bash
cargo install cargo-llvm-cov cargo-nextest
```

---

## 8. Profiling During Testing

The model node integrates with **Tracy** for real-time performance profiling.

### Basic profiling

```bash
# Build with Tracy support (enabled by default)
cargo build --release

# Run with Tracy broadcast enabled
./edgefirst-model --tracy --model model.tflite
```

### Advanced profiling (memory, sampling, system-tracing)

```bash
cargo build --profile profiling --features profiling
./target/profiling/edgefirst-model --tracy --model model.tflite
```

Connect the Tracy profiler GUI to visualize frame timing, inference latency, zone durations, and memory allocations. Tracy discovers the application automatically over the network.

---

## 9. Zenoh Testing Tips

Use Zenoh CLI tools to monitor and verify pub/sub communication during testing.

### Subscribe to all model outputs

```bash
z_sub -k "rt/model/**"
```

### Subscribe to the unified model output

```bash
z_sub -k "rt/model/output"
```

### Subscribe to a specific legacy topic

```bash
z_sub -k "rt/model/boxes2d"
```

### Publish a test message

```bash
z_pub -k "rt/camera/dma" -v "<payload>"
```

### Zenoh connectivity flags

The model node supports the following Zenoh connection options:

| Flag                        | Description                              |
| --------------------------- | ---------------------------------------- |
| `--mode <peer\|client>`     | Zenoh session mode (default: `peer`)     |
| `--connect <endpoint>`      | Connect to a specific Zenoh endpoint     |
| `--listen <endpoint>`       | Listen on a specific Zenoh endpoint      |
| `--no-multicast-scouting`   | Disable multicast discovery              |

These flags are useful for testing across network boundaries or when multicast is unavailable.

---

## Quick Reference

| Task                         | Command                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| Run unit tests               | `cargo test --lib`                                         |
| Run ignored (hardware) tests | `cargo test -- --ignored --test-threads=1`                 |
| Run integration tests        | `MODEL=model.tflite cargo test --test integration_test -- --include-ignored --test-threads=1` |
| Run benchmarks               | `cargo bench`                                              |
| Coverage report (HTML)       | `cargo llvm-cov --html`                                    |
| Coverage threshold check     | `cargo llvm-cov --fail-under-lines 70`                     |
| Format check                 | `cargo fmt --all --check`                                  |
| Lint check                   | `cargo clippy --all-targets --all-features -- -D warnings` |
| Profile with Tracy           | `cargo build --release && ./edgefirst-model --tracy ...`   |
