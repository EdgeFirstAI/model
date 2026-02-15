# Testing the EdgeFirst Model Node

This document describes how to test the EdgeFirst Model Node service, covering unit tests, integration testing, on-target testing, model loading, benchmarks, coverage, profiling, and Zenoh communication verification.

---

## 1. Unit Tests

Run unit tests with:

```bash
cargo test --lib
```

Unit tests are co-located with their implementation files inside `#[cfg(test)] mod tests` blocks. The following modules currently have unit tests:

- `masks.rs` -- mask slicing and filtering
- `nms.rs` -- non-maximum suppression
- `tracker.rs` -- ByteTrack multi-object tracking
- `kalman.rs` -- Kalman filter for tracking

**Test naming convention:** `test_<function>_<scenario>`

Examples:
- `test_decode_boxes_with_valid_input`
- `test_nms_removes_overlapping_boxes`
- `test_slice` (in `masks.rs`)

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

| Topic (default)              | CLI flag                    | Message Type                       | Description                                                  |
| ---------------------------- | --------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| `rt/model/boxes2d`           | `--detect-topic`            | `edgefirst_msgs/Detect`            | Detection bounding boxes                                     |
| `rt/model/info`              | `--info-topic`              | `edgefirst_msgs/ModelInfo`         | Model metadata                                               |
| `rt/model/mask`              | `--mask-topic`              | `edgefirst_msgs/Mask`              | Segmentation masks (raw)                                     |
| `rt/model/mask_compressed`   | `--mask-compressed-topic`   | `edgefirst_msgs/Mask`              | Compressed masks (when `--mask-compression` enabled)         |
| `rt/model/visualization`     | `--visual-topic`            | `foxglove_msgs/ImageAnnotations`   | Visualization overlays (when `--visualization` enabled)      |

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

### Test with mask compression

```bash
./edgefirst-model --model model.tflite --mask-compression
```

Hardware-dependent unit tests (marked `#[ignore]`) can be run on the device with:

```bash
cargo test -- --ignored --test-threads=1
```

---

## 4. Model Loading Verification

The model node auto-detects model configuration through the following resolution order:

1. **External config file** -- Provide a YAML or JSON file via `--edgefirst-config`. This overrides any embedded config.
2. **Embedded metadata** -- The model file may contain an `edgefirst.yaml` entry with model configuration.
3. **Shape-based heuristic guessing** -- When no config is found, the node inspects input/output tensor shapes to guess the model type (fallback).

Once loaded, model metadata is published on the info topic (`rt/model/info` by default). Verify correct loading by subscribing to this topic and inspecting the `ModelInfo` message.

---

## 5. Benchmarks

Benchmarks use the **divan** framework. Run all benchmarks with:

```bash
cargo bench
```

Benchmark test data is stored in `benches/benchmark_data/` and includes sample model outputs and configuration files used by the benchmark harness.

---

## 6. Coverage

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
- Core modules (`model.rs`, `nms.rs`, `tracker.rs`): **80%+**
- Public APIs: **100%**
- Hardware-specific paths: best effort (requires device)

Install coverage tools if not already available:

```bash
cargo install cargo-llvm-cov cargo-nextest
```

---

## 7. Profiling During Testing

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

## 8. Zenoh Testing Tips

Use Zenoh CLI tools to monitor and verify pub/sub communication during testing.

### Subscribe to all model outputs

```bash
z_sub -k "rt/model/**"
```

### Subscribe to a specific topic

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
| `--multicast-interface`     | Set the multicast scouting interface     |

These flags are useful for testing across network boundaries or when multicast is unavailable.

---

## Quick Reference

| Task                         | Command                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| Run unit tests               | `cargo test --lib`                                         |
| Run ignored (hardware) tests | `cargo test -- --ignored --test-threads=1`                 |
| Run benchmarks               | `cargo bench`                                              |
| Coverage report (HTML)       | `cargo llvm-cov --html`                                    |
| Coverage threshold check     | `cargo llvm-cov --fail-under-lines 70`                     |
| Format check                 | `cargo fmt --all --check`                                  |
| Lint check                   | `cargo clippy --all-targets --all-features -- -D warnings` |
| Profile with Tracy           | `cargo build --release && ./edgefirst-model --tracy ...`   |
