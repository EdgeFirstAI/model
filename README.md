# EdgeFirst Model Node

**Production-ready AI inference service with hardware-accelerated NPU inference, object tracking, and EdgeFirst Perception integration**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.90.0%2B-orange.svg)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20ARM64%20%7C%20x86__64-lightgrey.svg)]()

---

## Overview

The EdgeFirst Model Node is a high-performance AI inference service designed for edge AI perception systems. It consumes camera streams via zero-copy DMA, executes TensorFlow Lite models on dedicated NPU hardware, and publishes detection results, segmentation masks, and object tracks over Zenoh for seamless integration with robotics and vision ecosystems.

**Key Features:**

- **ROS2 Compatibility** - Standard `edgefirst_msgs` and `sensor_msgs` interfaces for drop-in integration
- **Hardware Acceleration** - NXP i.MX8 NPU inference with G2D preprocessing
- **Zero-Copy DMA** - Direct memory access for ultra-low latency vision pipelines
- **Object Detection** - YOLO detection models with auto-detection of model architecture
- **Instance Segmentation** - Semantic and instance segmentation
- **ByteTrack Tracking** - Multi-object tracking with Kalman filtering
- **Multi-Format Output** - Bounding boxes, masks, tracking IDs, and Foxglove visualization
- **Production Ready** - Tracy profiling, journald logging, comprehensive error handling

---

## EdgeFirst Perception Ecosystem

The EdgeFirst Model Node is the **AI inference layer** of the EdgeFirst Perception Middleware—a modular software stack for edge AI vision systems.

### Architecture Context

```mermaid
graph LR
    subgraph "Sensor Inputs"
        Camera["📷 Camera Service"]
        Radar["📡 Radar"]
        LiDAR["🔬 LiDAR"]
        IMU["⚖️ IMU"]
    end

    subgraph "EdgeFirst Perception Middleware"
        Zenoh["Zenoh Messaging<br/>(ROS2 CDR)"]
    end

    subgraph "Outputs & Applications"
        Vision["🤖 Vision Models<br/>(THIS REPO)"]
        Fusion["🔗 Fusion Models"]
        Recording["💾 Recording<br/>(MCAP)"]
        WebUI["🌐 Web UI<br/>(HTTPS)"]
    end

    Camera --> Zenoh
    Radar --> Zenoh
    LiDAR --> Zenoh
    IMU --> Zenoh

    Zenoh --> Vision
    Zenoh --> Fusion
    Zenoh --> Recording
    Zenoh --> WebUI

    style Vision fill:#f96,stroke:#333,stroke-width:3px
```

**What the Model Node Does:**

- **Consumes** camera DMA buffers from EdgeFirst Camera Node
- **Preprocesses** frames with hardware-accelerated G2D format conversion and scaling
- **Infers** using TFLite models on NPU (or CPU/GPU fallback)
- **Decodes** YOLO outputs into bounding boxes, scores, and labels
- **Tracks** objects across frames using ByteTrack multi-object tracking
- **Segments** images with semantic and instance segmentation models
- **Publishes** to Zenoh topics using ROS2 message formats

**Integration Points:**

- **Camera Service**: Zero-copy DMA buffer consumption for real-time inference
- **Fusion Models**: Combine vision detections with radar/LiDAR for multi-modal perception
- **Recording**: MCAP format recording for dataset collection and model training
- **Web UI**: Live visualization with Foxglove-compatible annotations
- **Custom Applications**: ROS2-compatible message access via zenoh-bridge-dds

**Learn More:** [EdgeFirst Perception Documentation](https://doc.edgefirst.ai/latest/perception/)

---

## Quick Start

### Prerequisites

**Hardware:**

- NXP i.MX8M Plus based platform (Maivin, Raivin) or compatible ARM64 device
- Minimum: 2GB RAM, quad-core ARM Cortex-A53
- NPU (Neural Processing Unit) for hardware-accelerated inference

**Software:**

- Linux kernel 5.10+ with V4L2 support
- Rust 1.90.0 or later (for building from source)
- OR: Pre-built binaries from [GitHub Releases](https://github.com/EdgeFirstAI/model/releases)
- TensorFlow Lite model file (.tflite or .rtm)

### Installation

**Option 1: Pre-built Binary (Recommended)**

```bash
# Download latest release for ARM64
wget https://github.com/EdgeFirstAI/model/releases/latest/download/edgefirst-model-linux-aarch64

# Make executable
chmod +x edgefirst-model-linux-aarch64

# Run with your model
./edgefirst-model-linux-aarch64 --model /path/to/model.tflite
```

**Option 2: Build from Source**

```bash
# Clone repository
git clone https://github.com/EdgeFirstAI/model.git
cd model

# Build release binary
cargo build --release

# Run
./target/release/edgefirst-model --model /path/to/model.tflite
```

**Option 3: Cross-Compile for ARM64**

```bash
# Add ARM64 target
rustup target add aarch64-unknown-linux-gnu

# Install cross-compilation toolchain
# (Debian/Ubuntu)
sudo apt-get install gcc-aarch64-linux-gnu

# Build for ARM64
cargo build --release --target aarch64-unknown-linux-gnu

# Binary at: target/aarch64-unknown-linux-gnu/release/edgefirst-model
```

### Basic Usage

**Object Detection with NPU Inference:**

```bash
edgefirst-model \
  --model yolov8n.tflite \
  --engine npu \
  --threshold 0.5
```

**Object Detection with ByteTrack Tracking:**

```bash
edgefirst-model \
  --model yolov8n.tflite \
  --track \
  --track-high-conf 0.7 \
  --track-iou 0.25
```

**Custom Topics and Visualization:**

```bash
edgefirst-model \
  --model model.tflite \
  --camera-topic rt/front_camera/dma \
  --visualization
```

**Re-enable Legacy Topics:**

```bash
edgefirst-model \
  --model model.tflite \
  --detect-topic rt/model/boxes2d \
  --mask-topic rt/model/mask
```

---

## Key Capabilities

### ROS2 Message Compatibility

The model node publishes standard ROS2 message types using CDR serialization, ensuring seamless integration with existing ROS2 ecosystems:

**Published Topics:**

| Topic (default) | Message Type | Description |
|----------------|--------------|-------------|
| `rt/model/output` | `edgefirst_msgs/Model` | **Unified model output** with boxes, masks, and timing |
| `rt/model/info` | `edgefirst_msgs/ModelInfo` | Model metadata, timing, and performance metrics |
| `rt/model/visualization` | `foxglove_msgs/ImageAnnotations` | Foxglove-compatible visualization overlays |
| *(disabled)* | `edgefirst_msgs/Detect` | Legacy detection boxes (set `DETECT_TOPIC=rt/model/boxes2d` to enable) |
| *(disabled)* | `edgefirst_msgs/Mask` | Legacy segmentation masks (set `MASK_TOPIC=rt/model/mask` to enable) |

**Subscribed Topics:**

| Topic (default) | Message Type | Description |
|----------------|--------------|-------------|
| `rt/camera/dma` | `edgefirst_msgs/DmaBuf` | Zero-copy DMA buffer metadata from camera |
| `rt/camera/info` | `sensor_msgs/CameraInfo` | Camera resolution and calibration info |

**ROS2 Bridge Integration:**

```bash
# Start camera node
edgefirst-camera --jpeg &

# Start model node
edgefirst-model --model model.tflite --visualization &

# Start zenoh-bridge-dds for ROS2 compatibility
zenoh-bridge-dds

# View with ROS2 tools
ros2 topic list
ros2 topic echo /rt/model/boxes2d
rviz2  # Visualize detections
```

### Hardware-Accelerated Inference Pipeline

The model node leverages NXP i.MX8M Plus hardware acceleration for maximum performance:

```mermaid
graph LR
    subgraph "Input"
        DMA["Camera DMA<br/>YUYV Buffer"]
    end

    subgraph "Preprocessing (G2D)"
        Convert["Format Convert<br/>YUYV → RGB"]
        Scale["Resize<br/>1920×1080 → 640×640"]
        Normalize["Normalize<br/>0-255 → 0.0-1.0"]
    end

    subgraph "Inference (NPU)"
        Model["TFLite Model<br/>Object Detection"]
    end

    subgraph "Postprocessing (CPU)"
        Decode["YOLO Decoder<br/>Extract Boxes"]
        NMS["Non-Max Suppression<br/>Filter Overlaps"]
        Track["ByteTrack<br/>Object Tracking"]
    end

    subgraph "Output"
        Pub["Zenoh Publisher<br/>rt/model/output"]
    end

    DMA --> Convert
    Convert --> Scale
    Scale --> Normalize
    Normalize --> Model
    Model --> Decode
    Decode --> NMS
    NMS --> Track
    Track --> Pub
```

**Performance Benefits:**

- **12-18ms total latency** (camera capture to published results)
- **30+ FPS** on NXP i.MX8M Plus with 640×640 models
- **Zero memory copies** from camera to inference
- **Hardware-accelerated** format conversion, scaling, and inference
- **Concurrent execution** - inference overlaps with next frame preprocessing

### Zero-Copy DMA Integration

The model node consumes DMA buffers directly from the camera service without copying:

```rust
// Pseudo-code: How the model node processes DMA buffers
subscriber.on_message(|dma_msg: DmaBuf| {
    // Map camera DMA buffer (zero-copy)
    let camera_buffer = map_dma_buffer(dma_msg.fd, dma_msg.offset)?;

    // Allocate DMA buffer for preprocessed image
    let preprocessed = allocate_dma_buffer(model_width, model_height)?;

    // Hardware-accelerated preprocessing (G2D)
    g2d_convert_and_scale(camera_buffer, preprocessed)?;

    // Run inference on NPU (zero-copy DMA input)
    let outputs = model.infer(preprocessed)?;

    // Decode and publish results
    let boxes = decode_yolo_outputs(outputs)?;
    publish_detections(boxes)?;
});
```

**Latency Breakdown (Typical 640×640 YOLO Model):**

| Stage | Time | Hardware |
|-------|------|----------|
| DMA buffer mapping | < 0.1ms | CPU |
| G2D preprocessing | 2-3ms | G2D Engine |
| NPU inference | 8-12ms | NPU |
| Decode + NMS | 1-2ms | CPU |
| ByteTrack tracking | 0.5-1ms | CPU |
| Zenoh publish | < 0.1ms | CPU |
| **Total** | **12-18ms** | - |

### Object Detection and Tracking

**Supported Model Architectures:**

- **YOLO**: YOLOv5, YOLOv8, YOLOv10, YOLO11, YOLO26 (detection and segmentation)

**Detection Features:**

- Configurable score threshold (default: 0.45)
- Non-maximum suppression (NMS) with tunable IoU threshold
- Max detections limit to control performance
- Label offset for class index mapping

**ByteTrack Multi-Object Tracking:**

ByteTrack is a state-of-the-art tracking algorithm that maintains object identity across frames:

```mermaid
graph TB
    subgraph "Frame N"
        Det["Detections<br/>YOLO Output"]
        High["High Confidence<br/>Score ≥ 0.7"]
        Low["Low Confidence<br/>0.45 ≤ Score < 0.7"]
    end

    subgraph "Tracking"
        Active["Active Tracks"]
        Lost["Lost Tracks"]
        Match1["First Match<br/>High Conf ↔ Active"]
        Match2["Second Match<br/>Low Conf ↔ Lost"]
        Predict["Kalman Prediction"]
    end

    subgraph "Output"
        Tracks["Tracked Objects<br/>with UUIDs"]
    end

    Det --> High
    Det --> Low

    Predict --> Match1
    High --> Match1
    Match1 --> Active

    Low --> Match2
    Active --> Lost
    Lost --> Match2
    Match2 --> Active

    Active --> Tracks
```

**Tracking Configuration:**

```bash
edgefirst-model \
  --model model.tflite \
  --track \
  --track-high-conf 0.7        # High confidence threshold
  --track-iou 0.25             # IoU threshold for association
  --track-update 0.25          # Kalman filter update factor
  --track-extra-lifespan 0.5   # Seconds to keep lost tracks
```

### Segmentation and Masking

The model node supports both semantic segmentation and instance segmentation:

**Semantic Segmentation:**

- Full-frame pixel-wise classification
- Output: Class label per pixel
- Example: Road, sidewalk, building, sky

**Instance Segmentation:**

- Object detection + per-instance masks
- Output: Bounding box + mask for each detected object
- Example: YOLOv8-seg

### Unified Model Output (`rt/model/output`)

The `rt/model/output` topic publishes an `edgefirst_msgs/Model` message that combines detection boxes, segmentation masks, and detailed timing information in a single message. It is published on every frame for **all model types** (detection, semantic segmentation, and instance segmentation).

**How it handles each model type:**

| Model Type | `boxes` field | `masks` field |
|------------|---------------|---------------|
| Detection only (e.g. YOLOv8n) | Detected boxes with scores, labels, tracks | Empty |
| Semantic segmentation (e.g. DeepLab) | Empty | Single `Mask` with `boxed: false` |
| Instance segmentation (e.g. YOLOv8n-seg) | Detected boxes | One `Mask` per box with `boxed: true` |

For instance segmentation, each entry in the `masks` array corresponds to the box at the same index in the `boxes` array, with `boxed: true` indicating the mask is cropped to the bounding box region.

The `Model` message also includes per-stage timing fields (`input_time`, `model_time`, `output_time`, `decode_time`) using `Duration` instead of the `Time` type used by the older `Detect` message, providing clearer semantics for duration measurements.

### Legacy Topics (Opt-In)

Prior to the unified output, detection results and segmentation masks were published on separate topics. These legacy topics are now **disabled by default** and must be explicitly enabled via environment variable or CLI flag:

- **`rt/model/boxes2d`** (`edgefirst_msgs/Detect`) — Enable with `DETECT_TOPIC=rt/model/boxes2d` or `--detect-topic rt/model/boxes2d`. Contains bounding boxes with scores, labels, and tracks, plus timing fields using `Time`. Does not include any mask data.
- **`rt/model/mask`** (`edgefirst_msgs/Mask`) — Enable with `MASK_TOPIC=rt/model/mask` or `--mask-topic rt/model/mask`. Publishes a single full-frame mask for semantic segmentation models only.

New subscribers should prefer `rt/model/output` which provides a complete view of all model outputs in a single message.

---

## Configuration

### Command-Line Options

```bash
edgefirst-model --help
```

**Essential Options:**

- `--model <PATH>` - Path to TFLite model file (required)
- `--engine <ENGINE>` - Inference engine: `npu`, `gpu`, `cpu` (default: `npu`)
- `--threshold <FLOAT>` - Detection score threshold (default: `0.45`)
- `--iou <FLOAT>` - NMS IoU threshold (default: `0.45`)
- `--max-boxes <N>` - Maximum detections per frame (default: `100`)

**Tracking Options:**

- `--track` - Enable ByteTrack object tracking
- `--track-high-conf <FLOAT>` - High confidence threshold (default: `0.7`)
- `--track-iou <FLOAT>` - Tracking IoU threshold (default: `0.25`)
- `--track-update <FLOAT>` - Kalman filter update factor (default: `0.25`)
- `--track-extra-lifespan <SECS>` - Lost track lifespan in seconds (default: `0.5`)

**Filtering Options:**

- `--classes <CLASSES>` - Space-separated class label names to include in output (default: all)

**Topic Configuration:**

- `--camera-topic <TOPIC>` - Camera DMA topic (default: `rt/camera/dma`)
- `--output-topic <TOPIC>` - Unified model output topic (default: `rt/model/output`)
- `--info-topic <TOPIC>` - Model info topic (default: `rt/model/info`)
- `--detect-topic <TOPIC>` - Legacy detection topic (default: empty/disabled)
- `--mask-topic <TOPIC>` - Legacy mask topic (default: empty/disabled)

**Visualization:**

- `--visualization` - Enable Foxglove visualization messages
- `--visual-topic <TOPIC>` - Visualization topic (default: `rt/model/visualization`)
- `--labels <MODE>` - Label annotation mode: `index`, `label`, `score`, `label-score`, `track`
- `--camera-info-topic <TOPIC>` - Camera info topic for resolution (default: `rt/camera/info`)

**Zenoh Configuration:**

- `--mode <peer|client|router>` - Zenoh participant mode (default: `peer`)
- `--connect <ENDPOINT>` - Connect to Zenoh router
- `--listen <ENDPOINT>` - Listen for Zenoh connections
- `--no-multicast-scouting` - Disable multicast discovery

**Debugging:**

- `--tracy` - Enable Tracy profiler integration

**See full options:** `edgefirst-model --help`

### Environment Variables

Configuration can also be set via environment variables. See `model.default` for the full list of supported variables.

```bash
export MODEL=/models/yolov8n.tflite
export ENGINE=npu
export THRESHOLD=0.5
export TRACK=true
export OUTPUT_TOPIC=rt/model/output
export DETECT_TOPIC=rt/model/boxes2d  # Re-enable legacy detect topic

edgefirst-model  # Uses environment configuration
```

### Model Metadata Configuration

Models can include embedded metadata (edgefirst.yaml) for automatic configuration:

```yaml
name: YOLOv8n Object Detection
version: 1.0.0
description: Nano YOLO model optimized for NXP i.MX8M Plus
author: Ultralytics
license: AGPL-3.0

outputs:
  - type: detection
    name: output0
    shape: [1, 84, 8400]
    format: yolo

  - type: boxes
    name: boxes
    shape: [1, 8400, 4]

  - type: scores
    name: scores
    shape: [1, 8400, 80]
```

**Embedding Metadata:**

```bash
# Add edgefirst.yaml to existing TFLite model
zip model.tflite edgefirst.yaml
```

---

## Profiling

The model node includes Tracy profiler integration for performance analysis. See [CONTRIBUTING.md](CONTRIBUTING.md#profiling-and-performance-analysis) for setup instructions.

**Quick Start:**

```bash
# Run with Tracy profiler enabled
edgefirst-model --tracy --model model.tflite

# Connect Tracy profiler GUI to analyze inference timing
# Download Tracy: https://github.com/wolfpld/tracy/releases
```

---

## Platform Support

### Tested Platforms

| Platform | Architecture | Status | Notes |
|----------|--------------|--------|-------|
| Maivin + Raivin | ARM64 (i.MX8M Plus) | ✅ Fully Supported | Primary target, NPU + G2D acceleration |
| NXP i.MX8M Plus EVK | ARM64 | ✅ Supported | Hardware acceleration available |
| Generic ARM64 Linux | ARM64 | ⚠️ Partial | Software fallback (no NPU/G2D) |
| x86_64 Linux | x86_64 | ⚠️ Development Only | CPU inference only, slower |

### Model Compatibility

**Supported Formats:**

- **TensorFlow Lite** (.tflite) - Primary format, full NPU support
- **RTM** (.rtm) - Experimental, feature-gated (`--features rtm`)

**Inference Engines:**

- **NPU** (default) - NXP i.MX8M Plus Neural Processing Unit (fastest)
- **GPU** - GPU delegate (moderate performance)
- **CPU** - Software fallback (slowest, but universal)

**Model Architecture Support:**

- Object Detection: YOLO (v5, v8, v10, v11, v26)
- Instance Segmentation: YOLOv8-seg
- Semantic Segmentation: DeepLab

---

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/EdgeFirstAI/model.git
cd model

# Build with all features
cargo build --release

# Run tests
cargo test

# Run benchmarks (requires hardware)
cargo bench

# Generate documentation
cargo doc --no-deps --open
```

### Project Structure

```
model/
├── src/
│   ├── main.rs          # Application entry, Zenoh session, main inference loop
│   ├── lib.rs           # Public library interface, TrackerBox wrapper, DmaBuf handling
│   ├── model.rs         # Model trait, enum_dispatch, model config guessing
│   ├── tflite_model.rs  # TFLite model loading and inference via NPU
│   ├── rtm_model.rs     # RTM/VAAL model support (feature-gated)
│   ├── buildmsgs.rs     # Zenoh message construction (CDR serialization)
│   ├── masks.rs         # Segmentation mask publishing (legacy mask topic)
│   ├── args.rs          # CLI argument parsing (Clap)
│   └── fps.rs           # FPS monitoring
├── tflitec-sys/         # TFLite C API FFI bindings (internal)
├── benches/             # Divan benchmarks
├── Cargo.toml           # Project dependencies
└── README.md            # This file
```

**See also:**

- [ARCHITECTURE.md](ARCHITECTURE.md) - Internal architecture and design documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines and development workflow
- [SECURITY.md](SECURITY.md) - Security policy and vulnerability reporting

---

## Troubleshooting

### Common Issues

**Problem: "Model file not found"**

```bash
# Verify model path
ls -lh /path/to/model.tflite

# Use absolute path
edgefirst-model --model /absolute/path/to/model.tflite
```

**Problem: "NPU delegate initialization failed"**

```bash
# Check NPU delegate library
ls -lh /usr/lib/libvx_delegate.so

# Fallback to CPU
edgefirst-model --model model.tflite --engine cpu
```

**Problem: "No camera frames received"**

```bash
# Ensure camera node is running
edgefirst-camera --camera /dev/video0 &

# Check Zenoh connectivity
zenoh-cli query "/rt/**"

# Verify DMA topic matches
edgefirst-model --model model.tflite --camera-topic rt/camera/dma
```

**Problem: "Low FPS or high latency"**

```bash
# Check model complexity
# Smaller models (e.g., YOLOv8n) run faster than larger ones (YOLOv8x)

# Reduce max detections
edgefirst-model --model model.tflite --max-boxes 50

# Disable tracking if not needed
edgefirst-model --model model.tflite  # No --track flag

# Use NPU engine
edgefirst-model --model model.tflite --engine npu
```

**Problem: "Tracking IDs unstable"**

```bash
# Increase track lifespan for occlusions
edgefirst-model --model model.tflite --track --track-extra-lifespan 1.0

# Reduce IoU threshold for more lenient matching
edgefirst-model --model model.tflite --track --track-iou 0.15

# Increase Kalman update factor for smoother prediction
edgefirst-model --model model.tflite --track --track-update 0.1
```

**Problem: "Segmentation masks empty or incorrect"**

```bash
# Verify model supports segmentation
# Model must output mask coefficients or full masks

# Check class filter
edgefirst-model --model model.tflite --classes ""  # All classes

# Check unified output for mask data
z_sub -k "rt/model/output"
```

### Logging

```bash
# Set log level
RUST_LOG=debug edgefirst-model --model model.tflite

# Filter specific module
RUST_LOG=edgefirst_model::tflite_model=trace edgefirst-model --model model.tflite

# View systemd journal logs
journalctl -u edgefirst-model -f
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

**Third-Party Components:** See [NOTICE](NOTICE) for required attributions.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code style guidelines
- Development workflow
- Pull request process
- Testing requirements

**Found a bug?** [Open an issue](https://github.com/EdgeFirstAI/model/issues)

**Have a feature request?** [Start a discussion](https://github.com/orgs/EdgeFirstAI/discussions)

---

## Support

**Community Resources:**

- 📚 **Documentation**: https://doc.edgefirst.ai/latest/perception/
- 💬 **Discussions**: https://github.com/orgs/EdgeFirstAI/discussions
- 🐛 **Issues**: https://github.com/EdgeFirstAI/model/issues

**Commercial Support:**

- **EdgeFirst Studio**: Integrated deployment, monitoring, and management
- **Professional Services**: Training, custom development, enterprise support
- **Contact**: support@au-zone.com

---

## Acknowledgments

Built with by the EdgeFirst team at [Au-Zone Technologies](https://au-zone.com)

For questions or support, see our [Contributing Guide](CONTRIBUTING.md) or open an issue on [GitHub](https://github.com/EdgeFirstAI/model/issues).

**Powered by:**

- [Zenoh](https://zenoh.io/) - Efficient pub/sub middleware
- [Tokio](https://tokio.rs/) - Async runtime
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Lightweight ML inference
- [ROS2 CDR](https://design.ros2.org/) - Message serialization
- [NXP i.MX8](https://www.nxp.com/products/processors-and-microcontrollers/arm-processors/i-mx-applications-processors/i-mx-8-applications-processors:IMX8-SERIES) - Hardware acceleration platform
