---
name: Hardware Compatibility Report
about: Report compatibility results for specific hardware platforms
title: '[HARDWARE] '
labels: hardware, compatibility
assignees: ''
---

## Hardware Platform

**Board/SoM:**
- Manufacturer: [e.g., NXP, Raspberry Pi, Custom]
- Model: [e.g., i.MX8M Plus EVK, Maivin 2.0]
- CPU: [e.g., ARM Cortex-A53, x86_64]
- RAM: [e.g., 2GB, 4GB, 8GB]

**NPU/Accelerator:**
- Type: [e.g., VeriSilicon NPU, None]
- Delegate: [e.g., libvx_delegate.so, CPU-only]

**Operating System:**
- Distribution: [e.g., Yocto Kirkstone, Ubuntu 22.04]
- Kernel version: [e.g., 5.15.52, 6.1.0]
- Rust version: [e.g., 1.90.0]

## Test Results

**edgefirst-model version:** [e.g., 0.1.0, commit SHA]

**Model tested:**
```bash
edgefirst-model --model yolov8n.tflite --engine npu --track
```

**Results:**
- [ ] Builds successfully
- [ ] Runs without errors
- [ ] NPU inference works
- [ ] G2D preprocessing works
- [ ] Object detection works
- [ ] Object tracking works
- [ ] Segmentation works
- [ ] Partial functionality (see notes)
- [ ] Does not work (see logs)

## Performance Metrics

**Inference:**
- Model: [e.g., YOLOv8n 640x640]
- NPU inference: ___ ms
- CPU inference: ___ ms
- Total pipeline: ___ ms
- FPS: ___ frames/sec

**Resource Usage:**
- CPU usage: ___% (single core)
- Memory footprint: ___ MB
- Power consumption: ___ W (if measured)

## Known Issues

List any issues, workarounds, or limitations discovered on this platform.

**Example:**
> NPU delegate fails to load YOLOv8x model due to unsupported operators. Falls back to CPU.

## Logs

<details>
<summary>Full logs (click to expand)</summary>

```
# Paste journalctl or stderr output
```

</details>

## Hardware Acceleration

**NPU (VeriSilicon):**
- [ ] Available and working
- [ ] Available but issues (see notes)
- [ ] Not available on this platform
- [ ] Not tested

**G2D (NXP):**
- [ ] Available and working
- [ ] Available but issues (see notes)
- [ ] Not available on this platform
- [ ] Not tested

## Additional Context

Any other details about hardware-specific behavior, configuration requirements, or platform quirks.

## Checklist

- [ ] I have tested with the latest version
- [ ] I have included all hardware details
- [ ] I have provided performance metrics
- [ ] I have attached relevant logs
- [ ] I confirm this report is for EdgeFirst Model compatibility (not a bug report)
