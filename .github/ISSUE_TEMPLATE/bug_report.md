---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Run command: `edgefirst-model --model model.tflite`
2. Observe behavior: ...
3. See error: ...

## Expected Behavior

A clear description of what you expected to happen.

## Actual Behavior

What actually happened (including error messages).

## Environment

**Hardware:**
- Platform: [e.g., NXP i.MX8M Plus, x86_64 desktop]
- NPU: [e.g., VeriSilicon NPU available, CPU-only]
- Camera: [e.g., MIPI CSI-2, USB UVC]

**Software:**
- OS: [e.g., Linux 5.15, Yocto Kirkstone]
- Rust version: [e.g., 1.90.0]
- edgefirst-model version: [e.g., 0.1.0, commit SHA]

**Configuration:**
```bash
# Paste your command-line arguments or configuration
edgefirst-model --model yolov8n.tflite --engine npu --track
```

**Model:**
- Architecture: [e.g., YOLOv8n, SSD]
- Input size: [e.g., 640x640]
- Task: [e.g., detection, segmentation, instance segmentation]

## Logs

<details>
<summary>Logs (click to expand)</summary>

```
# Paste journalctl output or stderr logs
RUST_LOG=debug edgefirst-model --model model.tflite
```

</details>

## Additional Context

Any other context about the problem (e.g., happens only with specific models, works on CPU but not NPU).

## Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have tested with the latest version
- [ ] I have included all relevant logs and environment details
- [ ] I have provided steps to reproduce the issue
