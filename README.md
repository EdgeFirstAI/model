# ADIS Detect Service

The ADIS Detect Service is based on the DeepView Detection Service but updated for the two-stage model required for the high-resolution ADIS model.

# Offline

An offline application is provided, detect-offline, which does not use VSL or ZeroMQ for IPC but instead runs detection on a provided list of image files and produces matching files with the extension changed to .txt that contains extended DarkNet annotations.  The extended DarkNet format adds a score field after the label index.  The format is as follows, box coordinates are normalized.  The score field is not present by default, but can be enabled with the --score command-line parameter.

```
label_index score center_x center_y width height
```
