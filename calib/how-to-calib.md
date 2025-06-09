# Camera Calibration Instructions

This document provides instructions for calibrating the cameras in the 4D SDK using ROS 2.

## Prerequisites

Ensure you have the following installed and set up:
- ROS 2 (Humble or later)
- `camera_calibration` package
- The cameras are publishing images on the topics `/camera/fourd/right/image_raw` and `/camera/fourd/left/image_raw`.

## Start the ros2 driver

Disable the rectification flag and start the ros2 driver of stereo 4d.

```bash
python3 stereo_4d_ros2.py
```

## Launch the calibration tool

```bash
ros2 run camera_calibration cameracalibrator --no-service-check --approximate 0.1 --size 8x6 --square 0.03 right:=stereo_4d/right_raw/image left:=stereo_4d/left_raw/image right_camera:=stereo_4d/right_raw left_camera:=stereo_4d/left_raw --fix-principal-point
```
