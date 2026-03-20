# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The **4D SDK** is a Python library for interfacing with stereo 4D cameras. It provides a `Stereo4DCameraHandler` class that manages camera connections, image streaming, and camera calibration. The SDK integrates with ROS2 for use in robotics applications.

## Development Commands

```bash
# Install dependencies
pip3 install -r requirements.txt

# Install as editable package
pip3 install -e .

# Run examples (no camera needed for syntax/import checks)
python3 examples/simple_stream.py
python3 examples/image_saver.py
python3 examples/stream_monitor.py
python3 examples/image_viewer.py       # requires ROS2
python3 examples/stereo_4d_ros2.py     # requires ROS2

# Camera calibration (requires ROS2 + camera_calibration package)
ros2 run camera_calibration cameracalibrator --no-service-check --approximate 0.1 \
  --size 8x6 --square 0.03 \
  right:=stereo_4d/right_raw/image left:=stereo_4d/left_raw/image \
  right_camera:=stereo_4d/right_raw left_camera:=stereo_4d/left_raw \
  --fix-principal-point
```

No formal test suite exists. Verify changes by running examples with a connected camera and checking frame reception/calibration loading.

## Core Architecture

### `stereo_4d/` module

**`Stereo4DCameraHandler`** (`stereo_4d/stereo_4d.py`) — central class managing the full camera lifecycle:
- Connects to camera over ZMQ (default `172.31.1.77:5555` for SUB, `:5556` for PUB)
- Background `RepeatTimer` threads handle: frame reception (`__receiver`, 10ms interval), status heartbeats (`__status_sender`, 1s interval), and connection monitoring (`__run`, 1s interval)
- `start()` kicks off `__start_sequence` in a separate thread: pings camera IP, sends start command via PUB socket, waits for status + first frame
- Frames arrive as JSON with hex-encoded JPEG data on the SUB socket, decoded via `cv2.imdecode`
- The side-by-side stereo image (left|right concatenated horizontally) can be rectified internally when `rectify_internally=True` and stereo maps are computed from received intrinsics
- Auto-reconnects when status heartbeat times out (20s) or frame drop exceeds `max_frame_drop_percent` (95%)

**Data flow:** Camera → ZMQ SUB → `__receiver` → `__handle_message` → dispatches by `msg_type`:
- `"frame"` → decode image, optionally rectify, store in `__last_frame`, fire `__frame_callback`
- `"intrinsics"` → populate `left_camera_info`/`right_camera_info`, compute stereo rectification maps via `cv2.stereoRectify` + `cv2.initUndistortRectifyMap`
- `"status"` → update heartbeat timestamp

**`Stereo4DFrame`** — holds `timestamp`, `frame_id`, and `image` (numpy array).

**`Stereo4DCameraInfo`** — holds per-camera calibration: `k` (intrinsics), `d` (distortion), `r`, `p`, `extrinsic_matrix`, `rect_matrix`.

### Examples

- `simple_stream.py` — minimal: instantiate handler, start, poll `get_last_frame()`
- `image_saver.py` — captures frames every 2s, saves PNGs, shows live preview with OpenCV window
- `stream_monitor.py` — monitors stream health, logs dropout events to file
- `image_viewer.py` — ROS2 node with interactive stereo topic switching
- `stereo_4d_ros2.py` — full ROS2 driver publishing stereo images

### Calibration (`calib/`)

- `stereo_calibration.py` — stereo camera calibration utility
- `calib_ost_to_yaml.py` — converts OST calibration format to YAML
- `how-to-calib.md` — step-by-step calibration guide using ROS2 camera_calibration

## Key Implementation Notes

- Image resolution is currently hardcoded to 1920x1080 in `__init_stereo_rectify_maps` (marked with TODO)
- Stereo width is computed as `full_width // 2` (left and right images are concatenated horizontally)
- `rectify_stereo_images()` returns a 4-tuple: `(rectified_left, rectified_right, left_rect_k, right_rect_k)`
- `CustomLogger` wraps Python's `logging` with `throttle_sec` and `log_once` options to control log verbosity
- Package requires Python >= 3.8; core deps: `opencv-python`, `numpy`, `zmq`; optional: `scipy` (calibration), `rclpy`/`cv_bridge` (ROS2)
