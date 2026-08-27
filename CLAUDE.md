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

There is no full test suite. `python3 tests/test_exposure_frame.py` covers the dual-exposure
parsing and bookkeeping without a camera; everything else is verified by running the examples
against a connected camera and checking frame reception/calibration loading.

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

**`Stereo4DFrame`** — holds `timestamp`, `frame_id`, `image` (numpy array) and `exposure`
(`{"label": "short"|"long"|"unknown", "left": {...}, "right": {...}, "lux"}`), with
`exposure_label` and `exposure_time_us` shortcuts.

**Dual exposure** — opt-in: the camera comes up with its AE running and the first
`set_exposure_pair()` switches it to dual, for as long as that connection lasts. It alternates
a short exposure (for a bright light) and a long one (for the room), interleaved on the same
stream and labelled per frame. Both are absolute
microseconds with the AE off, so every change lands on the next frame and nothing rebuilds the
camera. `set_exposure_pair()` and `set_awb_gains()` send the commands;
`get_last_frame("short"|"long")` returns the newest frame of each, `get_exposure_fps()` their
delivery rates, and `get_exposure_stats()` the camera's telemetry (`out_of_sync_percent` is
how often a pair was dropped for the eyes being too far apart in time; `cadence_percent` is
how often a frame wore the exposure that was asked for). Labels come from
what the sensor measured, never from what was requested. See `examples/exposure_pair_viewer.py`
and the Dual Exposure section of the firmware README.

**Exposure modes** — `set_auto_exposure(mode)` takes four: `"manual"` (two absolute targets,
15Hz each), `"normal"` (AE, stock Pi metering, one stream at 30Hz — the default), `"highlight"`
(AE capping the brightest 2%, one stream at 30Hz) and `"ae_dual"` (both profiles alternating,
15Hz each, `highlight` on `short` and `normal` on `long`). `normal` is the default so a client
that has never heard of labels is not handed a stream where half the frames are deliberately
wrong. The profiles live in the camera's `custom_ov5647.json` and are passed by name; switching
costs a frame, not a rebuild. `ae_dual` has no targets to label against, so each eye learns
what its own AE settles on under each profile and splits on the midpoint.
`get_exposure_stats()["ae_mode"]` reports the current mode, `["ae_learned"]` the learned pair. Every frame is labelled `"short"` while it runs (nothing to tell apart),
the targets keep being recorded so `set_auto_exposure("manual")` resumes them, and
`get_exposure_stats()["auto"]` reports the mode. It is a runtime control, so it costs a frame,
not a rebuild.

**`Stereo4DCameraInfo`** — holds per-camera calibration: `k` (intrinsics), `d` (distortion), `r`, `p`, `extrinsic_matrix`, `rect_matrix`.

### Examples

- `simple_stream.py` — minimal: instantiate handler, start, poll `get_last_frame()`
- `image_saver.py` — captures frames every 2s, saves PNGs, shows live preview with OpenCV window
- `stream_monitor.py` — monitors stream health, logs dropout events to file
- `image_viewer.py` — ROS2 node with interactive stereo topic switching
- `stereo_4d_ros2.py` — full ROS2 driver publishing stereo images
- `exposure_pair_viewer.py` — interactive dual-exposure tuner: both exposures side by side, retuned live over ZMQ

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
