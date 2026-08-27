# 4D SDK

Python SDK for the **4D Stereo Camera**. Provides stereo image streaming, rectification, and calibration utilities over ZMQ. Optional ROS2 integration for robotics applications.

## Requirements

- Python >= 3.8
- Core: `opencv-python`, `numpy`, `zmq`
- Optional: `scipy` (calibration), `rclpy` + `cv_bridge` (ROS2)

## Installation

```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Install as editable package
pip3 install -e .
```

For ROS2 features, also install:

```bash
sudo apt install ros-${ROS_DISTRO}-compressed-image-transport ros-${ROS_DISTRO}-image-transport
```

## Quick Start

```python
from stereo_4d import Stereo4DCameraHandler
import time

handler = Stereo4DCameraHandler(show_stream=True, rectify_internally=True)
handler.start(wait=True)

try:
    while True:
        frame = handler.get_last_frame()
        if frame is not None:
            print(f"Frame {frame.frame_id}: {frame.image.shape}")
        time.sleep(0.1)
except KeyboardInterrupt:
    handler.stop()
```

## Examples

| Example | Description | Requires |
|---------|-------------|----------|
| `examples/simple_stream.py` | Minimal streaming with live preview | Camera |
| `examples/image_saver.py` | Captures frames periodically and saves PNGs | Camera |
| `examples/stream_monitor.py` | Monitors stream health and logs dropouts | Camera |
| `examples/image_viewer.py` | Interactive stereo topic viewer | Camera + ROS2 |
| `examples/stereo_4d_ros2.py` | Full ROS2 driver publishing stereo images | Camera + ROS2 |
| `examples/exposure_pair_viewer.py` | Interactive dual-exposure tuner: both exposures side by side, retuned live | Camera |

Run any example with:

```bash
python3 examples/simple_stream.py
```

## Architecture

**`Stereo4DCameraHandler`** manages the full camera lifecycle:

- Connects to the camera over ZMQ (default `172.31.1.77:5555`)
- Background threads handle frame reception, status heartbeats, and connection monitoring
- Frames arrive as JSON with hex-encoded JPEG data, decoded via OpenCV
- Stereo rectification can be applied internally when `rectify_internally=True`
- Auto-reconnects on heartbeat timeout or excessive frame drops

**Key classes:**

- `Stereo4DFrame` — holds `timestamp`, `frame_id`, `image` (numpy array) and `exposure`
- `Stereo4DCameraInfo` — per-camera calibration data (intrinsics, distortion, rectification)

## Dual Exposure

One exposure cannot hold both a bright light and the room around it. The camera can alternate
two separately correct exposures — a short one metered for the light, a long one for the rest —
and tags every frame with which one it is. Both arrive interleaved on the same stream, so the
handler keeps the newest of each:

```python
handler.set_exposure_pair(short_us=4000, long_us=25000)

short = handler.get_last_frame("short")
long_frame = handler.get_last_frame("long")
print(short.exposure_label, short.exposure_time_us)   # "short" 4013

handler.get_exposure_fps()     # {"short": 9.8, "long": 9.9} - each is half the frame rate
handler.get_exposure_stats()   # targets, measured exposures, out_of_sync/cadence percent
```

Both targets are absolute microseconds pushed straight at the sensor with the AE off, so a
change lands on the **next frame** and nothing ever rebuilds the camera. Each exposure is
delivered at half the capture rate — 15 Hz each at 30 fps — and both are clamped below the
frame period (31333us at 30 fps).

A frame caught while an exposure change was still landing is labelled `"unknown"` rather than
guessed at, and `get_last_frame("short"|"long")` never returns one. `get_exposure_stats()`
reports `cadence_percent` (how often a frame wore the exposure that was asked for) and
`out_of_sync_percent` (how often a pair was dropped for the eyes being too far apart).

Tune both exposures interactively against a live scene with:

```bash
python3 examples/exposure_pair_viewer.py --ip 172.31.1.77
```

## Tests

```bash
python3 tests/test_exposure_frame.py   # no camera needed
```

## Calibration

Calibration tools are in `calib/`. See [`calib/how-to-calib.md`](calib/how-to-calib.md) for a step-by-step guide using ROS2's `camera_calibration` package.

```bash
# Example: run stereo calibration with ROS2
ros2 run camera_calibration cameracalibrator --no-service-check --approximate 0.1 \
  --size 8x6 --square 0.03 \
  right:=stereo_4d/right_raw/image left:=stereo_4d/left_raw/image \
  right_camera:=stereo_4d/right_raw left_camera:=stereo_4d/left_raw \
  --fix-principal-point
```
