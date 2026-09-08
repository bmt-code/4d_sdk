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
| `examples/check_calibration.py` | Checks a live calibration against a checkerboard | Camera |
| `calibration/calibrate.py` | All-in-one calibration: capture, filter, solve, check | Camera (or a folder) |

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
handler keeps the newest of each.

**It is opt-in.** The camera comes up with its AE running, and `set_exposure_pair()` is what
turns dual on. Call it once; the camera stays in dual for as long as this handler stays connected,
and returns to auto when the connection ends. A client that never calls it gets the single
auto-metered stream the camera has always produced.

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

### Exposure modes

`set_auto_exposure(mode)` picks who chooses the exposure and how many streams come back.
`get_last_frame("short")` and `get_last_frame("long")` both fill in the alternating modes;
in the single-stream modes every frame arrives as `"short"` and the `"long"` side stops
updating.

| mode | streams | rate | what it does |
|---|---|---|---|
| `"manual"` | short + long | 15 Hz each | no AE; the pair last set through `set_exposure_pair()` runs. |
| `"normal"` | one | 30 Hz | stock Raspberry Pi metering: pull the brightest 2% up to mid-grey. The default. |
| `"highlight"` | one | 30 Hz | hold that 2% between 0.2 and 0.4 instead — a bright light stops blowing out, at the cost of a darker room. |
| `"ae_dual"` | short + long | 15 Hz each | the ISP's unmerged HDR: two AGC channels side by side, one metering the beam (`short`) and one the room (`long`). Same shape as `manual`, except the AE picks both exposures so each follows the scene. |

Switching between `manual`, `normal` and `highlight` costs a frame. Crossing into or out of
`ae_dual` **rebuilds the camera** — about 1.5 s with no frames — because `HdrMode` only applies
at configure time. Worth knowing if you drive the mode from a control loop.

Whenever the AE runs it carries its own limits — the two eyes meter independently, and under a
single profile one exposure cannot hold the bright light and the room at once.

```python
handler.set_auto_exposure("normal")      # stock AE, one stream
handler.set_auto_exposure("highlight")   # AE that protects the bright light, one stream
handler.set_auto_exposure("ae_dual")     # two ISP channels at once, short + long
handler.set_auto_exposure("manual")      # back to the pair set through set_exposure_pair
```

Every frame arrives labelled `"short"` while auto runs, since there is nothing to tell apart,
so `get_last_frame("short")` sees the full rate and `get_last_frame("long")` stops updating.
`get_exposure_stats()["ae_mode"]` says which profile is running. Like the pair, it lands on the next
frame — nothing rebuilds.

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

One command does the whole thing -- capture, drop the blurry images, calibrate, and
check the result on a live rectified stream with epipolar lines:

```bash
python3 calibration/calibrate.py

# or from images you already have
python3 calibration/calibrate.py --images examples/images_stereo_4d --grid 9x6 --square 25
```

Every run writes to `calibration/sessions/<name>_<timestamp>/` -- the calibration, a
report with per-image reprojection errors, and symlinks to whatever was rejected and
why. See [`calibration/README.md`](calibration/README.md).

The last prompts offer to copy the result into the local `4d_firmware/calib/` and to
rsync it straight onto the camera at
`bmt@172.31.1.77:~/4d_firmware/calib/stereo_calibration.yaml`.

`calibration/reference_calibration.yaml` is the last known-good calibration for this rig
(fx 2284, baseline 293.4 mm), kept as a sanity check on new results.
