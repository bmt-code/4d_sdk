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

# Camera calibration (capture, filter, solve, check, deploy)
python3 calibration/calibrate.py
python3 calibration/calibrate.py --images examples/images_stereo_4d --grid 9x6 --square 25 --offline

```

`python3 tests/test_capture_smoke.py` draws the capture window without a camera, and
`python3 -m pyflakes calibration/*.py` catches the rest: the capture loop cannot be run on a
desk, so a rename that leaves it calling a drawing function with an argument nobody set
otherwise only surfaces when someone is stood in front of the camera. Run both after
touching `capture.py`.

There is no full test suite. `python3 tests/test_exposure_frame.py` covers the dual-exposure
parsing and bookkeeping without a camera, and the calibration pipeline can be exercised end
to end on the committed images with
`python3 calibration/calibrate.py --images examples/images_stereo_4d --grid 9x6 --square 25 --yes --no-install`.
Everything else is verified by running the examples against a connected camera and checking
frame reception/calibration loading.

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
(AE capping the brightest 2%, one stream at 30Hz) and `"ae_dual"` (the ISP's unmerged HDR: two
AGC channels side by side, beam on `short` and room on `long`, 15Hz each). `normal` is the
default so a client that has never heard of labels is not handed a stream where half the frames
are deliberately wrong. Switching among `manual`/`normal`/`highlight` costs a frame — those are
runtime controls. Crossing the `ae_dual` boundary **rebuilds the camera** (~1.5s): `HdrMode` only
applies at configure time, and writing it live makes the ISP return `HdrChannelNone` frames that
the pairing drops, seen as one pane blinking. The tuning the HDR channels need is still written
at every build regardless of mode.

Under the single-profile modes every frame is labelled `"short"` (nothing to tell apart). The
manual targets keep being recorded whatever runs, so `set_auto_exposure("manual")` resumes
them. `get_exposure_stats()["ae_mode"]` reports the current mode and `["hdr_active"]` whether
the two channels are really running. Do **not** alternate `AeConstraintMode` frame by frame:
`rpi.agc` filters toward its target at speed 0.2, so a two-frame square wave comes back as its
mean and both exposures land in the same place.

**`Stereo4DCameraInfo`** — holds per-camera calibration: `k` (intrinsics), `d` (distortion), `r`, `p`, `extrinsic_matrix`, `rect_matrix`.

### Examples

- `simple_stream.py` — minimal: instantiate handler, start, poll `get_last_frame()`
- `image_saver.py` — captures frames every 2s, saves PNGs, shows live preview with OpenCV window
- `stream_monitor.py` — monitors stream health, logs dropout events to file
- `image_viewer.py` — ROS2 node with interactive stereo topic switching
- `stereo_4d_ros2.py` — full ROS2 driver publishing stereo images
- `exposure_pair_viewer.py` — interactive dual-exposure tuner: both exposures side by side, retuned live over ZMQ
- `check_calibration.py` — live checkerboard check of the deployed calibration; its
  `find_checkerboard`, `points_3d_from_stereo` and `check_grid_quality` are reused by
  `calibration/verify.py`

### Calibration (`calibration/`)

**Stereo placements ask for the board upright** (`Target.roll_deg = 90`): the strip both
cameras see is narrow and tall, so an upright board reaches the overlap where a landscape
one does not. `Target.rvec()` is the single source of that convention -- roll in the board's
own plane, then yaw about the camera's vertical. Anything rebuilding a board at a placement
must use it; composing from `yaw_deg` alone silently drops the roll and puts a landscape
board where an upright one was asked for, which broke two tests at once.

**Guide geometry, two traps.** The drawn guide is `Target.guide_full()` -- the whole
board, a square wider on every side than the inner corners a detector reports -- so any
inset or fit check sized from `guide_corners()` is wrong by that square: doing so once put
92 of 108 guides partly off the sensor while the test said all was well. Placements are
now clamped by `Target.fit_into_frame()`, which re-projects and nudges until the drawn
board is inside, because a board projects larger and more skewed the further off-axis it
sits, so a margin measured at the centre still clips at the edges. Second trap:
`np.linspace(a, b, 1)` returns `[a]`, so `TargetPlan._spread` exists to put a single
position in the middle of the frame rather than against the margin -- which is where the
one close-up frame, the only thing pinning corner distortion, least wants to be.

**The shutter has three gates beyond the guide**, and two of them have bitten already:
the board must be still, sharp, and have *moved since the last saved frame*. That last one
is measured on the **worst-moving corner**, never an average — yaw pivots the board about
its centre, so a genuine pose change shifts the median only 20-30 px and an average-based
gate refuses the very pose it just asked for, sitting at a full stillness gauge while the
hint reads "change the pose". And the shot cooldown starts when a frame is *saved*, not
when one is attempted.

**Capture tolerances trade against pose discrimination.** The three poses at one placement
are separated mostly by *where yaw puts the board's projected centre*, not by its shape, so
forgiving a centring error forgives most of the separation with it. Position was
deliberately loosened anyway (`POSITION_WEIGHT = 0.5` charges a centring error at half, so
position tolerance is twice the shape tolerance -- about 9% of the board's width). The cost,
recorded band by band in `test_the_three_poses_stay_distinct`:

- **square on is still refused for a turned target** -- the tilt requirement -- in every
  band but `far`, where the board subtends a quarter of the frame and its yaw skews the
  corners by less than the position slack allows;
- **which way the board is turned is no longer enforced.** A symmetric checkerboard yawed
  one way and rotated 180 deg in its own plane projects the same quad as the opposite yaw,
  so direction only ever showed up as that small centre shift.

Two things that are *not* knobs on this: raising `ACCEPT_SCORE` alone (it gates distance
and yaw too), and widening the pose yaw (the ambiguity is geometric, not a matter of
separation). Roll is free by design -- `derotate` takes it out before scoring, since a
rolled board is exactly as good a calibration frame and charging it at the board's
half-diagonal spent the whole tolerance on 4.5 degrees of wrist.

**Capture protocol** (determined empirically on this rig, and what the pipeline is built
around): two close-ups with the board hard against the left and right of the field —
one eye each, it cannot be in both at that range, and these are what fit focal length and
corner distortion; then both-eye frames working the image corners at several distances,
no deeper than ~1800 mm. Hold the board still per shot: the eyes are not exposed
simultaneously, so a moving board lands in different places in the two images. Omitting
the close-ups measurably biases fx (2550 vs a reference 2284 on one dataset).

**`calibration/`** is the all-in-one pipeline and the entry point people should use:
`python3 calibration/calibrate.py` walks capture → filter → solve → check with no paths
edited in source. Modules:

- `calibrate.py` — CLI and wizard; owns the session directory and the prompts. The
  image-source prompt lists past sessions (newest first by mtime, borrowed image
  folders flagged) to pick by number, and `resolve_source()` accepts either a
  session directory or a raw image folder wherever a path is given, `--images`
  included.
- `capture.py` — stage 1, guided capture: shows **only the eye the current target belongs
  to**, with the header and footer in their own bands of canvas above and below rather
  than painted over the frame. SPACE forces a shot; captures raw
  (`rectify_internally=False`). Both this and `verify.py`
  render at `--preview-width` (1600) instead of the sensor's 3840, and gate every live
  board search behind `find_corners_fast` (half-size, FAST_CHECK only, ~6 ms). The
  ungated full search costs ~300 ms an eye when there is no board to find and is what
  makes the windows feel stuck — never call `find_corners`/`find_checkerboard` straight
  from a display loop.
- `sound.py` — two cue tones, rendered to WAV once and played by whichever of
  `paplay`/`pw-play`/`aplay`/`ffplay` exists, spawned detached so nothing can stall the
  capture loop (`play()` returns in well under a millisecond). Every failure path is quiet:
  no player, no audio, no writable temp dir and capture just runs silently. `near` fires on
  the rising edge of a matched pose only, with a repeat guard, since the score dithers
  around the threshold and would otherwise machine-gun.
- `window.py` — `open_window` and `bring_to_front`, shared by both views.
  `bring_to_front` toggles `WND_PROP_TOPMOST` on and straight off, which raises the
  window without pinning it; it must be called *after* the first `imshow` (an empty
  window has nothing to raise) and the geometry survives the toggle. The Qt5 backend
  implements the setter but not the getter, so the property never reads back.
- `quality.py` — stage 2, board detection (FAST_CHECK then full, sub-pixel refined) and
  the blur filter. A frame is only rejected when **neither** eye saw the board.
  Sharpness is Laplacian variance **inside the board's bounding box**,
  thresholded at `--blur-ratio` × the set's median — full-frame variance is
  scene-dominated and no fixed constant works. Corners detected here are handed to
  stage 3, so filtering costs nothing extra. `--include`/`--exclude` override both
  filters by frame number (`27`, `027`, `frame_027.png` and ranges like `27-34` all
  work); an included frame the normal detector missed is retried with
  `findChessboardCornersSB`, and one whose board is not in both eyes is reported as
  impossible rather than kept.
- `solve.py` — stage 3. **Each eye's intrinsics are fitted from every frame that eye
  saw the board in, including the frames where only one eye did**; only the both-eye
  frames go into `stereoCalibrate` (with `CALIB_FIX_INTRINSIC`). The mono fit is local
  (`calibrate_eye`) because `StereoCalibration.calibrate_cameras` assumes both eyes
  contribute the same frames; the stereo step and `save_calibration` are still that
  class's. Defaults to the 5-coefficient distortion model (`--rational` restores the
  14-coefficient one, which fits degenerately). Pruning narrows what an image is used
  for rather than discarding it: an eye over `--max-mono-reproj` is dropped for that eye,
  a pair over `--max-reproj` **keeps its intrinsics and leaves the extrinsics fit**, and
  only a frame both eyes fail on is rejected.
- `verify.py` — stage 4, rectifies with the **fresh** calibration (not the camera's
  deployed one), draws epipolar lines, and when a board is in shot reports y-RMS
  (vertical disparity of matched corners) plus the metric grid check reused from
  `examples/check_calibration.py`. `--offline` pages through saved images instead, and
  is the automatic fallback when the camera does not come up.

Output per run: `calibration/sessions/<name>_<timestamp>/` with `images/` (or a symlink
to the source folder), `rejected/` (symlinks + `reasons.txt`; nothing is moved or
deleted), `stereo_calibration.yaml`, `params.yaml` and `report.txt`. The last prompt
offers to copy the YAML into the sibling `4d_firmware/calib/stereo_calibration.yaml`,
backing up what is there, and then to rsync it onto the camera at
`bmt@<--ip>:~/4d_firmware/calib/stereo_calibration.yaml` (`--send`/`--no-send`/`--send-to`;
`rsync --backup` keeps the unit's previous file). Every send first runs
`ssh-keygen -f ~/.ssh/known_hosts -R <host>`: the unit is reflashed often and a changed host
key otherwise blocks ssh outright. That trades away the host-key check on that address, which
is acceptable on a private link to a reimaged device and should not be copied elsewhere. Both prompts default to no, and `--yes`
alone never deploys. The firmware only reads the calibration at start-up, so the run
prints the `systemctl restart stereo_4d.service` line afterwards.

`calibration/` is self-contained: the calibration maths lives in `solve.py`
(`object_points`, `calibrate_eye`, `stereo_calibrate`, `save_calibration`,
`CALIB_CRITERIA`), and the only outside code it borrows is `find_checkerboard`,
`points_3d_from_stereo` and `check_grid_quality` from `examples/check_calibration.py`.
`calibration/reference_calibration.yaml` is the last known-good calibration of this rig
(fx 2284, baseline 293.4 mm), read by nothing and kept only to compare fresh results
against.

The old `calib/` directory is gone (`git checkout HEAD~ -- calib/` restores it): its
`stereo_calibration.py` is superseded by `calibration/solve.py`, and its ROS2 route
(`calib_ost_to_yaml.py`, `stereo_calib.txt`, `how-to-calib.md`) went with it now that
deployment is rsync-direct to the camera.

## Key Implementation Notes

- Image resolution is currently hardcoded to 1920x1080 in `__init_stereo_rectify_maps` (marked with TODO)
- Stereo width is computed as `full_width // 2` (left and right images are concatenated horizontally)
- `rectify_stereo_images()` returns a 4-tuple: `(rectified_left, rectified_right, left_rect_k, right_rect_k)`
- `CustomLogger` wraps Python's `logging` with `throttle_sec` and `log_once` options to control log verbosity
- Package requires Python >= 3.8; core deps: `opencv-python`, `numpy`, `zmq`, `PyYAML`, `tqdm`; optional: `scipy` (calibration checks), `rclpy`/`cv_bridge` (ROS2)
