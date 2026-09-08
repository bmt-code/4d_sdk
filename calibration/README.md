# All-in-one stereo calibration

One command from an empty folder to a checked calibration. Nothing to edit in source,
no paths to keep in your head.

```bash
python3 calibration/calibrate.py
```

It walks four stages, asking only what it cannot work out:

1. **Images** — capture from the camera, or reuse a set you already have: past runs
   are listed to pick from by number, and any path works too.
2. **Filter** — find the board in every image, drop the ones with no board and the
   soft ones.
3. **Solve** — each eye's intrinsics from every frame *that eye* saw the board in,
   then the geometry between them from the frames where both did.
4. **Check** — live rectified stream with epipolar lines. Hold the checkerboard up and
   the numbers appear on their own.

Everything a run produces goes into `calibration/sessions/<name>_<timestamp>/`:

```
images/                  captured frames, or a symlink to the folder you pointed at
rejected/                symlinks to what was dropped, plus reasons.txt
stereo_calibration.yaml  the result, in the keys the firmware and the SDK read
params.yaml              board and settings, so a re-run on this folder skips the prompts
report.txt               RMS errors, per-image reprojection, what was rejected and why
```

Nothing is ever moved or deleted — rejects are symlinks.

## Reusing an image set

Answering `f` at the first prompt lists your past runs, newest first, with how many
images each holds and what board it used:

```
Past sessions:
   1  stereo_camera_20260908-175643    69 images   9x6, 25mm, 15 stereo
   2  monoaware_20260908-174240        86 images   9x6, 25mm, 22 stereo   -> .../stereo_camera_20260908-170301/images
   3  stereo_camera_20260908-165717    35 images   9x6, 25mm   -> examples/images_stereo_4d

Pick a number, or type the path to a session or image folder [1]:
```

A session whose images are borrowed from elsewhere says where they live. Instead of a
number you can type a path, and either form is accepted: a session directory (its
`images/` is found for you) or a plain folder of side-by-side PNGs. `--images` takes both
too, so a path copied out of a previous run works whichever half you copied. Picking a
session also picks up the board settings it recorded, so those prompts default correctly.

## Capturing

The capture window behaves like `examples/image_saver.py`: a shot every couple of
seconds with a white flash for the shutter. On top of that it tells you whether the
board is currently visible in each eye, so you can see your coverage as you move it.

| Key | |
|---|---|
| `SPACE` | shoot now |
| `q` / `Esc` | done capturing, carry on to the next stage |

The coverage map in the top right of each eye fills a cell in once a saved board has
put corners in it, so you can see which parts of the image still need visiting — the
corners above all.

Both windows raise themselves to the front when the first frame lands, so the preview is
not buried behind the terminal that started it — they are raised, not pinned, so they
behave normally once up.

Both render at `--preview-width` (1600 px) rather than the sensor's 3840, and the
live board indicator runs on a half-size copy with `CALIB_CB_FAST_CHECK` only. The full
search costs ~300 ms an eye when there is nothing to find — most frames, while you are
carrying the board around — which is enough to make the window feel stuck. The half-size
pass costs about 6 ms and, measured over a full session, misses none of the boards the
full search finds.

### The capture protocol

Determined empirically on this rig. Two distinct kinds of frame, and they do different
jobs:

**1. Close-ups, one eye at a time.** Two frames with the board held very close: one
against the left of the field, one against the right. The board does **not** need to be
in both eyes — at that range it cannot be. These fit each eye's focal length and its
distortion out at the image corners, which is the part a stereo-only set leaves
unconstrained. Skipping them is what puts fx 10% out.

**2. Both eyes, corners covered, several distances.** Work the board through the corners
and edges of the frame with both eyes seeing it, repeating at a few distances. **Do not
go past about 1800 mm** — beyond that the board is too small in frame to add anything.
These are the only frames that fit the geometry between the eyes.

**Hold the board still for each shot.** The two eyes are not exposed at the same instant,
so a board in motion lands in different places in the left and right image. Such a frame
looks perfect in each eye alone and is wrong as a pair — see the note on stereo error
below. Raise `--interval` if two seconds does not give you time to settle.

## Filtering

Sharpness is the Laplacian variance measured **inside the board's bounding box**, not
over the whole frame — a full-frame number mostly tells you what else is in the room.
The threshold is a fraction of the set's own median (`--blur-ratio`, default `0.6`), so
it follows your lighting instead of needing a magic constant.

`--include` overrides the filters for frames you know you want:
`--include 27-34,45,55` keeps them whatever the blur check and the pruner say. Names are
loose -- `27`, `027`, `frame_027` and `frame_027.png` all mean the same frame -- and
ranges work. An included frame the normal detector missed gets a second pass with
`findChessboardCornersSB`, which copes with blur and steep angles at about half a second
an eye. If the board is not in **either** eye there is nothing to detect, and the run
says so instead of pretending. `--exclude` is the inverse.

A frame with a low mono error but a high stereo one was captured while the board was
moving: each eye is sharp and fits fine on its own, but the two eyes did not see the
board in the same place. The pruner keeps its intrinsics and takes it out of the
extrinsics on its own — forcing it back in drags the baseline off.

### What each frame is used for

A frame where only one eye saw the board still fits that eye's intrinsics — it is kept,
not rejected. Only a frame where **neither** eye saw it is dropped.

After the first fit, two errors are computed per frame: each eye's own reprojection
error, and — for the both-eye frames — a stereo error, from solving the board pose in the
left eye and pushing it through the rig into the right. Then:

- an eye over `--max-mono-reproj` (default 1.0 px) is dropped for that eye alone;
- a pair whose eyes are each fine but disagree with each other by more than
  `--max-reproj` (default 1.5 px) **leaves the extrinsics fit and keeps its
  intrinsics** — that pattern means the board moved between the two exposures, so the
  frame still says something true about each lens and nothing true about the baseline;
- a frame both eyes fail on is dropped entirely.

Then the fit is re-run once. `report.txt` gives all three errors per frame and what each
one ended up used for.

## Checking

The check rectifies with the calibration you just computed, not with whatever YAML is
currently deployed on the camera. Two numbers, both live:

- **y-RMS** — vertical disparity between matched corners, in pixels. This is the direct
  measure of how well the pair is rectified. Green under 0.5 px, amber under 1.0, red
  above.
- **grid mean / max** — the corners triangulated and compared against the real board,
  in millimetres.

| Key | |
|---|---|
| `s` | save a snapshot into the session |
| `n` / `p` | page through images (offline mode) |
| `q` / `Esc` | finish |

`--offline` reviews the session's own images instead of a live stream, so you can
calibrate and check a folder with no camera on the network. It falls back to this by
itself if the camera does not come up.

## Deploying

Two prompts at the end, both defaulting to no:

1. Copy the result into the sibling `4d_firmware/calib/stereo_calibration.yaml` — the
   tree that gets rsynced to a unit. Whatever was there is backed up first.
2. **Send it to the camera** — rsyncs onto the unit at
   `bmt@172.31.1.77:~/4d_firmware/calib/stereo_calibration.yaml`, which is the copy the
   running firmware actually reads. The host follows `--ip`, so pointing at another unit
   deploys to that one; `--send-to` overrides the whole destination. The file already on
   the camera is kept as `stereo_calibration.yaml.bak-<timestamp>`.

The firmware only reads the calibration at start-up, so it prints the restart line for
you afterwards:

```bash
ssh bmt@172.31.1.77 sudo systemctl restart stereo_4d.service
```

`--send` and `--no-send` decide it without being asked. **`--yes` on its own does not
send** — an unattended run says what it is skipping rather than writing to hardware
quietly.

## Options

```
--ip 172.31.1.77        camera address
--name stereo_camera    session name

--capture               capture instead of asking
--images DIR            calibrate from an existing folder
--interval 2.0          seconds between automatic shots

--no-blur-filter        keep the soft images
--blur-ratio 0.6        drop below this fraction of the median sharpness
--blur-min N            absolute Laplacian-variance floor on top of the ratio
--include 27-34,45      keep these frames whatever the filters say (repeatable)
--exclude 12,19         drop these frames whatever the filters say (repeatable)

--max-mono-reproj 1.0   per-eye pruning threshold in pixels
--grid 9x6              inner corners, columns x rows
--square 25             square size in millimetres
--rational              14-coefficient distortion model (see below)
--no-prune              keep the high-error pairs
--max-reproj 1.5        stereo pruning threshold in pixels
--workers N             detection threads

--offline               review saved images instead of a live stream
--line-spacing 20       displayed pixels between epipolar lines
--preview-width 1600    width the capture and check windows render at
--no-check              skip the check entirely

--install / --no-install    decide the 4d_firmware copy without being asked
--send / --no-send          decide the camera deploy without being asked
--send-to DEST              rsync destination (default bmt@<--ip>:~/4d_firmware/calib/...)
-y, --yes                   take every default, no prompts
```

### On `--rational`

The default is the plain five-coefficient distortion model. The rational model buys a
slightly lower RMS by fitting fourteen coefficients that come out degenerate — k1 in the
tens, k2 in the hundreds — which is what produced
an earlier calibration of this rig. The firmware feeds `distL` straight into
`cv2.initUndistortRectifyMap`, which takes either length, so use the rational model only
if you have a reason to.

## Files

`calibration/` is self-contained apart from `stereo_4d` itself and three helpers it
borrows from `examples/check_calibration.py` (`find_checkerboard`,
`points_3d_from_stereo`, `check_grid_quality`), which keeps working on its own.

`reference_calibration.yaml` is the last known-good calibration of this rig, produced by
the ROS2 `camera_calibration` route before this tool existed: fx 2284, baseline 293.4 mm.
It is not read by anything — it is there to compare a fresh result against.
