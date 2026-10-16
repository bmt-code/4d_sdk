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

The window draws **the board itself** at the pose it wants -- the wanted pose projected
through a nominal camera, rendered as the board's own grid. Match the shape and the tool
shoots by itself. The outline is **red when you are far off, amber as you close, green when
it is right**, so you steer by the colour instead of reading numbers.

The view is **mirrored**: you stand in front of the camera looking at the screen, and an
unmirrored view sends you the wrong way every time.

| Key | |
|---|---|
| `SPACE` | shoot now, whatever the guide says |
| `s` | skip this pose |
| `x` | skip all three poses at this spot |
| `q` / `Esc` | done capturing, carry on to the next stage |

### One measure, not three

Position, distance and tilt are all in the guide already, so there is **one** gate: how far
the detected corners sit from the guide's, as a fraction of the guide's width. Separate
gates for size and tilt were computed alongside the guide rather than from it, and
disagreed with the picture on screen by up to 15% -- refusing the very pose being drawn.

It is the **worst** corner distance, not the mean, and that matters. Yaw moves the board's
edges while the middle stays put, so a mean dilutes exactly the difference the three poses
are made of: measured as a mean, a square-on board scores no better against a turned guide
than a well-matched board does, and the poses collapse into each other. Taken as the worst
corner the gap is threefold.

The miss is split in two before it is judged. **Roll comes out entirely** — a board tilted
in its own plane is exactly as good a calibration frame as a level one, and charging it at
the board's half-diagonal used to spend the whole tolerance on four degrees of wrist.
**Where the board sits is charged at half weight**, and the shape is then measured against
the same pose re-projected where the board actually is, so being off-centre is not charged
twice. What is left to match:

| | tolerance |
|---|---|
| off the outline | ±9% of the board's width |
| too near or too far | ±6% |
| yaw | ±12° |
| roll | free |

The slack on position is not free of consequence: the poses at one placement are separated
mostly by where yaw puts the board's centre, so forgiving position forgives some of that.
Square on is still refused for a turned target everywhere but the far band. Which *way* you
turn it is no longer enforced — a checkerboard yawed one way and spun 180° in its own plane
is the same picture as the opposite yaw, so it never was reliably enforceable. Follow the
guide; the tool will not argue.

One more gate the guide cannot express: the board has to have **actually moved** since the
last saved frame, measured on its worst-moving corner. Without it a stationary board
satisfies every queued placement in turn and the tool fires three times over while you
stand there. Measured on the worst corner rather than the average because turning the
board pivots it about its centre — the middle corners barely move, so an average makes a
real pose change look like standing still.

Two further gates the guide cannot express: the board must be **still** (the eyes are
not exposed together, so a moving board lands in two different places and the pruner drops
the pair later), and it must be **sharp**. "Still" is 5 px of corner motion per check held
for 0.3 s -- set from what the desync needs rather than from what looks still, since two
eyes a couple of milliseconds apart turn even 40 px/s into under a tenth of a pixel between
them. Both only apply once the pose is matched --
before that you are still moving on purpose, which is why the hold-still bar only appears
once the outline turns green.

### The ladder

Every position is worked three ways: square on, then turned so the left edge comes towards
the camera, then the right. Turn on the spot; you do not walk between the three.

**Per-eye placements** — these fit each camera's own lens, and the closest of them cannot
be in both eyes whatever you do:

| Band | Board fills | Roughly | Positions per eye | Frames |
|---|---|---|---|---|
| very close | 50% of the frame | 0.48 m | 1 | 6 |
| close | 40% | 0.59 m | 2x2 | 24 |
| mid | 31% | 0.77 m | 3x2 | 36 |

**Both-eye placements** — the only ones that say anything about where one camera sits
relative to the other. They start further out because they have to: the cameras are 293 mm
apart, so at 0.48 m a board filling one eye is entirely outside the other.

Held the same way round as every other band. They sit well inside the overlap rather than
at the edge of it: a placement that only just clears the far camera stops clearing it as
soon as you land a little off the guide, and a stereo frame the second camera cannot see
contributes nothing at all. Each one now tolerates about 370 px of drift at worst before
the far camera loses the board.

| Band | Board fills | Roughly | Positions | Frames |
|---|---|---|---|---|
| stereo near | 28% | 0.85 m | 2x2 | 12 |
| stereo far | 20% | 1.19 m | 2x2 | 12 |
| | | | | **90 total** |

It starts almost filling the frame and works out. The first band is one position, and that
single frame is what pins the distortion at the very edge of the field -- nothing further
away replaces it. A band is named by how much of the frame the board spans rather than by a
distance, because that is what you can see; the metres fall out of the focal length.

The guides are drawn with the reference lens distortion, not through a pinhole. Without it
a real board at the far band misses a pinhole guide by more than the whole tolerance.

Both windows render at `--preview-width` (1920, so 1920x540 for the 3840x1080 stereo frame),
and the live board search runs on a half-size copy with `CALIB_CB_FAST_CHECK` only: the full
search costs ~300 ms an eye when there is nothing to find, which is most frames while the
board is being carried around, and is what makes the window feel stuck.

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
2. **Send it to the camera** — clears the unit's entry from `~/.ssh/known_hosts`, then
   rsyncs onto the unit at
   `bmt@172.31.1.77:~/4d_firmware/calib/stereo_calibration.yaml`, which is the copy the
   running firmware actually reads. The host follows `--ip`, so pointing at another unit
   deploys to that one; `--send-to` overrides the whole destination. The file already on
   the camera is kept as `stereo_calibration.yaml.bak-<timestamp>`.

The `ssh-keygen -R` first is because the unit is reflashed often and comes back with a new
host key, which stops ssh connecting until the stale entry is gone. It does mean whatever
answers on that address next is accepted without checking — fine on a private link to a
device you reimage, not something to copy onto anything routable.

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
--shots 100             frames to collect; the guides are sized to reach this

--no-blur-filter        keep the soft images
--blur-ratio 0.6        drop below this fraction of the median sharpness
--blur-min N            absolute Laplacian-variance floor on top of the ratio
--include 27-34,45      keep these frames whatever the filters say (repeatable)
--exclude 12,19         drop these frames whatever the filters say (repeatable)

--max-mono-reproj 1.0   per-eye pruning threshold in pixels
--grid 9x6              inner corners, columns x rows
--square 25             square size in millimetres
--no-rational           plain 5-coefficient distortion model (see below)
--no-prune              keep the high-error pairs
--max-reproj 1.5        stereo pruning threshold in pixels
--workers N             detection threads

--offline               review saved images instead of a live stream
--line-spacing 20       displayed pixels between epipolar lines
--preview-width 1920    width the capture and check windows render at
--no-check              skip the check entirely

--install / --no-install    decide the 4d_firmware copy without being asked
--send / --no-send          decide the camera deploy without being asked
--send-to DEST              rsync destination (default bmt@<--ip>:~/4d_firmware/calib/...)
-y, --yes                   take every default, no prompts
```

### On the distortion model

The default is the **rational** model: eight coefficients — six radial
(k1 k2 k3 k4 k5 k6) and two tangential (p1 p2) — chosen on this rig's own early tests,
where it measured more accurate than the plain five. Being a ratio of two radial
polynomials rather than one, its coefficients are not comparable to the plain model's
term by term; the numerator and denominator terms only mean anything together.

The vector in the YAML is fourteen long, not eight: OpenCV always returns the thin-prism
(s1..s4) and tilted-sensor (taux, tauy) slots, and they stay zero unless
`CALIB_THIN_PRISM_MODEL` / `CALIB_TILTED_MODEL` are asked for, which this pipeline never
does. The six zeros change nothing — `initUndistortRectifyMap` produces identical maps
with or without them.

`--no-rational` falls back to the five-coefficient model. The firmware feeds `distL`
straight into `cv2.initUndistortRectifyMap`, which takes either length, so both deploy
the same way.

## Tests

`python3 tests/test_calibration_synthetic.py` checks the pipeline against a rig it
invents: known intrinsics, distortion and extrinsics, a checkerboard projected into both
eyes from poses that follow the capture protocol above, sub-pixel noise on the corners,
and then the real solve. It asserts the recovered focal length, principal point, baseline
and rotation against the truth, that the fitted lens model agrees with the real one out at
the image corners, that triangulated boards measure correctly in millimetres, that the
pruner takes a moved board out of the extrinsics while keeping its intrinsics, and that
the saved YAML is what the firmware reads.

It also measures the reason the close-ups exist: with them the fitted lens agrees with the
truth to about 1 px across the frame, without them about 15 px.

No camera and no images needed; about nine seconds.

## Files

`calibration/` is self-contained apart from `stereo_4d` itself and three helpers it
borrows from `examples/check_calibration.py` (`find_checkerboard`,
`points_3d_from_stereo`, `check_grid_quality`), which keeps working on its own.

`reference_calibration.yaml` is the last known-good calibration of this rig, produced by
the ROS2 `camera_calibration` route before this tool existed: fx 2284, baseline 293.4 mm.
It is not read by anything — it is there to compare a fresh result against.
