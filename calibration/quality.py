"""Checkerboard detection and image-quality filtering.

Detection runs once here and the corners are handed to :mod:`calibration.solve`, so
filtering costs nothing on top of the calibration it feeds.
"""

import concurrent.futures
import os

import cv2
import numpy as np
from tqdm import tqdm

# Same flags examples/check_calibration.py detects with.
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
FIND_FLAGS_FAST = FIND_FLAGS | cv2.CALIB_CB_FAST_CHECK
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
SUBPIX_WINDOW = (11, 11)

# How far outside the board's bounding box the sharpness crop reaches, in pixels.
SHARPNESS_PAD = 24

# Downscale used by the live detector. Half size keeps full recall against the full
# search; a third starts losing boards.
PREVIEW_SCALE = 0.5


def split_stereo(image):
    """Split a side-by-side stereo frame into (left, right)."""
    half = image.shape[1] // 2
    return image[:, :half], image[:, half:]


def to_gray(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def find_corners(gray, grid, refine=True):
    """Locate the inner corners of ``grid`` = (cols, rows), sub-pixel refined.

    Tries the cheap FAST_CHECK pass first and falls back to the full search, the same
    two-stage approach as examples/check_calibration.py.
    """
    found, corners = cv2.findChessboardCorners(gray, grid, flags=FIND_FLAGS_FAST)
    if not found:
        found, corners = cv2.findChessboardCorners(gray, grid, flags=FIND_FLAGS)
    if not found:
        return None
    if refine:
        corners = cv2.cornerSubPix(
            gray, corners, SUBPIX_WINDOW, (-1, -1), SUBPIX_CRITERIA
        )
    return corners


def find_corners_sb(gray, grid):
    """Second-opinion detector, used only on images you asked to force-include.

    findChessboardCornersSB copes with blur, glare and steep angles the classic
    detector gives up on, at roughly half a second an eye -- too slow to be the default
    over a whole session, worth it for a handful of images.
    """
    found, corners = cv2.findChessboardCornersSB(
        gray, grid, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    )
    return corners if found else None


def find_corners_fast(gray, grid, scale=PREVIEW_SCALE):
    """Cheap board detector for live overlays, in full-resolution coordinates.

    The full search costs ~300 ms an eye when there is nothing to find, which is most
    frames while you are moving the board around -- far too slow to run in a display
    loop. FAST_CHECK on a half-size copy costs a few milliseconds and, measured over a
    real session, misses none of the boards the full search finds. Corners are rough:
    good enough to point at, not to calibrate from.
    """
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    found, corners = cv2.findChessboardCorners(small, grid, flags=FIND_FLAGS_FAST)
    return corners / scale if found else None


def sharpness(gray, corners, pad=SHARPNESS_PAD):
    """Laplacian variance inside the board's bounding box.

    Measured on the board rather than the whole frame: full-frame variance is dominated
    by whatever else is in the room, so a global threshold either keeps everything or
    nothing.
    """
    pts = corners.reshape(-1, 2)
    h, w = gray.shape[:2]
    x0 = max(0, int(pts[:, 0].min()) - pad)
    x1 = min(w, int(pts[:, 0].max()) + pad)
    y0 = max(0, int(pts[:, 1].min()) - pad)
    y1 = min(h, int(pts[:, 1].max()) + pad)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


class Detection:
    """One image's detection result.

    An eye that saw the board is usable for that eye's intrinsics even if the other eye
    missed it -- a board held close against the left edge cannot be in both eyes at
    once, and those are exactly the frames that pin down the distortion out at the
    image corners. Only frames where the board is in *both* eyes can say anything about
    the geometry between them.
    """

    def __init__(self, path, corners_left=None, corners_right=None, sharpness=0.0,
                 reason=None):
        self.path = path
        self.corners_left = corners_left
        self.corners_right = corners_right
        self.sharpness = sharpness
        self.reason = reason  # set once the image is out of the fit entirely
        self.forced = False   # named in --include: no filter touches it

        # Pruning narrows what an image is used for rather than throwing it away.
        self.drop_left = False
        self.drop_right = False
        self.drop_stereo = False

    @property
    def ok(self):
        return self.reason is None

    @property
    def has_left(self):
        return self.corners_left is not None

    @property
    def has_right(self):
        return self.corners_right is not None

    @property
    def has_both(self):
        return self.has_left and self.has_right

    @property
    def use_left(self):
        return self.ok and self.has_left and not self.drop_left

    @property
    def use_right(self):
        return self.ok and self.has_right and not self.drop_right

    @property
    def use_stereo(self):
        return self.use_left and self.use_right and not self.drop_stereo

    @property
    def eyes(self):
        """'both', 'left' or 'right' -- what this image still contributes to."""
        if self.use_stereo:
            return "both"
        return "left" if self.use_left else "right" if self.use_right else "none"

    @property
    def name(self):
        return os.path.basename(self.path)


def _detect_one(path, grid):
    image = cv2.imread(path)
    if image is None:
        return Detection(path, reason="unreadable")
    if image.shape[1] % 2:
        return Detection(path, reason="odd width, not a side-by-side pair")

    left, right = split_stereo(image)
    gray_left, gray_right = to_gray(left), to_gray(right)

    corners_left = find_corners(gray_left, grid)
    corners_right = find_corners(gray_right, grid)
    if corners_left is None and corners_right is None:
        return Detection(path, reason="no board in either eye")

    # Whichever eyes saw it, the blurriest one is what limits the fit.
    sharp = min(
        sharpness(gray, corners)
        for gray, corners in ((gray_left, corners_left), (gray_right, corners_right))
        if corners is not None
    )
    return Detection(path, corners_left, corners_right, sharp)


def detect_all(paths, grid, workers=None):
    """Detect the board in every image. Returns detections in input order."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(lambda p: _detect_one(p, grid), paths),
                desc="Detecting board",
                total=len(paths),
                unit="img",
            )
        )


def filter_blurry(detections, ratio=0.6, floor=None):
    """Flag the soft images among those that have a board.

    The threshold is relative to the median sharpness of the set, so it follows the
    lighting and the lens instead of needing a magic number per session. ``floor`` adds
    an optional absolute minimum on top.
    """
    usable = [d for d in detections if d.ok]
    if not usable:
        return

    values = np.array([d.sharpness for d in usable])
    threshold = float(np.median(values)) * ratio
    if floor is not None:
        threshold = max(threshold, floor)

    for det in usable:
        if det.sharpness < threshold and not det.forced:
            det.reason = f"blurry ({det.sharpness:.1f} < {threshold:.1f})"
    return threshold


def image_shape(paths):
    """Per-eye (h, w, c) of the first readable image, matching StereoCalibration."""
    for path in paths:
        image = cv2.imread(path)
        if image is not None:
            h, w, c = image.shape
            return (h, w // 2, c)
    return None


def stem(name):
    """'frame_027.png', 'frame_027', '027' and '27' all reduce to the same key.

    Zero padding is dropped so a frame can be named the way it is easiest to type.
    """
    name = os.path.splitext(os.path.basename(str(name)))[0]
    _, _, tail = name.rpartition("frame_")
    tail = tail or name
    return str(int(tail)) if tail.isdigit() else tail


def parse_frame_spec(values):
    """Frame selectors -> a set of stems.

    Accepts anything the eye writes: '27', 'frame_027.png', comma- or space-separated
    lists, and numeric ranges like '27-34'.
    """
    wanted = set()
    for value in values or []:
        for token in value.replace(",", " ").split():
            low, dash, high = token.partition("-")
            if dash and low.strip().isdigit() and high.strip().isdigit():
                for number in range(int(low), int(high) + 1):
                    wanted.add(str(number))
            else:
                wanted.add(stem(token))
    return wanted


def apply_overrides(detections, include, exclude, grid):
    """Honour --include / --exclude. Returns (forced, undetectable, excluded, unknown).

    An included image that had no board gets a second pass with the stronger detector;
    one that still has no board in both eyes cannot be calibrated from, whatever was
    asked for, and is reported rather than silently kept.
    """
    by_stem = {stem(d.path): d for d in detections}
    forced, undetectable, excluded = [], [], []

    for key in sorted(include):
        det = by_stem.get(key)
        if det is None:
            continue
        if not det.has_both:
            _retry_with_sb(det, grid)
        if not (det.has_left or det.has_right):
            undetectable.append(det)
            continue
        det.forced = True
        det.reason = None
        det.drop_left = det.drop_right = det.drop_stereo = False
        forced.append(det)

    for key in sorted(exclude):
        det = by_stem.get(key)
        if det is not None:
            det.forced = False
            det.reason = "excluded by --exclude"
            excluded.append(det)

    unknown = sorted((include | exclude) - set(by_stem))
    return forced, undetectable, excluded, unknown


def _retry_with_sb(det, grid):
    """Second pass over the eyes that came up empty, with the stronger detector."""
    image = cv2.imread(det.path)
    if image is None:
        return
    left, right = split_stereo(image)
    gray_left, gray_right = to_gray(left), to_gray(right)

    if det.corners_left is None:
        det.corners_left = find_corners_sb(gray_left, grid)
    if det.corners_right is None:
        det.corners_right = find_corners_sb(gray_right, grid)

    found = [
        (gray, corners)
        for gray, corners in ((gray_left, det.corners_left), (gray_right, det.corners_right))
        if corners is not None
    ]
    if not found:
        det.reason = "no board in either eye (both detectors)"
        return

    det.sharpness = min(sharpness(gray, corners) for gray, corners in found)
    det.reason = None
