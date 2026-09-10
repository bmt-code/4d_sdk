"""Where to put the board next, in what pose, and whether the frame you took counts.

The capture study (``tests/capture_pattern_study.py``) settled what a good set looks like:
about a hundred frames over several depth bands, swept across both cameras' fields, and
tilted. A near-fronto-parallel set was the worst configuration measured, costing more
focal-length error than halving the shot count -- but a set with *no* square-on frames is
not what was measured either, so each position is worked three ways: square on, yawed one
way, yawed the other.

The guide is the board itself. Rather than a rectangle to fill, the wanted pose is
projected through a nominal camera and drawn as the board's own grid, so the operator
matches a shape instead of interpreting a box, and the yaw is legible in the drawing
rather than having to be spelled out in a label.

Nothing here needs a real calibration: the nominal focal length only scales a drawing, and
being a few per cent out costs nothing. Drawing lives in :mod:`calibration.hud`; this
module only produces geometry.
"""

import os

import cv2
import numpy as np
import yaml

# Nominal rig, used only to size and place the guides. Round numbers on purpose: these are
# a template for "a camera roughly like ours", not a measurement, and a precise-looking
# constant invites the reader to trust it further than it deserves. The reference
# calibration is read over the top of them when it is there.
#
# REFERENCE_WIDTH has to be declared rather than inferred. The obvious shortcut -- take it
# as twice the principal point -- is wrong by however far the principal point sits off
# centre, which on this rig is 2%, and that error lands straight on every guide.
NOMINAL_FX = 2280.0
REFERENCE_WIDTH = 1920.0
REFERENCE_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "reference_calibration.yaml")

# The ladder of bands. A band is named by how much of the frame width the board spans, not
# by a distance: "almost fills the screen" is something an operator can see, and the metres
# fall out of it. Each entry is (name, fill fraction, grid of positions per eye).
#
# It starts very close and works out. The first band is one position filling the frame --
# that single frame is what pins the distortion at the very edge of the field, and nothing
# further away can replace it. Every position is then worked at three poses.
#
#   very close  1 position   x 3 poses x 2 eyes =   6 frames
#   close       2x2          x 3       x 2      =  24
#   mid         3x3          x 3       x 2      =  54
#   far         2x2          x 3       x 2      =  24
#                                                 ---
#                                                 108
BANDS = (
    ("very close", 0.70, (1, 1)),
    ("close",      0.55, (2, 2)),
    ("mid",        0.38, (3, 3)),
    ("far",        0.25, (2, 2)),
)

# The three poses worked at every position, as a yaw in degrees. Square on comes first: it
# is the easiest to hit, and landing it is what tells the operator they are in the right
# place before they start turning the board about.
#
# The name describes what the operator sees; the sign is whatever produces it, and
# ``Target.yaw_sign`` carries the measured convention -- a positive rotation about the
# vertical axis swings the board's right edge *away*, so the measured sign is the opposite
# of the rotation's. The guide has to satisfy its own acceptance test, which is what pins
# this down (and caught it inverted once).
POSES = (
    ("square on", 0.0),
    ("turn left edge in", -25.0),
    ("turn right edge in", 25.0),
)

# A tilted board reaches further than a flat one, so guides are inset by this much extra;
# without it the outermost position of every band is unreachable.
TILT_HEADROOM = 1.30

# One number decides whether a frame counts: how far the detected corners sit from the
# guide's, as a fraction of the guide's width. The guide already says where the board goes,
# how far away, and at what angle -- so position, distance and tilt are all in this one
# measure, and separate gates for each could only ever disagree with the picture on screen.
#
# Scale-free by construction, so the same thresholds hold from the closest band to the
# furthest.
ACCEPT_SCORE = 0.028     # green: close enough to shoot
NEAR_SCORE = 0.12        # amber: recognisably the right pose, not there yet

def nominal_distortion():
    """Distortion coefficients for drawing the guide, from the reference calibration.

    A guide projected through a pinhole is not the shape a real board makes: the lens bends
    the outer field, and at the far band that disagreement is larger than the whole
    matching tolerance -- so a perfectly placed board fails to match a guide drawn without
    it. Unlike focal length these do not scale with resolution; they act on normalised
    coordinates.

    An empty array is a clean fallback: the guide is then a pinhole one, which is only
    wrong out at the edges.
    """
    try:
        with open(REFERENCE_YAML) as handle:
            data = yaml.safe_load(handle) or {}
        return np.asarray(data["distL"], dtype=float).reshape(-1)[:5]
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return np.zeros(5)


def nominal_fx(image_width=None):
    """Focal length for the guides, scaled to the resolution actually streaming.

    Focal length scales with resolution, so a rig streaming at a width other than
    ``REFERENCE_WIDTH`` needs the value scaled -- otherwise every guide comes out the wrong
    size and the board reads as permanently "too far away" however close it is held.
    """
    fx = NOMINAL_FX
    try:
        with open(REFERENCE_YAML) as handle:
            data = yaml.safe_load(handle) or {}
        fx = float(np.asarray(data["mtxL"], dtype=float)[0][0])
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        pass
    if image_width:
        fx *= image_width / REFERENCE_WIDTH
    # Rounded: it scales a drawing, and a guide sized to a tenth of a pixel is a fiction.
    return round(fx)


def board_size_m(grid, square_m):
    """Outer extent of the inner-corner grid, in metres."""
    return ((grid[0] - 1) * square_m, (grid[1] - 1) * square_m)


def board_points(grid, square_m):
    """The board's inner corners in its own frame, centred on the board."""
    objp = np.zeros((grid[0] * grid[1], 3), np.float32)
    objp[:, :2] = np.indices(grid).T.reshape(-1, 2)
    objp *= square_m
    objp[:, 0] -= (grid[0] - 1) * square_m / 2
    objp[:, 1] -= (grid[1] - 1) * square_m / 2
    return objp


def quad(corners, grid):
    """The four outer corners of a detected board, in detection order.

    ``findChessboardCorners`` returns the grid row-major with x varying fastest, so the
    quad is the first corner, the end of the first row, the start of the last row, and the
    last corner.
    """
    pts = np.asarray(corners).reshape(-1, 2)
    cols = grid[0]
    return pts[0], pts[cols - 1], pts[-cols], pts[-1]


def edges(corners, grid):
    """Lengths of the board quad's four sides: top, bottom, left, right."""
    top_left, top_right, bottom_left, bottom_right = quad(corners, grid)
    return (float(np.linalg.norm(top_right - top_left)),
            float(np.linalg.norm(bottom_right - bottom_left)),
            float(np.linalg.norm(bottom_left - top_left)),
            float(np.linalg.norm(bottom_right - top_right)))


def tilt_score(corners, grid):
    """Raw perspective foreshortening of a board quad, unsigned.

    A board square to the camera projects to a parallelogram: opposite edges come out the
    same length. Tilt it and the near edge grows while the far edge shrinks.

    Not comparable across distances -- the same physical tilt foreshortens far less on a
    board that subtends less of the frame. Use :func:`tilt` to threshold.
    """
    top, bottom, left, right = edges(corners, grid)
    if min(top, bottom, left, right) < 1e-6:
        return 0.0
    return float(max(abs(1.0 - top / bottom), abs(1.0 - left / right)))


def yaw_score(corners, grid):
    """Signed foreshortening about the vertical axis.

    Positive when the left edge is the longer one, which is the board's left side turned
    towards the camera. This is what lets the tool say "turn it the other way" instead of
    leaving the operator to guess which way it meant.
    """
    _, _, left, right = edges(corners, grid)
    if left + right < 1e-6:
        return 0.0
    return float(2.0 * (left - right) / (left + right))


def apparent_width(corners, grid):
    """Mean of the board's two horizontal edges, in pixels."""
    top, bottom, _, _ = edges(corners, grid)
    return (top + bottom) / 2


def _normalise(value, corners, grid, fx):
    width = apparent_width(corners, grid)
    if width < 1e-6:
        return 0.0
    return float(value * fx / width)


def tilt(corners, grid, fx):
    """Physical tilt of the board, near enough, with no calibration needed.

    Dividing the raw foreshortening by the board's angular size removes the distance term
    that makes :func:`tilt_score` incomparable between bands. What is left tracks the slant
    itself: about 0.02 per degree, holding to within 5% from half a metre out to 1.7 m, so
    one threshold covers every band.
    """
    return _normalise(tilt_score(corners, grid), corners, grid, fx)


def yaw(corners, grid, fx):
    """Signed counterpart of :func:`tilt`, in the same units."""
    return _normalise(yaw_score(corners, grid), corners, grid, fx)


def centre(corners):
    pts = np.asarray(corners).reshape(-1, 2)
    return pts.mean(axis=0)


class Target:
    """One placement: an eye, a spot in it, a distance, and one of the three poses."""

    def __init__(self, eye, band, depth, box, pose, yaw_deg, fx, grid,
                 square_m, image_size, dist=None):
        self.eye = eye
        self.band = band
        self.depth = depth
        self.box = box                      # (x, y, w, h) in that eye's full-res pixels
        self._guide = None
        self.pose = pose                    # human-readable pose name
        self.yaw_deg = yaw_deg
        # The sign :func:`yaw` will report for this pose. A positive rotation about the
        # vertical axis swings the board's right edge *away*, so the measured sign is the
        # opposite of the rotation's -- kept as its own field rather than negated at the
        # point of use, because the double negative is what got this inverted once already.
        self.yaw_sign = -float(np.sign(yaw_deg))
        self.fx = fx
        self.grid = grid
        self.square_m = square_m
        self.image_size = image_size
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=float)
        self.taken = 0
        self.shots = 1

    @property
    def done(self):
        return self.taken >= self.shots

    @property
    def flat(self):
        return abs(self.yaw_deg) < 1e-6

    @property
    def position(self):
        """What identifies the spot, ignoring which of the three poses this is."""
        return (self.eye, self.box)

    def guide_corners(self):
        """The board's own corners, projected at the pose being asked for. Cached."""
        if self._guide is not None:
            return self._guide
        img_w, img_h = self.image_size
        K = np.array([[self.fx, 0.0, img_w / 2],
                      [0.0, self.fx, img_h / 2],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        rvec = np.array([0.0, np.radians(self.yaw_deg), 0.0])

        x, y, w, h = self.box
        cx, cy = x + w / 2, y + h / 2
        tvec = np.array([(cx - K[0, 2]) * self.depth / self.fx,
                         (cy - K[1, 2]) * self.depth / self.fx,
                         self.depth], dtype=np.float64)

        objp = board_points(self.grid, self.square_m)
        points, _ = cv2.projectPoints(objp, rvec, tvec, K, self.dist)
        self._guide = points.reshape(-1, 2)
        return self._guide

    @property
    def expected_width(self):
        """How wide the board reads when it matches the guide.

        Measured off the guide that is actually drawn, not from ``fx * board / depth``
        computed alongside it. Those two disagree: a guide off to the side of the frame or
        turned away from the camera projects narrower than the on-axis, square-on formula
        says, so a separately-estimated gate refuses the very pose the picture is asking
        for. Taking it from the drawing keeps the gate and the guide the same object.
        """
        return apparent_width(self.guide_corners(), self.grid)

    def score(self, corners, grid):
        """How far the detected board sits from the guide, 0 is a perfect match.

        The *worst* corner-to-corner distance, divided by the guide's width so the number
        means the same thing in every band.

        Worst rather than mean, and that choice is load-bearing. Yaw moves the board's
        edges while the middle stays put, so a mean dilutes exactly the difference the
        three poses are made of: measured as a mean, a square-on board scores no worse
        against a turned guide than a well-matched board does, and the poses collapse into
        each other. Taken as the worst corner the gap is threefold, which is what lets one
        number stand in for position, distance and tilt at once.

        Both corner orderings are tried: a board turned end for end detects in reverse, and
        that is a board in the right place, not a miss.
        """
        if corners is None:
            return float("inf")
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        guide = self.guide_corners()
        if pts.shape != guide.shape:
            return float("inf")

        width = apparent_width(guide, grid)
        if width < 1e-6:
            return float("inf")
        forward = float(np.max(np.linalg.norm(pts - guide, axis=1)))
        reversed_ = float(np.max(np.linalg.norm(pts[::-1] - guide, axis=1)))
        return min(forward, reversed_) / width

    def hint(self, corners, grid):
        """What to say about a board that is not there yet.

        The score decides whether a frame counts; this only decides the wording, so it can
        be loose about which fault it names when several apply at once.
        """
        if corners is None:
            return "show the board"

        guide = self.guide_corners()
        width = apparent_width(guide, grid)
        offset = centre(corners) - centre(guide)
        if np.linalg.norm(offset) > width * 0.25:
            return "move onto the outline"

        ratio = apparent_width(corners, grid) / width
        if ratio < 0.82:
            return "come closer"
        if ratio > 1.22:
            return "move back"

        want = self.yaw_sign
        got = np.sign(yaw(corners, grid, self.fx))
        if want and got and want != got:
            return "turn it the other way"
        if self.flat and tilt(corners, grid, self.fx) > 0.20:
            return "hold it square on"
        return "match the outline"

    def accepts(self, corners, grid):
        """Does this detection satisfy the target? Returns (ok, reason)."""
        value = self.score(corners, grid)
        if value <= ACCEPT_SCORE:
            return True, "ok"
        return False, self.hint(corners, grid)


class TargetPlan:
    """The whole capture, as a worked-through list of targets.

    Every position is visited three times -- square on, then turned each way -- so frames
    arrive in threes and the operator turns the board on the spot rather than walking back
    and forth across the room for every one.
    """

    def __init__(self, grid, square_m, image_size, total_shots=None, fx=None):
        self.grid = grid
        self.square_m = square_m
        self.image_size = image_size            # (w, h) of one eye
        self.board_w, self.board_h = board_size_m(grid, square_m)
        self.fx = fx or nominal_fx(image_size[0])
        self.dist = nominal_distortion()

        self.targets = []
        img_w = image_size[0]
        for band, fill, shape in BANDS:
            # The fill fraction says how wide the board should read; the distance that
            # puts it there follows from the focal length.
            width_px = fill * img_w
            height_px = width_px * self.board_h / self.board_w
            depth = self.fx * self.board_w / width_px
            for eye in ("left", "right"):
                for box in self._boxes(width_px, height_px, shape):
                    for pose, yaw_deg in POSES:
                        self.targets.append(
                            Target(eye, band, depth, box, pose, yaw_deg,
                                   self.fx, grid, square_m, image_size, self.dist))
        self.index = 0

    def _boxes(self, width_px, height_px, shape):
        """Guide centres laid over one eye, inset so a tilted board stays on the sensor."""
        cols, rows = shape
        img_w, img_h = self.image_size
        margin_x = min(width_px / 2 * TILT_HEADROOM, img_w * 0.46) + 8
        margin_y = min(height_px / 2 * TILT_HEADROOM, img_h * 0.46) + 8
        xs = np.linspace(margin_x, img_w - margin_x, cols)
        ys = np.linspace(margin_y, img_h - margin_y, rows)
        return [(x - width_px / 2, y - height_px / 2, width_px, height_px)
                for y in ys for x in xs]

    @property
    def current(self):
        while self.index < len(self.targets) and self.targets[self.index].done:
            self.index += 1
        return self.targets[self.index] if self.index < len(self.targets) else None

    @property
    def finished(self):
        return self.current is None

    @property
    def taken(self):
        return sum(target.taken for target in self.targets)

    @property
    def wanted(self):
        return sum(target.shots for target in self.targets)

    def band_progress(self):
        out = {}
        for target in self.targets:
            taken, wanted = out.get(target.band, (0, 0))
            out[target.band] = (taken + target.taken, wanted + target.shots)
        return out

    def pose_progress(self):
        """Where this position stands: (done, total) over its three poses."""
        target = self.current
        if target is None:
            return (0, 0)
        here = [t for t in self.targets if t.position == target.position]
        return (sum(1 for t in here if t.done), len(here))

    def check(self, corners_left, corners_right):
        """Would a frame with these corners satisfy the target being asked for?

        Strictly the current target. An earlier version credited any outstanding
        placement, which sounded accommodating and behaved badly: standing still in one
        spot satisfied a run of queued targets and the tool fired several times over
        without the operator having moved at all.
        """
        target = self.current
        if target is None:
            return False, "done"
        corners = corners_left if target.eye == "left" else corners_right
        return target.accepts(corners, self.grid)

    def credit(self, corners_left, corners_right):
        """Record a saved frame against the current target. True if it counted."""
        target = self.current
        if target is None:
            return False
        ok, _ = self.check(corners_left, corners_right)
        if ok:
            target.taken += 1
        return ok

    def skip(self):
        """Give up on the current pose."""
        target = self.current
        if target is not None:
            target.shots = target.taken
            self.index += 1

    def skip_position(self):
        """Give up on all three poses here -- some placements a room does not allow."""
        target = self.current
        if target is None:
            return
        for other in self.targets:
            if other.position == target.position and not other.done:
                other.shots = other.taken

    def completed(self, eye):
        return [t for t in self.targets if t.eye == eye and t.done]
