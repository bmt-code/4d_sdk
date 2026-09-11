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
# One eye is enough for these: they exist to fit each camera's own lens, and the closest
# of them cannot be in both eyes whatever you do. The stereo ladder below is separate.
#
#   very close  1 position   x 3 poses x 2 eyes =   6 frames
#   close       2x2          x 3       x 2      =  24
#   mid         3x2          x 3       x 2      =  36
#                                                 ---
#                                                  66
BANDS = (
    # 0.50, down from the 0.70 this started at. The whole board is drawn now, not just the
    # inner corners, and yawed 33 degrees it swings wider still: anything above 0.50 runs
    # off the top and bottom of the frame at this band, and a guide nobody can fill is
    # worse than one held a little further back.
    ("very close", 0.50, (1, 1)),
    ("close",      0.40, (2, 2)),
    # Three columns, two rows. The middle row went: top and bottom already bracket the
    # vertical field, and a row through the centre mostly repeats what the band below it
    # covers at a slightly different size.
    ("mid",        0.31, (3, 2)),
)

# Placements the board has to be in *both* eyes for. Nothing else in the plan constrains
# the geometry between the cameras -- the bands above are per-eye by construction, and a
# calibration built only from those has no way to know where one camera sits relative to
# the other.
#
# They start further out than the mono bands because they have to. The cameras are 293 mm
# apart, so at 0.48 m a board filling one eye is entirely outside the other; the overlap
# only opens up past about 0.75 m, and widens from there.
#
# The board is asked for upright here, stood on its end. The strip both cameras can see is
# narrow and tall, and a board turned on its end fits it at ranges where a landscape one
# does not reach the overlap at all -- which is also why these fills look small next to the
# per-eye bands: an upright board spans the frame with its short side.
#
#   stereo near  2x2  x 3 poses = 12 frames
#   stereo far   2x2  x 3       = 12
#                               ---
#                                24
STEREO_BANDS = (
    ("stereo near", 0.19, (2, 2)),
    ("stereo far",  0.15, (2, 2)),
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
# The yaw is what separates the three poses, and it sets the ceiling on ACCEPT_SCORE
# below: two poses closer together than the tolerance accept each other's boards and the
# tilt requirement quietly evaporates. Measured, worst pair over every band, the closest
# a wrong pose comes is 0.040 at 25 degrees and 0.065 at 33 -- so the wider turn is what
# buys the looser tolerance, not a free choice alongside it.
POSES = (
    ("square on", 0.0),
    ("turn left edge in", -33.0),
    ("turn right edge in", 33.0),
)

# Clearance kept between the drawn board and the edge of the frame, in pixels. The inset
# itself is measured from the projected board rather than guessed at: see
# TargetPlan._margins.
GUIDE_CLEARANCE = 12

# The whole board is this much wider than the inner-corner grid the detector reports: for
# a 9x6 board, 10 squares across against 8.
FULL_BOARD_RATIO = 1.25

# Slack kept between a stereo placement and the edge of the other camera's field.
STEREO_MARGIN_PX = 90

# One number decides whether a frame counts: how far the detected corners sit from the
# guide's, as a fraction of the guide's width. The guide already says where the board goes,
# how far away, and at what angle -- so position, distance and tilt are all in this one
# measure, and separate gates for each could only ever disagree with the picture on screen.
#
# Scale-free by construction, so the same thresholds hold from the closest band to the
# furthest.
# The ceiling is 0.75 x the closest a wrong pose scores (0.065 at the yaw above), which
# is the margin tests/test_calibration_synthetic.py holds the design to.
# How much of the score a centring error is charged. Where the board sits in the frame is
# the one part of the pose the guide asks for that costs nothing to be loose about: the
# calibration wants boards spread over the field, and two hand-widths either side of the
# outline is still a different part of the field. Distance and yaw carry the information
# and stay on the full tolerance. At 0.5 a centring error is charged half, so the tolerance
# on position alone is twice the tolerance on shape -- about 9% of the board's width.
POSITION_WEIGHT = 0.5

ACCEPT_SCORE = 0.46      # green: close enough to shoot
NEAR_SCORE = 1.00        # amber: recognisably the right pose, not there yet

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


def nominal_baseline():
    """Distance between the two cameras, in metres, for sizing the overlap.

    Only used to work out where a board can be seen by both eyes at once, so the
    reference rig's value is close enough for any rig of this shape.
    """
    try:
        with open(REFERENCE_YAML) as handle:
            data = yaml.safe_load(handle) or {}
        return float(np.linalg.norm(np.asarray(data["T"], dtype=float)))
    except (OSError, KeyError, TypeError, ValueError):
        return 0.293


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


def full_board_points(grid, square_m):
    """Every square corner of the whole board, outer ring included, centred on it.

    The inner corners a detector returns stop one square short of the board's edge on
    every side, so a guide drawn from them is a whole square smaller than the thing in
    the operator's hands and there is nothing for the board's own edge to line up with.
    This is the full lattice: for a 9x6 inner-corner board, the 11x8 corners of its 10x7
    squares. Projected through the same pose it gives a guide the real board sits on
    exactly.

    Only ever drawn, never scored -- the detector has no opinion about the outer ring.
    """
    cols, rows = grid
    nx, ny = cols + 2, rows + 2
    pts = np.zeros((nx * ny, 3), np.float32)
    pts[:, :2] = np.indices((nx, ny)).T.reshape(-1, 2)
    pts *= square_m
    pts[:, 0] -= (nx - 1) * square_m / 2
    pts[:, 1] -= (ny - 1) * square_m / 2
    return pts


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


def derotate(pts, guide):
    """``pts`` turned about its own centre to best line up with ``guide``.

    Roll -- the board tilted in its own plane -- is the one error the guide asks nothing
    about and the score used to punish hardest: measured at the board's half-diagonal, four
    and a half degrees of it spent the entire tolerance on its own, before any error in
    where the board was, how far away, or how far turned. And it buys nothing. A rolled
    board is exactly as good a calibration frame as a level one; the yaw is what breaks the
    focal-length/depth ambiguity, and roll adds no information either way.

    So it is taken out before scoring, by the 2D Kabsch rotation. Rotating about the
    detection's own centre leaves the two things that do matter untouched -- the offset
    between the centres, and the scale -- and being rigid it cannot undo foreshortening, so
    a square-on board still cannot pass as a turned one.

    Returns the rotated points and the angle removed, in radians.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    centre_pts = pts.mean(axis=0)
    a = pts - centre_pts
    b = guide - guide.mean(axis=0)
    theta = np.arctan2(float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
                       float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    cos, sin = np.cos(theta), np.sin(theta)
    return a @ np.array([[cos, sin], [-sin, cos]]) + centre_pts, theta


def centre(corners):
    pts = np.asarray(corners).reshape(-1, 2)
    return pts.mean(axis=0)


class Target:
    """One placement: an eye, a spot in it, a distance, and one of the three poses."""

    def __init__(self, eye, band, depth, box, pose, yaw_deg, fx, grid,
                 square_m, image_size, dist=None, stereo=False, roll_deg=0.0):
        self.eye = eye
        # True when the board has to land in both cameras at once. The guide is still
        # drawn in one eye -- the operator has to aim at something -- but the other eye
        # has to see it too for the frame to count.
        self.stereo = stereo
        # In-plane rotation of the board. 90 means held upright, which is how a stereo
        # placement is asked for: the strip both cameras can see is narrow and tall, and a
        # board turned on its end fits it where a landscape one does not.
        self.roll_deg = roll_deg
        self.band = band
        self.depth = depth
        self.box = box                      # (x, y, w, h) in that eye's full-res pixels
        # What the box was before fit_into_frame nudged it. The three poses at one spot
        # get nudged by different amounts -- a yawed board reaches further -- so the live
        # box cannot identify the spot, and grouping by it split every position into three
        # singletons, quietly breaking "skip this spot" and the pose counter with it.
        self.origin = box
        self._guide = None
        self._guide_full = None
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

    def fit_into_frame(self, clearance=GUIDE_CLEARANCE, passes=5, min_centre_x=None):
        """Nudge this placement until the whole drawn board is on the sensor.

        Measuring an inset once at the centre of the frame is not enough: a board projects
        larger and more skewed the further off-axis it sits, so a margin that clears at the
        middle still clips at the edges. Moving the box and re-projecting converges in a
        pass or two, and is exact rather than an estimate. Returns False if the board is
        simply too big to fit at this depth, which is a band that wants pulling back.
        """
        img_w, img_h = self.image_size
        for _ in range(passes):
            pts = self.guide_full().reshape(-1, 2)
            dx = (max(0.0, clearance - pts[:, 0].min())
                  - max(0.0, pts[:, 0].max() - (img_w - clearance)))
            dy = (max(0.0, clearance - pts[:, 1].min())
                  - max(0.0, pts[:, 1].max() - (img_h - clearance)))
            if min_centre_x is not None:
                # A stereo placement has a floor on how far left it may sit: past it the
                # board has slid off the other camera entirely. Fitting to the frame alone
                # walked four of them straight out of the overlap.
                dx = max(dx, min_centre_x - (self.box[0] + self.box[2] / 2))
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                return True
            x, y, w, h = self.box
            self.box = (x + dx, y + dy, w, h)
            self._guide = None
            self._guide_full = None
        pts = self.guide_full().reshape(-1, 2)
        return (pts[:, 0].min() >= 0 and pts[:, 1].min() >= 0
                and pts[:, 0].max() < img_w and pts[:, 1].max() < img_h)

    @property
    def done(self):
        return self.taken >= self.shots

    @property
    def upright(self):
        """True when this placement wants the board stood on its end."""
        return abs(self.roll_deg) > 1e-6

    @property
    def flat(self):
        return abs(self.yaw_deg) < 1e-6

    @property
    def position(self):
        """What identifies the spot, ignoring which of the three poses this is."""
        return (self.eye, self.band, self.origin)

    def guide_corners(self):
        """The board's own corners, projected at the pose being asked for. Cached."""
        if self._guide is not None:
            return self._guide
        self._guide = self.guide_at(None)
        return self._guide

    def guide_full(self):
        """The whole board's square corners at this pose, as (rows+2, cols+2, 2).

        What gets drawn. :meth:`guide_corners` stays the inner grid, because that is what
        the detector reports and so what the score has to compare against.
        """
        if self._guide_full is None:
            self._guide_full = self._project(full_board_points(self.grid, self.square_m),
                                             None).reshape(self.grid[1] + 2,
                                                           self.grid[0] + 2, 2)
        return self._guide_full

    def guide_at(self, centre_px):
        """The same pose, projected as it would look centred on ``centre_px``.

        Scoring against a guide nailed to the target box charges a centring error twice:
        once as the offset itself, and again as the shape disagreement that offset causes,
        because a board a hand's width to the left genuinely projects a different shape
        from one dead on the outline. Re-projecting the guide where the board actually is
        removes the second charge and leaves the first, so position can be forgiven on its
        own terms.

        It also keeps what tells the poses apart. Yaw is what shifts a board's projected
        centre and skews its rows, and both survive here because the guide is rebuilt with
        the real perspective for that spot -- where simply sliding the guide sideways would
        have thrown them away, and with them the difference between turning the board one
        way and the other.
        """
        return self._project(board_points(self.grid, self.square_m), centre_px)

    def rvec(self):
        """This placement's board orientation, as a Rodrigues vector.

        The single source of the convention -- rolled in its own plane first, then yawed
        about the camera's vertical. Anything reconstructing a board at this pose should
        ask for it here rather than rebuilding it from ``yaw_deg``, which silently drops
        the roll and puts a landscape board where an upright one was asked for.
        """
        roll, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(self.roll_deg)]))
        yaw, _ = cv2.Rodrigues(np.array([0.0, np.radians(self.yaw_deg), 0.0]))
        return cv2.Rodrigues(yaw @ roll)[0]

    def _project(self, objp, centre_px):
        """``objp`` in board coordinates, projected at this pose, centred on ``centre_px``
        (or on the target's own box when that is None)."""
        img_w, img_h = self.image_size
        K = np.array([[self.fx, 0.0, img_w / 2],
                      [0.0, self.fx, img_h / 2],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        rvec = self.rvec()

        if centre_px is None:
            x, y, w, h = self.box
            cx, cy = x + w / 2, y + h / 2
        else:
            cx, cy = float(centre_px[0]), float(centre_px[1])
        tvec = np.array([(cx - K[0, 2]) * self.depth / self.fx,
                         (cy - K[1, 2]) * self.depth / self.fx,
                         self.depth], dtype=np.float64)

        points, _ = cv2.projectPoints(objp, rvec, tvec, K, self.dist)
        return points.reshape(-1, 2)

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

        The worse of two things, each divided by the guide's width so the number means the
        same in every band: how far the board's centre sits from the guide's (charged at
        POSITION_WEIGHT), and the worst corner-to-corner distance once that centring error
        and the roll are taken out.

        Worst rather than mean, and that choice is load-bearing. Yaw moves the board's
        edges while the middle stays put, so a mean dilutes exactly the difference the
        three poses are made of: measured as a mean, a square-on board scores no worse
        against a turned guide than a well-matched board does, and the poses collapse into
        each other. Taken as the worst corner the gap is threefold, which is what lets one
        number stand in for position, distance and tilt at once.

        Both corner orderings are tried: a board turned end for end detects in reverse, and
        that is a board in the right place, not a miss. Roll is removed first -- see
        :func:`derotate` -- so the number left is error in position, distance and yaw, which
        are the three the guide is actually asking for.
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

        # How far off the outline the board sits, charged at POSITION_WEIGHT ...
        position = float(np.linalg.norm(pts.mean(axis=0) - guide.mean(axis=0))) / width

        # ... and what shape it is, against the same pose re-projected where it actually
        # is, so being off-centre is not charged a second time as a shape error.
        #
        # The re-placement is the guide's own centre moved by the offset, not the board's
        # centroid handed over directly: a yawed board's corners do not average to the
        # point its centre projects to, so passing the centroid in would rebuild a slightly
        # different guide and a perfect match would stop scoring zero.
        x, y, w, h = self.box
        offset = pts.mean(axis=0) - guide.mean(axis=0)
        here = self.guide_at((x + w / 2 + offset[0], y + h / 2 + offset[1]))
        best = float("inf")
        for candidate in (pts, pts[::-1]):
            turned, _ = derotate(candidate, here)
            best = min(best, float(np.max(np.linalg.norm(turned - here, axis=1))))
        return max(best / width, position * POSITION_WEIGHT)

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
        for band, fill, shape in BANDS:
            width_px, height_px, depth = self._geometry(fill, upright=False)
            for eye in ("left", "right"):
                for box in self._boxes(width_px, height_px, shape, depth):
                    for pose, yaw_deg in POSES:
                        target = Target(eye, band, depth, box, pose, yaw_deg,
                                        self.fx, grid, square_m, image_size, self.dist)
                        target.fit_into_frame()
                        self.targets.append(target)

        self.baseline = nominal_baseline()
        self._stereo_floor = {}
        for band, fill, shape in STEREO_BANDS:
            width_px, height_px, depth = self._geometry(fill, upright=True)
            for box in self._stereo_boxes(width_px, height_px, shape, depth):
                for pose, yaw_deg in POSES:
                    target = Target("left", band, depth, box, pose, yaw_deg,
                                    self.fx, grid, square_m, image_size, self.dist,
                                    stereo=True, roll_deg=90.0)
                    target.fit_into_frame(
                        min_centre_x=self._stereo_floor.get(round(depth, 4)))
                    self.targets.append(target)
        self.index = 0

    def _stereo_boxes(self, width_px, height_px, shape, depth):
        """Placements the right eye can also see, laid out in the left eye's frame.

        A point at ``x`` in the left image sits at ``x - fx*baseline/depth`` in the right,
        so anything left of that disparity has fallen off the right sensor entirely. The
        usable strip is therefore pushed towards the right-hand side of the left image,
        and it widens as the board goes further away.
        """
        cols, rows = shape
        img_w, img_h = self.image_size
        margin_x, margin_y = self._margins(depth, roll_deg=90.0)
        disparity = self.fx * self.baseline / depth
        # The margin is generous on purpose: the disparity is worked out from a nominal
        # baseline and focal length, and a placement that only just clears the edge of the
        # other sensor stops clearing it as soon as either is a few per cent out.
        # margin_x already measures the drawn board's own half-width at this pose.
        low = disparity + margin_x + STEREO_MARGIN_PX
        high = img_w - margin_x
        if low >= high:                       # no overlap at this depth; nothing to place
            return []
        self._stereo_floor[round(depth, 4)] = low
        xs = self._spread(low, high, cols)
        ys = self._spread(margin_y, img_h - margin_y, rows)
        return [(x - width_px / 2, y - height_px / 2, width_px, height_px)
                for y in ys for x in xs]

    def _geometry(self, fill, upright):
        """(width, height, depth) for a band. ``fill`` is how much of the frame the board
        spans across; an upright board spans its short side, so the same fill puts it
        closer than a landscape one would."""
        width_px = fill * self.image_size[0]
        across, along = ((self.board_h, self.board_w) if upright
                         else (self.board_w, self.board_h))
        return width_px, width_px * along / across, self.fx * across / width_px

    def _margins(self, depth, roll_deg=0.0):
        """How far a guide's centre must stay from the edge, in pixels.

        Measured off the whole board actually drawn, at the most demanding of the three
        poses, rather than reasoned about from the inner grid: the drawn board is a square
        wider on every side than the corners the detector reports, and a yawed one swings
        its near edge wider still. Sizing the inset from the inner grid put nine guides in
        ten partly off the sensor -- visible as a board clipped by the frame edge, with no
        way to fill it.
        """
        img_w, img_h = self.image_size
        centre = np.array([img_w / 2, img_h / 2])
        half_w = half_h = 0.0
        for _, yaw_deg in POSES:
            probe = Target("left", "probe", depth,
                           (centre[0] - 1, centre[1] - 1, 2, 2), "probe", yaw_deg,
                           self.fx, self.grid, self.square_m, self.image_size, self.dist,
                           roll_deg=roll_deg)
            pts = probe.guide_full().reshape(-1, 2)
            half_w = max(half_w, float(np.max(np.abs(pts[:, 0] - centre[0]))))
            half_h = max(half_h, float(np.max(np.abs(pts[:, 1] - centre[1]))))
        return (min(half_w + GUIDE_CLEARANCE, img_w / 2),
                min(half_h + GUIDE_CLEARANCE, img_h / 2))

    @staticmethod
    def _spread(low, high, count):
        """``count`` positions across the span. One goes in the middle.

        np.linspace(a, b, 1) returns [a], so a band with a single position used to land it
        against the margin instead of in the centre of the frame -- which is where the one
        close-up frame, the only thing pinning the distortion at the edge of the field,
        least wants to be.
        """
        if count <= 1:
            return np.array([(low + high) / 2])
        return np.linspace(low, high, count)

    def _boxes(self, width_px, height_px, shape, depth):
        """Guide centres laid over one eye, inset so the drawn board stays on the sensor."""
        cols, rows = shape
        img_w, img_h = self.image_size
        margin_x, margin_y = self._margins(depth)
        xs = self._spread(margin_x, img_w - margin_x, cols)
        ys = self._spread(margin_y, img_h - margin_y, rows)
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
        ok, reason = target.accepts(corners, self.grid)
        if ok and target.stereo:
            other = corners_right if target.eye == "left" else corners_left
            if other is None:
                return False, "the other camera cannot see it"
        return ok, reason

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
