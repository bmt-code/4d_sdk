#!/usr/bin/env python3
"""Ground-truth tests for the calibration pipeline. No camera and no images needed:

    python3 tests/test_calibration_synthetic.py

A rig with known intrinsics, distortion and extrinsics is invented, a checkerboard is
projected into both eyes from a set of poses that follows the documented capture
protocol, sub-pixel noise is added, and the result is pushed through the real
``calibration.solve.solve``. The recovered calibration is then compared against the
truth it was generated from, so the fit is measured rather than merely exercised.

Distortion coefficients are never compared directly: the default rational model fits
eight (k1 k2 p1 p2 k3 k4 k5 k6) to a five-coefficient truth, so the two are not the same
numbers even when they describe the same lens. What is compared is what the coefficients
*do* -- where a pixel undistorts to, and where a board triangulates to.
"""
import os
import sys
import tempfile

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from calibration.quality import Detection  # noqa: E402
from calibration.solve import object_points, solve  # noqa: E402
from calibration.targets import (  # noqa: E402
    ACCEPT_SCORE,
    NEAR_SCORE,
    POSES,
    TargetPlan,
    tilt,
    tilt_score,
)

FAILURES = []

# --- the rig these tests invent -------------------------------------------------------
# Numbers taken from calibration/reference_calibration.yaml, so the synthetic camera sits
# in the same part of parameter space as the real one.
IMAGE_SIZE = (1920, 1080)          # per eye, (w, h)
GRID = (9, 6)                      # inner corners
SQUARE_M = 0.025

K_LEFT = np.array([[2284.44, 0.0, 939.29],
                   [0.0, 2275.51, 523.54],
                   [0.0, 0.0, 1.0]])
K_RIGHT = np.array([[2284.44, 0.0, 969.38],
                    [0.0, 2275.51, 528.66],
                    [0.0, 0.0, 1.0]])
DIST_LEFT = np.array([0.032649, 0.476611, -0.001971, 0.012093, -2.333679])
DIST_RIGHT = np.array([0.116451, 0.214441, 0.001170, 0.006300, -2.870412])

# Left-to-right rotation and translation. T is in metres, as the firmware expects.
R_TRUE = cv2.Rodrigues(np.array([-0.015500, -0.045400, 0.007200]))[0]
T_TRUE = np.array([[-0.293319], [0.003779], [-0.005097]])
BASELINE_TRUE = float(np.linalg.norm(T_TRUE))

# Sub-pixel corner noise. cornerSubPix on a sharp board lands around here.
CORNER_NOISE_PX = 0.08

# Middle of the board in its own frame, so a pose can be aimed by where the board's
# centre lands rather than by where its origin corner does.
BOARD_CENTER = np.array([
    (GRID[0] - 1) * SQUARE_M / 2.0,
    (GRID[1] - 1) * SQUARE_M / 2.0,
    0.0,
])


def check(name, got, want):
    if got == want:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {got!r} != {want!r}")
        FAILURES.append(name)


def check_close(name, got, want, tol, unit=""):
    """Assert a number lands within ``tol`` of the truth, and say by how much."""
    delta = abs(got - want)
    if delta <= tol:
        print(f"ok   {name}: {got:.4f} vs {want:.4f} ({delta:.4f} <= {tol} {unit})")
    else:
        print(f"FAIL {name}: {got:.4f} vs {want:.4f} (off by {delta:.4f} > {tol} {unit})")
        FAILURES.append(name)


def check_below(name, got, limit, unit=""):
    if got <= limit:
        print(f"ok   {name}: {got:.4f} <= {limit} {unit}")
    else:
        print(f"FAIL {name}: {got:.4f} > {limit} {unit}")
        FAILURES.append(name)


# --- synthetic capture ----------------------------------------------------------------

def corners_in_frame(objp, rvec, tvec, K, dist):
    """Where the board lands in one eye, or None if any of it misses the sensor.

    A pose you could not have captured is not an observation, so it is dropped rather
    than fed in as one. Points behind the camera are rejected outright: projectPoints
    happily returns a mirrored image for those.
    """
    rotation = cv2.Rodrigues(rvec)[0]
    if np.min((rotation @ objp.T).T[:, 2] + tvec.reshape(3)[2]) <= 0.05:
        return None

    corners, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    corners = corners.reshape(-1, 2)
    if corners.min() < 0:
        return None
    if corners[:, 0].max() >= IMAGE_SIZE[0] or corners[:, 1].max() >= IMAGE_SIZE[1]:
        return None
    return corners


def visible(objp, rvec, tvec, K, dist):
    return corners_in_frame(objp, rvec, tvec, K, dist) is not None


def project(objp, rvec, tvec, K, dist, rng):
    """corners_in_frame plus the sub-pixel noise a real detector leaves behind."""
    corners = corners_in_frame(objp, rvec, tvec, K, dist)
    if corners is None:
        return None
    corners = corners + rng.normal(0.0, CORNER_NOISE_PX, corners.shape)
    return corners.reshape(-1, 1, 2).astype(np.float32)


def right_eye_pose(rvec, tvec):
    """The same board pose seen from the right eye."""
    rotation = cv2.Rodrigues(rvec)[0]
    rvec_right = cv2.Rodrigues(R_TRUE @ rotation)[0]
    tvec_right = R_TRUE @ tvec.reshape(3, 1) + T_TRUE
    return rvec_right, tvec_right


def stereo_poses(rng, count=22):
    """Both-eye frames: the board worked around the frame at several distances.

    Kept away from the outer corners on purpose. At a distance both eyes have to see the
    board, so it lives in the overlapping part of the two fields -- which is why a
    stereo-only set leaves the corner distortion unmeasured. Poses that do not land on
    both sensors are skipped and replaced, so the set is the size it says it is.
    """
    objp = object_points(GRID, SQUARE_M)
    poses = []
    attempt = 0
    while len(poses) < count and attempt < 40 * count:
        angle = 2.0 * np.pi * attempt / count
        depth = 0.7 + 1.1 * (attempt % 5) / 4.0        # 0.7 m .. 1.8 m
        attempt += 1
        rvec = np.array([
            0.30 * np.sin(angle) + rng.normal(0, 0.05),
            0.30 * np.cos(angle) + rng.normal(0, 0.05),
            rng.normal(0, 0.20),
        ])
        tvec = np.array([
            0.22 * np.cos(angle) * depth,
            0.14 * np.sin(angle) * depth,
            depth,
        ]) - cv2.Rodrigues(rvec)[0] @ BOARD_CENTER

        if not visible(objp, rvec, tvec, K_LEFT, DIST_LEFT):
            continue
        rvec_r, tvec_r = right_eye_pose(rvec, tvec)
        if not visible(objp, rvec_r, tvec_r, K_RIGHT, DIST_RIGHT):
            continue
        poses.append((rvec, tvec))
    return poses


def close_up_poses():
    """The two kinds of close-up, found rather than guessed.

    A board held close enough to reach one eye's image corners cannot be in the other
    eye -- at 0.6 m the two fields overlap over a narrow band -- so these poses are
    searched for: candidates near the camera are projected into both eyes, only those
    landing in exactly one are kept, and the one reaching furthest into each image
    quadrant is taken. That is the frame that constrains the focal length and the
    distortion out at that corner, and no both-eye frame can replace it.

    Deterministic: no rng, so the two halves of the close-up comparison differ by
    nothing else.
    """
    objp = object_points(GRID, SQUARE_M)
    best = {}

    for depth in (0.5, 0.6, 0.7, 0.85):
        for tilt in ((0.15, 0.25, 0.03), (-0.20, -0.30, -0.05)):
            rvec = np.array(tilt)
            offset = cv2.Rodrigues(rvec)[0] @ BOARD_CENTER
            for tx in np.arange(-0.32, 0.62, 0.02):
                for ty in np.arange(-0.22, 0.22, 0.02):
                    tvec = np.array([tx, ty, depth]) - offset
                    left = corners_in_frame(objp, rvec, tvec, K_LEFT, DIST_LEFT)
                    rvec_r, tvec_r = right_eye_pose(rvec, tvec)
                    right = corners_in_frame(objp, rvec_r, tvec_r, K_RIGHT, DIST_RIGHT)

                    if (left is None) == (right is None):
                        continue  # in both eyes, or in neither: not a close-up
                    eye = "left" if left is not None else "right"
                    corners = left if left is not None else right
                    principal = (K_LEFT if eye == "left" else K_RIGHT)[:2, 2]

                    # Furthest corner reached, recorded per image quadrant so the set
                    # covers all four rather than piling into one.
                    offsets = corners - principal
                    for quadrant in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                        mask = ((np.sign(offsets[:, 0]) == quadrant[0])
                                & (np.sign(offsets[:, 1]) == quadrant[1]))
                        if not mask.any():
                            continue
                        reach = float(np.max(np.linalg.norm(offsets[mask], axis=1)))
                        key = (eye, quadrant)
                        if key not in best or reach > best[key][0]:
                            best[key] = (reach, rvec, tvec)

    return [(rvec, tvec) for _, rvec, tvec in best.values()]


def make_detections(rng, with_close_ups=True):
    """A synthetic image set as a list of Detection objects, ready for solve()."""
    objp = object_points(GRID, SQUARE_M)
    detections = []

    for rvec, tvec in stereo_poses(rng):
        left = project(objp, rvec, tvec, K_LEFT, DIST_LEFT, rng)
        rvec_r, tvec_r = right_eye_pose(rvec, tvec)
        right = project(objp, rvec_r, tvec_r, K_RIGHT, DIST_RIGHT, rng)
        if left is None or right is None:
            continue
        detections.append(
            Detection(f"/synthetic/frame_{len(detections):03d}.png", left, right, 100.0)
        )

    if with_close_ups:
        for rvec, tvec in close_up_poses():
            # One eye each: whichever of the two the board actually lands in.
            left = project(objp, rvec, tvec, K_LEFT, DIST_LEFT, rng)
            rvec_r, tvec_r = right_eye_pose(rvec, tvec)
            right = project(objp, rvec_r, tvec_r, K_RIGHT, DIST_RIGHT, rng)
            if left is None and right is None:
                continue
            detections.append(
                Detection(f"/synthetic/close_{len(detections):03d}.png", left, right, 100.0)
            )

    return detections


def run_solve(rng, with_close_ups=True, rational=True, prune=True):
    out = os.path.join(tempfile.mkdtemp(), "stereo_calibration.yaml")
    detections = make_detections(rng, with_close_ups)
    result = solve(
        detections, GRID, SQUARE_M, (IMAGE_SIZE[1], IMAGE_SIZE[0], 3), out,
        rational=rational, prune=prune,
    )
    result["detections"] = detections
    return result


def rotation_error_deg(R_fit):
    """Angle of the residual rotation between the fitted and the true R, in degrees."""
    residual = R_fit @ R_TRUE.T
    angle = np.arccos(np.clip((np.trace(residual) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle))


def undistort_error_px(K_fit, dist_fit, K_true, dist_true, margin=0):
    """Worst disagreement between two lens models, in pixels, over the whole frame.

    Both models are asked where a grid of pixels really points, and the answers are put
    back through the true camera matrix so the difference reads in pixels. This is what
    the distortion coefficients are *for*, and unlike the coefficients themselves it is
    comparable across the 5- and 8-coefficient models.
    """
    xs = np.linspace(margin, IMAGE_SIZE[0] - 1 - margin, 24)
    ys = np.linspace(margin, IMAGE_SIZE[1] - 1 - margin, 16)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    pts = grid.reshape(-1, 1, 2)

    norm_fit = cv2.undistortPoints(pts, K_fit, dist_fit).reshape(-1, 2)
    norm_true = cv2.undistortPoints(pts, K_true, dist_true).reshape(-1, 2)

    fx, fy = K_true[0, 0], K_true[1, 1]
    delta = (norm_fit - norm_true) * np.array([fx, fy])
    return float(np.max(np.linalg.norm(delta, axis=1)))


# --- tests ----------------------------------------------------------------------------

def test_recovers_intrinsics():
    """The headline numbers: focal length, principal point, baseline, rotation."""
    result = run_solve(np.random.default_rng(20260909))

    check_close("intrinsics: left fx", result["mtxL"][0, 0], K_LEFT[0, 0], 8.0, "px")
    check_close("intrinsics: left fy", result["mtxL"][1, 1], K_LEFT[1, 1], 8.0, "px")
    check_close("intrinsics: left cx", result["mtxL"][0, 2], K_LEFT[0, 2], 4.0, "px")
    check_close("intrinsics: left cy", result["mtxL"][1, 2], K_LEFT[1, 2], 4.0, "px")
    check_close("intrinsics: right fx", result["mtxR"][0, 0], K_RIGHT[0, 0], 8.0, "px")
    check_close("intrinsics: right fy", result["mtxR"][1, 1], K_RIGHT[1, 1], 8.0, "px")

    baseline = float(np.linalg.norm(np.asarray(result["T"]).reshape(3)))
    check_close("extrinsics: baseline", baseline * 1000, BASELINE_TRUE * 1000, 0.75, "mm")
    check_below("extrinsics: rotation error", rotation_error_deg(result["R"]), 0.10, "deg")

    check_below("fit: left mono RMS", result["rms_left"], 0.25, "px")
    check_below("fit: right mono RMS", result["rms_right"], 0.25, "px")
    check_below("fit: stereo RMS", result["rms_stereo"], 0.25, "px")


def test_distortion_behaves_like_the_truth():
    """The recovered lens model has to agree with the real one out at the corners.

    Compared as pixels, not as coefficients: the default rational model fits eight
    numbers to a five-number truth and will never reproduce them.
    """
    result = run_solve(np.random.default_rng(20260910))

    left_err = undistort_error_px(result["mtxL"], result["distL"], K_LEFT, DIST_LEFT)
    right_err = undistort_error_px(result["mtxR"], result["distR"], K_RIGHT, DIST_RIGHT)
    # Measured over 15 seeds the worst corner lands around 3.5 px, so 5 px is a real
    # bound rather than a rubber stamp: without the close-ups this reads about 15 px.
    check_below("distortion: left model agrees over the whole frame", left_err, 5.0, "px")
    check_below("distortion: right model agrees over the whole frame", right_err, 5.0, "px")

    # Inside the outermost 60 px, where the board actually reaches, the agreement has to
    # be better -- a fit that is only right in the middle fails here.
    inner_left = undistort_error_px(result["mtxL"], result["distL"], K_LEFT, DIST_LEFT,
                                    margin=60)
    check_below("distortion: left model agrees away from the very edge",
                inner_left, 4.0, "px")


def test_close_ups_pin_down_the_corners():
    """The claim the capture protocol rests on, measured.

    Dropping the one-eye close-ups leaves the outer field unobserved, and the fit there
    is free to drift. This asserts the direction of the effect, not a specific size.
    """
    with_close = run_solve(np.random.default_rng(4242), with_close_ups=True)
    without = run_solve(np.random.default_rng(4242), with_close_ups=False)

    err_with = undistort_error_px(with_close["mtxL"], with_close["distL"],
                                  K_LEFT, DIST_LEFT)
    err_without = undistort_error_px(without["mtxL"], without["distL"],
                                     K_LEFT, DIST_LEFT)
    print(f"     corner agreement: {err_with:.2f} px with close-ups, "
          f"{err_without:.2f} px without")
    check("protocol: close-ups improve the corners", err_with < err_without, True)

    counts_with = with_close["counts"]
    counts_without = without["counts"]
    check("protocol: close-ups add mono frames but no stereo pairs",
          (counts_with["left"] > counts_without["left"]
           and counts_with["stereo"] == counts_without["stereo"]), True)


def test_plain_model_also_recovers_the_rig():
    """--no-rational has to keep working; the truth here is a 5-coefficient lens."""
    result = run_solve(np.random.default_rng(20260911), rational=False)

    check("model: plain writes 5 coefficients", np.asarray(result["distL"]).size, 5)
    check_close("model: plain left fx", result["mtxL"][0, 0], K_LEFT[0, 0], 8.0, "px")
    baseline = float(np.linalg.norm(np.asarray(result["T"]).reshape(3)))
    check_close("model: plain baseline", baseline * 1000, BASELINE_TRUE * 1000, 1.0, "mm")

    for name, coefficient, truth in (
        ("k1", result["distL"].reshape(-1)[0], DIST_LEFT[0]),
        ("k2", result["distL"].reshape(-1)[1], DIST_LEFT[1]),
    ):
        # Only meaningful for the plain model, where the fit and the truth are the same
        # parameterisation.
        check_close(f"model: plain recovers {name}", float(coefficient), truth, 0.05)


def test_rational_fits_eight_coefficients():
    """The rational model estimates 8: k1 k2 p1 p2 k3 k4 k5 k6 -- 6 radial, 2 tangential.

    OpenCV returns them in a 14-long vector all the same, the last six being the thin
    prism (s1..s4) and tilted sensor (taux, tauy) terms, which are only estimated with
    CALIB_THIN_PRISM_MODEL / CALIB_TILTED_MODEL and stay zero here. This pins both facts
    down: what is fitted, and what lands in the file.
    """
    result = run_solve(np.random.default_rng(20260912), rational=True)
    coefficients = np.asarray(result["distL"]).reshape(-1)

    check("model: rational estimates 8 coefficients",
          int(np.count_nonzero(coefficients[:8])), 8)
    check("model: thin-prism and tilt terms are left at zero",
          bool(np.all(coefficients[8:] == 0.0)), True)
    check("model: OpenCV still hands back a 14-long vector", coefficients.size, 14)

    # The six trailing zeros carry no information: the maps come out identical without
    # them, so the file's length is a quirk of the return shape, not of the fit.
    K = result["mtxL"]
    full = cv2.initUndistortRectifyMap(K, coefficients.reshape(1, -1), np.eye(3), K,
                                       IMAGE_SIZE, cv2.CV_16SC2)
    trimmed = cv2.initUndistortRectifyMap(K, coefficients[:8].reshape(1, -1), np.eye(3),
                                          K, IMAGE_SIZE, cv2.CV_16SC2)
    check("model: the trailing zeros change nothing",
          bool(np.array_equal(full[0], trimmed[0]) and np.array_equal(full[1], trimmed[1])),
          True)


def test_triangulation_is_accurate_in_millimetres():
    """End to end: rectify with the fitted calibration and measure a board in 3D.

    The same quantity the check stage puts on screen -- pairwise corner distances against
    the real grid -- so a regression here is one an operator would actually see.
    """
    result = run_solve(np.random.default_rng(20260913))
    rng = np.random.default_rng(99)

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        result["mtxL"], result["distL"], result["mtxR"], result["distR"],
        IMAGE_SIZE, result["R"], np.asarray(result["T"]).reshape(3, 1),
        alpha=0, flags=cv2.CALIB_ZERO_DISPARITY,
    )
    fx = P1[0, 0]
    baseline = abs(float(P2[0, 3] / P2[0, 0]))
    objp = object_points(GRID, SQUARE_M)

    worst_mean = 0.0
    checked = 0
    for rvec, tvec in stereo_poses(rng, count=6):
        left = project(objp, rvec, tvec, K_LEFT, DIST_LEFT, rng)
        rvec_r, tvec_r = right_eye_pose(rvec, tvec)
        right = project(objp, rvec_r, tvec_r, K_RIGHT, DIST_RIGHT, rng)
        if left is None or right is None:
            continue

        # Rectify the observed corners, then triangulate them the way verify.py does.
        rect_left = cv2.undistortPoints(left, result["mtxL"], result["distL"],
                                        R=R1, P=P1).reshape(-1, 2)
        rect_right = cv2.undistortPoints(right, result["mtxR"], result["distR"],
                                         R=R2, P=P2).reshape(-1, 2)

        y_rms = float(np.sqrt(np.mean((rect_left[:, 1] - rect_right[:, 1]) ** 2)))
        check_below(f"rectification: y-RMS at {tvec[2] * 1000:.0f} mm", y_rms, 0.5, "px")

        disparity = np.clip(rect_left[:, 0] - rect_right[:, 0], 1e-6, None)
        depth = fx * baseline / disparity
        points = np.stack([
            (rect_left[:, 0] - P1[0, 2]) * depth / fx,
            (rect_left[:, 1] - P1[1, 2]) * depth / P1[1, 1],
            depth,
        ], axis=1) * 1000.0

        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        model = objp * 1000.0
        model_distances = np.linalg.norm(model[:, None, :] - model[None, :, :], axis=2)
        idx = np.triu_indices_from(distances, k=1)
        worst_mean = max(worst_mean, float(np.mean(np.abs(distances[idx] - model_distances[idx]))))
        checked += 1

    check("triangulation: poses measured", checked > 0, True)
    check_below("triangulation: worst mean grid error", worst_mean, 2.0, "mm")


def test_pruner_keeps_intrinsics_of_a_moved_board():
    """A board that moved between the two exposures.

    Each eye is sharp and fits fine on its own; the pair is a lie. The documented
    behaviour is that such a frame leaves the extrinsics fit and keeps its intrinsics.
    """
    rng = np.random.default_rng(777)
    detections = make_detections(rng)

    victim = detections[3]
    # Shift the right eye only: the board was somewhere else when that eye was exposed.
    victim.corners_right = (victim.corners_right + np.array([9.0, 6.0],
                                                            dtype=np.float32))

    out = os.path.join(tempfile.mkdtemp(), "stereo_calibration.yaml")
    solve(detections, GRID, SQUARE_M, (IMAGE_SIZE[1], IMAGE_SIZE[0], 3), out)

    check("prune: moved board left the extrinsics fit", victim.use_stereo, False)
    check("prune: moved board kept its left eye", victim.use_left, True)
    check("prune: moved board kept its right eye", victim.use_right, True)
    check("prune: moved board was not rejected outright", victim.ok, True)


def test_pruning_a_clean_set_changes_nothing():
    """No frame in a clean synthetic set should be pruned at the default thresholds."""
    result = run_solve(np.random.default_rng(20260914))
    check("prune: clean set is left alone", result["pruned"], [])


def _board_at(depth, degrees, eye="left", offset=(0.0, 0.0)):
    """Corners of a board held ``degrees`` off square at ``depth``, or None if off frame."""
    objp = object_points(GRID, SQUARE_M)
    K, dist = (K_LEFT, DIST_LEFT) if eye == "left" else (K_RIGHT, DIST_RIGHT)
    rvec = np.array([0.0, np.radians(degrees), 0.0])
    tvec = (np.array([offset[0], offset[1], depth])
            - cv2.Rodrigues(rvec)[0] @ BOARD_CENTER)
    corners = corners_in_frame(objp, rvec, tvec, K, dist)
    return None if corners is None else corners.reshape(-1, 1, 2).astype(np.float32)


def test_tilt_is_the_same_number_at_every_distance():
    """The guide thresholds one tilt value across three depth bands, so it has to be
    distance-independent. The raw foreshortening is not -- that is the whole reason the
    normalised form exists."""
    raw, normalised = [], []
    for depth in (0.5, 1.0, 1.7):
        corners = _board_at(depth, 20)
        check(f"tilt: a 20 deg board is in frame at {depth} m", corners is not None, True)
        if corners is None:
            continue
        raw.append(tilt_score(corners, GRID))
        normalised.append(tilt(corners, GRID, K_LEFT[0, 0]))

    spread = max(normalised) / min(normalised)
    check_below("tilt: normalised spread across the bands", spread, 1.10, "x")
    # And the raw score really does vary, so the normalisation is doing work.
    check("tilt: raw score varies with distance", max(raw) / min(raw) > 2.5, True)


def test_tilt_tracks_the_angle():
    """Roughly 0.02 per degree, so a threshold can be reasoned about in degrees."""
    for degrees, expected in ((10, 0.20), (20, 0.37), (30, 0.57)):
        corners = _board_at(1.0, degrees)
        if corners is None:
            check(f"tilt: {degrees} deg board in frame", False, True)
            continue
        check_close(f"tilt: {degrees} deg reads about {expected}",
                    tilt(corners, GRID, K_LEFT[0, 0]), expected, 0.05)

    flat = _board_at(1.0, 3)
    check("tilt: a nearly flat board reads near zero",
          tilt(flat, GRID, K_LEFT[0, 0]) < 0.10, True)


def test_the_guide_satisfies_its_own_target():
    """The drawn guide is the answer to the question the tool is asking.

    Project each target's wanted pose, feed those corners back into that same target's
    acceptance, and it has to pass. This is what pins down the yaw sign convention -- a
    positive rotation about the vertical axis swings the board's right edge away, so the
    measured sign is the opposite of the rotation's, and getting that backwards makes the
    tool tell the operator to turn the board the other way from the picture it is showing
    them. It was backwards when first written.
    """
    plan = TargetPlan(GRID, SQUARE_M, IMAGE_SIZE)
    failures = []
    for target in plan.targets:
        corners = target.guide_corners().reshape(-1, 1, 2).astype(np.float32)
        ok, reason = target.accepts(corners, GRID)
        if not ok:
            failures.append(f"{target.band}/{target.pose}: {reason}")
    if failures:
        print(f"     {failures[:3]}")
    check("guide: every guide passes its own acceptance", len(failures), 0)


def test_the_guide_stays_on_the_sensor():
    """A guide drawn partly off the image is a placement nobody can fill."""
    plan = TargetPlan(GRID, SQUARE_M, IMAGE_SIZE)
    width, height = IMAGE_SIZE
    off = []
    for target in plan.targets:
        pts = target.guide_corners()
        if (pts[:, 0].min() < 0 or pts[:, 1].min() < 0
                or pts[:, 0].max() >= width or pts[:, 1].max() >= height):
            off.append(f"{target.band}/{target.eye}/{target.pose}")
    if off:
        print(f"     {off[:3]}")
    check("guide: every guide fits inside the frame", len(off), 0)


def test_a_real_board_can_satisfy_every_target():
    """The guides are geometry; this checks them against the actual camera model.

    Projected through the real intrinsics and distortion rather than the nominal pinhole
    the guides are drawn with, and required to land in frame and be accepted.
    """
    plan = TargetPlan(GRID, SQUARE_M, IMAGE_SIZE)
    objp = object_points(GRID, SQUARE_M)
    unreachable = []

    for target in plan.targets:
        K, dist = ((K_LEFT, DIST_LEFT) if target.eye == "left"
                   else (K_RIGHT, DIST_RIGHT))
        x, y, w, h = target.box
        depth = target.depth
        point = np.array([(x + w / 2 - K[0, 2]) * depth / K[0, 0],
                          (y + h / 2 - K[1, 2]) * depth / K[1, 1],
                          depth])
        # Sweep around the pose the target asks for, allowing the operator some latitude.
        angles = ([-3, 0, 3] if target.flat
                  else [target.yaw_deg * k for k in (0.85, 0.95, 1.0, 1.05, 1.15)])
        found = False
        for degrees in angles:
            rvec = np.array([0.0, np.radians(degrees), 0.0])
            tvec = point - cv2.Rodrigues(rvec)[0] @ BOARD_CENTER
            corners = corners_in_frame(objp, rvec, tvec, K, dist)
            if corners is None:
                continue
            ok, _ = target.accepts(corners.reshape(-1, 1, 2).astype(np.float32), GRID)
            if ok:
                found = True
                break
        if not found:
            unreachable.append(f"{target.band}/{target.eye}/{target.pose}")

    if unreachable:
        print(f"     unreachable: {unreachable[:4]}")
    check("targets: a real board can satisfy every placement", len(unreachable), 0)


def test_the_three_poses_stay_distinct():
    """The score has to tell the poses apart, or the tilt requirement quietly evaporates.

    This is the check that forced the score to be the *worst* corner distance rather than
    the mean: measured as a mean, a square-on board scores no worse against a turned guide
    than a well-matched board does, and all three poses accept each other.
    """
    plan = TargetPlan(GRID, SQUARE_M, IMAGE_SIZE)
    by_position = {}
    for target in plan.targets:
        by_position.setdefault(target.position, []).append(target)

    worst_gap = float("inf")
    failures = []
    for here in by_position.values():
        for target in here:
            for other in here:
                if other is target:
                    continue
                corners = other.guide_corners().reshape(-1, 1, 2).astype(np.float32)
                value = target.score(corners, GRID)
                worst_gap = min(worst_gap, value / ACCEPT_SCORE)
                if target.accepts(corners, GRID)[0]:
                    failures.append(f"{target.band}/{target.pose} accepted {other.pose}")

    if failures:
        print(f"     {failures[:3]}")
    check("poses: no pose accepts another's board", len(failures), 0)
    check_below("poses: closest confusion, as a fraction of the threshold",
                1 / worst_gap, 0.75, "x")

    # And a board that is turned the wrong way is named as such.
    target = [t for t in plan.targets if not t.flat][0]
    mirror = [t for t in plan.targets
              if t.position == target.position and not t.flat and t is not target][0]
    ok, reason = target.accepts(
        mirror.guide_corners().reshape(-1, 1, 2).astype(np.float32), GRID)
    check("poses: the wrong direction is refused", ok, False)
    check("poses: and says which way", reason, "turn it the other way")


def test_every_position_is_worked_three_ways():
    plan = TargetPlan(GRID, SQUARE_M, IMAGE_SIZE)
    by_position = {}
    for target in plan.targets:
        by_position.setdefault(target.position, []).append(target)

    poses_each = {len(v) for v in by_position.values()}
    check("poses: every position gets the full set", poses_each, {len(POSES)})
    first_pose = {v[0].pose for v in by_position.values()}
    check("poses: square on comes first", first_pose, {POSES[0][0]})

    signs = {t.yaw_sign for v in by_position.values() for t in v}
    check("poses: both turn directions are asked for", signs, {-1.0, 0.0, 1.0})


def test_the_score_is_the_only_gate():
    """One measure decides acceptance, and it behaves like a distance.

    The guide already carries position, distance and tilt, so a separate gate for each
    could only ever disagree with the picture on screen -- which is exactly what happened
    when they were separate.
    """
    plan = TargetPlan(GRID, SQUARE_M, IMAGE_SIZE)
    target = plan.current
    guide = target.guide_corners()

    perfect = guide.reshape(-1, 1, 2).astype(np.float32)
    check("score: a perfect match scores zero",
          round(target.score(perfect, GRID), 6), 0.0)
    check("score: and is accepted", target.accepts(perfect, GRID)[0], True)

    # Monotone in displacement, which is what makes a colour ramp meaningful.
    scores = [target.score((guide + np.array([shift, 0.0])).reshape(-1, 1, 2)
                           .astype(np.float32), GRID)
              for shift in (0, 20, 60, 150, 400)]
    check("score: rises with displacement",
          all(b > a for a, b in zip(scores, scores[1:])), True)
    check("score: a small nudge still passes",
          target.accepts((guide + np.array([8.0, 0.0])).reshape(-1, 1, 2)
                         .astype(np.float32), GRID)[0], True)
    check("score: a large one does not",
          target.accepts((guide + np.array([150.0, 0.0])).reshape(-1, 1, 2)
                         .astype(np.float32), GRID)[0], False)
    check("score: no board is infinite", target.score(None, GRID), float("inf"))

    # A board turned end for end detects in reverse order; that is the same board.
    check("score: reversed corner order matches too",
          round(target.score(guide[::-1].reshape(-1, 1, 2).astype(np.float32), GRID), 6),
          0.0)


def test_the_bands_start_close_and_work_out():
    plan = TargetPlan(GRID, SQUARE_M, IMAGE_SIZE)
    order, seen = [], set()
    for target in plan.targets:
        if target.band not in seen:
            seen.add(target.band)
            order.append(target)

    widths = [t.expected_width for t in order]
    check("bands: each is further out than the last",
          all(a > b for a, b in zip(widths, widths[1:])), True)
    check("bands: the first nearly fills the frame",
          widths[0] / IMAGE_SIZE[0] > 0.6, True)
    check("bands: the first is a single position",
          sum(1 for t in plan.targets if t.band == order[0].band) // len(POSES), 2)
    check("bands: the whole ladder is about a hundred frames",
          90 <= plan.wanted <= 120, True)


def test_saved_yaml_matches_the_firmware_contract():
    """The file the camera reads: six keys, distortion on one row, T in metres."""
    out = os.path.join(tempfile.mkdtemp(), "stereo_calibration.yaml")
    detections = make_detections(np.random.default_rng(20260915))
    solve(detections, GRID, SQUARE_M, (IMAGE_SIZE[1], IMAGE_SIZE[0], 3), out)

    with open(out) as handle:
        data = yaml.safe_load(handle)

    check("yaml: keys", sorted(data), ["R", "T", "distL", "distR", "mtxL", "mtxR"])
    check("yaml: E and F are not written", "E" in data or "F" in data, False)
    check("yaml: mtxL is 3x3", np.asarray(data["mtxL"]).shape, (3, 3))
    check("yaml: distL is one row", np.asarray(data["distL"]).shape[0], 1)
    check("yaml: R is 3x3", np.asarray(data["R"]).shape, (3, 3))

    baseline = float(np.linalg.norm(np.asarray(data["T"]).reshape(3)))
    check("yaml: T is in metres, not millimetres", 0.1 < baseline < 1.0, True)

    # The firmware reads exactly these and rebuilds rectification itself; if this stops
    # working the camera cannot use the file.
    K = np.array(data["mtxL"])
    dist = np.array(data["distL"])
    R = np.array(data["R"])
    T = np.array(data["T"]).reshape(3, 1)
    R1, _, P1, _, _, _, _ = cv2.stereoRectify(
        K, dist, np.array(data["mtxR"]), np.array(data["distR"]),
        IMAGE_SIZE, R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY,
    )
    maps = cv2.initUndistortRectifyMap(K, dist, R1, P1, IMAGE_SIZE, cv2.CV_16SC2)
    check("yaml: the camera can build rectification maps from it",
          maps[0].shape[:2], (IMAGE_SIZE[1], IMAGE_SIZE[0]))


def main():
    for test in (
        test_recovers_intrinsics,
        test_distortion_behaves_like_the_truth,
        test_close_ups_pin_down_the_corners,
        test_plain_model_also_recovers_the_rig,
        test_rational_fits_eight_coefficients,
        test_triangulation_is_accurate_in_millimetres,
        test_pruner_keeps_intrinsics_of_a_moved_board,
        test_pruning_a_clean_set_changes_nothing,
        test_tilt_is_the_same_number_at_every_distance,
        test_tilt_tracks_the_angle,
        test_the_guide_satisfies_its_own_target,
        test_the_guide_stays_on_the_sensor,
        test_a_real_board_can_satisfy_every_target,
        test_the_three_poses_stay_distinct,
        test_every_position_is_worked_three_ways,
        test_the_score_is_the_only_gate,
        test_the_bands_start_close_and_work_out,
        test_saved_yaml_matches_the_firmware_contract,
    ):
        print(f"\n--- {test.__name__} ---")
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
