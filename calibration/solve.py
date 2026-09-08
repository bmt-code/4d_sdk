"""Stage 3: turn detected corners into a stereo calibration.

Each eye's intrinsics are fitted from every frame *that eye* saw the board in, including
the close-up frames where the board is only in one of them -- those are what pin down
the distortion out at the image corners. Only the frames where both eyes saw it can say
anything about the geometry between them, so those alone go into the stereo fit.
"""

import logging

import cv2
import numpy as np
import yaml

CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 1000, 1e-8)

# Below these counts a fit is not worth trusting, and pruning must not take a set under
# them.
MIN_MONO_IMAGES = 6
MIN_STEREO_PAIRS = 6


def parse_grid(text):
    """'9x6' or '9,6' -> (9, 6) inner corners, columns first."""
    parts = text.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 'cols x rows', got {text!r}")
    cols, rows = (int(p.strip()) for p in parts)
    if cols < 3 or rows < 3:
        raise ValueError("A grid needs at least 3x3 inner corners")
    return cols, rows


def object_points(grid, square_size):
    """The board's corners in its own frame, x varying fastest.

    Same order findChessboardCorners returns, so the two line up index for index.
    """
    objp = np.zeros((np.prod(grid), 3), np.float32)
    objp[:, :2] = np.indices(grid).T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate_eye(objpoints, imgpoints, image_shape, flags):
    """One camera's intrinsics, fitted per eye so the two can draw on different
    frames."""
    mtx_init = np.eye(3, dtype=np.float32)
    mtx_init[0, 2] = image_shape[1] / 2
    mtx_init[1, 2] = image_shape[0] / 2
    dist_init = np.zeros(5, dtype=np.float32)

    rms, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape[:2][::-1], mtx_init, dist_init,
        criteria=CALIB_CRITERIA, flags=flags,
    )
    return rms, mtx, dist


def stereo_calibrate(objpoints, imgpoints_left, imgpoints_right,
                     mtxL, distL, mtxR, distR, image_shape, flags):
    """The geometry between the eyes, with the intrinsics held fixed."""
    rms, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtxL, distL, mtxR, distR, image_shape[:2][::-1],
        criteria=CALIB_CRITERIA, flags=flags,
    )
    return rms, mtxL, distL, mtxR, distR, R, T, E, F


def save_calibration(path, mtxL, distL, mtxR, distR, R, T, E, F, precision=12):
    """Write the eight keys the firmware and the SDK read.

    Distortion is flattened to one row to match the shape the ROS-derived calibrations
    use; the firmware feeds it straight to initUndistortRectifyMap, which takes either.
    """
    data = {
        "mtxL": np.round(mtxL, precision).tolist(),
        "distL": np.round(np.asarray(distL).reshape(1, -1), precision).tolist(),
        "mtxR": np.round(mtxR, precision).tolist(),
        "distR": np.round(np.asarray(distR).reshape(1, -1), precision).tolist(),
        "R": np.round(R, precision).tolist(),
        "T": np.round(T, precision).tolist(),
        "E": np.round(E, precision).tolist(),
        "F": np.round(F, precision).tolist(),
    }
    with open(path, "w") as handle:
        yaml.dump(data, handle, default_flow_style=False)
    logging.info(f"Calibration written to {path}")


def mono_reprojection_error(objp, corners, mtx, dist):
    """RMS reprojection error of one board in one eye, in pixels."""
    ok, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    if not ok:
        return float("inf")
    projected, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    errors = np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def stereo_reprojection_error(objp, corners_left, corners_right,
                              mtxL, distL, mtxR, distR, R, T):
    """RMS reprojection error of one pair, over both eyes, in pixels.

    The board pose is solved from the left eye and pushed through the rig's R|T into the
    right, so this measures stereo consistency rather than each eye's own fit. A frame
    with a small mono error and a large one here was captured while the board moved:
    each eye is sharp, but they did not see it in the same place.
    """
    ok, rvec, tvec = cv2.solvePnP(objp, corners_left, mtxL, distL)
    if not ok:
        return float("inf")

    projected_left, _ = cv2.projectPoints(objp, rvec, tvec, mtxL, distL)
    rotation, _ = cv2.Rodrigues(rvec)
    rvec_right, _ = cv2.Rodrigues(R @ rotation)
    tvec_right = R @ tvec + T.reshape(3, 1)
    projected_right, _ = cv2.projectPoints(objp, rvec_right, tvec_right, mtxR, distR)

    errors = np.concatenate([
        np.linalg.norm(projected_left.reshape(-1, 2) - corners_left.reshape(-1, 2), axis=1),
        np.linalg.norm(projected_right.reshape(-1, 2) - corners_right.reshape(-1, 2), axis=1),
    ])
    return float(np.sqrt(np.mean(errors ** 2)))


def _sets(detections):
    left = [d for d in detections if d.use_left]
    right = [d for d in detections if d.use_right]
    stereo = [d for d in detections if d.use_stereo]
    return left, right, stereo


def _check_counts(left, right, stereo):
    if len(left) < MIN_MONO_IMAGES or len(right) < MIN_MONO_IMAGES:
        raise SystemExit(
            f"Need at least {MIN_MONO_IMAGES} images per eye; have "
            f"{len(left)} left and {len(right)} right."
        )
    if len(stereo) < MIN_STEREO_PAIRS:
        raise SystemExit(
            f"Need at least {MIN_STEREO_PAIRS} frames with the board in both eyes; "
            f"have {len(stereo)}."
        )


def _fit(detections, grid, square_m, image_shape, rational):
    """One full mono + stereo fit."""
    left, right, stereo = _sets(detections)
    _check_counts(left, right, stereo)

    objp = object_points(grid, square_m)

    mono_flags = cv2.CALIB_RATIONAL_MODEL if rational else 0
    print(f"Fitting intrinsics: {len(left)} left, {len(right)} right")
    rms_left, mtxL, distL = calibrate_eye(
        [objp] * len(left), [d.corners_left for d in left], image_shape, mono_flags
    )
    rms_right, mtxR, distR = calibrate_eye(
        [objp] * len(right), [d.corners_right for d in right], image_shape, mono_flags
    )

    # The stereo step only refines the geometry between the eyes; the intrinsics above
    # are held fixed, which is why they are allowed to come from different frames.
    print(f"Fitting extrinsics: {len(stereo)} stereo pair(s)")
    rms_stereo, mtxL, distL, mtxR, distR, R, T, E, F = stereo_calibrate(
        [objp] * len(stereo),
        [d.corners_left for d in stereo],
        [d.corners_right for d in stereo],
        mtxL, distL, mtxR, distR, image_shape,
        flags=cv2.CALIB_FIX_INTRINSIC | mono_flags,
    )

    errors = {}
    for det in detections:
        entry = {}
        if det.use_left:
            entry["left"] = mono_reprojection_error(objp, det.corners_left, mtxL, distL)
        if det.use_right:
            entry["right"] = mono_reprojection_error(objp, det.corners_right, mtxR, distR)
        if det.use_stereo:
            entry["stereo"] = stereo_reprojection_error(
                objp, det.corners_left, det.corners_right,
                mtxL, distL, mtxR, distR, R, T,
            )
        errors[det.name] = entry

    result = {
        "mtxL": mtxL, "distL": distL, "mtxR": mtxR, "distR": distR,
        "R": R, "T": T, "E": E, "F": F,
        "rms_left": rms_left,
        "rms_right": rms_right,
        "rms_stereo": rms_stereo,
        "errors": errors,
        "counts": {"left": len(left), "right": len(right), "stereo": len(stereo)},
    }
    return result


def _prune(detections, errors, max_reproj, max_mono):
    """Narrow what each image is used for. Returns a list of (detection, note).

    An eye whose board does not reproject is dropped for that eye alone; a pair whose
    two eyes are each fine but disagree with one another keeps its intrinsics and only
    leaves the stereo fit. Nothing is thrown away that still has something to say.
    """
    notes = []
    for det in detections:
        if det.forced:
            continue
        entry = errors.get(det.name, {})

        bad_left = entry.get("left", 0.0) > max_mono
        bad_right = entry.get("right", 0.0) > max_mono
        if bad_left and bad_right:
            det.reason = (
                f"reprojection {entry['left']:.2f}/{entry['right']:.2f} px "
                f"> {max_mono} px in both eyes"
            )
            notes.append((det, det.reason))
            continue
        if bad_left:
            det.drop_left = True
            notes.append((det, f"left eye dropped, {entry['left']:.2f} px > {max_mono} px"))
        if bad_right:
            det.drop_right = True
            notes.append((det, f"right eye dropped, {entry['right']:.2f} px > {max_mono} px"))

        if det.use_stereo and entry.get("stereo", 0.0) > max_reproj:
            det.drop_stereo = True
            notes.append((
                det,
                f"out of the stereo fit, {entry['stereo']:.2f} px > {max_reproj} px "
                "(the eyes disagree -- the board moved between them)",
            ))
    return notes


def solve(detections, grid, square_m, image_shape, out_yaml,
          rational=False, prune=True, max_reproj=1.5, max_mono=1.0):
    """Calibrate, optionally prune, and refit once.

    ``square_m`` is in metres, so T comes out in metres like the firmware expects.
    """
    result = _fit(detections, grid, square_m, image_shape, rational)
    result["first_pass"] = {"rms_stereo": result["rms_stereo"], **result["counts"]}
    result["pruned"] = []

    if prune:
        notes = _prune(detections, result["errors"], max_reproj, max_mono)
        if notes:
            left, right, stereo = _sets(detections)
            print(
                f"\nPruning {len(notes)} entr(ies); refitting on {len(left)} left, "
                f"{len(right)} right, {len(stereo)} stereo."
            )
            refit = _fit(detections, grid, square_m, image_shape, rational)
            refit["first_pass"] = result["first_pass"]
            refit["pruned"] = [(d.name, note) for d, note in notes]
            result = refit

    save_calibration(
        out_yaml,
        result["mtxL"], result["distL"], result["mtxR"], result["distR"],
        result["R"], result["T"], result["E"], result["F"],
    )
    result["used"] = [d for d in detections if d.ok]
    result["stereo_used"] = [d for d in detections if d.use_stereo]
    result["yaml"] = out_yaml
    return result


def format_report(result, grid, square_m, source, rejected, blur_threshold):
    """Human-readable summary of a run; written to the session's report.txt."""
    cols, rows = grid
    counts = result["counts"]
    mono_only = len(result["used"]) - counts["stereo"]

    lines = [
        "Stereo 4D calibration report",
        "=" * 68,
        f"Source          : {source}",
        f"Board           : {cols}x{rows} inner corners, {square_m * 1000:.2f} mm squares",
        "",
        "Images",
        "-" * 68,
        f"  left eye     : {counts['left']}",
        f"  right eye    : {counts['right']}",
        f"  stereo pairs : {counts['stereo']}",
        f"  one eye only : {mono_only}  (intrinsics only)",
        f"  rejected     : {len(rejected)}",
    ]
    if blur_threshold is not None:
        lines.append(f"  blur cut-off : {blur_threshold:.1f} (Laplacian var on the board)")

    lines += [
        "",
        "RMS error (pixels)",
        "-" * 68,
        f"  left mono   : {result['rms_left']:.4f}",
        f"  right mono  : {result['rms_right']:.4f}",
        f"  stereo      : {result['rms_stereo']:.4f}",
    ]
    first = result.get("first_pass") or {}
    if result.get("pruned"):
        lines.append(
            f"  (before pruning: {first['rms_stereo']:.4f} over "
            f"{first['stereo']} stereo pairs)"
        )

    baseline = float(np.linalg.norm(np.asarray(result["T"]).reshape(3)))
    lines += [
        "",
        f"Baseline        : {baseline * 1000:.2f} mm",
        f"Left  fx, fy    : {result['mtxL'][0, 0]:.2f}, {result['mtxL'][1, 1]:.2f}",
        f"Right fx, fy    : {result['mtxR'][0, 0]:.2f}, {result['mtxR'][1, 1]:.2f}",
        f"Distortion coef : {np.asarray(result['distL']).size} per eye",
        "",
        "Per-image reprojection error (pixels)",
        "-" * 68,
        f"  {'left':>8} {'right':>8} {'stereo':>8}  used   image",
    ]

    def sort_key(det):
        entry = result["errors"].get(det.name, {})
        return -max(entry.values() or [0.0])

    for det in sorted(result["used"], key=sort_key):
        entry = result["errors"].get(det.name, {})
        cell = lambda key: f"{entry[key]:8.3f}" if key in entry else " " * 8
        lines.append(
            f"  {cell('left')} {cell('right')} {cell('stereo')}  "
            f"{det.eyes:<5}  {det.name}"
        )

    if result.get("pruned"):
        lines += ["", "Pruned", "-" * 68]
        for name, note in result["pruned"]:
            lines.append(f"  {name}: {note}")

    if rejected:
        lines += ["", "Rejected images", "-" * 68]
        for det in rejected:
            lines.append(f"  {det.name}: {det.reason}")

    return "\n".join(lines) + "\n"
