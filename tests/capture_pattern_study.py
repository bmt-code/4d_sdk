#!/usr/bin/env python3
"""Capture-pattern study on the synthetic ground-truth rig.

    python3 tests/capture_pattern_study.py            # the eight patterns
    python3 tests/capture_pattern_study.py --counts   # shot-count sweep
    python3 tests/capture_pattern_study.py --noise    # what corner noise matches the bench

Not a test -- nothing here asserts. It answers "how should we capture?" by running whole
capture patterns against a rig whose intrinsics, distortion and extrinsics are known, and
scoring the calibration that comes out against that truth.

A pattern says how many boards to place in each depth band, how wide to sweep them across
the two cameras' combined field, and how hard to tilt them. Nothing forces a board into
one eye or both: poses are laid across the field and the visibility falls out of the
geometry, as it does on the bench. A placement landing on neither sensor costs an
*attempt*, not a frame -- an operator only presses the shutter on a board the tool can
see -- so bands fill to a target of saved frames and the attempt count records how hard
the band was to fill.

The headline metric is the one an operator actually reads: ``check_grid_quality`` -- the
mean disagreement between a triangulated board's pairwise corner distances and the real
grid, in millimetres, from examples/check_calibration.py. It is reported two ways; see
:func:`score`.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_calibration_synthetic as t  # noqa: E402
from calibration.quality import Detection  # noqa: E402
from calibration.solve import object_points, solve  # noqa: E402
from examples.check_calibration import check_grid_quality, points_3d_from_stereo  # noqa: E402

SEEDS = 6

# Half-field of one sensor per metre of depth, from the true intrinsics.
HALF_X = t.IMAGE_SIZE[0] / 2 / t.K_LEFT[0, 0]
HALF_Y = t.IMAGE_SIZE[1] / 2 / t.K_LEFT[1, 1]
# Where the right camera sits in the left camera's frame.
RIGHT_EYE_X = float((-t.R_TRUE.T @ t.T_TRUE)[0, 0])

# Distances the grid check is evaluated at, in metres. The number an operator quotes
# depends on how far away they hold the board, so it is reported per distance. Below
# ~0.8 m a centred board no longer fits in both eyes at once, so there is nothing to
# triangulate and no grid check to run.
CHECK_DEPTHS = (0.8, 1.0, 1.4, 1.8)
CHECK_BOARDS = 6

# The bench standard: board centred in the shared field at 1 m, held square to the rig.
# This is the pose the quoted grid figure is measured at, so the model is anchored to it
# rather than to an average over distances.
BENCH_DEPTH = 1.0

TILTS = {"low": (0.04, 0.16), "medium": (0.15, 0.42), "high": (0.34, 0.70)}


# --- capture --------------------------------------------------------------------------

def layout(n, z, span, rng, tilt, jitter=0.35):
    """``n`` board poses spread over the combined field of both cameras at depth ``z``.

    ``span`` scales how far the sweep reaches: 1.0 covers both fields edge to edge, 0.5
    stays central. Positions are a jittered grid, then shuffled so a band that fills
    early still draws from the whole sweep rather than from one corner of it.
    """
    x_lo = -HALF_X * z * span
    x_hi = RIGHT_EYE_X + HALF_X * z * span
    y_lo, y_hi = -HALF_Y * z * span, HALF_Y * z * span

    cols = max(1, int(round(np.sqrt(n * (x_hi - x_lo) / max(y_hi - y_lo, 1e-6)))))
    rows = max(1, int(np.ceil(n / cols)))
    lo, hi = TILTS[tilt]

    poses = []
    for i in range(n):
        col, row = i % cols, (i // cols) % rows
        x = x_lo + (x_hi - x_lo) * ((col + 0.5) / cols) + rng.normal(0, jitter * (x_hi - x_lo) / cols)
        y = y_lo + (y_hi - y_lo) * ((row + 0.5) / rows) + rng.normal(0, jitter * (y_hi - y_lo) / rows)

        axis = rng.normal(0, 1, 3)
        axis[2] *= 0.4
        axis /= np.linalg.norm(axis)
        rvec = axis * rng.uniform(lo, hi)

        poses.append((rvec, np.array([x, y, z]) - cv2.Rodrigues(rvec)[0] @ t.BOARD_CENTER))
    rng.shuffle(poses)
    return poses


def capture(pattern, seed):
    """Run one pattern. Returns the detections and where the frames landed."""
    rng = np.random.default_rng(seed)
    objp = object_points(t.GRID, t.SQUARE_M)
    dets = []
    tally = {"left_only": 0, "right_only": 0, "both": 0, "attempts": 0}
    by_band = {}

    for band, (z, n) in pattern["bands"].items():
        counts = {"left_only": 0, "right_only": 0, "both": 0, "attempts": 0}
        by_band[band] = counts
        if n == 0:
            continue
        saved = 0
        for rvec, tvec in layout(n * 6, z, pattern["span"], rng, pattern["tilt"]):
            if saved >= n:
                break
            counts["attempts"] += 1
            left = t.project(objp, rvec, tvec, t.K_LEFT, t.DIST_LEFT, rng)
            rvec_r, tvec_r = t.right_eye_pose(rvec, tvec)
            right = t.project(objp, rvec_r, tvec_r, t.K_RIGHT, t.DIST_RIGHT, rng)
            if left is None and right is None:
                continue
            key = "both" if (left is not None and right is not None) else (
                "left_only" if left is not None else "right_only")
            counts[key] += 1
            saved += 1
            dets.append(Detection(f"/s/{band}_{len(dets):03d}.png", left, right, 100.0))
        for k, v in counts.items():
            tally[k] += v
    return dets, tally, by_band


# --- scoring --------------------------------------------------------------------------

def _project(objp, rvec, tvec, K, dist, rng):
    """Project with detector noise when ``rng`` is given, perfectly when it is None."""
    if rng is not None:
        return t.project(objp, rvec, tvec, K, dist, rng)
    corners = t.corners_in_frame(objp, rvec, tvec, K, dist)
    return None if corners is None else corners.reshape(-1, 1, 2).astype(np.float32)


def grid_check(result, seed):
    """``check_grid_quality`` against the true board, rectified with this calibration.

    Measured twice, because the single number an operator reads is two things added
    together:

    ``as measured`` puts the usual corner noise on the evaluation board. This is what the
    check stage shows on screen, and at these distances it is dominated by that noise
    rather than by the calibration.

    ``calibration only`` detects the evaluation board perfectly. That strips the
    measurement noise out and leaves the calibration's own contribution -- the part a
    capture pattern can actually change. A pattern comparison has to be read here; the
    measured column will barely move between good patterns.
    """
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        result["mtxL"], result["distL"], result["mtxR"], result["distR"],
        t.IMAGE_SIZE, result["R"], np.asarray(result["T"]).reshape(3, 1),
        alpha=0, flags=cv2.CALIB_ZERO_DISPARITY,
    )
    K_rect = P1[:3, :3]
    baseline = abs(float(P2[0, 3] / P2[0, 0]))
    objp = object_points(t.GRID, t.SQUARE_M)
    square_mm = t.SQUARE_M * 1000.0

    out = {}
    worst_y = 0.0
    for noisy in (True, False):
        for depth in CHECK_DEPTHS:
            rng = np.random.default_rng(seed + 90000 + int(depth * 100))
            means, maxes = [], []
            for k in range(CHECK_BOARDS):
                angle = 2.0 * np.pi * k / CHECK_BOARDS
                rvec = np.array([0.22 * np.sin(angle), 0.22 * np.cos(angle), 0.05])
                tvec = np.array([RIGHT_EYE_X / 2 + 0.10 * np.cos(angle) * depth,
                                 0.08 * np.sin(angle) * depth,
                                 depth]) - cv2.Rodrigues(rvec)[0] @ t.BOARD_CENTER

                noise = rng if noisy else None
                left = _project(objp, rvec, tvec, t.K_LEFT, t.DIST_LEFT, noise)
                rvec_r, tvec_r = t.right_eye_pose(rvec, tvec)
                right = _project(objp, rvec_r, tvec_r, t.K_RIGHT, t.DIST_RIGHT, noise)
                if left is None or right is None:
                    continue

                rl = cv2.undistortPoints(left, result["mtxL"], result["distL"], R=R1, P=P1).reshape(-1, 2)
                rr = cv2.undistortPoints(right, result["mtxR"], result["distR"], R=R2, P=P2).reshape(-1, 2)
                if noisy:
                    worst_y = max(worst_y, float(np.sqrt(np.mean((rl[:, 1] - rr[:, 1]) ** 2))))
                points_mm = points_3d_from_stereo(rl, rr, K_rect, K_rect, baseline) * 1000
                mean_err, max_err = check_grid_quality(points_mm, t.GRID, square_mm)
                means.append(float(mean_err))
                maxes.append(float(max_err))

            tag = ("measured" if noisy else "calib") + f"_{int(depth * 100)}"
            out[tag + "_mean"] = float(np.mean(means)) if means else float("nan")
            out[tag + "_max"] = float(np.mean(maxes)) if maxes else float("nan")

    for kind in ("measured", "calib"):
        out["grid_" + kind] = float(np.nanmean(
            [out[f"{kind}_{int(d * 100)}_mean"] for d in CHECK_DEPTHS]))
    out["y_rms_px"] = worst_y
    return out


def bench_grid(result, seed, boards=12, noisy=True):
    """The standard bench measurement: centred board at 1 m, square to the rig.

    Same computation as :func:`grid_check`, but at the single pose the shop figure is
    quoted at, so the synthetic model can be anchored to a real number.
    """
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        result["mtxL"], result["distL"], result["mtxR"], result["distR"],
        t.IMAGE_SIZE, result["R"], np.asarray(result["T"]).reshape(3, 1),
        alpha=0, flags=cv2.CALIB_ZERO_DISPARITY,
    )
    K_rect = P1[:3, :3]
    baseline = abs(float(P2[0, 3] / P2[0, 0]))
    objp = object_points(t.GRID, t.SQUARE_M)
    square_mm = t.SQUARE_M * 1000.0
    rng = np.random.default_rng(seed + 4242)

    means, maxes = [], []
    for _ in range(boards):
        # Centred in the shared field, square to the rig apart from a slight tilt: a
        # board held perfectly flat is a degenerate pose nobody actually achieves.
        rvec = rng.normal(0, 0.04, 3)
        tvec = np.array([RIGHT_EYE_X / 2, 0.0, BENCH_DEPTH]) - cv2.Rodrigues(rvec)[0] @ t.BOARD_CENTER

        noise = rng if noisy else None
        left = _project(objp, rvec, tvec, t.K_LEFT, t.DIST_LEFT, noise)
        rvec_r, tvec_r = t.right_eye_pose(rvec, tvec)
        right = _project(objp, rvec_r, tvec_r, t.K_RIGHT, t.DIST_RIGHT, noise)
        if left is None or right is None:
            continue
        rl = cv2.undistortPoints(left, result["mtxL"], result["distL"], R=R1, P=P1).reshape(-1, 2)
        rr = cv2.undistortPoints(right, result["mtxR"], result["distR"], R=R2, P=P2).reshape(-1, 2)
        points_mm = points_3d_from_stereo(rl, rr, K_rect, K_rect, baseline) * 1000
        mean_err, max_err = check_grid_quality(points_mm, t.GRID, square_mm)
        means.append(float(mean_err))
        maxes.append(float(max_err))

    if not means:
        return {"bench_mean": float("nan"), "bench_max": float("nan")}
    return {"bench_mean": float(np.mean(means)), "bench_max": float(np.mean(maxes))}


def score(result, seed):
    row = {
        "fx": abs(result["mtxL"][0, 0] - t.K_LEFT[0, 0]),
        "baseline_mm": abs(float(np.linalg.norm(np.asarray(result["T"]).reshape(3))) * 1000
                           - t.BASELINE_TRUE * 1000),
        "rotation_deg": t.rotation_error_deg(result["R"]),
        "lens_px": max(
            t.undistort_error_px(result["mtxL"], result["distL"], t.K_LEFT, t.DIST_LEFT),
            t.undistort_error_px(result["mtxR"], result["distR"], t.K_RIGHT, t.DIST_RIGHT),
        ),
    }
    row.update(grid_check(result, seed))
    row.update(bench_grid(result, seed))
    row["bench_calib"] = bench_grid(result, seed, noisy=False)["bench_mean"]
    return row


METRICS = ["fx", "baseline_mm", "rotation_deg", "lens_px", "y_rms_px",
           "grid_measured", "grid_calib", "bench_mean", "bench_max", "bench_calib"]
METRICS += [f"{kind}_{int(d * 100)}_{stat}" for kind in ("measured", "calib")
            for d in CHECK_DEPTHS for stat in ("mean", "max")]


def run(pattern, seeds=SEEDS):
    runs, tallies, bands, failures = [], [], [], 0
    for seed in range(1, seeds + 1):
        dets, tally, by_band = capture(pattern, seed)
        out = os.path.join(tempfile.mkdtemp(), "c.yaml")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = solve(dets, t.GRID, t.SQUARE_M,
                               (t.IMAGE_SIZE[1], t.IMAGE_SIZE[0], 3), out)
        except SystemExit as error:
            print(f"    seed {seed} FAILED: {error}")
            failures += 1
            continue
        row = score(result, seed)
        row["stereo_used"] = result["counts"]["stereo"]
        row["pruned"] = len(result["pruned"])
        runs.append(row)
        tallies.append(tally)
        bands.append(by_band)

    if not runs:
        return None

    summary = {"name": pattern["name"], "shots": pattern["shots"],
               "runs": len(runs), "failed_seeds": failures}
    for key in METRICS:
        values = [r[key] for r in runs]
        summary[key + "_mean"] = float(np.nanmean(values))
        summary[key + "_p90"] = float(np.nanpercentile(values, 90))
    for key in ("stereo_used", "pruned"):
        summary[key] = float(np.mean([r[key] for r in runs]))
    for key in ("left_only", "right_only", "both", "attempts"):
        summary[key] = float(np.mean([x[key] for x in tallies]))
    summary["bands"] = {
        band: {k: float(np.mean([b[band][k] for b in bands]))
               for k in ("left_only", "right_only", "both", "attempts")}
        for band in pattern["bands"]
    }
    return summary


def pattern(name, per_band, span, tilt, depths=(0.5, 1.0, 1.7)):
    close, mid, far = depths
    return {"name": name, "span": span, "tilt": tilt, "shots": sum(per_band),
            "bands": {"close": (close, per_band[0]), "medium": (mid, per_band[1]),
                      "far": (far, per_band[2])}}


PATTERNS = [
    pattern("A · baseline 90",          (30, 30, 30), span=1.00, tilt="medium"),
    pattern("B · half the shots 45",    (15, 15, 15), span=1.00, tilt="medium"),
    pattern("C · double the shots 180", (60, 60, 60), span=1.00, tilt="medium"),
    pattern("D · narrow sweep 90",      (30, 30, 30), span=0.55, tilt="medium"),
    pattern("E · low tilt 90",          (30, 30, 30), span=1.00, tilt="low"),
    pattern("F · high tilt 90",         (30, 30, 30), span=1.00, tilt="high"),
    pattern("G · no close band 90",     (0, 45, 45),  span=1.00, tilt="medium"),
    pattern("H · close-heavy 90",       (50, 25, 15), span=1.00, tilt="medium"),
]

COUNTS = [30, 45, 60, 90, 120, 180, 240]


def report(summary):
    print(f"    landed: {summary['both']:.0f} both, {summary['left_only']:.0f} L-only, "
          f"{summary['right_only']:.0f} R-only (from {summary['attempts']:.0f} attempts)")
    print(f"    BENCH grid @1m centred: {summary['bench_mean_p90']:.3f} mm mean, "
          f"{summary['bench_max_p90']:.3f} mm max   |  {summary['bench_calib_p90']:.4f} mm calibration-only")
    print(f"    lens {summary['lens_px_p90']:6.2f}px  fx {summary['fx_p90']:5.2f}px  "
          f"base {summary['baseline_mm_p90']:.3f}mm  y {summary['y_rms_px_p90']:.3f}px")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", action="store_true", help="shot-count sweep")
    parser.add_argument("--noise", action="store_true",
                        help="find the corner noise that reproduces a bench grid error")
    parser.add_argument("--target-grid", type=float, default=0.15,
                        help="bench grid error in mm to match, with --noise")
    parser.add_argument("--corner-noise", type=float, default=None,
                        help="override the detector noise, in pixels")
    parser.add_argument("--out", help="write the summary to this JSON file")
    args = parser.parse_args()

    if args.corner_noise is not None:
        t.CORNER_NOISE_PX = args.corner_noise
        print(f"corner noise set to {args.corner_noise} px\n")

    if args.noise:
        # Calibrate once, then re-measure the grid check at a range of detector noise
        # levels. Whichever reproduces the bench figure is this rig's real corner noise.
        dets, _, _ = capture(PATTERNS[0], 1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = solve(dets, t.GRID, t.SQUARE_M,
                           (t.IMAGE_SIZE[1], t.IMAGE_SIZE[0], 3),
                           os.path.join(tempfile.mkdtemp(), "c.yaml"))
        base = t.CORNER_NOISE_PX
        print(f"bench pose: centred board at {BENCH_DEPTH:.1f} m, "
              f"target mean {args.target_grid:.3f} mm\n")
        print(f"{'noise px':>9} | {'bench mean':>11} | {'bench max':>10} | "
              + " | ".join(f"{d:.1f} m".rjust(7) for d in CHECK_DEPTHS))
        best, best_gap = None, float("inf")
        for noise in (0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.40, 0.50):
            t.CORNER_NOISE_PX = noise
            b = bench_grid(result, 1, boards=24)
            g = grid_check(result, 1)
            gap = abs(b["bench_mean"] - args.target_grid)
            if gap < best_gap:
                best, best_gap = noise, gap
            row = " | ".join(f"{g[f'measured_{int(d * 100)}_mean']:7.3f}" for d in CHECK_DEPTHS)
            print(f"{noise:9.2f} | {b['bench_mean']:11.3f} | {b['bench_max']:10.3f} | {row}")
        print(f"\nclosest to {args.target_grid:.3f} mm at the bench pose: "
              f"{best:.2f} px corner noise")
        t.CORNER_NOISE_PX = base
        return 0

    rows = []
    if args.counts:
        for total in COUNTS:
            per = total // 3
            spec = pattern(f"{total} shots", (per, per, per), span=1.00, tilt="medium")
            print(f"{total} shots")
            summary = run(spec)
            if summary:
                summary["total"] = total
                rows.append(summary)
                report(summary)
    else:
        for spec in PATTERNS:
            print(f"{spec['name']}  ({spec['shots']} shots)")
            summary = run(spec)
            if summary:
                rows.append(summary)
                report(summary)

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
