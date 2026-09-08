"""Stage 4: look at the calibration you just computed.

A live stream with epipolar lines drawn across both rectified eyes. Whenever a
checkerboard is in shot the calibration check runs on its own: vertical disparity of
the matched corners (the direct measure of how well the pair is rectified) and the
metric grid check from examples/check_calibration.py.

Rectification is done here from the fresh calibration, not by the camera -- the camera
is still running whatever YAML was last deployed to it.
"""

import os
import time

import cv2
import numpy as np

from calibration.quality import find_corners_fast, split_stereo, to_gray
from calibration.window import bring_to_front, open_window
from examples.check_calibration import (
    check_grid_quality,
    find_checkerboard,
    points_3d_from_stereo,
)
from stereo_4d import Stereo4DCameraHandler

WINDOW = "4D calibration check"

# Alternating epipolar line colours, so a row is easy to follow across the pair.
LINE_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

# The rectified pair is 3840x1080; the canvas is built at display size so the overlay
# is drawn once, small, instead of being rescaled by the GUI every frame.
DEFAULT_DISPLAY_WIDTH = 1600

FONT = cv2.FONT_HERSHEY_SIMPLEX
GOOD = (0, 255, 0)
FAIR = (0, 200, 255)
POOR = (0, 0, 255)

# Vertical-disparity thresholds in pixels: under half a pixel the rows line up as well
# as sub-pixel corner detection can tell.
Y_RMS_GOOD = 0.5
Y_RMS_FAIR = 1.0

# Even gated, the check does not need to run on every frame.
CHECK_INTERVAL = 0.2

# Height of the darkened strip the two HUD lines are written into.
HUD_HEIGHT = 62


def rectification(calibration, size):
    """Rectification transforms and remap tables for a per-eye ``size`` of (w, h)."""
    mtxL = np.asarray(calibration["mtxL"], dtype=np.float64)
    distL = np.asarray(calibration["distL"], dtype=np.float64)
    mtxR = np.asarray(calibration["mtxR"], dtype=np.float64)
    distR = np.asarray(calibration["distR"], dtype=np.float64)
    R = np.asarray(calibration["R"], dtype=np.float64)
    T = np.asarray(calibration["T"], dtype=np.float64).reshape(3, 1)

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, size, R, T,
        alpha=0, flags=cv2.CALIB_ZERO_DISPARITY,
    )
    map_left = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, size, cv2.CV_16SC2)
    map_right = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, size, cv2.CV_16SC2)

    return {
        "P1": P1, "P2": P2, "Q": Q,
        "map_left": map_left, "map_right": map_right,
        # P2[0, 3] is -fx * baseline, so this is the true rectified baseline in metres.
        "baseline": abs(float(P2[0, 3] / P2[0, 0])),
        "K": P1[:3, :3],
    }


def draw_epipolar_lines(image, spacing):
    """Horizontal lines every ``spacing`` px, colours alternating line by line."""
    height, width = image.shape[:2]
    for i, y in enumerate(range(0, height, spacing)):
        cv2.line(image, (0, y), (width, y), LINE_COLORS[i % len(LINE_COLORS)], 1)


def measure(rect_left, rect_right, grid, square_mm, K, baseline_m):
    """Run the calibration check on a rectified pair. None if no board is in shot.

    Gated on the cheap detector: the full search costs ~290 ms a pair when there is no
    board to find, which is most of the time you are walking the target around, and that
    is what makes the window feel stuck.
    """
    if (find_corners_fast(to_gray(rect_left), grid) is None
            or find_corners_fast(to_gray(rect_right), grid) is None):
        return None

    corners_left, corners_right = find_checkerboard(
        rect_left.copy(), rect_right.copy(), grid
    )
    if corners_left is None:
        return None

    y_rms = float(
        np.sqrt(np.mean((corners_left[:, 1] - corners_right[:, 1]) ** 2))
    )
    points_mm = points_3d_from_stereo(
        corners_left, corners_right, K, K, baseline_m
    ) * 1000
    mean_err, max_err = check_grid_quality(points_mm, grid, square_mm)

    return {
        "corners_left": corners_left,
        "corners_right": corners_right,
        "y_rms": y_rms,
        "mean_err": float(mean_err),
        "max_err": float(max_err),
        "distance": float(np.mean(np.linalg.norm(points_mm, axis=1))),
    }


def _verdict_color(y_rms):
    if y_rms < Y_RMS_GOOD:
        return GOOD
    return FAIR if y_rms < Y_RMS_FAIR else POOR


def _render(rect_left, rect_right, result, grid, spacing, subtitle,
            display_width=DEFAULT_DISPLAY_WIDTH):
    # Shrink first, then draw: the lines and corners come out crisp at the size they are
    # actually looked at, and nothing is drawn at a resolution nobody sees.
    eye_width = max(1, display_width // 2)
    scale = eye_width / rect_left.shape[1]
    eye_height = round(rect_left.shape[0] * scale)
    left, right = (
        cv2.resize(eye, (eye_width, eye_height), interpolation=cv2.INTER_AREA)
        for eye in (rect_left, rect_right)
    )

    # Spacing is in displayed pixels: the lines are there to be read off the screen,
    # and tying them to the sensor resolution just makes them denser on a bigger sensor.
    draw_epipolar_lines(left, spacing)
    draw_epipolar_lines(right, spacing)

    if result is not None:
        for image, corners in ((left, result["corners_left"]),
                               (right, result["corners_right"])):
            cv2.drawChessboardCorners(
                image, grid,
                (corners * scale).reshape(-1, 1, 2).astype(np.float32), True
            )

    canvas = np.hstack((left, right))

    if result is None:
        headline, color = "Show the checkerboard to check the calibration", FAIR
    else:
        headline = (
            f"y-RMS {result['y_rms']:.2f} px | grid mean {result['mean_err']:.2f} mm "
            f"| max {result['max_err']:.2f} mm | dist {result['distance']:.0f} mm"
        )
        color = _verdict_color(result["y_rms"])

    # The epipolar lines run right under the text, so the HUD gets its own dark band.
    band = canvas[:HUD_HEIGHT]
    cv2.addWeighted(band, 0.25, np.zeros_like(band), 0.75, 0, band)

    cv2.putText(canvas, headline, (12, 26), FONT, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 50), FONT, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    return canvas


def _snapshot(canvas, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"check_{time.strftime('%H%M%S')}.png")
    cv2.imwrite(path, canvas)
    print(f"Saved {path}")
    return path


def _open_window(display_width):
    open_window(WINDOW, (display_width, display_width // 4))


def verify_live(calibration, grid, square_mm, ip="172.31.1.77", spacing=20,
                out_dir=".", timeout=60.0, display_width=DEFAULT_DISPLAY_WIDTH):
    """Live rectified preview. Returns False if the camera never came up."""
    handler = Stereo4DCameraHandler(ip=ip, rectify_internally=False)
    print(f"Connecting to camera at {ip} for the live check ...")
    if not handler.start(wait=True, timeout=timeout):
        handler.stop()
        return False

    rect = None
    result = None
    last_check = 0.0
    best = None
    raised = False
    _open_window(display_width)
    print("Live check running. 's' saves a snapshot, 'q' finishes.")

    try:
        while True:
            frame = handler.get_last_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            left, right = split_stereo(frame.image)
            if rect is None:
                rect = rectification(calibration, (left.shape[1], left.shape[0]))
                print(f"Rectified baseline: {rect['baseline'] * 1000:.1f} mm")

            rect_left = cv2.remap(left, *rect["map_left"], cv2.INTER_LINEAR)
            rect_right = cv2.remap(right, *rect["map_right"], cv2.INTER_LINEAR)

            now = time.time()
            if now - last_check > CHECK_INTERVAL:
                last_check = now
                result = measure(rect_left, rect_right, grid, square_mm,
                                 rect["K"], rect["baseline"])
                if result is not None and (best is None or result["y_rms"] < best["y_rms"]):
                    best = result

            canvas = _render(rect_left, rect_right, result, grid, spacing,
                             "s snapshot   q done", display_width)
            cv2.imshow(WINDOW, canvas)
            if not raised:
                bring_to_front(WINDOW)
                raised = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                _snapshot(canvas, out_dir)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)
        handler.stop()

    return best if best is not None else True


def verify_offline(calibration, grid, square_mm, paths, spacing=20, out_dir=".",
                   display_width=DEFAULT_DISPLAY_WIDTH):
    """Page through the session's own images with the same overlay, no camera needed."""
    if not paths:
        print("No images to review.")
        return None

    index = 0
    rect = None
    best = None
    raised = False
    _open_window(display_width)
    print("Offline check. 'n'/'p' or arrow keys to page, 's' snapshot, 'q' finishes.")

    try:
        while True:
            image = cv2.imread(paths[index])
            if image is None:
                index = (index + 1) % len(paths)
                continue

            left, right = split_stereo(image)
            if rect is None:
                rect = rectification(calibration, (left.shape[1], left.shape[0]))
                print(f"Rectified baseline: {rect['baseline'] * 1000:.1f} mm")

            rect_left = cv2.remap(left, *rect["map_left"], cv2.INTER_LINEAR)
            rect_right = cv2.remap(right, *rect["map_right"], cv2.INTER_LINEAR)
            result = measure(rect_left, rect_right, grid, square_mm,
                             rect["K"], rect["baseline"])
            if result is not None and (best is None or result["y_rms"] < best["y_rms"]):
                best = result

            canvas = _render(
                rect_left, rect_right, result, grid, spacing,
                f"[{index + 1}/{len(paths)}] {os.path.basename(paths[index])}"
                "   n/p page   s snapshot   q done",
                display_width,
            )
            cv2.imshow(WINDOW, canvas)
            if not raised:
                bring_to_front(WINDOW)
                raised = True

            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("n"), 83, 84):  # n, right, down
                index = (index + 1) % len(paths)
            elif key in (ord("p"), 81, 82):  # p, left, up
                index = (index - 1) % len(paths)
            elif key == ord("s"):
                _snapshot(canvas, out_dir)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)

    return best
