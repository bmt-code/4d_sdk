"""Stage 1: pull calibration images off the camera.

Same flow as examples/image_saver.py -- a shot every couple of seconds with a white
flash for a shutter -- plus a board indicator so you can see coverage while you move
the target, and a manual trigger.
"""

import glob
import os
import time

import cv2
import numpy as np

from calibration.quality import find_corners_fast, split_stereo, to_gray
from calibration.window import bring_to_front, open_window
from stereo_4d import Stereo4DCameraHandler

WINDOW = "4D calibration capture"
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")

# The window is a fraction of the 3840x1080 frame, so the preview is built at display
# size and the overlay drawn on that: the text stays crisp and the GUI is not asked to
# rescale twelve megabytes every frame.
DEFAULT_PREVIEW_WIDTH = 1600

FONT = cv2.FONT_HERSHEY_SIMPLEX
OK_COLOR = (0, 255, 0)
BAD_COLOR = (0, 0, 255)
INFO_COLOR = (255, 0, 255)

# How often the board indicator re-runs. It is only a HUD hint, so it does not need
# to keep up with the stream.
BOARD_CHECK_INTERVAL = 0.3

# The coverage map divides each eye into this many cells. A cell lights up once a saved
# frame put board corners in it, so you can see which parts of the image -- the corners
# above all -- still need the board taken to them.
COVERAGE_GRID = (6, 4)
COVERAGE_CELL = 16
COVERAGE_MARGIN = 10

# Height of the darkened strip the overlay is written into.
HUD_HEIGHT = 112


def list_images(folder):
    """Every stereo image in ``folder``, sorted."""
    paths = []
    for pattern in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(paths)


def collect_folder(folder):
    """Validate an existing image folder and return its images."""
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise SystemExit(f"Not a directory: {folder}")

    paths = list_images(folder)
    if not paths:
        raise SystemExit(f"No images found in {folder}")

    probe = cv2.imread(paths[0])
    if probe is None:
        raise SystemExit(f"Could not read {paths[0]}")
    if probe.shape[1] % 2:
        raise SystemExit(
            f"{paths[0]} is {probe.shape[1]} px wide (odd) -- these do not look like "
            "side-by-side stereo frames."
        )

    print(f"Using {len(paths)} image(s) from {folder}")
    return paths


def _mark_coverage(covered, corners, eye_width, eye_height):
    """Record which cells of an eye this board landed in."""
    cols, rows = COVERAGE_GRID
    for x, y in corners.reshape(-1, 2):
        col = min(cols - 1, max(0, int(x / eye_width * cols)))
        row = min(rows - 1, max(0, int(y / eye_height * rows)))
        covered.add((col, row))


def _draw_coverage(image, covered, origin):
    """A small filled grid showing where the board has already been."""
    cols, rows = COVERAGE_GRID
    x0, y0 = origin
    for row in range(rows):
        for col in range(cols):
            top_left = (x0 + col * COVERAGE_CELL, y0 + row * COVERAGE_CELL)
            bottom_right = (top_left[0] + COVERAGE_CELL - 2,
                            top_left[1] + COVERAGE_CELL - 2)
            if (col, row) in covered:
                cv2.rectangle(image, top_left, bottom_right, OK_COLOR, -1)
            else:
                cv2.rectangle(image, top_left, bottom_right, (120, 120, 120), 1)


def _overlay(image, saved, next_in, board, grid, coverage=None):
    lines = [(f"Saved: {saved}", INFO_COLOR)]
    if grid is not None:
        left_ok, right_ok = board
        # One eye is enough: a board held close against an edge cannot be in both, and
        # those frames are what fit that eye's distortion at the image corners.
        lines.append((
            f"Board  L {'OK' if left_ok else '--'}  R {'OK' if right_ok else '--'}",
            OK_COLOR if (left_ok or right_ok) else BAD_COLOR,
        ))
    lines.append((f"Next shot in {max(0.0, next_in):.1f}s", INFO_COLOR))
    lines.append(("SPACE shoot now   q done", INFO_COLOR))

    band = image[:HUD_HEIGHT]
    cv2.addWeighted(band, 0.25, np.zeros_like(band), 0.75, 0, band)

    for i, (text, color) in enumerate(lines):
        cv2.putText(image, text, (12, 26 + i * 24), FONT, 0.65, color, 2, cv2.LINE_AA)

    if coverage is not None:
        half = image.shape[1] // 2
        width = COVERAGE_GRID[0] * COVERAGE_CELL
        for eye, origin_x in (("left", half - COVERAGE_MARGIN - width),
                              ("right", image.shape[1] - COVERAGE_MARGIN - width)):
            _draw_coverage(image, coverage[eye], (origin_x, COVERAGE_MARGIN))


def capture_images(out_dir, ip="172.31.1.77", interval=2.0, grid=None, timeout=60.0,
                   preview_width=DEFAULT_PREVIEW_WIDTH):
    """Stream from the camera and save side-by-side frames into ``out_dir``.

    Returns the list of saved paths. Frames are saved raw: rectifying them here would
    destroy the very distortion the calibration has to measure.
    """
    os.makedirs(out_dir, exist_ok=True)

    handler = Stereo4DCameraHandler(ip=ip, rectify_internally=False)
    print(f"Connecting to camera at {ip} ...")
    if not handler.start(wait=True, timeout=timeout):
        handler.stop()
        raise SystemExit("Camera did not start. Is it powered and on the network?")

    saved = []
    board = (False, False)
    coverage = {"left": set(), "right": set()} if grid is not None else None
    last_shot = time.time()
    last_board_check = 0.0

    preview_size = None
    raised = False
    open_window(WINDOW)
    print(f"Capturing to {out_dir}. SPACE for a manual shot, q when you are done.")

    def shoot(frame):
        path = os.path.join(out_dir, f"frame_{len(saved):03d}.png")
        cv2.imwrite(path, frame.image)
        saved.append(path)
        return path

    try:
        while True:
            time.sleep(1 / 30)
            frame = handler.get_last_frame()
            if frame is None:
                continue

            now = time.time()
            if grid is not None and now - last_board_check > BOARD_CHECK_INTERVAL:
                last_board_check = now
                left, right = split_stereo(frame.image)
                corners = {
                    "left": find_corners_fast(to_gray(left), grid),
                    "right": find_corners_fast(to_gray(right), grid),
                }
                board = (corners["left"] is not None, corners["right"] is not None)
                for eye, found in corners.items():
                    if found is not None:
                        _mark_coverage(coverage[eye], found,
                                       left.shape[1], left.shape[0])

            if preview_size is None:
                height, width = frame.image.shape[:2]
                preview_size = (preview_width, round(preview_width * height / width))
                cv2.resizeWindow(WINDOW, *preview_size)

            preview = cv2.resize(frame.image, preview_size,
                                 interpolation=cv2.INTER_NEAREST)
            _overlay(preview, len(saved), interval - (now - last_shot), board, grid,
                     coverage)
            cv2.imshow(WINDOW, preview)
            if not raised:
                # The camera is live and you are about to stand in front of it, so the
                # preview should not be buried behind the terminal that started it.
                bring_to_front(WINDOW)
                raised = True

            # Read the key every iteration -- image_saver.py only polls on the frames
            # it is not saving, which makes it feel unresponsive on the shutter.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            manual = key == ord(" ")
            if manual or now - last_shot > interval:
                last_shot = now
                print(f"Saved {shoot(frame)}")
                # Flash white as the shutter, exactly as image_saver.py does.
                cv2.imshow(WINDOW, 255 * np.ones_like(preview))
                cv2.waitKey(20)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)
        handler.stop()

    print(f"Captured {len(saved)} image(s).")
    return saved
