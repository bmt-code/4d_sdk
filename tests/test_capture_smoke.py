#!/usr/bin/env python3
"""Draws the capture window without a camera.

    python3 tests/test_capture_smoke.py

The capture loop cannot be exercised on a desk, so the drawing it does gets exercised
here instead: the same canvas, composed the same way, for the states the loop actually
passes through. Twice now a rename has left the loop calling a drawing function with an
argument nobody set, which shows up only when someone stands in front of a camera. This
catches that in a second, and nothing here needs a window -- the canvas is an array.

Run `python3 -m pyflakes calibration/*.py` alongside it: an undefined name inside the
loop body itself is out of reach from here, and that is exactly what pyflakes reports.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import calibration.capture as capture  # noqa: E402
from calibration.targets import TargetPlan  # noqa: E402

FAILURES = []
GRID = (9, 6)
SQUARE_M = 0.025
EYE = (1920, 1080)
PREVIEW_WIDTH = 1280


def check(name, ok):
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        FAILURES.append(name)


def compose(plan, state, corners, saved=7):
    """Exactly what the capture loop builds, minus the camera."""
    scale = PREVIEW_WIDTH / EYE[0]
    eye_h = round(EYE[1] * scale)
    canvas = np.zeros((capture.HUD_HEIGHT + eye_h + capture.FOOTER_HEIGHT,
                       PREVIEW_WIDTH, 3), np.uint8)
    canvas[capture.HUD_HEIGHT:capture.HUD_HEIGHT + eye_h] = 60
    capture._draw_hud(canvas, plan, saved, state)
    capture._draw_targets(canvas, plan, scale, capture.HUD_HEIGHT, state)
    capture._draw_footer(canvas, saved, corners)
    return canvas


def state_for(**over):
    base = {"hint": "Match the outline", "ready": False, "near": False, "matched": False,
            "score": None, "still": False, "sharp_ok": True, "still_progress": 0.0,
            "moved": True}
    base.update(over)
    return base


def main():
    plan = TargetPlan(GRID, SQUARE_M, EYE)
    board = plan.current.guide_corners().reshape(-1, 1, 2).astype(np.float32)

    cases = {
        "no board in shot": (state_for(hint="Show the board to the left camera"),
                             {"left": None, "right": None}),
        "board seen, not matched": (state_for(near=True, score=0.9),
                                    {"left": board, "right": None}),
        "matched, still moving": (state_for(near=True, matched=True, score=0.02,
                                            still_progress=0.4),
                                  {"left": board, "right": board}),
        "matched and held": (state_for(near=True, matched=True, ready=True, score=0.0,
                                       still=True, still_progress=1.0),
                             {"left": board, "right": board}),
        "soft": (state_for(near=True, matched=True, score=0.02, sharp_ok=False),
                 {"left": board, "right": None}),
    }
    for name, (state, corners) in cases.items():
        try:
            canvas = compose(plan, state, corners)
            ok = canvas.shape[1] == PREVIEW_WIDTH and canvas.any()
        except Exception as error:  # noqa: BLE001 -- the point is to catch anything
            print(f"     {type(error).__name__}: {error}")
            ok = False
        check(f"draws '{name}'", ok)

    # The right eye is shown for a right-eye target, and the guide lands in the frame.
    right = next(t for t in plan.targets if t.eye == "right")
    while plan.current is not right:
        plan.skip()
    try:
        canvas = compose(plan, state_for(near=True, score=0.5), {"left": None, "right": board})
        ok = canvas.any()
    except Exception as error:  # noqa: BLE001
        print(f"     {type(error).__name__}: {error}")
        ok = False
    check("draws a right-eye target", ok)

    # And the finished state, which takes a different branch through the header.
    for target in plan.targets:
        target.taken = target.shots
    try:
        canvas = compose(plan, state_for(hint="All positions covered"),
                         {"left": None, "right": None})
        ok = plan.finished and canvas.any()
    except Exception as error:  # noqa: BLE001
        print(f"     {type(error).__name__}: {error}")
        ok = False
    check("draws the completed state", ok)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
