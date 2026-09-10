"""Stage 1: pull calibration images off the camera, guided.

The tool asks for one placement at a time -- a light-blue bracket drawn on the eye the
board should be in -- and shoots by itself once the board is in it, held still, sharp
enough and at an angle. The operator moves the board and watches the brackets fill in;
there is no interval to set and no shutter to press, though SPACE still forces a shot.

Three gates stand between a board being visible and a frame being saved, and each exists
because of something the study or the pipeline measured:

*Stillness* -- the two eyes are not exposed at the same instant, so a board in motion
lands in different places in the two images. Such a pair looks perfect in each eye alone
and is wrong as a pair; the pruner drops it from the extrinsics later. Gating on corner
motion enforces the hold-still rule mechanically instead of by instruction.

*Sharpness* -- Laplacian variance inside the board's own bounding box, judged against the
running median of what has been saved so far. A provisional call: stage 2 re-decides
against the whole set's median, which is the only threshold that follows the lighting.

*Tilt* -- a near-fronto-parallel set was the worst configuration in the capture study,
costing more focal-length error than halving the shot count. See
:mod:`calibration.targets`.

The board search still runs on a timer over a half-size copy with FAST_CHECK only: the
full search costs ~300 ms an eye when there is nothing to find, which is most frames while
the board is being carried around, and is what makes the window feel stuck.
"""

import glob
import os
import time

import cv2
import numpy as np

from calibration import hud
from calibration.quality import find_corners_fast, sharpness, split_stereo, to_gray
from calibration.targets import (
    ACCEPT_SCORE,
    NEAR_SCORE,
    TargetPlan,
)
from calibration.window import bring_to_front, open_window
from stereo_4d import Stereo4DCameraHandler

WINDOW = "4D calibration capture"
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")

# Full HD wide. The stereo frame is 3840x1080, so this renders at 1920x540 -- half the
# height of a 1080p screen, with the guides at half sensor scale. Anything narrower throws
# away screen for nothing: the operator is reading these labels from across the room.
DEFAULT_PREVIEW_WIDTH = 1920
DEFAULT_TOTAL_SHOTS = 100

# How often the board search re-runs. It drives the guides, not the stream.
BOARD_CHECK_INTERVAL = 0.12
# Corner motion below this, between consecutive checks, counts as still (full-res px).
STILL_MOTION_PX = 2.2
# How long it has to stay still before the shutter releases.
STILL_SECONDS = 0.35
# Sharpness must clear this fraction of the running median of what has been saved.
BLUR_RATIO = 0.6
# Never shoot the same target twice within this many seconds.
SHOT_COOLDOWN = 0.6
# The board has to have actually moved since the last saved frame, in full-res pixels of
# median corner displacement. Without this, one good position quietly satisfies the next
# queued target too and the tool fires several times over while the operator stands still.
MOVED_SINCE_SHOT_PX = 45.0
# The preview is mirrored: the operator stands in front of the camera looking at the
# screen, so an unmirrored image sends them the wrong way every time.
MIRROR_PREVIEW = True

HUD_HEIGHT = 138
FOOTER_HEIGHT = 34


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


class Stillness:
    """Tracks whether the board has stopped moving, and whether the stream is alive.

    A frozen stream reads as perfectly still, which is exactly the condition that would
    make the tool save the same stale image over and over, so the frame's own timestamp
    has to advance as well.
    """

    def __init__(self):
        self.reference = None
        self.since = None
        self.last_stamp = None
        self.motion = float("inf")

    def update(self, corners, stamp, now):
        if stamp is not None and stamp == self.last_stamp:
            self.since = None
            self.motion = float("inf")
            return False
        self.last_stamp = stamp

        if corners is None:
            self.reference = None
            self.since = None
            self.motion = float("inf")
            return False

        pts = np.asarray(corners).reshape(-1, 2)
        if self.reference is not None and self.reference.shape == pts.shape:
            self.motion = float(np.median(np.linalg.norm(pts - self.reference, axis=1)))
        else:
            self.motion = float("inf")
        self.reference = pts

        if self.motion <= STILL_MOTION_PX:
            if self.since is None:
                self.since = now
        else:
            self.since = None
        return self.held(now)

    def held(self, now):
        return self.since is not None and (now - self.since) >= STILL_SECONDS

    def progress(self, now):
        if self.since is None:
            return 0.0
        return min(1.0, (now - self.since) / STILL_SECONDS)


def _to_screen_x(x_full, eye, scale, width):
    """One x in an eye's full-res pixels -> x on the (mirrored) preview."""
    x = x_full * scale + (0 if eye == "left" else width // 2)
    return width - x if MIRROR_PREVIEW else x


def _to_screen(points, eye, scale, width):
    """Full-res points in one eye -> preview points, mirrored to match the operator."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    pts[:, 0] = pts[:, 0] * scale + (0 if eye == "left" else width // 2)
    pts[:, 1] *= scale
    if MIRROR_PREVIEW:
        pts[:, 0] = width - pts[:, 0]
    return pts


def _draw_targets(preview, plan, scale, state):
    """The wanted board pose, the position after it, and a dot per pose already covered."""
    width = preview.shape[1]
    target = plan.current

    for eye in ("left", "right"):
        for done in plan.completed(eye):
            x, y, w, h = done.box
            cx, cy = _to_screen([[x + w / 2, y + h / 2]], eye, scale, width)[0]
            cv2.circle(preview, (int(cx), int(cy)), 3, hud.DONE, -1, cv2.LINE_AA)

    if target is None:
        return

    # The guide is coloured by how close the board is to matching it -- red, through amber,
    # to green -- so the operator can steer by the outline instead of reading a number.
    score = state.get("score")
    color = hud.ACCENT if score is None else hud.match_color(score, ACCEPT_SCORE, NEAR_SCORE)
    hud.board_guide(preview, _to_screen(target.guide_corners(), target.eye, scale, width),
                    plan.grid, color, thickness=2)

    # The stillness bar appears only once the pose is right. Before that it is answering a
    # question nobody is asking: the operator is still moving on purpose.
    if state.get("matched"):
        x, y, w, h = target.box
        left_x = _to_screen_x(x + w, target.eye, scale, width) if MIRROR_PREVIEW \
            else _to_screen_x(x, target.eye, scale, width)
        y_bar = min(y * scale + h * scale + 10, preview.shape[0] - FOOTER_HEIGHT - 10)
        hud.progress_bar(preview, (left_x, y_bar, w * scale, 5),
                         state["still_progress"], hud.DONE)


def _draw_hud(preview, plan, saved, state):
    """Header: what to do now on the left, how far through on the right.

    Two cards rather than one full-width band. The close-band targets are nearly
    frame-sized and their top row sits under the header, so a bar spanning the width hides
    the very bracket the operator is aiming at; leaving the middle clear keeps the image
    visible where the guides actually are.
    """
    width = preview.shape[1]
    target = plan.current

    if target is None:
        hud.panel(preview, (0, 0, 420, HUD_HEIGHT))
        hud.label(preview, "Coverage complete", (16, 46), hud.DONE, scale=0.8)
        hud.label(preview, f"{saved} frames saved  -  press q to solve",
                  (16, 82), hud.INK_MUTED, scale=0.56, small=True)
        return

    # --- left card: the instruction ---
    done_here, total_here = plan.pose_progress()
    headline = (f"{target.eye.upper()} EYE   {target.band}   {target.depth:.1f} m"
                f"   pose {done_here + 1}/{total_here}")
    (head_w, _), _ = cv2.getTextSize(headline, hud.FONT, 0.72, 1)
    (hint_w, _), _ = cv2.getTextSize(state["hint"], hud.FONT_SMALL, 0.56, 1)
    card_w = max(head_w, hint_w, 300) + 32
    hud.panel(preview, (0, 0, card_w, HUD_HEIGHT))

    hud.label(preview, headline, (16, 34), hud.INK, scale=0.72)
    hud.label(preview, target.pose.upper(), (16, 62), hud.ACCENT, scale=0.62)
    hud.label(preview, state["hint"], (16, 88),
              hud.DONE if state["ready"] else hud.INK_MUTED, scale=0.56, small=True)

    x = 16
    score = state.get("score")
    chips = [(("match %d%%" % round(100 * max(0.0, 1 - score / NEAR_SCORE)))
              if score is not None and np.isfinite(score) else "no board",
              state.get("matched", False), True)]
    if state.get("matched"):
        chips.append(("still" if state["still"] else "moving", state["still"], True))
        chips.append(("sharp" if state["sharp_ok"] else "soft", state["sharp_ok"], True))
    for text, ok, shown in chips:
        if not shown:
            continue
        x += hud.chip(preview, text, (x, 100), hud.DONE if ok else hud.WARN,
                      filled=ok) + 7

    # --- right card: progress ---
    bands = plan.band_progress()
    parts = "   ".join(f"{name} {done}/{want}" for name, (done, want) in bands.items())
    (parts_w, _), _ = cv2.getTextSize(parts, hud.FONT_SMALL, 0.5, 1)
    right_w = max(parts_w + 32, 280)
    right_x = width - right_w
    hud.panel(preview, (right_x, 0, right_w, HUD_HEIGHT))

    counter = f"{plan.taken} / {plan.wanted}"
    (tw, _), _ = cv2.getTextSize(counter, hud.FONT, 0.72, 1)
    hud.label(preview, counter, (width - tw - 16, 34), hud.INK, scale=0.72)
    hud.progress_bar(preview, (right_x + 16, 52, right_w - 32, 6),
                     plan.taken / max(1, plan.wanted))
    hud.label(preview, parts, (width - parts_w - 16, 86), hud.INK_MUTED,
              scale=0.5, small=True)


def _draw_footer(preview, saved):
    """Key hints on the left, frame count on the right. Partial-width, as the header."""
    height, width = preview.shape[:2]
    y = height - FOOTER_HEIGHT
    keys = "space shoot   s skip pose   x skip spot   q done"
    spaced = " ".join(keys.upper())
    (kw, _), _ = cv2.getTextSize(spaced, hud.FONT_SMALL, 0.44, 1)
    hud.panel(preview, (0, y, kw + 32, FOOTER_HEIGHT), alpha=0.66)
    hud.caption(preview, keys, (16, y + 22), hud.INK_MUTED, scale=0.44)

    text = f"{saved} saved"
    (tw, _), _ = cv2.getTextSize(text, hud.FONT_SMALL, 0.54, 1)
    hud.panel(preview, (width - tw - 32, y, tw + 32, FOOTER_HEIGHT), alpha=0.66)
    hud.label(preview, text, (width - tw - 16, y + 22), hud.INK_MUTED,
              scale=0.54, small=True)


def capture_images(out_dir, ip="172.31.1.77", interval=None, grid=None, timeout=60.0,
                   preview_width=DEFAULT_PREVIEW_WIDTH, square_mm=25.0,
                   total_shots=DEFAULT_TOTAL_SHOTS):
    """Stream from the camera and save guided side-by-side frames into ``out_dir``.

    Returns the list of saved paths. Frames are saved raw: rectifying them here would
    destroy the very distortion the calibration has to measure. ``interval`` is accepted
    and ignored -- the shutter is driven by the targets now, not by a clock.
    """
    os.makedirs(out_dir, exist_ok=True)

    handler = Stereo4DCameraHandler(ip=ip, rectify_internally=False)
    print(f"Connecting to camera at {ip} ...")
    if not handler.start(wait=True, timeout=timeout):
        handler.stop()
        raise SystemExit("Camera did not start. Is it powered and on the network?")

    saved = []
    sharpness_log = []
    plan = None
    still = Stillness()
    last_shot_corners = None
    corners = {"left": None, "right": None}
    state = {"hint": "Looking for the board", "ready": False, "near": False,
             "matched": False, "score": None, "still": False, "sharp_ok": True,
             "still_progress": 0.0}

    preview_size = None
    scale = 1.0
    raised = False
    last_check = 0.0
    last_shot = 0.0
    open_window(WINDOW)
    print(f"Capturing to {out_dir}. Match the blue board outline; q when you are done.")

    def shoot(frame, reason):
        path = os.path.join(out_dir, f"frame_{len(saved):03d}.png")
        cv2.imwrite(path, frame.image)
        saved.append(path)
        print(f"Saved {os.path.basename(path)}  ({reason})")
        return path

    try:
        while True:
            frame = handler.get_last_frame()
            if frame is None:
                time.sleep(1 / 60)
                continue

            left, right = split_stereo(frame.image)
            if plan is None:
                plan = TargetPlan(grid, square_mm / 1000.0,
                                  (left.shape[1], left.shape[0]),
                                  total_shots=total_shots) if grid else None
                if plan:
                    print(f"Plan: {plan.wanted} frames, {len(plan.targets) // 3} "
                          f"positions x 3 poses, {left.shape[1]}x{left.shape[0]} per eye, "
                          f"fx {plan.fx}")
                    seen = []
                    for t_ in plan.targets:
                        if t_.band in seen:
                            continue
                        seen.append(t_.band)
                        count = sum(1 for o in plan.targets if o.band == t_.band)
                        print(f"  {t_.band:11} {count // 3:2d} positions, board reads "
                              f"{t_.expected_width:.0f} px "
                              f"({t_.expected_width / left.shape[1] * 100:.0f}% of frame) "
                              f"at about {t_.depth:.2f} m")

            now = time.time()
            if grid is not None and now - last_check > BOARD_CHECK_INTERVAL:
                last_check = now
                corners["left"] = find_corners_fast(to_gray(left), grid)
                corners["right"] = find_corners_fast(to_gray(right), grid)

                target = plan.current if plan else None
                wanted = corners[target.eye] if target else None
                still.update(wanted, getattr(frame, "timestamp", None), now)

                ok, reason = plan.check(corners["left"], corners["right"]) if plan else (False, "")
                score = target.score(wanted, grid) if (target and wanted is not None) \
                    else float("inf")
                state["score"] = score
                state["near"] = wanted is not None
                state["matched"] = bool(ok)
                state["still"] = still.held(now)
                state["still_progress"] = still.progress(now)

                sharp_ok = True
                if wanted is not None:
                    gray = to_gray(left if target.eye == "left" else right)
                    value = sharpness(gray, wanted)
                    if sharpness_log:
                        sharp_ok = value >= np.median(sharpness_log) * BLUR_RATIO
                    state["sharp_value"] = value
                state["sharp_ok"] = sharp_ok

                state["ready"] = bool(ok and state["still"] and sharp_ok)
                if plan and plan.finished:
                    state["hint"] = "All positions covered"
                elif not state["near"]:
                    state["hint"] = f"Show the board to the {target.eye} camera" if target else ""
                elif not ok:
                    state["hint"] = reason.capitalize()
                elif not sharp_ok:
                    state["hint"] = "Too soft -- steady it"
                elif not state["still"]:
                    state["hint"] = "Hold still"
                else:
                    state["hint"] = "Holding..."

                # The board must have moved since the last saved frame. One good
                # position otherwise satisfies the next queued pose as well, and the tool
                # fires two or three times over while the operator stands still.
                moved = True
                if last_shot_corners is not None and wanted is not None:
                    here = np.asarray(wanted).reshape(-1, 2)
                    if here.shape == last_shot_corners.shape:
                        moved = float(np.median(np.linalg.norm(
                            here - last_shot_corners, axis=1))) > MOVED_SINCE_SHOT_PX
                state["moved"] = moved
                if state["ready"] and not moved:
                    state["hint"] = "Change the pose"

                if state["ready"] and moved and now - last_shot > SHOT_COOLDOWN:
                    last_shot = now
                    if plan.credit(corners["left"], corners["right"]):
                        sharpness_log.append(state.get("sharp_value", 0.0))
                        shoot(frame, f"{target.band} {target.eye} {target.pose}")
                        last_shot_corners = np.asarray(wanted).reshape(-1, 2).copy()
                        still.since = None

            if preview_size is None:
                height, width = frame.image.shape[:2]
                preview_size = (preview_width, round(preview_width * height / width))
                scale = (preview_width // 2) / left.shape[1]
                cv2.resizeWindow(WINDOW, *preview_size)

            preview = cv2.resize(frame.image, preview_size, interpolation=cv2.INTER_NEAREST)
            if MIRROR_PREVIEW:
                # Mirrored, because the operator stands in front of the camera looking at
                # the screen: unmirrored, every instruction sends them the wrong way. The
                # image is flipped first and the overlays are placed in mirrored
                # coordinates, so the guides move with the operator but the text does not
                # come out backwards.
                preview = cv2.flip(preview, 1)
            cv2.line(preview, (preview_size[0] // 2, HUD_HEIGHT),
                     (preview_size[0] // 2, preview_size[1] - FOOTER_HEIGHT),
                     hud.RULE, 1, cv2.LINE_AA)
            if plan is not None:
                # Panels first, guides over them. The guides tile the whole frame, so a
                # card drawn last hides whichever placement sits behind it -- which is
                # exactly the one the operator is being asked to fill.
                _draw_hud(preview, plan, len(saved), state)
                _draw_targets(preview, plan, scale, state)
            _draw_footer(preview, len(saved))

            cv2.imshow(WINDOW, preview)
            if not raised:
                bring_to_front(WINDOW)
                raised = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and plan is not None:
                plan.skip()
                still.since = None
            if key == ord("x") and plan is not None:
                plan.skip_position()
                still.since = None
            if key == ord(" "):
                last_shot = time.time()
                if plan is not None:
                    plan.credit(corners["left"], corners["right"])
                    target = plan.current
                    if target is not None and corners[target.eye] is not None:
                        last_shot_corners = np.asarray(
                            corners[target.eye]).reshape(-1, 2).copy()
                shoot(frame, "manual")
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
