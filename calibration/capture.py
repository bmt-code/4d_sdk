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

from calibration import hud, sound
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
# Held to what the desync actually needs rather than to what looks still: the two eyes are
# a couple of milliseconds apart, so even 5 px per 0.12 s check -- about 40 px/s -- puts
# under a tenth of a pixel between them, far inside the corner noise. Tighter than this
# only fights the operator's hands.
STILL_MOTION_PX = 5.0
# How long it has to stay still before the shutter releases.
STILL_SECONDS = 0.30
# Sharpness must clear this fraction of the running median of what has been saved.
BLUR_RATIO = 0.6
# Never shoot the same target twice within this many seconds.
SHOT_COOLDOWN = 0.6
# The board has to have actually moved since the last saved frame, in full-res pixels of
# the *worst* corner's displacement. Without some such gate a stationary board satisfies
# every queued target in turn and the tool fires several times over while the operator
# stands still -- especially now the match tolerance is loose enough not to object.
#
# Measured on the worst corner, not the median. Turning the board pivots it about its
# centre, so the middle corners barely move and the median stays around 20-30 px for a
# perfectly good pose change -- under the old 45 px median gate the tool refused the very
# thing it had just asked for, sat at a full stillness gauge, and told the operator to
# change the pose they had already changed. The worst corner never moves less than 45 px
# for a real re-pose, and only detector noise for a board held still, so one threshold
# separates them with room to spare.
MOVED_SINCE_SHOT_PX = 25.0
# The preview is mirrored: the operator stands in front of the camera looking at the
# screen, so an unmirrored image sends them the wrong way every time.
MIRROR_PREVIEW = True

HUD_HEIGHT = 132
FOOTER_HEIGHT = 32


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


def _to_screen(points, scale, origin_y, pane_x, pane_w):
    """Full-res points in one eye -> canvas pixels.

    Mirrored within the pane rather than across the canvas: with both eyes shown, flipping
    the whole strip would swap which camera is on which side.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2) * scale
    if MIRROR_PREVIEW:
        pts[:, 0] = pane_w - 1 - pts[:, 0]
    pts[:, 0] += pane_x
    pts[:, 1] += origin_y
    return pts


def _draw_targets(canvas, plan, scale, origin_y, state):
    """The wanted board pose, and a dot per pose already covered in this eye."""
    width = canvas.shape[1]
    target = plan.current
    if target is None:
        return


    for done in plan.completed(target.eye):
        x, y, w, h = done.box
        cx, cy = _to_screen([[x + w / 2, y + h / 2]], scale, origin_y, 0, width)[0]
        cv2.circle(canvas, (int(cx), int(cy)), 3, hud.DONE, -1, cv2.LINE_AA)

    # Coloured by how close the board is to matching -- red, through amber, to green -- so
    # the operator can steer by the outline instead of reading a number.
    score = state.get("score")
    color = hud.ACCENT if score is None else hud.match_color(score, ACCEPT_SCORE, NEAR_SCORE)
    lattice = target.guide_full()
    hud.board_guide(canvas,
                    _to_screen(lattice.reshape(-1, 2), scale, origin_y,
                               0, width).reshape(lattice.shape),
                    color, thickness=4)

    # The stillness bar appears only once the pose is right. Before that it is answering a
    # question nobody is asking: the operator is still moving on purpose.
    if state.get("matched"):
        pts = _to_screen(lattice.reshape(-1, 2), scale, origin_y, 0, width)
        bar_w = float(pts[:, 0].max() - pts[:, 0].min())
        y_bar = min(pts[:, 1].max() + 10, canvas.shape[0] - FOOTER_HEIGHT - 8)
        hud.progress_bar(canvas, (pts[:, 0].min(), y_bar, bar_w, 5),
                         state["still_progress"], hud.DONE)


def _draw_hud(canvas, plan, saved, state):
    """The header strip. Drawn above the image, never over it.

    Anything painted on the frame competes with the one thing the operator is trying to
    read -- their board against the guide -- and the close bands put a guide under wherever
    the header would sit. So the header gets its own band of canvas instead.
    """
    width = canvas.shape[1]
    canvas[:HUD_HEIGHT] = hud.PANEL
    cv2.line(canvas, (0, HUD_HEIGHT - 1), (width, HUD_HEIGHT - 1), hud.RULE, 1)

    target = plan.current
    if target is None:
        hud.label(canvas, "Coverage complete", (16, 40), hud.DONE, scale=0.8)
        hud.label(canvas, f"{saved} frames saved  -  press q to solve",
                  (16, 74), hud.INK_MUTED, scale=0.56, small=True)
        return

    done_here, total_here = plan.pose_progress()
    hud.label(canvas, f"{target.eye.upper()} EYE   {target.band}   {target.depth:.2f} m",
              (16, 30), hud.INK, scale=0.68)
    stance = "UPRIGHT" if target.upright else "FLAT"
    hud.label(canvas, f"{stance}, {target.pose.upper()}   (pose {done_here + 1}/{total_here})",
              (16, 58), hud.ACCENT, scale=0.6)
    hud.label(canvas, state["hint"], (16, 84),
              hud.DONE if state["ready"] else hud.INK_MUTED, scale=0.56, small=True)

    x = 16
    score = state.get("score")
    chips = [(("match %d%%" % round(100 * max(0.0, 1 - score / NEAR_SCORE)))
              if score is not None and np.isfinite(score) else "no board",
              state.get("matched", False))]
    if state.get("matched"):
        chips.append(("still" if state["still"] else "moving", state["still"]))
        chips.append(("sharp" if state["sharp_ok"] else "soft", state["sharp_ok"]))
    for text, ok in chips:
        x += hud.chip(canvas, text, (x, HUD_HEIGHT - 34),
                      hud.DONE if ok else hud.WARN, filled=ok) + 7

    # --- right: progress ---
    counter = f"{plan.taken} / {plan.wanted}"
    (tw, _), _ = cv2.getTextSize(counter, hud.FONT, 0.72, 1)
    hud.label(canvas, counter, (width - tw - 16, 32), hud.INK, scale=0.72)
    bands = plan.band_progress()
    parts = "   ".join(f"{name} {done}/{want}" for name, (done, want) in bands.items())
    (pw, _), _ = cv2.getTextSize(parts, hud.FONT_SMALL, 0.5, 1)
    hud.progress_bar(canvas, (width - pw - 16, 46, pw, 6),
                     plan.taken / max(1, plan.wanted))
    hud.label(canvas, parts, (width - pw - 16, 76), hud.INK_MUTED, scale=0.5, small=True)


def _draw_footer(canvas, saved, corners):
    """The footer strip, also outside the image.

    Board visibility is read straight off the corners the loop already found, rather than
    mirrored into a second variable -- keeping two copies in step is what left this
    referring to a name nobody set.
    """
    height, width = canvas.shape[:2]
    y = height - FOOTER_HEIGHT
    canvas[y:] = hud.PANEL
    cv2.line(canvas, (0, y), (width, y), hud.RULE, 1)
    # label(), not caption(): caption letter-spaces for headings, which turns a row of key
    # hints into a wall the eye cannot skim.
    hud.label(canvas, "SPACE shoot    s skip pose    x skip spot    q done",
              (16, y + 21), hud.INK_MUTED, scale=0.46, small=True)

    text = f"{saved} saved"
    (tw, _), _ = cv2.getTextSize(text, hud.FONT_SMALL, 0.54, 1)
    hud.label(canvas, text, (width - tw - 16, y + 21), hud.INK_MUTED,
              scale=0.54, small=True)

    left_ok = corners.get("left") is not None
    right_ok = corners.get("right") is not None
    seen = f"board  L {'ok' if left_ok else '--'}  R {'ok' if right_ok else '--'}"
    (sw, _), _ = cv2.getTextSize(seen, hud.FONT_SMALL, 0.46, 1)
    hud.label(canvas, seen, (width - tw - sw - 44, y + 21),
              hud.DONE if (left_ok or right_ok) else hud.INK_MUTED, scale=0.46, small=True)


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
    was_matched = False
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
        sound.play("shot")
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

                # On the way in only: the cue is "you are on it, now hold still", and
                # repeating it while they hold still is the opposite of helpful.
                if state["matched"] and not was_matched:
                    sound.play("near")
                was_matched = state["matched"]
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
                        moved = float(np.max(np.linalg.norm(
                            here - last_shot_corners, axis=1))) > MOVED_SINCE_SHOT_PX
                state["moved"] = moved
                if state["ready"] and not moved:
                    state["hint"] = "Change the pose"

                if state["ready"] and moved and now - last_shot > SHOT_COOLDOWN:
                    # The cooldown starts when a frame is actually saved. Starting it on
                    # the attempt meant a refused credit silently ate the next 0.6 s too.
                    if plan.credit(corners["left"], corners["right"]):
                        last_shot = now
                        sharpness_log.append(state.get("sharp_value", 0.0))
                        shoot(frame, f"{target.band} {target.eye} {target.pose}")
                        last_shot_corners = np.asarray(wanted).reshape(-1, 2).copy()
                        still.since = None

            # One eye, always -- the left unless a placement belongs to the right. Two
            # panes halve the size of the thing being aimed at, and the idle one only
            # distracts. A stereo placement is no exception: it is specified in one eye and
            # aimed at in one eye, and whether the other camera can see it is a yes/no the
            # footer already answers.
            target_now = plan.current if plan else None
            if preview_size is None:
                scale = preview_width / left.shape[1]
                eye_h = round(left.shape[0] * scale)
                preview_size = (preview_width, HUD_HEIGHT + eye_h + FOOTER_HEIGHT)
                cv2.resizeWindow(WINDOW, *preview_size)

            band_h = preview_size[1] - HUD_HEIGHT - FOOTER_HEIGHT
            shown = left if (target_now is None or target_now.eye == "left") else right
            eye = cv2.resize(shown, (preview_width, band_h),
                             interpolation=cv2.INTER_NEAREST)
            if MIRROR_PREVIEW:
                # Mirrored, because the operator stands in front of the camera looking at
                # the screen: unmirrored, every instruction sends them the wrong way. The
                # guides are placed in mirrored coordinates too, so they move with the
                # operator while the text stays the right way round.
                eye = cv2.flip(eye, 1)
            scale = preview_width / shown.shape[1]

            canvas = np.zeros((preview_size[1], preview_size[0], 3), np.uint8)
            canvas[HUD_HEIGHT:HUD_HEIGHT + eye.shape[0]] = eye
            if plan is not None:
                _draw_hud(canvas, plan, len(saved), state)
                _draw_targets(canvas, plan, scale, HUD_HEIGHT, state)
            _draw_footer(canvas, len(saved), corners)
            preview = canvas

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
