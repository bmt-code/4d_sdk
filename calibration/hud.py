"""The look of the capture and check windows.

One palette and a few drawing primitives, so every label in the tool is styled in one
place instead of each view picking its own colours. Cool blues on a dark translucent
panel: the guides have to sit over a live camera image of a room without competing with
it, and blue is the one hue a printed checkerboard and most rooms do not contain.

Everything here takes and returns BGR, because that is what OpenCV draws in. The palette
is written as RGB and converted once, so the values can be read against a normal colour
picker.
"""

import cv2
import numpy as np


def _bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


# --- palette --------------------------------------------------------------------------
ACCENT = _bgr((122, 190, 255))        # light blue: the target you are aiming at
ACCENT_DIM = _bgr((70, 112, 155))     # the targets queued behind it
DONE = _bgr((86, 214, 164))           # covered ground
WARN = _bgr((255, 190, 92))           # nearly right, keep going
BAD = _bgr((240, 106, 106))           # not acceptable
INK = _bgr((233, 241, 250))           # primary text
INK_MUTED = _bgr((150, 172, 194))     # secondary text
PANEL = _bgr((10, 18, 30))            # panel ground, drawn translucent
RULE = _bgr((44, 66, 94))

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def panel(image, box, alpha=0.72):
    """A translucent dark panel with a hairline top edge, drawn in place."""
    x, y, w, h = [int(v) for v in box]
    x, y = max(0, x), max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    region = image[y:y + h, x:x + w]
    ground = np.empty_like(region)
    ground[:] = PANEL
    cv2.addWeighted(ground, alpha, region, 1 - alpha, 0, region)
    cv2.line(image, (x, y), (x + w, y), RULE, 1, cv2.LINE_AA)


def label(image, text, origin, color=INK, scale=0.52, weight=1, small=False):
    font = FONT_SMALL if small else FONT
    cv2.putText(image, text, (int(origin[0]), int(origin[1])), font, scale,
                color, weight, cv2.LINE_AA)


def caption(image, text, origin, color=INK_MUTED, scale=0.42):
    """Uppercase, letter-spaced-ish label for section headings."""
    spaced = " ".join(text.upper())
    cv2.putText(image, spaced, (int(origin[0]), int(origin[1])), FONT_SMALL, scale,
                color, 1, cv2.LINE_AA)


def chip(image, text, origin, color, filled=False):
    """A small rounded status pill. Returns its width so chips can be laid in a row."""
    scale = 0.56
    (tw, th), _ = cv2.getTextSize(text, FONT_SMALL, scale, 1)
    pad_x, pad_y = 11, 8
    x, y = int(origin[0]), int(origin[1])
    w, h = tw + pad_x * 2, th + pad_y * 2
    box = (x, y, w, h)
    if filled:
        _rounded(image, box, color, thickness=-1)
        text_color = _bgr((8, 14, 24))
    else:
        _rounded(image, box, color, thickness=1)
        text_color = color
    cv2.putText(image, text, (x + pad_x, y + h - pad_y - 1), FONT_SMALL, scale,
                text_color, 1, cv2.LINE_AA)
    return w


def _rounded(image, box, color, thickness=1, radius=5):
    x, y, w, h = [int(v) for v in box]
    r = min(radius, h // 2, w // 2)
    if thickness < 0:
        cv2.rectangle(image, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(image, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r),
                       (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(image, (cx, cy), r, color, -1, cv2.LINE_AA)
        return
    cv2.line(image, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
    for (cx, cy), angle in (((x + r, y + r), 180), ((x + w - r, y + r), 270),
                            ((x + r, y + h - r), 90), ((x + w - r, y + h - r), 0)):
        cv2.ellipse(image, (cx, cy), (r, r), angle, 0, 90, color, thickness, cv2.LINE_AA)


def target_rect(image, box, color=ACCENT, thickness=2, corner=0.22, filled=False):
    """The place-the-board guide: a bracketed rectangle, corners only.

    Drawn as four corner brackets rather than a closed box so it frames the board without
    a continuous line running through the squares, which the detector has to see.
    """
    x, y, w, h = [int(v) for v in box]
    if filled:
        # Kept light: a close-band target covers half the pane, and a heavier wash there
        # dims the very board the operator is trying to line up.
        region = image[max(0, y):y + h, max(0, x):x + w]
        if region.size:
            tint = np.empty_like(region)
            tint[:] = color
            cv2.addWeighted(tint, 0.10, region, 0.90, 0, region)

    cut_x, cut_y = int(w * corner), int(h * corner)
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1),
                             (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(image, (px, py), (px + dx * cut_x, py), color, thickness, cv2.LINE_AA)
        cv2.line(image, (px, py), (px, py + dy * cut_y), color, thickness, cv2.LINE_AA)


def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def match_color(score, accept, near):
    """Red through amber to green as the board closes on the guide.

    A continuous ramp rather than three states: the operator is steering, and a colour that
    only changes at a threshold tells them nothing about which way they are moving.
    """
    if score <= accept:
        return DONE
    if score <= near:
        return _lerp(DONE, WARN, (score - accept) / max(near - accept, 1e-6))
    return _lerp(WARN, BAD, (score - near) / max(near, 1e-6))


def board_guide(image, points, grid, color=ACCENT, thickness=2):
    """The wanted board pose, drawn as the board's own grid.

    Rows and columns of the projected inner corners, plus a heavier outline and a marked
    origin corner. Drawn as lines rather than a filled shape so the live board stays
    readable underneath it, and so the yaw is visible in the convergence of the rows
    instead of needing a label to explain which way to turn.
    """
    cols, rows = grid
    pts = np.asarray(points, dtype=np.float32).reshape(rows, cols, 2)

    thin = max(1, thickness - 1)
    for r in range(rows):
        cv2.polylines(image, [pts[r].astype(np.int32)], False, color, thin, cv2.LINE_AA)
    for c in range(cols):
        cv2.polylines(image, [pts[:, c].astype(np.int32)], False, color, thin, cv2.LINE_AA)

    outline = np.array([pts[0, 0], pts[0, -1], pts[-1, -1], pts[-1, 0]], np.int32)
    cv2.polylines(image, [outline], True, color, thickness, cv2.LINE_AA)
    # One corner marked, so a board held upside down is obvious.
    cv2.circle(image, tuple(pts[0, 0].astype(int)), max(3, thickness + 2), color, -1,
               cv2.LINE_AA)


def progress_bar(image, box, fraction, color=ACCENT, ground=RULE):
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), ground, -1)
    filled = int(w * max(0.0, min(1.0, fraction)))
    if filled > 0:
        cv2.rectangle(image, (x, y), (x + filled, y + h), color, -1)


def crosshair(image, point, color=ACCENT, size=9):
    px, py = int(point[0]), int(point[1])
    cv2.line(image, (px - size, py), (px + size, py), color, 1, cv2.LINE_AA)
    cv2.line(image, (px, py - size), (px, py + size), color, 1, cv2.LINE_AA)
