#!/usr/bin/env python3
"""Interactive dual-exposure test for the 4D Stereo Camera.

The camera interleaves two exposures on one stream: a short one metered for the bright light
(a surgical beam) and a long one metered for the rest of the room. Every frame carries the
label of the exposure it was taken with, plus the exposure the sensor actually used.

This viewer shows the newest frame of each exposure side by side and lets you retune both
while watching the scene. Nothing here ever restarts the stream: every change lands on the
next frame.

    python3 examples/exposure_pair_viewer.py --ip 172.31.1.77

Keys
    q / a     short exposure -/+
    w / s     long exposure -/+
    e / d     analogue gain -/+ on the focused stream
    tab       move focus between short and long
    f         fine steps (x0.2)
    l         toggle the AWB lock
    n         cycle the mode: manual / normal / highlight / ae_dual
    p         dump the current short and long frames as PNGs
    esc / x   quit
"""

import argparse
import os
import time

import cv2
import numpy as np

from stereo_4d import Stereo4DCameraHandler

# Manual first, so the key reads as "leave the pair, try each AE mode, come back".
AE_CYCLE = ("manual", "normal", "highlight", "ae_dual")

# The modes that deliver two labels. The others fill only the short pane.
AE_ALTERNATING = ("manual", "ae_dual")

SHORT = "short"
LONG = "long"

SHORT_STEP_US = 250
LONG_STEP_US = 1000
GAIN_STEP = 0.5
MIN_US = 100
MAX_US = 200000
MIN_GAIN = 1.0
MAX_GAIN = 16.0

HUD_HEIGHT = 104
HUD_BG = (24, 24, 24)
HUD_FG = (235, 235, 235)
HUD_DIM = (150, 150, 150)
HUD_WARN = (60, 190, 250)
HUD_OK = (120, 220, 140)


class StreamView:
    """The newest frame of one exposure, and when it landed here."""

    def __init__(self, label):
        self.label = label
        self.frame = None
        self.arrived_at = None
        self.__last_timestamp = None

    def update(self, frame):
        if frame is None or frame.timestamp == self.__last_timestamp:
            return False
        self.__last_timestamp = frame.timestamp
        self.frame = frame
        self.arrived_at = time.time()
        return True

    @property
    def age_ms(self):
        if self.arrived_at is None:
            return None
        return (time.time() - self.arrived_at) * 1000.0

    def measured(self, key):
        if self.frame is None:
            return None
        return (self.frame.exposure.get("left") or {}).get(key)


def left_eye(image):
    """The camera sends left|right concatenated; the left eye is enough to judge exposure."""
    if image is None:
        return None
    return image[:, : image.shape[1] // 2]


def pane(view: StreamView, width, height):
    image = left_eye(view.frame.image) if view.frame is not None else None
    if image is None:
        canvas = np.full((height, width, 3), 40, dtype=np.uint8)
        cv2.putText(
            canvas, f"waiting for {view.label} frames", (20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_DIM, 1, cv2.LINE_AA,
        )
        return canvas
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def fmt(value, suffix="", digits=0):
    if value is None:
        return "--"
    if digits:
        return f"{value:.{digits}f}{suffix}"
    return f"{int(round(value))}{suffix}"


def draw_hud(canvas, state, views, stats, rates, notice):
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (width, HUD_HEIGHT), HUD_BG, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for row, label in enumerate((SHORT, LONG)):
        view = views[label]
        target_us = state[f"{label}_us"]
        target_gain = state[f"{label}_gain"]
        measured_us = view.measured("exposure_time_us")
        measured_gain = view.measured("analogue_gain")

        focused = state["focus"] == label
        colour = HUD_FG if focused else HUD_DIM
        marker = ">" if focused else " "
        text = (
            f"{marker} {label:<5} target {fmt(target_us,'us'):>8} g{target_gain:<4.1f} "
            f"| measured {fmt(measured_us,'us'):>8} g{fmt(measured_gain,'',1):<5} "
            f"| {fmt(rates.get(label), 'Hz', 1):>7} "
            f"| age {fmt(view.age_ms,'ms'):>7}"
        )
        cv2.putText(canvas, text, (12, 26 + row * 24), font, 0.55, colour, 1, cv2.LINE_AA)

    unknown = stats.get("unknown_percent")
    out_of_sync = stats.get("out_of_sync_percent")
    cadence = stats.get("cadence_percent")
    # Green while pairs are actually landing; desync is the honest read on that.
    summary_colour = HUD_OK if (out_of_sync or 0) <= 10 else HUD_WARN
    summary = (
        f"unknown {fmt(unknown,'%',1)} "
        f"| desync {fmt(out_of_sync,'%',1)} "
        f"| cadence {fmt(cadence,'%',1)} "
        f"| cam pairs S/L {stats.get('pairs_short','--')}/{stats.get('pairs_long','--')}"
    )
    cv2.putText(canvas, summary, (12, 72), font, 0.5, summary_colour, 1, cv2.LINE_AA)

    # Bottom row: what each toggle is bound to, and where it currently stands. The AE
    # segment is drawn on its own so it can be coloured -- dual is the working mode, an AE
    # profile is the exception, and which one is running has to be readable at a glance.
    ae_text = f"[n] mode {state['ae']}"
    if state["ae"] not in AE_ALTERNATING:
        ae_text += " - single stream, long pane idle"
    # Dim for manual, the working mode; coloured for anything the AE is driving.
    ae_colour = HUD_DIM if state["ae"] == "manual" else HUD_WARN
    cv2.putText(canvas, ae_text, (12, 90), font, 0.45, ae_colour, 1, cv2.LINE_AA)

    rest = (
        f"   | [l] awb {'locked' if state['awb_locked'] else 'auto'} "
        f"| [f] step {'fine' if state['fine'] else 'coarse'} "
        f"| [tab] focus {state['focus']}"
    )
    ae_width = cv2.getTextSize(ae_text, font, 0.45, 1)[0][0]
    cv2.putText(canvas, rest, (12 + ae_width, 90), font, 0.45, HUD_DIM, 1, cv2.LINE_AA)

    if notice:
        cv2.putText(
            canvas, notice, (12, height - 16), font, 0.6, HUD_WARN, 2, cv2.LINE_AA
        )
    return canvas


def clamp(value, low, high):
    return max(low, min(high, value))


def send_pair(handler, state):
    handler.set_exposure_pair(
        short_us=state["short_us"],
        long_us=state["long_us"],
        short_gain=state["short_gain"],
        long_gain=state["long_gain"],
    )


def dump_pair(views, dump_dir):
    os.makedirs(dump_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    written = []
    for label, view in views.items():
        if view.frame is None:
            continue
        measured = view.measured("exposure_time_us")
        path = os.path.join(dump_dir, f"{stamp}_{label}_{fmt(measured,'us')}.png")
        cv2.imwrite(path, view.frame.image)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Interactive dual-exposure tuner for the 4D Stereo Camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Keys", 1)[1],
    )
    parser.add_argument("--ip", default="172.31.1.77", help="camera IP address")
    parser.add_argument("--short-us", type=int, default=4000,
                        help="short exposure target in microseconds")
    parser.add_argument("--long-us", type=int, default=25000,
                        help="long exposure target in microseconds")
    parser.add_argument("--short-gain", type=float, default=1.0)
    parser.add_argument("--long-gain", type=float, default=2.0)
    parser.add_argument("--awb-gains", default="2.0,2.0",
                        help="red,blue gains used when the AWB lock is on")
    parser.add_argument("--pane-width", type=int, default=640,
                        help="width of each of the two panes")
    parser.add_argument("--dump-dir", default="exposure_dumps",
                        help="where the 'p' key writes PNG pairs")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="seconds to wait for the camera to start")
    args = parser.parse_args()

    awb_gains = tuple(float(x) for x in args.awb_gains.split(","))

    state = {
        "short_us": args.short_us,
        "long_us": args.long_us,
        "short_gain": args.short_gain,
        "long_gain": args.long_gain,
        # The camera comes up with its AE running; the send_pair() below is what takes it
        # out, so the viewer starts on the two manual targets.
        "ae": "manual",
        "focus": SHORT,
        "fine": False,
        "awb_locked": False,
    }

    handler = Stereo4DCameraHandler(ip=args.ip)
    print(f"Connecting to camera at {args.ip} ...")
    if not handler.start(wait=True, timeout=args.timeout):
        print("Camera did not start. Is it powered and on the network?")
        handler.stop()
        return

    send_pair(handler, state)

    pane_h = int(args.pane_width * 9 / 16)
    window = "4D dual exposure"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.pane_width * 2, pane_h + HUD_HEIGHT)

    views = {SHORT: StreamView(SHORT), LONG: StreamView(LONG)}
    notice = ""
    notice_until = 0.0
    def announce(text, seconds=2.5):
        nonlocal notice, notice_until
        notice = text
        notice_until = time.time() + seconds

    try:
        while True:
            for label, view in views.items():
                view.update(handler.get_last_frame(label))

            stats = handler.get_exposure_stats()
            rates = handler.get_exposure_fps()

            canvas = np.zeros((pane_h + HUD_HEIGHT, args.pane_width * 2, 3), dtype=np.uint8)
            canvas[HUD_HEIGHT:, : args.pane_width] = pane(views[SHORT], args.pane_width, pane_h)
            canvas[HUD_HEIGHT:, args.pane_width:] = pane(views[LONG], args.pane_width, pane_h)

            if notice and time.time() > notice_until:
                notice = ""
            draw_hud(canvas, state, views, stats, rates, notice)
            cv2.imshow(window, canvas)

            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (27, ord("x")):
                break

            scale = 0.2 if state["fine"] else 1.0
            short_step = max(10, int(SHORT_STEP_US * scale))
            long_step = max(10, int(LONG_STEP_US * scale))
            gain_step = max(0.1, GAIN_STEP * scale)
            changed = False

            if key == ord("q"):
                state["short_us"] = clamp(state["short_us"] - short_step, MIN_US, MAX_US)
                changed = True
            elif key == ord("a"):
                state["short_us"] = clamp(state["short_us"] + short_step, MIN_US, MAX_US)
                changed = True
            elif key == ord("w"):
                state["long_us"] = clamp(state["long_us"] - long_step, MIN_US, MAX_US)
                changed = True
            elif key == ord("s"):
                state["long_us"] = clamp(state["long_us"] + long_step, MIN_US, MAX_US)
                changed = True
            elif key in (ord("e"), ord("d")):
                delta = -gain_step if key == ord("e") else gain_step
                gain_key = f"{state['focus']}_gain"
                state[gain_key] = round(clamp(state[gain_key] + delta, MIN_GAIN, MAX_GAIN), 2)
                changed = True
            elif key == ord("\t"):
                state["focus"] = LONG if state["focus"] == SHORT else SHORT
            elif key == ord("f"):
                state["fine"] = not state["fine"]
            elif key in (ord("n"), ord("N")):
                state["ae"] = AE_CYCLE[(AE_CYCLE.index(state["ae"]) + 1) % len(AE_CYCLE)]
                handler.set_auto_exposure(state["ae"])
                announce({
                    "manual": "manual: the two absolute targets are back, 15Hz each",
                    "normal": "AE normal: one auto-metered stream at 30Hz",
                    "highlight": "AE highlight: one stream, metered for the bright light",
                    "ae_dual": "AE dual: two ISP channels, beam + room, 15Hz each",
                }[state["ae"]])
            elif key == ord("l"):
                state["awb_locked"] = not state["awb_locked"]
                handler.set_awb_gains(awb_gains if state["awb_locked"] else None)
                announce(f"awb {'locked to ' + str(awb_gains) if state['awb_locked'] else 'auto'}")
            elif key == ord("p"):
                written = dump_pair(views, args.dump_dir)
                announce(f"dumped {len(written)} frame(s) to {args.dump_dir}")

            if changed:
                # Every change lands on the next frame, so there is nothing to debounce.
                # Sending a pair is also what leaves the AE, so the HUD has to follow.
                state["ae"] = "manual"
                send_pair(handler, state)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        handler.stop()


if __name__ == "__main__":
    main()
