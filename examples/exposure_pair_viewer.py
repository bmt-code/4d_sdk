#!/usr/bin/env python3
"""Interactive dual-exposure test for the 4D Stereo Camera.

The camera interleaves two exposures on one stream: a short one metered for the bright light
(a surgical beam) and a long one metered for the rest of the room. Every frame carries the
label of the exposure it was taken with, plus the exposure the sensor actually used.

This viewer shows the newest frame of each exposure side by side and lets you retune both
while watching the scene. Nothing here restarts the stream except an engine switch to "hdr",
which has to rebuild the camera.

    python3 examples/exposure_pair_viewer.py --ip 172.31.1.77

Keys
    q / a     short exposure -/+
    w / s     long exposure -/+
    e / d     analogue gain -/+ on the focused stream
    tab       move focus between short and long
    f         fine steps (x0.2)
    m         cycle exposure engine: auto -> manual -> hdr
    n         cycle hold_n 1..4 (manual engine: frames held before switching)
    o         toggle the AE profile (hdr engine: the dual-exposure split on/off)
    v         toggle the hdr shutter/gain envelope (off = profiles choose the exposure)
    l         toggle the AWB lock
    k         toggle the AE lock (float / frozen)
    p         dump the current short and long frames as PNGs
    esc / x   quit
"""

import argparse
import os
import time

import cv2
import numpy as np

from stereo_4d import Stereo4DCameraHandler

SHORT = "short"
LONG = "long"
ENGINES = ["auto", "manual", "hdr"]

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
        channel = view.measured("hdr_channel")

        focused = state["focus"] == label
        colour = HUD_FG if focused else HUD_DIM
        marker = ">" if focused else " "
        text = (
            f"{marker} {label:<5} target {fmt(target_us,'us'):>8} g{target_gain:<4.1f} "
            f"| measured {fmt(measured_us,'us'):>8} g{fmt(measured_gain,'',1):<5} "
            f"| {fmt(rates.get(label), 'Hz', 1):>7} "
            f"| age {fmt(view.age_ms,'ms'):>7}"
        )
        if channel is not None:
            text += f" | ch{channel}"
        cv2.putText(canvas, text, (12, 26 + row * 24), font, 0.55, colour, 1, cv2.LINE_AA)

    phase_lock = stats.get("phase_lock_percent")
    unknown = stats.get("unknown_percent")
    phase_colour = HUD_OK if (phase_lock or 0) >= 90 else HUD_WARN
    if state["engine"] == "hdr":
        source = "envelope+profiles" if state["hdr_envelope"] else "profiles"
    elif state["engine"] == "manual":
        source = "targets"
    else:
        source = "AE"
    summary = (
        f"engine {state['engine']:<6} {source:<17} hold_n {state['hold_n']} "
        f"| phase-lock {fmt(phase_lock,'%',1)} | unknown {fmt(unknown,'%',1)}"
    )
    second = (
        f"profile {'split (on)' if state['profile_on'] else 'room (off)'} "
        f"| awb {'locked' if state['awb_locked'] else 'auto'} "
        f"| ae {'locked' if state['ae_locked'] else 'float'} "
        f"| step {'fine' if state['fine'] else 'coarse'}"
    )
    cv2.putText(canvas, summary, (12, 72), font, 0.5, phase_colour, 1, cv2.LINE_AA)
    cv2.putText(canvas, second, (12, 90), font, 0.45, HUD_DIM, 1, cv2.LINE_AA)

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
        engine=state["engine"],
        hold_n=state["hold_n"],
        hdr_envelope=state["hdr_envelope"],
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
    parser.add_argument("--engine", default="manual", choices=ENGINES,
                        help="exposure engine to start in (default: manual)")
    parser.add_argument("--short-us", type=int, default=4000,
                        help="short exposure target in microseconds")
    parser.add_argument("--long-us", type=int, default=25000,
                        help="long exposure target in microseconds")
    parser.add_argument("--short-gain", type=float, default=1.0)
    parser.add_argument("--long-gain", type=float, default=2.0)
    parser.add_argument("--hold-n", type=int, default=1,
                        help="manual engine: frames to hold each exposure before switching")
    parser.add_argument("--hdr-envelope", action="store_true",
                        help="hdr engine: cap each channel's shutter/gain at the targets "
                             "instead of letting its AE profile choose")
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
        "engine": args.engine,
        "short_us": args.short_us,
        "long_us": args.long_us,
        "short_gain": args.short_gain,
        "long_gain": args.long_gain,
        "hold_n": max(1, args.hold_n),
        "focus": SHORT,
        "fine": False,
        "awb_locked": False,
        "ae_locked": False,
        # Under the hdr engine this is the dual-exposure switch: on = short channel meters
        # the bright light while the long channel meters the room.
        "profile_on": True,
        "hdr_envelope": args.hdr_envelope,
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
    # The hdr engine applies new targets by rebuilding the camera, so firing on every
    # keypress would stall the stream for the whole time you are tuning. Wait for the
    # keypresses to stop first. The manual engine applies on the next frame, so it sends
    # straight away.
    pending_send_at = None
    HDR_SEND_DEBOUNCE_S = 0.6

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

            if pending_send_at is not None and time.time() >= pending_send_at:
                pending_send_at = None
                send_pair(handler, state)
                announce("applying new targets ...", 1.5)

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
            elif key == ord("n"):
                state["hold_n"] = state["hold_n"] % 4 + 1
                changed = True
            elif key == ord("m"):
                state["engine"] = ENGINES[(ENGINES.index(state["engine"]) + 1) % len(ENGINES)]
                pending_send_at = None
                send_pair(handler, state)
                # Any engine switch rebuilds the camera: manual needs its controls set when
                # the stream is configured, and hdr needs a patched tuning.
                announce(f"engine -> {state['engine']}, rebuilding camera ...", 4.0)
            elif key == ord("o"):
                state["profile_on"] = not state["profile_on"]
                handler.set_exposure_control(state["profile_on"])
                announce(
                    "AE profile: dual-exposure split"
                    if state["profile_on"] else "AE profile: both channels meter the room"
                )
            elif key == ord("v"):
                state["hdr_envelope"] = not state["hdr_envelope"]
                changed = True
                announce(
                    "hdr exposure from the targets (envelope on)"
                    if state["hdr_envelope"] else "hdr exposure from the AE profiles"
                )
            elif key == ord("l"):
                state["awb_locked"] = not state["awb_locked"]
                handler.set_awb_gains(awb_gains if state["awb_locked"] else None)
                announce(f"awb {'locked to ' + str(awb_gains) if state['awb_locked'] else 'auto'}")
            elif key == ord("k"):
                state["ae_locked"] = not state["ae_locked"]
                handler.set_exposure_lock(state["ae_locked"])
                announce(f"ae {'locked' if state['ae_locked'] else 'floating'}")
            elif key == ord("p"):
                written = dump_pair(views, args.dump_dir)
                announce(f"dumped {len(written)} frame(s) to {args.dump_dir}")

            if changed:
                if state["engine"] == "hdr":
                    pending_send_at = time.time() + HDR_SEND_DEBOUNCE_S
                else:
                    pending_send_at = None
                    send_pair(handler, state)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        handler.stop()


if __name__ == "__main__":
    main()
