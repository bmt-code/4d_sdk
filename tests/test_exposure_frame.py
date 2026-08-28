#!/usr/bin/env python3
"""Tests for the dual-exposure side of the SDK. No camera needed:

    python3 tests/test_exposure_frame.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from stereo_4d.stereo_4d import (  # noqa: E402
    EXPOSURE_LONG,
    EXPOSURE_SHORT,
    EXPOSURE_UNKNOWN,
    Stereo4DCameraHandler,
    Stereo4DFrame,
)

FAILURES = []

parse_exposure = Stereo4DCameraHandler._Stereo4DCameraHandler__parse_exposure


def check(name, got, want):
    if got == want:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {got!r} != {want!r}")
        FAILURES.append(name)


def bare_handler():
    """A handler with only the exposure bookkeeping set up.

    Constructing one for real opens sockets and starts threads; none of that is involved in
    what these tests cover.
    """
    handler = object.__new__(Stereo4DCameraHandler)
    prefix = "_Stereo4DCameraHandler__"
    setattr(handler, prefix + "last_frame", None)
    setattr(handler, prefix + "last_frame_by_label", {})
    setattr(handler, prefix + "label_arrival_times", {})
    setattr(handler, prefix + "exposure_stats", {})
    setattr(handler, prefix + "last_status_time", None)
    return handler


def frame_message(label="short", left_us=4013, right_us=4013):
    return {
        "msg_type": "frame",
        "timestamp": 123.0,
        "exposure": label,
        "exposure_time_us": {"left": left_us, "right": right_us},
        "analogue_gain": {"left": 1.0, "right": 1.5},
        "lux": 320.5,
    }


def test_parse():
    exposure = parse_exposure(frame_message())
    check("parse: label", exposure["label"], "short")
    check("parse: left exposure", exposure["left"]["exposure_time_us"], 4013)
    check("parse: right gain", exposure["right"]["analogue_gain"], 1.5)
    check("parse: lux", exposure["lux"], 320.5)


def test_parse_legacy_and_missing():
    # Firmware that predates the split reported one value for the pair.
    legacy = parse_exposure({"exposure_time_us": 5000, "analogue_gain": 2.0})
    check("legacy: scalar applies to both eyes",
          (legacy["left"]["exposure_time_us"], legacy["right"]["exposure_time_us"]),
          (5000, 5000))
    check("legacy: no label means unknown", legacy["label"], EXPOSURE_UNKNOWN)

    empty = parse_exposure({})
    check("empty: label unknown", empty["label"], EXPOSURE_UNKNOWN)
    check("empty: exposure absent", empty["left"]["exposure_time_us"], None)


def test_frame_accessors():
    frame = Stereo4DFrame(timestamp=1.0, exposure=parse_exposure(frame_message()))
    check("frame: exposure_label", frame.exposure_label, "short")
    check("frame: exposure_time_us", frame.exposure_time_us, 4013)

    unlabelled = Stereo4DFrame(timestamp=1.0)
    check("frame: no exposure means unknown", unlabelled.exposure_label, EXPOSURE_UNKNOWN)

    copied = frame.copy()
    copied.exposure["left"]["exposure_time_us"] = 999
    check("frame: copy does not share nested exposure", frame.exposure_time_us, 4013)


def test_last_frame_per_label():
    handler = bare_handler()
    record = handler._Stereo4DCameraHandler__record_exposure_arrival

    short = Stereo4DFrame(timestamp=1.0, exposure=parse_exposure(frame_message(EXPOSURE_SHORT, 4000, 4000)))
    long_frame = Stereo4DFrame(timestamp=1.05, exposure=parse_exposure(frame_message(EXPOSURE_LONG, 25000, 25000)))
    record(short)
    record(long_frame)

    check("per-label: short kept", handler.get_last_frame(EXPOSURE_SHORT).exposure_time_us, 4000)
    check("per-label: long kept", handler.get_last_frame(EXPOSURE_LONG).exposure_time_us, 25000)

    # A long frame arriving must not displace the newest short one -- that is the whole
    # point of keeping them apart.
    newer_long = Stereo4DFrame(timestamp=1.10, exposure=parse_exposure(frame_message(EXPOSURE_LONG, 26000, 26000)))
    record(newer_long)
    check("per-label: short survives a newer long",
          handler.get_last_frame(EXPOSURE_SHORT).exposure_time_us, 4000)
    check("per-label: long updated", handler.get_last_frame(EXPOSURE_LONG).exposure_time_us, 26000)

    unknown = Stereo4DFrame(timestamp=1.2, exposure=parse_exposure({}))
    record(unknown)
    check("per-label: unknown frames are not filed", handler.get_last_frame(EXPOSURE_UNKNOWN), None)
    check("per-label: missing label returns None", handler.get_last_frame("nonsense"), None)


def test_exposure_fps():
    handler = bare_handler()
    times = getattr(handler, "_Stereo4DCameraHandler__label_arrival_times")
    now = time.time()
    from collections import deque
    times[EXPOSURE_SHORT] = deque([now, now + 0.1, now + 0.2, now + 0.3])
    times[EXPOSURE_LONG] = deque([now])

    rates = handler.get_exposure_fps()
    check("fps: short measured over the window", round(rates[EXPOSURE_SHORT], 1), 10.0)
    check("fps: a single sample is not a rate", rates[EXPOSURE_LONG], 0.0)


def test_status_carries_exposure_stats():
    handler = bare_handler()
    status = handler._Stereo4DCameraHandler__handle_camera_status
    stats = {"cadence_percent": 99.8, "out_of_sync_percent": 0.2, "short_us": 4000}

    status({"msg_type": "status", "data": "started", "exposure_stats": stats})
    check("status: stats stored", handler.get_exposure_stats()["out_of_sync_percent"], 0.2)

    # A status without stats (older firmware) must not wipe what is known.
    status({"msg_type": "status", "data": "started"})
    check("status: stats kept when absent",
          handler.get_exposure_stats()["cadence_percent"], 99.8)

    check("status: getter returns a copy",
          handler.get_exposure_stats() is handler.get_exposure_stats(), False)


def frame_handler():
    """A handler wired up enough to push frame messages through __handle_frame_message."""
    handler = bare_handler()
    prefix = "_Stereo4DCameraHandler__"
    handler.rectify_internally = False
    handler.stereo_maps_set = False
    setattr(handler, prefix + "frame_count_total", 0)
    setattr(handler, prefix + "frame_drop_count", 0)
    setattr(handler, prefix + "frame_drop_percent", 0.0)
    setattr(handler, prefix + "frame_count_in_interval", 0)
    setattr(handler, prefix + "received_fps", 0)
    setattr(handler, prefix + "prev_fps_measured_time", time.time())
    setattr(handler, prefix + "fps_measurement_interval", 1.0)
    setattr(handler, prefix + "frame_event", threading.Event())
    setattr(handler, prefix + "frame_callback", None)
    # Decoding a real JPEG is not what this covers, and it would drag cv2 in.
    setattr(handler, prefix + "decode_frame", lambda _bytes: object())
    return handler


def test_fps_is_actually_counted():
    """get_fps() returned None forever: the interval counter was read and reset, never
    incremented, so every measurement divided zero by the elapsed time.

    Visible in examples/stereo_4d_ros2.py, which publishes "FPS: None".
    """
    handler = frame_handler()
    prefix = "_Stereo4DCameraHandler__"
    handle = getattr(handler, prefix + "handle_frame_message")

    for _ in range(20):
        handle(frame_message())
    check("fps: frames counted in the interval",
          getattr(handler, prefix + "frame_count_in_interval"), 20)
    check("fps: no measurement before the interval elapses", handler.get_fps(), None)

    # Backdate the window so the next frame closes it.
    setattr(handler, prefix + "prev_fps_measured_time", time.time() - 2.0)
    handle(frame_message())
    measured = handler.get_fps()
    check("fps: a rate is reported once the window closes",
          measured is not None and 9.0 < measured < 12.0, True)
    check("fps: the counter restarts for the next window",
          getattr(handler, prefix + "frame_count_in_interval"), 0)


def test_a_dropped_frame_is_counted_once():
    """The drop percentage used to be computed twice, once per branch."""
    handler = frame_handler()
    prefix = "_Stereo4DCameraHandler__"
    handle = getattr(handler, prefix + "handle_frame_message")
    setattr(handler, prefix + "decode_frame", lambda _bytes: None)

    for _ in range(4):
        handle(frame_message())
    check("drops: every undecodable frame counted",
          getattr(handler, prefix + "frame_drop_count"), 4)
    check("drops: reported as a percentage of all frames",
          round(getattr(handler, prefix + "frame_drop_percent")), 100)
    check("drops: nothing was filed as a frame", handler.get_last_frame(), None)


def main():
    for test in (
        test_parse,
        test_parse_legacy_and_missing,
        test_frame_accessors,
        test_last_frame_per_label,
        test_exposure_fps,
        test_status_carries_exposure_stats,
        test_fps_is_actually_counted,
        test_a_dropped_frame_is_counted_once,
    ):
        print(f"\n--- {test.__name__} ---")
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
