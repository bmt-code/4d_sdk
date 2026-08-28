#!/usr/bin/env python3
"""Check a camera's exposure stream from the command line, with no display.

The viewer needs a screen, which a unit in a rack does not have and a CI job never will. This
asks the same questions and prints numbers you can paste into a report:

    python3 examples/exposure_probe.py --ip 172.31.1.77 --mode ae_dual --seconds 60

What to look at, in the order it matters:

  short/long Hz     both near half the frame rate in a dual mode, one at the full rate in a
                    single one. Lopsided means frames are being lost between the pairing and
                    this client, not that the camera stopped producing an exposure.
  mislabelled       every frame's label against the exposure inside that same frame. Anything
                    but zero is a swapped pane.
  camera pairs      what the camera paired, against what actually left it. A gap between the
  vs sent           two is the wire; no gap and a lopsided rate is this client.
  dropped q/zmq     where the gap went. zmq is the publisher discarding at its high-water
                    mark, which was silent and uncounted before.
  desync            the two eyes landing on different moments. Under 1% is healthy.
  cadence           whether frames come back wearing the exposure the cadence asked for. Near
                    zero means timing.control_lag_frames is aimed at the wrong frame; the
                    camera logs which value to try.
  frame period      what the sensor runs against what the cadence assumes.
"""
import argparse
import collections
import sys
import time

from stereo_4d import Stereo4DCameraHandler

DUAL_MODES = ("manual", "ae_dual")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default="172.31.1.77")
    parser.add_argument("--mode", default="ae_dual",
                        choices=("manual", "normal", "highlight", "ae_dual"))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--settle", type=float, default=5.0,
                        help="seconds to let the mode land before measuring")
    args = parser.parse_args()

    handler = Stereo4DCameraHandler(ip=args.ip)
    if not handler.start(wait=True, timeout=30):
        print("Camera did not start. Is it powered and on the network?")
        handler.stop()
        return 1

    handler.set_auto_exposure(args.mode)
    # The mode lands on the next frame, but the controllers need a moment to settle and the
    # camera clears its counters as it switches -- measuring through that reports the
    # transition rather than the steady state.
    time.sleep(args.settle)

    last_timestamp = {}
    counts = collections.Counter()
    mislabelled = 0
    seen = 0

    deadline = time.time() + args.seconds
    while time.time() < deadline:
        for label in ("short", "long"):
            frame = handler.get_last_frame(label)
            if frame is None or frame.timestamp == last_timestamp.get(label):
                continue
            last_timestamp[label] = frame.timestamp
            counts[label] += 1
            seen += 1
            mislabelled += _is_mislabelled(handler, frame, args.mode)
        time.sleep(0.005)

    _report(handler, args, counts, seen, mislabelled)
    handler.stop()
    return 0


def _is_mislabelled(handler, frame, mode):
    """Does this frame's label match the exposure reported in the same frame?

    Only asked in the dual modes. A single exposure is reported as "short" whatever it is, so
    comparing it against a split between two targets says nothing -- an earlier version of
    this script called every frame in `normal` mismatched for exactly that reason.
    """
    if mode not in DUAL_MODES:
        return 0
    exposure_us = frame.exposure_time_us
    if exposure_us is None:
        return 0
    stats = handler.get_exposure_stats()
    short_us, long_us = stats.get("short_us"), stats.get("long_us")
    if not short_us or not long_us or abs(long_us - short_us) < 500:
        return 0
    implied = "short" if exposure_us < (short_us * long_us) ** 0.5 else "long"
    return int(implied != frame.exposure_label)


def _report(handler, args, counts, seen, mislabelled):
    stats = handler.get_exposure_stats()
    rates = handler.get_exposure_fps()
    send = stats.get("send") or {}

    def show(name, value):
        print(f"  {name:<24}{value}")

    print(f"\n--- {args.mode} on {args.ip}, {args.seconds:.0f}s ---")
    for label in ("short", "long"):
        print(f"  {label:<8}{counts[label] / args.seconds:6.2f} Hz polled   "
              f"{rates.get(label, 0.0):6.2f} Hz reported")
    show("mislabelled", f"{mislabelled} of {seen}")
    show("camera pairs S/L", f"{stats.get('pairs_short')}/{stats.get('pairs_long')}")
    show("sent S/L", f"{send.get('sent_short')}/{send.get('sent_long')}")
    show("dropped queue/zmq",
         f"{send.get('dropped_queue_full')}/{send.get('dropped_zmq_again')}")
    show("firmware label check", send.get("label_mismatch"))
    show("desync / unknown",
         f"{_pct(stats.get('out_of_sync_percent'))} / {_pct(stats.get('unknown_percent'))}")
    show("cadence", _pct(stats.get("cadence_percent")))
    show("frame period meas/ass",
         f"{stats.get('frame_period_measured_us')} / {stats.get('frame_period_us')}")
    show("targets S/L", f"{stats.get('short_us')} / {stats.get('long_us')}")
    if stats.get("ae_railed"):
        show("", "RAILED -- a controller wants more light than its band can give")


def _pct(value):
    return "--" if value is None else f"{value:.2f}%"


if __name__ == "__main__":
    sys.exit(main())
