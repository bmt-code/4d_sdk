#!/usr/bin/env python3
"""Check a camera's exposure stream from the CLI, no display (racked units, CI have no screen).
Same questions as the viewer, printable numbers:

    python3 examples/exposure_probe.py --ip 172.31.1.77 --mode ae_dual --seconds 60

Read in this order:

  short/long Hz     both near half frame rate in dual, one at full rate in single. Lopsided =
                    frames lost between pairing and this client, not camera stopping.
  mislabelled       frame label vs exposure inside that frame. Non-zero = swapped pane.
  camera pairs      what camera paired vs what left it. Gap = the wire; no gap plus lopsided
  vs sent           rate = this client.
  dropped q/zmq     where the gap went. zmq = publisher discarding at its high-water mark.
  desync            eyes on different moments. Under 1% healthy.
  cadence           do frames wear the exposure the cadence asked for. Near zero means
                    timing.control_lag_frames aims at the wrong frame; camera logs what to try.
  frame period      sensor rate vs what the cadence assumes.
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
    """Frame label vs exposure reported in the same frame. Dual modes only: a single exposure
    reports as "short" whatever it is, so splitting it against two targets says nothing (an
    older version called every `normal` frame mismatched for that reason)."""
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
