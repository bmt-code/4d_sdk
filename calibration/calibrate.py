"""All-in-one stereo calibration for the Stereo 4D camera.

One command walks the whole thing: capture (or point at a folder), drop the images
that are blurry or have no board, solve, then check the result on a live rectified
stream with epipolar lines drawn across it.

    python3 calibration/calibrate.py
    python3 calibration/calibrate.py --images examples/images_stereo_4d --grid 9x6 --square 25
    python3 calibration/calibrate.py --capture --grid 9x6 --square 25 --yes

Everything a run produces lands in calibration/sessions/<name>_<timestamp>/, so no
paths ever have to be edited in source.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from calibration.capture import capture_images, collect_folder, list_images
from calibration.quality import (
    apply_overrides,
    detect_all,
    filter_blurry,
    image_shape,
    parse_frame_spec,
)
from calibration.solve import format_report, parse_grid, solve
from calibration.verify import verify_live, verify_offline

SESSIONS_DIR = os.path.join(REPO_ROOT, "calibration", "sessions")
# The firmware's own copy in the sibling repo -- the tree that gets rsynced to a unit,
# and what stereo_cam.py reads as "../calib/stereo_calibration.yaml".
DEPLOY_PATH = os.path.join(
    os.path.dirname(REPO_ROOT), "4d_firmware", "calib", "stereo_calibration.yaml"
)

# The camera's own copy: what the running firmware actually reads.
DEPLOY_USER = "bmt"
DEPLOY_REMOTE_PATH = "~/4d_firmware/calib/stereo_calibration.yaml"
RESTART_COMMAND = "sudo systemctl restart stereo_4d.service"

DEFAULT_GRID = "9x6"
DEFAULT_SQUARE_MM = 25.0


def ask(question, default, assume_yes=False):
    """Prompt with an Enter-accepts default."""
    if assume_yes:
        print(f"{question} [{default}]: {default}")
        return default
    try:
        answer = input(f"{question} [{default}]: ").strip()
    except EOFError:
        raise SystemExit("\nNo input; stopping.")
    return answer or default


def ask_yes_no(question, default=True, assume_yes=False):
    suffix = "Y/n" if default else "y/N"
    if assume_yes:
        print(f"{question} [{suffix}]: {'y' if default else 'n'}")
        return default
    try:
        answer = input(f"{question} [{suffix}]: ").strip().lower()
    except EOFError:
        raise SystemExit("\nNo input; stopping.")
    if not answer:
        return default
    return answer.startswith("y")


def make_session(name):
    path = os.path.join(SESSIONS_DIR, f"{name}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(path)
    return path


def resolve_source(path):
    """A session directory or a plain image folder -> the folder holding the images.

    Both are accepted wherever a folder is asked for, so a path copied out of a previous
    run works whether it points at the session or at its images.
    """
    path = os.path.abspath(os.path.expanduser(path.strip().strip("\"'")))
    nested = os.path.join(path, "images")
    if os.path.isdir(nested):
        return nested
    return path


def list_sessions():
    """Past runs that still have images, newest first."""
    if not os.path.isdir(SESSIONS_DIR):
        return []

    sessions = []
    for name in os.listdir(SESSIONS_DIR):
        session = os.path.join(SESSIONS_DIR, name)
        images = os.path.join(session, "images")
        if not os.path.isdir(images):
            continue  # also skips a dangling symlink to a folder that has moved
        count = len(list_images(images))
        if not count:
            continue
        sessions.append({
            "name": name,
            "images": images,
            "count": count,
            # A session that borrowed another folder's images says where they live
            # rather than pretending they are its own.
            "target": os.path.realpath(images) if os.path.islink(images) else None,
            "summary": _session_summary(session),
            "when": os.path.getmtime(session),
        })
    # Session names do not share a prefix, so sort by when the run happened.
    return sorted(sessions, key=lambda entry: entry["when"], reverse=True)


def _session_summary(session):
    """One line of what a past run concluded, for the picker."""
    try:
        with open(os.path.join(session, "params.yaml")) as handle:
            params = yaml.safe_load(handle) or {}
    except OSError:
        return ""
    bits = []
    if params.get("grid"):
        bits.append(str(params["grid"]))
    if params.get("square_mm"):
        bits.append(f"{params['square_mm']:g}mm")
    if params.get("stereo_pairs"):
        bits.append(f"{params['stereo_pairs']} stereo")
    return ", ".join(bits)


def choose_source(args):
    """Pick an image folder: a past session by number, or any path."""
    if args.images:
        return resolve_source(args.images)

    sessions = list_sessions()
    if not sessions:
        print(
            "No past sessions to browse. Point at a folder of side-by-side PNGs, or "
            "re-run with --capture to shoot a new set."
        )
        return resolve_source(ask("Image folder", "examples/images_stereo_4d", args.yes))

    print("\nPast sessions:")
    width = max(len(s["name"]) for s in sessions)
    for i, entry in enumerate(sessions, 1):
        note = entry["summary"]
        if entry["target"]:
            borrowed = os.path.relpath(entry["target"], REPO_ROOT)
            note = f"{note}   -> {borrowed}" if note else f"-> {borrowed}"
        print(f"  {i:2}  {entry['name']:<{width}}  {entry['count']:4} images   {note}")

    while True:
        answer = ask(
            "\nPick a number, or type the path to a session or image folder",
            "1", args.yes,
        ).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(sessions):
            return sessions[int(answer) - 1]["images"]
        folder = resolve_source(answer)
        if os.path.isdir(folder):
            return folder
        print(f"No such directory: {folder}")
        if args.yes:
            raise SystemExit("Nothing to calibrate from.")


def stored_params(images_dir):
    """Board settings recorded by an earlier run on this folder, if any."""
    for candidate in (
        os.path.join(images_dir, "params.yaml"),
        os.path.join(os.path.dirname(images_dir.rstrip("/")), "params.yaml"),
    ):
        if os.path.exists(candidate):
            with open(candidate) as handle:
                data = yaml.safe_load(handle) or {}
            if "grid" in data and "square_mm" in data:
                print(f"Reusing the board settings recorded in {candidate}")
                return data["grid"], float(data["square_mm"])
    return None, None


def resolve_board(args, defaults=(None, None)):
    """Grid and square size, from flags, from a previous run, or by asking."""
    grid_default, square_default = defaults
    grid_text = args.grid or grid_default or DEFAULT_GRID
    square_default = args.square or square_default or DEFAULT_SQUARE_MM

    while True:
        text = grid_text if args.grid else ask(
            "Grid - inner corners, cols x rows", grid_text, args.yes
        )
        try:
            grid = parse_grid(text)
            break
        except ValueError as error:
            print(error)
            args.grid = None
            grid_text = DEFAULT_GRID

    while True:
        raw = args.square if args.square else ask(
            "Square size in mm", f"{square_default:g}", args.yes
        )
        try:
            square_mm = float(raw)
            if square_mm <= 0:
                raise ValueError("Square size must be positive")
            break
        except ValueError as error:
            print(error)
            args.square = None

    return grid, square_mm


def link_rejects(rejected, session):
    """Record the rejects as symlinks plus a reason file. Nothing is moved or deleted."""
    if not rejected:
        return None
    reject_dir = os.path.join(session, "rejected")
    os.makedirs(reject_dir, exist_ok=True)

    for det in rejected:
        link = os.path.join(reject_dir, det.name)
        if not os.path.lexists(link):
            os.symlink(os.path.abspath(det.path), link)

    reasons = os.path.join(reject_dir, "reasons.txt")
    with open(reasons, "w") as handle:
        for det in rejected:
            handle.write(f"{det.name}\t{det.reason}\t{det.path}\n")
    return reject_dir


def install(session_yaml):
    """Copy the result into the firmware repo, backing up what is there."""
    if not os.path.isdir(os.path.dirname(DEPLOY_PATH)):
        print(f"{os.path.dirname(DEPLOY_PATH)} is not here, so nothing was copied.")
        return False
    if os.path.exists(DEPLOY_PATH):
        backup = f"{DEPLOY_PATH}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(DEPLOY_PATH, backup)
        print(f"Backed up the previous calibration to {backup}")
    shutil.copy2(session_yaml, DEPLOY_PATH)
    print(f"Installed to {DEPLOY_PATH}")
    return True


def remote_destination(ip):
    """Where this calibration goes on the camera."""
    return f"{DEPLOY_USER}@{ip}:{DEPLOY_REMOTE_PATH}"


def send_to_camera(session_yaml, destination):
    """rsync the calibration onto the camera, keeping the copy that was there.

    Returns True once the file is across. The calibration is already saved locally
    either way, so a failure here is reported and the run still ends cleanly.
    """
    command = [
        "rsync", "-v",
        "--backup", f"--suffix=.bak-{time.strftime('%Y%m%d-%H%M%S')}",
        session_yaml, destination,
    ]
    # Flushed so the command appears above rsync's own output, not after it.
    print("  " + " ".join(command), flush=True)
    try:
        result = subprocess.run(command)
    except FileNotFoundError:
        print("rsync is not installed here, so the file was not sent.")
        return False

    if result.returncode != 0:
        print(f"rsync exited {result.returncode}; the camera was not updated.")
        return False

    print(f"\nSent to {destination}")
    print("The firmware reads it at start-up, so restart the service on the camera:")
    host, sep, _ = destination.partition(":")
    if sep and "/" not in host:
        print(f"  ssh {host} {RESTART_COMMAND}")
    else:
        print(f"  {RESTART_COMMAND}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ip", default="172.31.1.77", help="camera IP address")
    parser.add_argument("--name", default="stereo_camera",
                        help="session name, prefixed to the output directory")

    source = parser.add_argument_group("images")
    source.add_argument("--capture", action="store_true",
                        help="capture new images instead of asking")
    source.add_argument("--images", metavar="DIR",
                        help="calibrate from a past session directory or any folder "
                             "of side-by-side PNGs")
    source.add_argument("--interval", type=float, default=2.0,
                        help="seconds between automatic shots while capturing")

    quality = parser.add_argument_group("quality filter")
    quality.add_argument("--no-blur-filter", dest="blur_filter", action="store_false",
                         help="keep soft images instead of dropping them")
    quality.add_argument("--blur-ratio", type=float, default=0.6,
                         help="drop images below this fraction of the median sharpness")
    quality.add_argument("--blur-min", type=float,
                         help="absolute Laplacian-variance floor on top of the ratio")
    quality.add_argument("--include", action="append", metavar="FRAMES",
                         help="keep these frames whatever the filters say, e.g. "
                              "--include 27-34,45,55 (repeatable)")
    quality.add_argument("--exclude", action="append", metavar="FRAMES",
                         help="drop these frames whatever the filters say (repeatable)")

    fit = parser.add_argument_group("calibration")
    fit.add_argument("--grid", help="inner corners as 'cols x rows', e.g. 9x6")
    fit.add_argument("--square", help="checkerboard square size in mm")
    fit.add_argument("--rational", action="store_true",
                     help="use the 14-coefficient rational distortion model")
    fit.add_argument("--no-prune", dest="prune", action="store_false",
                     help="keep image pairs with a high reprojection error")
    fit.add_argument("--max-reproj", type=float, default=1.5,
                     help="stereo reprojection error in px above which a pair leaves "
                          "the extrinsics fit (it still feeds the intrinsics)")
    fit.add_argument("--max-mono-reproj", type=float, default=1.0,
                     help="per-eye reprojection error in px above which that eye's "
                          "corners are dropped")
    fit.add_argument("--workers", type=int,
                     help="threads used for board detection")

    check = parser.add_argument_group("check")
    check.add_argument("--offline", action="store_true",
                       help="review the saved images instead of a live stream")
    check.add_argument("--line-spacing", type=int, default=20,
                       help="displayed pixels between epipolar lines")
    check.add_argument("--preview-width", type=int, default=1600,
                       help="width the capture and check windows render at")
    check.add_argument("--no-check", dest="check", action="store_false",
                       help="skip the verification stage")

    deploy = parser.add_argument_group("deploy")
    deploy.add_argument("--install", dest="install", action="store_true", default=None,
                        help="copy the result into the local 4d_firmware/calib/")
    deploy.add_argument("--no-install", dest="install", action="store_false",
                        help="never touch the local 4d_firmware checkout")
    deploy.add_argument("--send", dest="send", action="store_true", default=None,
                        help="rsync the result onto the camera")
    deploy.add_argument("--no-send", dest="send", action="store_false",
                        help="never touch the camera")
    deploy.add_argument("--send-to", metavar="DEST",
                        help="rsync destination (default "
                             f"{DEPLOY_USER}@<--ip>:{DEPLOY_REMOTE_PATH})")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="accept every default without prompting")
    return parser.parse_args()


def main():
    args = parse_args()
    session = make_session(args.name)
    print(f"Session: {session}\n")

    # --- 1. images ------------------------------------------------------------
    choice = "c" if args.capture else "f" if args.images else None
    if choice is None:
        choice = "c" if args.yes else ask(
            "Capture new images [c] or use an existing folder [f]?", "c"
        ).lower()[:1]

    board_defaults = (None, None)
    if choice == "c":
        grid, square_mm = resolve_board(args)
        images_dir = os.path.join(session, "images")
        paths = capture_images(images_dir, ip=args.ip, interval=args.interval,
                               grid=grid, timeout=60.0,
                               preview_width=args.preview_width)
        source = images_dir
    else:
        source = choose_source(args)
        paths = collect_folder(source)
        os.symlink(source, os.path.join(session, "images"))
        board_defaults = stored_params(source)
        grid, square_mm = resolve_board(args, board_defaults)

    if not paths:
        raise SystemExit("No images to calibrate from.")

    shape = image_shape(paths)
    if shape is None:
        raise SystemExit("None of the images could be read.")
    print(f"\nPer-eye image size: {shape[1]}x{shape[0]}")

    # --- 2. detect and filter -------------------------------------------------
    print(f"Board: {grid[0]}x{grid[1]} inner corners, {square_mm:g} mm squares\n")
    detections = detect_all(paths, grid, workers=args.workers)

    include = parse_frame_spec(args.include)
    exclude = parse_frame_spec(args.exclude)
    if include or exclude:
        forced, undetectable, excluded, unknown = apply_overrides(
            detections, include, exclude, grid
        )
        if forced:
            print(f"\nForcing in {len(forced)}: "
                  f"{', '.join(d.name for d in forced)}")
        if undetectable:
            print(
                f"\nCannot include {len(undetectable)} of them -- no board in either "
                "eye, so there is nothing to detect:"
            )
            for det in undetectable:
                print(f"  {det.name}: {det.reason}")
        if excluded:
            print(f"\nExcluding {len(excluded)}: "
                  f"{', '.join(d.name for d in excluded)}")
        if unknown:
            print(f"\nNo such frame(s): {', '.join(unknown)}")

    drop_blurry = args.blur_filter and ask_yes_no(
        "\nDrop blurry images automatically?", True, args.yes
    )
    blur_threshold = None
    if drop_blurry:
        blur_threshold = filter_blurry(detections, args.blur_ratio, args.blur_min)

    kept = [d for d in detections if d.ok]
    rejected = [d for d in detections if not d.ok]
    no_board = sum(1 for d in rejected if d.reason.startswith("no board"))
    both = sum(1 for d in kept if d.has_both)
    print(
        f"\nKept {len(kept)}/{len(detections)} "
        f"(no board: {no_board}, blurry: {len(rejected) - no_board})"
    )
    print(
        f"  {both} with the board in both eyes, {len(kept) - both} in one eye only "
        "(those still fit that eye's intrinsics)"
    )
    link_rejects(rejected, session)

    # --- 3. solve -------------------------------------------------------------
    out_yaml = os.path.join(session, "stereo_calibration.yaml")
    print()
    result = solve(
        kept, grid, square_mm / 1000.0, shape, out_yaml,
        rational=args.rational, prune=args.prune, max_reproj=args.max_reproj,
        max_mono=args.max_mono_reproj,
    )
    # Pruning moves pairs into the reject list, so re-collect before reporting.
    rejected = [d for d in detections if not d.ok]
    link_rejects(rejected, session)

    with open(os.path.join(session, "params.yaml"), "w") as handle:
        yaml.dump({
            "grid": f"{grid[0]}x{grid[1]}",
            "square_mm": square_mm,
            "source": source,
            "ip": args.ip,
            "rational": args.rational,
            "prune": args.prune,
            "max_reproj": args.max_reproj,
            "max_mono_reproj": args.max_mono_reproj,
            "blur_ratio": args.blur_ratio if drop_blurry else None,
            "include": sorted(include, key=len) or None,
            "exclude": sorted(exclude, key=len) or None,
            "images_used": len(result["used"]),
            "stereo_pairs": result["counts"]["stereo"],
            "images_rejected": len(rejected),
        }, handle, default_flow_style=False)

    report = format_report(result, grid, square_mm / 1000.0, source, rejected,
                           blur_threshold)
    report_path = os.path.join(session, "report.txt")
    with open(report_path, "w") as handle:
        handle.write(report)
    print("\n" + report)

    # --- 4. check -------------------------------------------------------------
    calibration = {key: np.asarray(result[key])
                   for key in ("mtxL", "distL", "mtxR", "distR", "R", "T")}
    best = None
    if args.check:
        if not args.offline:
            best = verify_live(calibration, grid, square_mm, ip=args.ip,
                               spacing=args.line_spacing, out_dir=session,
                               display_width=args.preview_width)
            if best is False:
                print("Camera did not start -- falling back to the saved images.")
                args.offline = True
        if args.offline:
            # Only the both-eye frames can be rectified into a pair to look at.
            best = verify_offline(calibration, grid, square_mm,
                                  [d.path for d in result["stereo_used"]],
                                  spacing=args.line_spacing, out_dir=session,
                                  display_width=args.preview_width)

    if isinstance(best, dict):
        summary = (
            f"\nBest check seen: y-RMS {best['y_rms']:.3f} px, "
            f"grid mean {best['mean_err']:.2f} mm, max {best['max_err']:.2f} mm, "
            f"at {best['distance']:.0f} mm\n"
        )
        print(summary)
        with open(report_path, "a") as handle:
            handle.write(summary)

    # --- 5. deploy ------------------------------------------------------------
    print(f"Calibration written to {out_yaml}")
    should_install = args.install
    if should_install is None:
        should_install = ask_yes_no(f"Copy it over {DEPLOY_PATH}?", False, args.yes)
    if should_install:
        install(out_yaml)

    destination = args.send_to or remote_destination(args.ip)
    should_send = args.send
    if should_send is None:
        # Writing to the camera is not something --yes should do on its own: an
        # unattended run says what it is skipping instead of deploying quietly.
        if args.yes:
            should_send = False
            print(f"Not sending to {destination} -- pass --send to deploy.")
        else:
            should_send = ask_yes_no(f"Send it to the camera at {destination}?", False)
    if should_send:
        send_to_camera(out_yaml, destination)


if __name__ == "__main__":
    main()
