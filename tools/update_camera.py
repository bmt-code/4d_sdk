#!/usr/bin/env python3
"""Update the firmware of a Stereo 4D camera (a Raspberry Pi 5) from the computer it is cabled to.

One file, Python >= 3.8 standard library only, so it runs on any computer without the SDK
installed: copy it over, or fetch it straight from GitHub (the SDK repository is public):

    curl -fsSLO https://raw.githubusercontent.com/bmt-code/4d_sdk/main/tools/update_camera.py
    python3 update_camera.py                     # same as: network
    python3 update_camera.py preflight
    python3 update_camera.py update --ref v2.1.0

Commands
    preflight   says exactly which layer is broken, from the cable to GitHub. Changes nothing,
                except that it offers to install this computer's SSH key or forget a changed
                host key when those are what is broken
    connect     preflight up to the firmware folder, without the internet checks; retries
                while the camera boots (--retries)
    network     (default) stop the camera's Ethernet link from carrying its default route
    wifi        give the camera internet over Wi-Fi (a network it sees, or this computer's)
    update      all of it: preflight, network, Wi-Fi, back up the calibration, pull the
                firmware, put the calibration back, install.sh, wait for the reboot, check

Nothing waits forever: every remote command has a timeout, and the reboot has a deadline.

Secrets (the GitHub token the camera pulls the private firmware with, a Wi-Fi password, the
camera's password) are asked for on the terminal, or with --secrets-stdin read as one JSON
object from stdin ({"token": ..., "wifi_psk": ..., "camera_password": ...}). They are never
saved, never put on a command line, and never printed.

Exit codes: 0 ok, 2 usage, 10 no link, 11 no SSH server, 12 SSH login refused, 13 host key
changed, 14 no usable sudo, 15 no firmware found, 16 locked out after failed logins, 20 camera has no internet, 21 Wi-Fi
failed, 30 git failed, 31 calibration not preserved, 40 install.sh failed, 41 camera did not
come back, 50 service unhealthy, 130 interrupted.
"""

import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime

DEFAULT_IP = "172.31.1.77"
DEFAULT_USER = "bmt"
FIRMWARE_URL = "https://github.com/bmt-code/stereo_4d.git"
SERVICE = "stereo_4d.service"
CALIBRATION = "calib/stereo_calibration.yaml"
BACKUPS = "~/.st4d_backup"  # on the camera, outside the firmware folder
ZMQ_PORTS = (5555, 5556)

# What the camera must reach: git pulls from GitHub, install.sh runs apt and pip
INTERNET = (
    ("github.com", "https://github.com", True),
    ("pypi.org", "https://pypi.org/simple/", False),
    ("archive.raspberrypi.com", "http://archive.raspberrypi.com/debian/", False),
    ("deb.debian.org", "http://deb.debian.org/debian/", False),
)

EXIT = {
    "ok": 0, "usage": 2, "link": 10, "ssh_port": 11, "auth": 12, "host_key": 13,
    "sudo": 14, "firmware": 15, "blocked": 16, "internet": 20, "wifi": 21, "git": 30, "calibration": 31,
    "install": 40, "reboot": 41, "health": 50, "interrupted": 130,
}


class Failure(Exception):
    """A step failed: what went wrong, what to do about it, and the exit code."""

    def __init__(self, kind, message, hint=None):
        super().__init__(message)
        self.kind, self.message, self.hint = kind, message, hint


# --------------------------------------------------------------------------- output

class Out:
    """Human output on a terminal, or one JSON object per line with --json (for wrappers
    such as the Orion deployer, which render it themselves)."""

    def __init__(self, as_json):
        self.json = as_json
        color = not as_json and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self.c = {k: (v if color else "") for k, v in
                  {"g": "\033[32m", "r": "\033[31m", "y": "\033[33m", "b": "\033[1m",
                   "d": "\033[2m", "x": "\033[0m"}.items()}

    def emit(self, **event):
        print(json.dumps(event), flush=True)

    def step(self, title):
        if self.json:
            return self.emit(event="step", title=title)
        print(f"\n{self.c['b']}== {title}{self.c['x']}", flush=True)

    def info(self, text):
        if self.json:
            return self.emit(event="info", text=text)
        print(f"   {text}", flush=True)

    def log(self, text):
        """A line of remote output, e.g. install.sh."""
        if self.json:
            return self.emit(event="log", text=text)
        print(f"   {self.c['d']}| {text}{self.c['x']}", flush=True)

    def warn(self, text):
        if self.json:
            return self.emit(event="warn", text=text)
        print(f" {self.c['y']}!{self.c['x']} {text}", flush=True)

    def check(self, name, ok, detail="", hint=None, warn=False):
        if self.json:
            return self.emit(event="check", name=name, ok=ok, warn=warn, detail=detail, hint=hint)
        mark = (f"{self.c['y']}!" if warn else f"{self.c['g']}✔") if ok else f"{self.c['r']}✘"
        print(f" {mark}{self.c['x']} {name}{': ' + detail if detail else ''}", flush=True)
        if hint and (not ok or warn):
            for line in hint.splitlines():
                print(f"     {self.c['d']}→ {line}{self.c['x']}", flush=True)

    def result(self, ok, kind, message, hint=None):
        if self.json:
            return self.emit(event="result", ok=ok, kind=kind, message=message, hint=hint,
                             exit_code=EXIT.get(kind, 1))
        if ok:
            print(f"\n{self.c['g']}{self.c['b']}✔ {message}{self.c['x']}", flush=True)
        else:
            print(f"\n{self.c['r']}{self.c['b']}✘ {message}{self.c['x']}", flush=True)
            for line in (hint or "").splitlines():
                print(f"  → {line}", flush=True)


# --------------------------------------------------------------------------- local helpers

def run(cmd, timeout, input=None, env=None):
    """subprocess.run with a timeout that is never optional. Returns (rc, stdout, stderr);
    rc is None when it timed out."""
    try:
        r = subprocess.run(cmd, input=input, capture_output=True, timeout=timeout, env=env,
                           encoding="utf-8", errors="replace",
                           stdin=None if input is not None else subprocess.DEVNULL)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        def text(b):
            return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
        return None, text(e.stdout), text(e.stderr)
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: command not found"


def last_line(text, default="no output"):
    lines = [l for l in (text or "").strip().splitlines()
             if l.strip() and "setlocale" not in l]  # the camera lacks the locale ssh forwards
    return lines[-1].strip() if lines else default


class Secrets:
    """Asked once, on the terminal or from --secrets-stdin; kept in memory only."""

    def __init__(self, from_stdin, interactive):
        self.values, self.interactive = {}, interactive
        if from_stdin:
            raw = sys.stdin.readline()
            try:
                self.values = {k: v for k, v in json.loads(raw or "{}").items() if v}
            except ValueError:
                raise Failure("usage", "--secrets-stdin expects one JSON object on stdin")

    def get(self, key, prompt):
        if key not in self.values:
            if not self.interactive:
                return None
            self.values[key] = getpass.getpass(f"   {prompt}: ").strip()
        return self.values[key] or None


def ask_yes(question, default, interactive, assume_yes):
    if assume_yes:
        return True
    if not interactive:
        return default
    answer = input(f"   {question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not answer else answer.startswith("y")


# --------------------------------------------------------------------------- camera over ssh

# When sudo on the camera needs a password: every command gets a sudo() that drops -n and
# asks SUDO_ASKPASS instead. The askpass lives in /dev/shm (memory, gone at reboot), 0700,
# for this run only; the password never touches a command line or a disk.
SUDO_PREFIX = ('export SUDO_ASKPASS={path}; '
               'sudo() {{ [ "$1" = -n ] && shift; command sudo -A "$@"; }}; export -f sudo; ')
ASKPASS_WRITE = r"""
set -eu
d=/dev/shm/st4d-$(id -u)
umask 077; mkdir -p "$d"; chmod 700 "$d"
cat > "$d/askpass"; chmod 700 "$d/askpass"
echo "$d/askpass"
"""


class Camera:
    def __init__(self, args):
        self.ip, self.user = args.ip, args.user
        self.known_hosts = os.path.expanduser(args.known_hosts)
        self.key = args.identity
        self.sudo_password, self.askpass = None, None

    def provide_askpass(self):
        """(Re)creates the askpass on the camera; needed again after a reboot."""
        self.askpass = None
        body = f"#!/bin/sh\ncat <<'ST4D_PW_END'\n{self.sudo_password}\nST4D_PW_END\n"
        rc, text, err = self.script(ASKPASS_WRITE, timeout=15, input=body)
        if rc != 0 or not text.strip():
            raise Failure("sudo", f"Could not set up sudo on the camera: {last_line(err)}")
        self.askpass = text.strip()

    def remove_askpass(self):
        if self.askpass:
            self.askpass = None
            self.run("rm -rf /dev/shm/st4d-$(id -u)", timeout=10)

    def ssh_base(self, batch=True, connect_timeout=8):
        opts = ["ssh", "-o", f"ConnectTimeout={connect_timeout}",
                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
                "-o", f"UserKnownHostsFile={self.known_hosts}",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "LogLevel=ERROR", "-o", "ControlMaster=no"]
        if batch:
            opts += ["-o", "BatchMode=yes"]
        if self.key:
            opts += ["-i", self.key, "-o", "IdentitiesOnly=yes"]
        return opts + [f"{self.user}@{self.ip}"]

    def run(self, command, timeout=30, input=None):
        """A shell command on the camera. Returns (rc, out, err), rc None on timeout."""
        if self.askpass:
            command = SUDO_PREFIX.format(path=shlex.quote(self.askpass)) + command
        return run(self.ssh_base() + [command], timeout=timeout, input=input)

    def script(self, script, *args, timeout=60, input=None):
        """A bash script on the camera, with positional args; stdin stays free for secrets."""
        quoted = " ".join(shlex.quote(str(a)) for a in args)
        return self.run(f"bash -c {shlex.quote(script)} bash {quoted}", timeout=timeout, input=input)

    def must(self, command, what, timeout=30, kind="health", input=None):
        rc, out, err = self.run(command, timeout=timeout, input=input)
        if rc is None:
            raise Failure(kind, f"Timed out after {timeout}s: {what}")
        if rc != 0:
            raise Failure(kind, f"Could not {what}: {last_line(err or out)}")
        return out

    def boot_id(self):
        rc, out, _ = self.run("cat /proc/sys/kernel/random/boot_id", timeout=12)
        return out.strip() if rc == 0 and out.strip() else None


# --------------------------------------------------------------------------- preflight checks

def check_local_link(cam, out):
    """The route to the camera leaves through a cable, not docker, a VPN or a router."""
    rc, text, err = run(["ip", "-j", "route", "get", cam.ip], timeout=5)
    if rc != 0:
        out.check("Route to the camera", False, last_line(err),
                  "This computer has no route to the camera: is the Ethernet cable plugged in?")
        return False
    route = (json.loads(text) or [{}])[0]
    dev, src, via = route.get("dev", "?"), route.get("prefsrc", "?"), route.get("gateway")
    virtual = re.match(r"(docker|br-|veth|virbr|tailscale|tun|tap|wg|zt|lo)", dev)
    if via or virtual:
        out.check("Route to the camera", False, f"through {via or dev} (src {src}), not over a cable",
                  f"{cam.ip} must be on a directly cabled network card. Either the cable's card has no\n"
                  f"address in the camera's subnet (unplugged, or not on DHCP: the camera hands out\n"
                  f"172.31.1.2-76), or another network (VPN, docker, router) claims {cam.ip}'s range.")
        return False
    out.check("Route to the camera", True, f"{dev}, this computer is {src}")
    # This computer's own internet must not go into the camera link either
    rc, text, _ = run(["ip", "-j", "-4", "route", "show", "default"], timeout=5)
    defaults = sorted(json.loads(text or "[]") if rc == 0 else [], key=lambda r: r.get("metric", 0))
    if defaults and defaults[0].get("dev") == dev:
        out.check("This computer's default route", True,
                  f"via {defaults[0].get('gateway')} on {dev}, the camera's cable", warn=True,
                  hint=f"This computer sends its own internet traffic into the camera link. Set\n"
                       f"ipv4.never-default yes on {dev}'s NetworkManager connection.")
    return True


def check_ping(cam, out):
    rc, text, _ = run(["ping", "-c", "2", "-W", "1", cam.ip], timeout=6)
    found = re.search(r"time[=<]([\d.]+) ms", text or "")
    if rc == 0 and found:
        ms = float(found.group(1))
        out.check("Ping", True, f"{ms:.2f} ms", warn=ms > 1,
                  hint="Over 1 ms: is the camera on a direct cable to this computer?")
        return True
    rc, neigh, _ = run(["ip", "neigh", "show", cam.ip], timeout=5)
    if "lladdr" in (neigh or ""):
        out.check("Ping", False, "no reply, but something answers ARP at that address",
                  "The camera is on the cable but does not answer ping: is it still booting?")
    else:
        out.check("Ping", False, "nothing answers at that address",
                  "Check the camera has power (it takes ~40 s to boot) and the cable is in both ends.")
    return False


# What OpenSSH >= 9.8 (the camera runs 10.0) says instead of its greeting while
# PerSourcePenalties shuts this computer out: every failed login (and every connection
# dropped before logging in) adds seconds, and from 15 s on, sshd refuses the address
# outright, for up to 10 minutes. A client retrying a refused key keeps itself locked out.
BLOCKED = "Not allowed at this time"
BLOCKED_HINT = ("sshd on the camera locks out an address after repeated failed logins (OpenSSH\n"
                "PerSourcePenalties). Usually: this computer has no key on the camera and kept retrying.\n"
                "Stop retrying: it clears by itself within minutes, or at once if the camera is\n"
                "power-cycled. Then install a key (the script offers it, or ssh-copy-id).")


def check_ssh_port(cam, out):
    """Returns "ok", "blocked" (sshd refuses this address for now), or "" when broken."""
    try:
        with socket.create_connection((cam.ip, 22), timeout=5) as s:
            s.settimeout(8)
            banner = s.recv(64).decode(errors="replace").strip()
    except socket.timeout:
        out.check("SSH server", False, "port 22 open but no SSH greeting within 8 s",
                  "sshd on the camera is stuck or overloaded. Power-cycle the camera.")
        return ""
    except OSError as e:
        out.check("SSH server", False, f"port 22: {e.strerror or e}",
                  "The camera answers ping but runs no SSH server (still booting, or sshd disabled).")
        return ""
    if BLOCKED in banner:
        out.check("SSH server", False, f"refuses this computer for now: '{BLOCKED}'", BLOCKED_HINT)
        return "blocked"
    if not banner.startswith("SSH-"):
        out.check("SSH server", False, f"port 22 answers {banner!r}, not SSH",
                  f"Another device has {cam.ip}. Unplug everything else on that cable.")
        return ""
    out.check("SSH server", True, banner)
    return "ok"


def check_login(cam, out):
    """Logs in with a key. Returns None when it works, else the failure kind."""
    rc, text, err = cam.run("true", timeout=20)
    if rc == 0:
        out.check("SSH login", True, f"{cam.user}@{cam.ip} with a key")
        return None
    if rc is None:
        out.check("SSH login", False, "timed out after 20 s",
                  "TCP and the greeting work but the login hangs: power-cycle the camera.")
        return "ssh_port"
    if "REMOTE HOST IDENTIFICATION HAS CHANGED" in err or "Host key verification failed" in err:
        out.check("SSH login", False, f"the camera's host key is not the one {cam.known_hosts} knows",
                  f"Expected after the camera is reflashed or swapped (every camera is {cam.ip}).\n"
                  f"Rerun with --forget-host-key, or: ssh-keygen -f {cam.known_hosts} -R {cam.ip}")
        return "host_key"
    if "Permission denied" in err:
        out.check("SSH login", False, f"{cam.user}@{cam.ip} refused this computer's key "
                  "(ping and SSH work: this is the 'pings but cannot connect' case)",
                  "Install a key with the camera's password: rerun interactively, or\n"
                  f"ssh-copy-id {cam.user}@{cam.ip}")
        return "auth"
    if "Connection closed" in err or "Connection reset" in err or BLOCKED in err:
        out.check("SSH login", False, "the camera closed the connection before the login", BLOCKED_HINT)
        return "blocked"
    out.check("SSH login", False, last_line(err))
    return "ssh_port"


def check_sudo(cam, out, secrets):
    """Root on the camera: passwordless sudo if it has it, else sudo with the camera's
    password, asked once and kept in memory."""
    rc, _, err = cam.run("sudo -n true", timeout=15)
    if rc == 0:
        out.check("sudo", True, "passwordless")
        return True
    password = secrets.get("camera_password", f"Password of {cam.user} on the camera, for sudo (not saved)")
    if not password:
        out.check("sudo", False, "the camera's sudo needs a password and none was given",
                  "Rerun interactively and type it, or pass camera_password with --secrets-stdin.")
        return False
    rc, _, err = cam.run("sudo -S -k -p '' true", timeout=20, input=password + "\n")
    if rc != 0:
        secrets.values.pop("camera_password", None)
        out.check("sudo", False, "the camera refused that password for sudo",
                  f"It is {cam.user}'s login password on the camera.")
        return False
    cam.sudo_password = password
    cam.provide_askpass()
    rc, _, err = cam.run("sudo -n true", timeout=15)
    if rc != 0:
        out.check("sudo", False, f"sudo with the password failed: {last_line(err)}")
        return False
    out.check("sudo", True, "with the camera's password (kept in memory for this run)")
    return True


FIND_FIRMWARE = r"""
root=$(systemctl cat stereo_4d.service 2>/dev/null \
  | sed -n 's|^ExecStart=[^ ]*python3[^ ]* \(/[^ ]*\)/scripts/img_sender\.py.*|\1|p' | tail -n 1)
if [ -z "$root" ] || [ ! -f "$root/scripts/img_sender.py" ]; then
  for d in "$1" ~/4d_firmware ~/stereo_4d; do
    [ -n "$d" ] && [ -f "$d/scripts/img_sender.py" ] && { root=$d; break; }
  done
fi
[ -n "$root" ] && [ -f "$root/scripts/img_sender.py" ] || exit 3
cd "$root"
echo "root=$root"
echo "version=$(git describe --tags --always 2>/dev/null || echo unknown)"
echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "dirty=$(git status --porcelain 2>/dev/null | wc -l)"
echo "calibration=$([ -f calib/stereo_calibration.yaml ] && echo yes || echo no)"
echo "service=$(systemctl is-active stereo_4d.service 2>/dev/null || true)"
"""


def find_firmware(cam, out, hint_dir=None):
    rc, text, err = cam.script(FIND_FIRMWARE, hint_dir or "", timeout=20)
    if rc != 0:
        out.check("Firmware", False, "no firmware folder found" if rc == 3 else last_line(err),
                  "stereo_4d.service does not point at a firmware checkout. Pass --firmware-dir.")
        return None
    info = dict(l.split("=", 1) for l in text.splitlines() if "=" in l)
    dirty = int(info.get("dirty", "0") or 0)
    out.check("Firmware", True, f"{info['root']} at {info['version']} ({info['commit']})"
              f"{f', {dirty} local changes' if dirty else ''}")
    out.check("Calibration", True,
              CALIBRATION if info.get("calibration") == "yes" else f"no {CALIBRATION}",
              "The camera runs on the default calibration: calibrate it after the update.",
              warn=info.get("calibration") != "yes")
    out.check("Service", True, f"{SERVICE} {info.get('service') or 'missing'}",
              warn=info.get("service") != "active",
              hint="Not running now; the update reinstalls it. Logs: sudo journalctl -u stereo_4d.service")
    return info


ROUTES = r"""
wired=$(nmcli -g DEVICE,TYPE device status 2>/dev/null | awk -F: '$2=="ethernet"{print $1; exit}')
echo "wired=${wired:-eth0}"
echo "wired_con=$(nmcli -g GENERAL.CONNECTION device show "${wired:-eth0}" 2>/dev/null)"
nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device status 2>/dev/null | awk -F: '$2=="wifi"{print "wifi="$1":"$3":"$4}'
ip -4 route show default | sed 's/^/v4=/'
ip -6 route show default | sed 's/^/v6=/'
echo "dns=$(getent ahostsv4 github.com | awk 'NR==1{print $1}')"
echo "github4=$(ip -4 route get 140.82.112.3 2>/dev/null | head -n 1)"
"""


# Prints "<name> <http code>" or "<name> <curl's error>" per (name, url) pair
PROBE = r"""
while [ $# -ge 2 ]; do
  err=$(mktemp)
  code=$(curl -4 -sS -o /dev/null -w '%{http_code}' --max-time 8 "$2" 2>"$err")
  case "$code" in 000|"") echo "$1 $(head -n 1 "$err" | sed 's/^curl: ([0-9]*) //')";; *) echo "$1 $code";; esac
  rm -f "$err"; shift 2
done
"""


def read_routes(cam):
    rc, text, err = cam.script(ROUTES, timeout=20)
    if rc != 0:
        raise Failure("internet", f"Could not read the camera's routes: {last_line(err)}")
    info = {"v4": [], "v6": [], "wifi": []}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key in info and isinstance(info[key], list):
            info[key].append(value)
        else:
            info[key] = value
    return info


def wired_defaults(routes):
    wired = routes.get("wired", "eth0")
    return [r for r in routes["v4"] + routes["v6"] if f" dev {wired} " in f" {r} "]


def check_internet(cam, out, routes=None):
    """Routes, DNS and HTTPS from the camera. Returns True when it reaches what git needs."""
    routes = routes or read_routes(cam)
    bad = wired_defaults(routes)
    if bad:
        out.check("Camera routes", False, f"the Ethernet link to this computer is a default route: {bad[0]}",
                  f"IPv4 internet goes into the cable, where nothing routes it. IPv6 can still leave over\n"
                  f"Wi-Fi, which is why google.com (IPv6) works and github.com (IPv4 only) does not.\n"
                  f"Fix: python3 {os.path.basename(__file__)} network")
    else:
        out.check("Camera routes", True,
                  "; ".join(r.split(" proto")[0] for r in routes["v4"]) or "no IPv4 default route",
                  warn=not routes["v4"], hint="No default route at all: the camera is on no Wi-Fi.")
    wifi = [w.split(":") for w in routes["wifi"]]
    connected = [w for w in wifi if len(w) > 2 and w[1] == "connected"]
    out.check("Camera Wi-Fi", True, warn=not connected, detail=
              f"{connected[0][0]} on '{connected[0][2]}'" if connected
              else (f"{wifi[0][0]} {wifi[0][1]}" if wifi else "no Wi-Fi device"),
              hint=f"Join a network: python3 {os.path.basename(__file__)} wifi")
    out.check("Camera DNS", bool(routes.get("dns")), f"github.com is {routes.get('dns') or 'not resolved'}",
              "The camera cannot resolve names: no DNS server from its Wi-Fi.")
    args = [a for name, url, _ in INTERNET for a in (name, url)]
    rc, text, _ = cam.script(PROBE, *args, timeout=9 * len(INTERNET) + 15)
    results = dict(l.split(" ", 1) for l in (text or "").splitlines() if " " in l)
    ok_git = True
    for name, _, required in INTERNET:
        code = results.get(name, "no answer (timed out)").strip()
        good = bool(re.fullmatch(r"[1-4]\d\d", code))
        ok_git &= good or not required
        out.check(f"Camera → {name} (IPv4)", good, f"HTTP {code}" if good else code,
                  warn=not good and not required,
                  hint=None if required else "install.sh needs it for apt/pip")
    return ok_git and not bad


# --------------------------------------------------------------------------- fixes

def install_key(cam, out, secrets, interactive):
    """The camera refused this computer's key: install one with the camera's password."""
    key = cam.key
    if not key:
        for name in ("id_ed25519", "id_ecdsa", "id_rsa"):
            path = os.path.expanduser(f"~/.ssh/{name}")
            if os.path.exists(path) and os.path.exists(path + ".pub"):
                key = path
                break
    if not key:
        key = os.path.expanduser("~/.ssh/id_ed25519")
        os.makedirs(os.path.dirname(key), mode=0o700, exist_ok=True)
        rc, _, err = run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", key,
                          "-C", f"{getpass.getuser()}@{socket.gethostname()}"], timeout=30)
        if rc != 0:
            raise Failure("auth", f"Could not create an SSH key: {last_line(err)}")
        out.info(f"Created {key}")
    copy = ["ssh-copy-id", "-i", key + ".pub", "-o", "ConnectTimeout=8",
            "-o", f"UserKnownHostsFile={cam.known_hosts}", "-o", "StrictHostKeyChecking=accept-new",
            f"{cam.user}@{cam.ip}"]
    password = secrets.values.get("camera_password")
    if interactive and not password:
        out.info(f"Installing {key}.pub on {cam.user}@{cam.ip}: type the camera's password")
        try:
            rc = subprocess.run(copy, timeout=120).returncode
        except subprocess.TimeoutExpired:
            rc = None
        if rc != 0:
            raise Failure("auth", "ssh-copy-id did not install the key", "Check the camera's password.")
        return
    if not password:
        raise Failure("auth", f"{cam.user}@{cam.ip} refused this computer's key and no camera password was given",
                      f"Run ssh-copy-id {cam.user}@{cam.ip}, or pass camera_password with --secrets-stdin.")
    # Non-interactive: ssh reads the password from a private askpass script, never argv
    with tempfile.TemporaryDirectory() as folder:
        askpass = os.path.join(folder, "askpass")
        with open(askpass, "w") as f:
            f.write('#!/bin/sh\nprintf "%s\\n" "$ST4D_PASSWORD"\n')
        os.chmod(askpass, 0o700)
        env = {**os.environ, "SSH_ASKPASS": askpass, "SSH_ASKPASS_REQUIRE": "force",
               "DISPLAY": os.environ.get("DISPLAY", "none"), "ST4D_PASSWORD": password}
        try:
            r = subprocess.run(copy[:1] + ["-o", "NumberOfPasswordPrompts=1"] + copy[1:],
                               env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=60, start_new_session=True)
            rc, err = r.returncode, r.stderr + r.stdout
        except subprocess.TimeoutExpired:
            rc, err = None, "timed out"
    if rc != 0:
        if "Permission denied" in err:
            raise Failure("auth", f"{cam.user}@{cam.ip} refused the password")
        if "Connection closed" in err or BLOCKED in err:
            raise Failure("blocked", "The camera locked this computer out after failed logins", BLOCKED_HINT)
        raise Failure("auth", f"Could not install the key: {last_line(err)}")
    cam.key = key


NETWORK_FIX = r"""
set -eu
wired=$(nmcli -g DEVICE,TYPE device status | awk -F: '$2=="ethernet"{print $1; exit}')
wired=${wired:-eth0}
con=$(nmcli -g GENERAL.CONNECTION device show "$wired" 2>/dev/null || true)
[ -n "$con" ] || con="Wired connection 1"
# Saved, for every boot: no gateway, never the default route. The address is untouched.
sudo -n nmcli connection modify "$con" ipv4.gateway "" ipv4.never-default yes ipv6.never-default yes
# Live, without taking the link down (that would cut this very session)
for fam in -4 -6; do
  while ip $fam route show default dev "$wired" | grep -q .; do
    sudo -n ip $fam route del default dev "$wired" || break
  done
done
echo "fixed $con on $wired"
"""


def fix_network(cam, out):
    routes = read_routes(cam)
    bad = wired_defaults(routes)
    if not bad:
        out.check("Ethernet is not a default route", True, f"{routes.get('wired_con') or routes.get('wired')}")
        return routes
    rc, text, err = cam.script(NETWORK_FIX, timeout=30)
    if rc != 0:
        raise Failure("internet", f"Could not fix the camera's routes: {last_line(err or text)}")
    routes = read_routes(cam)
    if wired_defaults(routes):
        raise Failure("internet", f"The Ethernet link is still a default route: {wired_defaults(routes)[0]}")
    out.check("Ethernet is not a default route", True,
              f"removed '{bad[0].split(' proto')[0]}'; {last_line(text)} (saved: ipv4.never-default)")
    return routes


def local_wifi():
    """(connection name, ssid) of this computer's active Wi-Fi, or None."""
    rc, text, _ = run(["nmcli", "-t", "-f", "NAME,TYPE", "connection", "show", "--active"], timeout=10)
    for line in (text or "").splitlines() if rc == 0 else []:
        name, _, kind = line.rpartition(":")
        if kind == "802-11-wireless":
            name = name.replace("\\:", ":")
            rc, ssid, _ = run(["nmcli", "-g", "802-11-wireless.ssid", "connection", "show", name], timeout=10)
            return name, (ssid or "").strip() or name
    return None


def local_wifi_psk(name, interactive):
    cmd = ["nmcli", "-s", "-g", "802-11-wireless-security.psk", "connection", "show", name]
    rc, text, _ = run(cmd, timeout=10)
    if rc == 0 and text.strip():
        return text.strip()
    if interactive:  # system connections hide the password from users without sudo
        try:
            r = subprocess.run(["sudo", *cmd], capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except subprocess.TimeoutExpired:
            pass
    return None


def keyfile_escape(value):
    return value.replace("\\", "\\\\").replace("\n", "\\n")


WIFI_JOIN = r"""
set -eu
name=$1; file="/etc/NetworkManager/system-connections/$2.nmconnection"
# The profile, with its password, arrives on stdin and goes straight into a root-only file
sudo -n sh -c 'umask 077; cat > "$1"; chmod 600 "$1"' sh "$file"
sudo -n nmcli connection reload
sudo -n nmcli --wait 45 connection up "$name"
"""


def camera_scan(cam):
    rc, text, _ = cam.run("sudo -n nmcli -t -f SSID,SIGNAL,SECURITY device wifi list --rescan yes", timeout=30)
    seen = {}
    for line in (text or "").splitlines() if rc == 0 else []:
        parts = re.split(r"(?<!\\):", line)
        if len(parts) >= 3 and parts[0]:
            ssid = parts[0].replace("\\:", ":")
            if ssid not in seen or int(parts[1] or 0) > int(seen[ssid][0] or 0):
                seen[ssid] = (parts[1], parts[2])
    return sorted(seen.items(), key=lambda kv: -int(kv[1][0] or 0))


def join_wifi(cam, out, args, secrets, interactive):
    """Puts the camera on a Wi-Fi network and waits until it reaches GitHub."""
    ssid, psk = args.wifi_ssid, secrets.values.get("wifi_psk")
    scan = camera_scan(cam)
    if scan:
        out.info("Networks the camera sees: " + ", ".join(f"{s} ({sig}%)" for s, (sig, _) in scan[:8]))
    if not ssid and args.share_wifi is not False:
        mine = local_wifi()
        if mine and (args.share_wifi or ask_yes(f"Put the camera on this computer's Wi-Fi '{mine[1]}'?",
                                                True, interactive, False)):
            ssid = mine[1]
            psk = psk or local_wifi_psk(mine[0], interactive)
            if psk is None:
                out.warn(f"Could not read the password of '{mine[0]}' on this computer")
    if not ssid and interactive and scan:
        for i, (s, (sig, sec)) in enumerate(scan[:15], 1):
            out.info(f"{i:2}. {s}  {sig}%  {sec or 'open'}")
        pick = input("   Network number (or a name): ").strip()
        ssid = scan[int(pick) - 1][0] if pick.isdigit() and 0 < int(pick) <= len(scan[:15]) else pick
    if not ssid:
        raise Failure("wifi", "The camera has no internet and no Wi-Fi network was given",
                      "Pass --wifi-ssid NAME (password via the prompt or --secrets-stdin), or\n"
                      "--share-wifi to use this computer's network.")
    security = dict(scan).get(ssid, ("", "WPA2"))[1]
    open_net = security in ("", "--")
    if not open_net and not psk:
        psk = secrets.get("wifi_psk", f"Password of Wi-Fi '{ssid}'")
        if not psk:
            raise Failure("wifi", f"No password for Wi-Fi '{ssid}'")
    if ssid not in dict(scan):
        out.warn(f"The camera does not see '{ssid}' right now; trying anyway")
    key_mgmt = "sae" if "WPA3" in security and "WPA2" not in security else "wpa-psk"
    name = f"st4d-{re.sub(r'[^A-Za-z0-9_.-]', '_', ssid)}"
    profile = [
        "[connection]", f"id={name}", f"uuid={uuid.uuid4()}", "type=wifi", "autoconnect=true",
        "autoconnect-priority=10", "", "[wifi]", "mode=infrastructure", f"ssid={keyfile_escape(ssid)}", "",
    ]
    if not open_net:
        profile += ["[wifi-security]", f"key-mgmt={key_mgmt}", f"psk={keyfile_escape(psk)}", ""]
    profile += ["[ipv4]", "method=auto", "", "[ipv6]", "method=auto", ""]
    out.info(f"Joining '{ssid}'…")
    rc, text, err = cam.script(WIFI_JOIN, name, name, timeout=75, input="\n".join(profile))
    if rc != 0:
        reason = "timed out" if rc is None else last_line(err or text)
        if "Secrets were required" in (err or "") or "802-1x" in (err or ""):
            reason = "wrong password"
        raise Failure("wifi", f"The camera could not join '{ssid}': {reason}")
    for _ in range(12):
        rc, code, _ = cam.run("curl -4 -sS -o /dev/null -w '%{http_code}' --max-time 8 https://github.com",
                              timeout=15)
        if rc == 0 and re.fullmatch(r"[1-4]\d\d", code.strip()):
            out.check("Camera Wi-Fi", True, f"on '{ssid}', reaches github.com")
            return
        time.sleep(5)
    raise Failure("wifi", f"The camera joined '{ssid}' but does not reach github.com",
                  "A captive portal or a firewall on that network? Try another network.")


# --------------------------------------------------------------------------- update steps

def token_problem(token, ref, url):
    """Checked from this computer when it has internet: the token reads the firmware, and
    the version exists. Returns None, a problem string, or 'offline'."""
    with tempfile.TemporaryDirectory() as folder:
        askpass = os.path.join(folder, "askpass")
        with open(askpass, "w") as f:
            f.write('#!/bin/sh\nprintf "%s\\n" "$ST4D_TOKEN"\n')
        os.chmod(askpass, 0o700)
        rc, text, err = run(
            ["git", "-c", "credential.helper=", "ls-remote", "--heads", "--tags",
             url.replace("https://", "https://x-access-token@")], timeout=30,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": askpass, "ST4D_TOKEN": token})
    if rc is None or "Could not resolve host" in err or ("unable to access" in err and "403" not in err):
        return "offline"
    if rc != 0:
        return f"GitHub refused the token: {last_line(err)}"
    refs = {l.split("\t")[1].replace("refs/heads/", "").replace("refs/tags/", "").replace("^{}", "")
            for l in text.splitlines() if "\t" in l}
    if ref not in refs:
        return f"'{ref}' is not a branch or tag of the firmware"
    return None


PULL = r"""
set -euo pipefail
ROOT=$1; B=$2; REF=$3; URL=$4; CAL=$5
cd "$ROOT"
export ST4D_TOKEN=$(head -n 1)
askpass=$(mktemp); chmod 700 "$askpass"
printf '#!/bin/sh\nprintf "%%s\\n" "$ST4D_TOKEN"\n' > "$askpass"
echo "at $(git describe --tags --always 2>/dev/null || echo '?')"

before=none
if [ -f "$CAL" ]; then
  mkdir -p "$B"; cp -p "$CAL" "$B/"
  cmp -s "$CAL" "$B/$(basename "$CAL")" || { echo "the calibration backup differs from the original" >&2; exit 31; }
  sync
  before=$(sha256sum < "$CAL" | cut -d' ' -f1)
  echo "calibration backed up to $B/ (sha256 ${before:0:16})"
else
  echo "no $CAL: the camera runs on the default calibration"
fi

sha() { [ -f "$CAL" ] && sha256sum < "$CAL" | cut -d' ' -f1 || echo none; }
restore() {
  if [ "$before" = none ]; then
    [ -f "$CAL" ] && echo "WARNING: $REF tracks $CAL in git: this is ANOTHER camera's calibration. Recalibrate." >&2
    return 0
  fi
  [ "$(sha)" = "$before" ] || { mkdir -p "$(dirname "$CAL")"; cp -p "$B/$(basename "$CAL")" "$CAL"; echo "calibration restored from $B/"; }
  [ "$(sha)" = "$before" ] || { echo "the calibration is not the backed up one" >&2; return 1; }
  echo "calibration intact (sha256 ${before:0:16})"
}
trap 'rc=$?; rm -f "$askpass"; restore || rc=31; exit $rc' EXIT

git status --short | sed 's/^/discarding: /'
git reset -q --hard
GIT_ASKPASS=$askpass GIT_TERMINAL_PROMPT=0 timeout 300 git -c credential.helper= \
  -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
  fetch -q --force --tags "$URL" '+refs/heads/*:refs/remotes/origin/*' || exit 30
if git rev-parse -q --verify "refs/tags/$REF^{commit}" >/dev/null; then target="refs/tags/$REF"
elif git rev-parse -q --verify "refs/remotes/origin/$REF" >/dev/null; then target="refs/remotes/origin/$REF"
else echo "'$REF' is not a tag or branch of the firmware" >&2; exit 30; fi
git checkout -q --force --detach "$target"
echo "now at $(git describe --tags --always) ($(git rev-parse --short HEAD))"
"""


def pull_firmware(cam, out, root, ref, token, url):
    stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    rc, home, _ = cam.run("echo $HOME", timeout=15)
    backup = BACKUPS.replace("~", home.strip() or "~") + f"/{stamp}"
    authed = url.replace("https://", "https://x-access-token@") if token else url
    rc, text, err = cam.script(PULL, root, backup, ref, authed, CALIBRATION, timeout=420,
                               input=(token or "") + "\n")
    for line in (text or "").splitlines():
        out.log(line)
    for line in (err or "").splitlines():
        if line.strip():
            out.log(line)
    if rc is None:
        raise Failure("git", "Timed out pulling the firmware (7 min)",
                      "The camera's internet is too slow or keeps dropping. Run preflight.")
    if rc == 31:
        raise Failure("calibration", f"The calibration could not be preserved; the backup is in {backup}")
    if rc != 0:
        reason = last_line(err)
        hint = None
        if "Authentication failed" in err or "403" in err or "could not read Username" in err:
            hint = "The token cannot read bmt-code/stereo_4d: it needs Contents: read on that repository."
        elif "Could not resolve host" in err or "Failed to connect" in err:
            hint = "The camera lost the internet. Run preflight."
        raise Failure("git", f"git failed on the camera: {reason}", hint)
    return backup


INSTALL_START = r"""
set -eu
ROOT=$1; LOG=$2
# Detached, so neither a dropped SSH session nor install.sh's own network changes kill it.
# The log is in /var/tmp, which survives the reboot install.sh ends with.
sudo -n setsid nohup bash -c 'cd "$1"; bash install.sh > "$2" 2>&1; echo "__ST4D_EXIT__ $?" >> "$2"' \
  bash "$ROOT" "$LOG" < /dev/null > /dev/null 2>&1 &
sleep 1
echo started
"""


def run_install(cam, out, root, timeout, quiet_after=600):
    """Runs install.sh detached on the camera and follows its log until it reports
    'Installation complete' (it then reboots) or fails. Never longer than timeout."""
    log = f"/var/tmp/stereo4d_install_{datetime.now():%Y%m%d_%H%M%S}.log"
    rc, text, err = cam.script(INSTALL_START, root, log, timeout=30)
    if rc != 0 or "started" not in text:
        raise Failure("install", f"Could not start install.sh: {last_line(err or text)}")
    out.info(f"install.sh runs on the camera, log in {log} (apt upgrade can take several minutes)")
    offset, deadline, last_output, unreachable = 0, time.time() + timeout, time.time(), 0
    while time.time() < deadline:
        time.sleep(3)
        rc, chunk, _ = cam.run(f"sudo -n tail -c +{offset + 1} {log}", timeout=20)
        if rc != 0:
            unreachable += 1
            if unreachable * 3 > 90:
                raise Failure("install", "Lost the camera for 90 s while install.sh ran, before it finished",
                              f"When it is back, see the log: ssh {cam.user}@{cam.ip} cat {log}")
            continue
        unreachable = 0
        complete = chunk[:chunk.rfind("\n") + 1]  # a line still being written waits for the next poll
        if complete:
            offset += len(complete.encode())
            last_output = time.time()
            for line in complete.splitlines():
                if line.startswith("__ST4D_EXIT__"):
                    code = line.split()[-1]
                    raise Failure("install", f"install.sh failed (exit {code})",
                                  f"Full log: ssh {cam.user}@{cam.ip} cat {log}")
                out.log(line)
                if "Installation complete" in line:
                    return log
        elif time.time() - last_output > quiet_after:
            out.warn(f"install.sh has printed nothing for {quiet_after // 60} min; still waiting")
            last_output = time.time()
    raise Failure("install", f"install.sh did not finish within {timeout // 60} min",
                  f"It may still be running. Log: ssh {cam.user}@{cam.ip} cat {log}")


def wait_reboot(cam, out, old_boot, timeout):
    """Waits for a new boot id. Never longer than timeout."""
    out.info(f"Waiting for the camera to reboot (up to {timeout // 60} min)…")
    start, seen_down = time.time(), False
    while time.time() - start < timeout:
        now = cam.boot_id()
        if now is None:
            seen_down = True
        elif now != old_boot:
            out.check("Reboot", True, f"back after {time.time() - start:.0f} s")
            return
        time.sleep(4)
    if not seen_down:
        raise Failure("reboot", f"The camera did not reboot within {timeout // 60} min of install.sh")
    raise Failure("reboot", f"The camera did not come back within {timeout // 60} min",
                  "Check its power and cable; then run preflight. The firmware is pulled and\n"
                  "install.sh finished, so a plain power-cycle is safe.")


HEALTH = r"""
for i in $(seq 30); do systemctl is-active --quiet stereo_4d.service && break; sleep 2; done
r1=$(systemctl show -p NRestarts --value stereo_4d.service); r1=${r1:-0}
sleep 12
echo "active=$(systemctl is-active stereo_4d.service)"
r2=$(systemctl show -p NRestarts --value stereo_4d.service); echo "restarts=$(( ${r2:-0} - r1 ))"
echo "ports=$(sudo -n ss -ltn | grep -cE ':(5555|5556) ')"
echo "version=$(git -C "$1" describe --tags --always)"
[ -f "$1/$2" ] && echo "calibration=$(sha256sum < "$1/$2" | cut -c1-16)" || echo "calibration=none"
"""


def check_health(cam, out, root, backup):
    rc, text, err = cam.script(HEALTH, root, CALIBRATION, timeout=120)
    if rc is None:
        raise Failure("health", "Timed out checking the service")
    info = dict(l.split("=", 1) for l in text.splitlines() if "=" in l)
    problems = []
    if info.get("active") != "active":
        problems.append(f"{SERVICE} is {info.get('active') or 'missing'}")
    if info.get("restarts", "0") not in ("0", ""):
        problems.append(f"{SERVICE} restarted {info['restarts']} times in 12 s")
    if int(info.get("ports") or 0) < 2:
        problems.append(f"not listening on ports {ZMQ_PORTS[0]} and {ZMQ_PORTS[1]}")
    out.check("Service", not problems, "; ".join(problems) or f"active, stable, ports {ZMQ_PORTS} open")
    out.check("Firmware", True, info.get("version", "?"))
    if backup:
        rc2, saved, _ = cam.run(f"sha256sum < {shlex.quote(backup)}/stereo_calibration.yaml | cut -c1-16", 15)
        same = rc2 == 0 and saved.strip() == info.get("calibration")
        out.check("Calibration", same, "same bytes as before the update" if same else "DIFFERS from the backup",
                  f"Restore it: cp {backup}/stereo_calibration.yaml {root}/{CALIBRATION}")
        if not same:
            problems.append("calibration changed")
    if problems:
        _, journal, _ = cam.run(f"sudo -n journalctl -u {SERVICE} -n 25 --no-pager", timeout=20)
        for line in (journal or "").splitlines():
            out.log(line)
        raise Failure("health", "The camera is not healthy after the update: " + "; ".join(problems))
    try:
        import zmq  # optional: the SDK has it, a bare computer may not
    except ImportError:
        out.info("pyzmq not installed here: skipped listening for the camera's status messages")
        return
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVTIMEO, 8000)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{cam.ip}:{ZMQ_PORTS[0]}")
    try:
        for _ in range(40):
            if b'"msg_type": "status"' in sock.recv_multipart()[0]:
                out.check("Status messages", True, f"received on tcp://{cam.ip}:{ZMQ_PORTS[0]}")
                return
    except zmq.Again:
        pass
    finally:
        sock.close()
    raise Failure("health", f"No status message from tcp://{cam.ip}:{ZMQ_PORTS[0]} within 8 s")


# --------------------------------------------------------------------------- commands

def wait_unblocked(cam, out, limit):
    """Waits, without logging in (which would extend it), for sshd to lift the lockout.
    Probes every 20 s: a probe never logs in, so it costs 1 s of penalty, far less than
    the wait gives back."""
    out.info(f"Waiting up to {limit} s for the camera to accept this computer again, without logging in…")
    start = time.time()
    while time.time() - start < limit:
        time.sleep(20)
        try:
            with socket.create_connection((cam.ip, 22), timeout=5) as s:
                s.settimeout(8)
                if s.recv(64).decode(errors="replace").startswith("SSH-"):
                    out.check("SSH server", True, f"accepts this computer again after {time.time() - start:.0f} s")
                    return
        except OSError:
            pass
    raise Failure("blocked", f"The camera still refuses this computer after {limit} s (failed-login lockout)",
                  BLOCKED_HINT)


def connect(cam, out, args, secrets, interactive, retries):
    """Layers from the cable to a root shell on the camera, retried while it boots.
    Raises a Failure naming the first layer that does not work."""
    out.step(f"Camera {cam.user}@{cam.ip}")
    if not check_local_link(cam, out):
        raise Failure("link", "No direct route to the camera")
    for attempt in range(1, retries + 1):
        pinged = check_ping(cam, out)
        port = check_ssh_port(cam, out) if pinged else ""
        if port == "blocked":
            wait_unblocked(cam, out, args.blocked_wait)
            port = "ok"
        if port == "ok":
            break
        if attempt == retries:
            raise Failure("ssh_port" if pinged else "link",
                          f"The camera at {cam.ip} {'runs no usable SSH server' if pinged else 'does not answer'}"
                          f" ({retries} {'try' if retries == 1 else 'tries'})")
        out.info(f"Retrying in 10 s (try {attempt}/{retries}, Ctrl+C to stop)…")
        time.sleep(10)
    problem = check_login(cam, out)
    if problem == "blocked":
        wait_unblocked(cam, out, args.blocked_wait)
        problem = check_login(cam, out)
    if problem == "host_key":
        if args.forget_host_key or ask_yes(f"Forget the old host key of {cam.ip} and trust this camera?",
                                           False, interactive, False):
            run(["ssh-keygen", "-f", cam.known_hosts, "-R", cam.ip], timeout=10)
            problem = check_login(cam, out)
    if problem == "auth":
        install_key(cam, out, secrets, interactive)
        problem = check_login(cam, out)
    if problem:
        raise Failure(problem, {
                                "blocked": "The camera keeps refusing this computer (failed-login lockout)",
                                "host_key": "The camera's host key changed",
                                "auth": "The camera refused the SSH login",
                                "ssh_port": "SSH to the camera does not work"}[problem])
    if not check_sudo(cam, out, secrets):
        raise Failure("sudo", f"No root on the camera: {cam.user}'s sudo was not usable")


def cmd_connect(cam, out, args, secrets, interactive):
    connect(cam, out, args, secrets, interactive, retries=args.retries)
    out.step("Firmware")
    if find_firmware(cam, out, args.firmware_dir) is None:
        raise Failure("firmware", "No firmware folder found on the camera")
    return f"Logged into {cam.user}@{cam.ip} and found its firmware"


def cmd_preflight(cam, out, args, secrets, interactive):
    connect(cam, out, args, secrets, interactive, retries=args.retries)
    out.step("Firmware")
    info = find_firmware(cam, out, args.firmware_dir)
    out.step("Camera internet")
    online = check_internet(cam, out)
    if info is None:
        raise Failure("firmware", "No firmware folder found on the camera")
    if not online:
        raise Failure("internet", "The camera does not reach GitHub",
                      f"python3 {os.path.basename(__file__)} network, then wifi (update does both)")
    return "Preflight passed: the camera can be updated"


def cmd_network(cam, out, args, secrets, interactive):
    connect(cam, out, args, secrets, interactive, retries=args.retries)
    out.step("Camera routes")
    fix_network(cam, out)
    check_internet(cam, out)
    return "The camera's Ethernet link no longer carries its default route"


def ensure_internet(cam, out, args, secrets, interactive):
    out.step("Camera internet")
    routes = fix_network(cam, out)
    if check_internet(cam, out, routes):
        return
    out.step("Camera Wi-Fi")
    join_wifi(cam, out, args, secrets, interactive)


def cmd_wifi(cam, out, args, secrets, interactive):
    connect(cam, out, args, secrets, interactive, retries=args.retries)
    ensure_internet(cam, out, args, secrets, interactive)
    return "The camera reaches GitHub"


def cmd_update(cam, out, args, secrets, interactive):
    if not args.ref:
        raise Failure("usage", "update needs --ref: the firmware branch or tag to install (e.g. v2.1.0 or main)")
    token = secrets.get("token", "GitHub token that can read bmt-code/stereo_4d (not saved)")
    if token:
        problem = token_problem(token, args.ref, args.repo)
        if problem == "offline":
            out.info("This computer cannot reach GitHub: the token is checked on the camera instead")
        elif problem:
            raise Failure("git", problem)
    connect(cam, out, args, secrets, interactive, retries=args.retries)
    out.step("Firmware")
    info = find_firmware(cam, out, args.firmware_dir)
    if info is None:
        raise Failure("firmware", "No firmware folder found on the camera")
    root = info["root"]
    ensure_internet(cam, out, args, secrets, interactive)

    out.step(f"Pull {args.ref}")
    backup = pull_firmware(cam, out, root, args.ref, token, args.repo)
    if info.get("calibration") != "yes":
        backup = None

    if args.no_install:
        out.step("Restart the service")
        cam.must(f"sudo -n systemctl restart {SERVICE}", "restart the service", timeout=60)
    else:
        out.step("install.sh")
        boot = cam.boot_id()
        run_install(cam, out, root, args.install_timeout)
        wait_reboot(cam, out, boot, args.reboot_timeout)
        if cam.sudo_password:  # /dev/shm was emptied by the reboot
            cam.provide_askpass()
        # An older install.sh puts the gateway back on Ethernet at every install
        out.step("Camera routes after the reboot")
        fix_network(cam, out)

    out.step("Health")
    check_health(cam, out, root, backup)
    return f"Camera updated to {args.ref}"


COMMANDS = {"connect": cmd_connect, "preflight": cmd_preflight, "network": cmd_network, "wifi": cmd_wifi, "update": cmd_update}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="See docs/camera_update.md")
    p.add_argument("command", nargs="?", default="network", choices=COMMANDS,
                   help="what to do (default network)")
    p.add_argument("--ip", default=DEFAULT_IP, help=f"camera address (default {DEFAULT_IP})")
    p.add_argument("--user", default=DEFAULT_USER, help=f"camera user (default {DEFAULT_USER})")
    p.add_argument("--ref", help="update: firmware branch or tag to install")
    p.add_argument("--repo", default=FIRMWARE_URL, help="firmware repository (HTTPS)")
    p.add_argument("--identity", help="SSH private key to log in with")
    p.add_argument("--known-hosts", default="~/.ssh/known_hosts", help="known_hosts file for the camera")
    p.add_argument("--forget-host-key", action="store_true",
                   help="trust a camera whose host key changed (reflashed or swapped)")
    p.add_argument("--firmware-dir", help="firmware folder on the camera, when the service does not say")
    p.add_argument("--wifi-ssid", help="Wi-Fi network for the camera, when it has no internet")
    share = p.add_mutually_exclusive_group()
    share.add_argument("--share-wifi", action="store_true", default=None,
                       help="give the camera this computer's Wi-Fi without asking")
    share.add_argument("--no-share-wifi", dest="share_wifi", action="store_false")
    p.add_argument("--no-install", action="store_true", help="update: pull and restart, skip install.sh")
    p.add_argument("--retries", type=int, default=None, help="tries to reach the camera, 10 s apart (default: 6 for update, else 1)")
    p.add_argument("--blocked-wait", type=int, default=180,
                   help="seconds to wait out sshd's failed-login lockout (default 180)")
    p.add_argument("--install-timeout", type=int, default=2400, help="seconds for install.sh (default 2400)")
    p.add_argument("--reboot-timeout", type=int, default=360, help="seconds for the reboot (default 360)")
    p.add_argument("--secrets-stdin", action="store_true",
                   help='read {"token","wifi_psk","camera_password"} as one JSON line from stdin')
    p.add_argument("--json", action="store_true", help="one JSON event per line, for wrappers")
    p.add_argument("--yes", "-y", action="store_true", help="never prompt; fail where a prompt was needed")
    args = p.parse_args(argv)

    out = Out(args.json)
    interactive = not (args.yes or args.json) and sys.stdin.isatty() and not args.secrets_stdin
    try:
        if shutil.which("ssh") is None:
            raise Failure("usage", "No ssh client on this computer (apt install openssh-client)")
        secrets = Secrets(args.secrets_stdin, interactive)
        if args.retries is None:
            args.retries = 6 if args.command == "update" else 1
        cam = Camera(args)
        try:
            message = COMMANDS[args.command](cam, out, args, secrets, interactive)
        finally:
            cam.remove_askpass()
        out.result(True, "ok", message)
        return 0
    except Failure as f:
        out.result(False, f.kind, f.message, f.hint)
        return EXIT.get(f.kind, 1)
    except KeyboardInterrupt:
        out.result(False, "interrupted", "Interrupted")
        return EXIT["interrupted"]


if __name__ == "__main__":
    sys.exit(main())
