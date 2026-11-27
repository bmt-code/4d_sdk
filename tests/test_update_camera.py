"""tools/update_camera.py without a camera: a fake camera replays what a Pi answers, here
one whose Ethernet carries a default route via 172.31.1.1, and the checks must name that,
the fix must remove it, and the JSON a wrapper reads must hold.

    python3 tests/test_update_camera.py
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import update_camera as uc  # noqa: E402

BROKEN_ROUTES = """wired=eth0
wired_con=Wired connection 1
wifi=wlan0:connected:Test WiFi
v4=default via 172.31.1.1 dev eth0 proto static metric 100
v4=default via 192.168.1.1 dev wlan0 proto dhcp src 192.168.1.50 metric 600
v6=default via fe80::1 dev wlan0 proto ra metric 600 pref medium
dns=140.82.112.3
github4=140.82.112.3 via 172.31.1.1 dev eth0 src 172.31.1.77
"""
FIXED_ROUTES = BROKEN_ROUTES.replace("v4=default via 172.31.1.1 dev eth0 proto static metric 100\n", "")
# IPv4 dies in the cable; what the camera's curl says
PROBE_BROKEN = ("github.com Failed to connect to github.com port 443 after 3001 ms: Couldn't connect to server\n"
                "pypi.org Failed to connect to pypi.org port 443: Connection timed out\n"
                "archive.raspberrypi.com Connection timed out after 8001 milliseconds\n"
                "deb.debian.org Connection timed out after 8001 milliseconds\n")
PROBE_OK = "github.com 200\npypi.org 200\narchive.raspberrypi.com 200\ndeb.debian.org 200\n"


class FakeCamera(uc.Camera):
    """Answers like the broken Pi until NETWORK_FIX runs, then like a fixed one."""

    def __init__(self):
        self.ip, self.user, self.known_hosts, self.key = "172.31.1.77", "bmt", "/dev/null", None
        self.fixed, self.commands = False, []

    def run(self, command, timeout=30, input=None):
        self.commands.append(command)
        if "nmcli connection modify" in command and "never-default" in command:
            self.fixed = True
            return 0, "fixed Wired connection 1 on eth0\n", ""
        if "ip -4 route show default" in command:
            return 0, FIXED_ROUTES if self.fixed else BROKEN_ROUTES, ""
        if "curl -4" in command:
            return 0, PROBE_OK if self.fixed else PROBE_BROKEN, ""
        return 1, "", "unexpected command"


def events(fn, *args):
    out, buf = uc.Out(as_json=True), io.StringIO()
    with redirect_stdout(buf):
        try:
            value = fn(*(a if a != "OUT" else out for a in args))
        except uc.Failure as f:
            value = f
    return value, [json.loads(l) for l in buf.getvalue().splitlines()]


def test_broken_route_is_named():
    cam = FakeCamera()
    online, evs = events(uc.check_internet, cam, "OUT")
    assert online is False
    routes = next(e for e in evs if e["name"] == "Camera routes")
    assert not routes["ok"] and "172.31.1.1" in routes["detail"] and "github.com" in routes["hint"]
    github = next(e for e in evs if e["name"].endswith("github.com (IPv4)"))
    assert not github["ok"] and "Couldn't connect" in github["detail"]
    assert next(e for e in evs if e["name"] == "Camera Wi-Fi")["ok"]  # Wi-Fi is up: not the problem


def test_network_fix_removes_it_without_touching_the_link():
    cam = FakeCamera()
    _, evs = events(uc.fix_network, cam, "OUT")
    assert cam.fixed
    fix = next(c for c in cam.commands if "never-default" in c)
    assert "connection up" not in fix and "connection down" not in fix and "device disconnect" not in fix
    assert any(e.get("name") == "Ethernet is not a default route" and e["ok"] for e in evs)
    online, _ = events(uc.check_internet, cam, "OUT")
    assert online is True


def test_already_fixed_changes_nothing():
    cam = FakeCamera()
    cam.fixed = True
    events(uc.fix_network, cam, "OUT")
    assert not any("nmcli connection modify" in c for c in cam.commands)


def test_wired_defaults_ignores_wifi():
    routes = {"wired": "eth0", "v4": ["default via 1.2.3.4 dev wlan0 metric 600"],
              "v6": ["default via fe80::1 dev eth0 proto ra"]}
    assert uc.wired_defaults(routes) == ["default via fe80::1 dev eth0 proto ra"]


def test_result_event_carries_exit_code():
    out, buf = uc.Out(as_json=True), io.StringIO()
    with redirect_stdout(buf):
        out.result(False, "auth", "refused", "ssh-copy-id")
    event = json.loads(buf.getvalue())
    assert event == {"event": "result", "ok": False, "kind": "auth", "message": "refused",
                     "hint": "ssh-copy-id", "exit_code": 12}


def test_keyfile_escape():
    assert uc.keyfile_escape("a\\b\nc") == "a\\\\b\\nc"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
