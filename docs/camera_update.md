# Updating a Stereo 4D camera

`tools/update_camera.py` updates the camera's firmware from the computer the camera is
cabled to. It is one file that needs only Python 3.8+ and `ssh`. It does not need the
SDK installed.

## What you need

- The computer the camera is plugged into by Ethernet. Its network card gets an address
  from the camera (172.31.1.2–76), so leave the card on DHCP.
- Python 3.8 or newer and `ssh`. Ubuntu has both.
- The camera's password (user `bmt`). It is asked for once per run: to install this
  computer's SSH key if it has none, and for `sudo` if the camera asks for one.
- A GitHub token that can read `bmt-code/stereo_4d`, because the firmware repository is
  private. Use a fine-grained token with **Contents: read** on that repository only, and a
  short expiry. It is asked for at the prompt and never saved.
- Internet for the camera. If the camera has none, the script puts it on a Wi-Fi network:
  this computer's own network, or one you pick. Have the Wi-Fi password ready.

## Running it

```bash
curl -fsSLO https://raw.githubusercontent.com/bmt-code/4d_sdk/main/tools/update_camera.py

python3 update_camera.py                        # same as network: fix the camera's routes
python3 update_camera.py preflight              # what works, what does not, and why
python3 update_camera.py update --ref TAG       # a version tag, or a branch such as main
```

From an SDK checkout, run `python3 tools/update_camera.py ...` instead.

`update` does every step and asks only for what it needs:

1. **Reach the camera.** It checks each layer in turn: route, ping, SSH server, login,
   sudo, and the firmware folder. It retries for one minute while the camera boots. If
   the camera refuses this computer's key, it offers to install one using the camera's
   password.
2. **Fix the camera's routes.** The Ethernet link must not be the camera's default route.
   See [Troubleshooting](#troubleshooting).
3. **Wi-Fi.** If the camera cannot reach GitHub, the script offers this computer's Wi-Fi,
   or lists the networks the camera sees and asks which one to join.
4. **Back up the calibration** to `~/.st4d_backup/<date>/` on the camera. Then it discards
   local changes, pulls `--ref`, and checks the calibration is byte for byte the same.
5. **Run `install.sh`.** This takes several minutes for apt and pip, then the camera
   reboots. The script follows the log live. The log stays on the camera in
   `/var/tmp/stereo4d_install_<date>.log`.
6. **Wait for the reboot**, up to 6 minutes. Then it checks the routes again, and fixes them
   if needed.
7. **Check the service.** It must be running, not restarting, with ports 5555 and 5556
   open, the calibration unchanged, and status messages arriving if `pyzmq` is installed.

Every step has a time limit. If a step fails, the script stops and prints what failed and
what to do about it.

## Other commands

| Command | Does |
|---|---|
| `preflight` | Checks everything and changes nothing. If SSH login fails, it offers to install a key or forget a changed host key. |
| `connect` | The preflight checks up to the firmware folder, with no internet checks. |
| `network` | The default when no command is given. Removes the bad default route from the camera's Ethernet link, live and saved for every boot. |
| `wifi` | `network`, then puts the camera on Wi-Fi if it still has no internet. |

Useful options:

- `--wifi-ssid NAME` picks the camera's network. `--share-wifi` uses this computer's
  network without asking.
- `--forget-host-key` trusts a camera that was reflashed or swapped. Every camera has the
  same address, so a new one looks like a changed host key.
- `--no-install` pulls the firmware and restarts the service, and skips `install.sh`.
- `--install-timeout` and `--reboot-timeout` set the time limits.
- `--ip` and `--user` set the camera's address and user. The defaults are 172.31.1.77 and
  bmt.
- `--json --secrets-stdin` are for wrapper programs: they print one JSON event per line
  and read `{"token", "wifi_psk", "camera_password"}` from stdin.

Exit codes name the failure: 10 no link, 11 no SSH server, 12 login refused, 13 host key
changed, 14 no usable sudo, 15 no firmware, 16 camera refusing logins, 20 no internet, 21 Wi-Fi, 30 git, 31
calibration, 40 install.sh, 41 reboot, 50 service unhealthy.

## Troubleshooting

**"Pings but cannot connect".** Ping working only shows that the cable and the address
are fine. `preflight` checks the next layers one at a time and reports the first one that
fails:

| Check that fails | Meaning | Fix |
|---|---|---|
| SSH server | Port 22 is closed, or answers without SSH | The camera is still booting, or another device has the address |
| SSH login: refused this computer's key | This computer has no key on the camera | Rerun interactively and type the camera's password, or run `ssh-copy-id bmt@172.31.1.77` |
| SSH server: refuses this computer for now (`Not allowed at this time`), or login: connection closed | The camera is refusing logins from this computer for now | Stop retrying. The script waits (`--blocked-wait`, default 180 s); otherwise wait a few minutes or power-cycle the camera. Then install a key |
| SSH login: host key | The camera was reflashed or swapped | `--forget-host-key` |
| sudo | `install.sh` and `nmcli` need root. Passwordless sudo is not required: when the camera's sudo asks for a password, the script asks for the camera's password once and keeps it in memory. The camera holds it only in `/dev/shm`, which is memory and is cleared at reboot, and only until the run ends | Type the camera password at the prompt, or pass `camera_password` with `--secrets-stdin` |
| Route to the camera: through … | A VPN or docker network overlaps 172.31.1.0/24, or the cable's card has no address | Leave the card on DHCP and remove the overlapping network |

**The camera is on Wi-Fi but cannot reach GitHub.** If the Ethernet link is the camera's
default route, internet traffic goes into the cable instead of over Wi-Fi, and git, apt and
pip fail. `update_camera.py network` fixes it without dropping the link to this computer:

```bash
sudo nmcli connection modify "Wired connection 1" ipv4.gateway "" ipv4.never-default yes ipv6.never-default yes
sudo ip -4 route del default dev eth0      # the live route, until the next boot
```

To diagnose by hand, run these on the camera:

```bash
ip route; ip -6 route                    # a "default ... dev eth0" line is the problem
ping -4 -c 2 github.com
getent ahostsv4 github.com               # DNS
curl -4 -sSI https://github.com | head -n 1
```

## Security notes

- Secrets are never written to disk on this computer and never appear on a command line.
  The GitHub token reaches the camera over SSH stdin. The Wi-Fi password goes straight
  into a root-only NetworkManager profile on the camera.
- New host keys are accepted on first contact (`StrictHostKeyChecking=accept-new`). A
  *changed* key is refused unless you pass `--forget-host-key`.
- Revoke the GitHub token once the update is done.
