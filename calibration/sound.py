"""Two short tones for the capture window: nearly there, and saved.

The operator is holding a board at arm's length, often looking at it rather than at the
screen. A sound carries where a colour change does not.

Everything here fails quietly. A machine with no audio, no player, or no writable temp
directory still captures -- it just does it silently, which is exactly how this behaved
before the sounds existed.
"""

import atexit
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import wave

SAMPLE_RATE = 44100

# Players in order of preference. The first one present wins; each takes a file path as
# its only argument and is spawned detached, so none of them can stall the capture loop.
PLAYERS = ("paplay", "pw-play", "aplay", "ffplay")
PLAYER_FLAGS = {"ffplay": ("-nodisp", "-autoexit", "-loglevel", "quiet")}

# A rising pair for "that will do", a brighter double click for "saved". Kept short: they
# fire while the operator is holding still, and anything with a tail invites them to move
# before it finishes.
TONES = {
    "near": ((660, 0.055), (880, 0.055)),
    "shot": ((1320, 0.040), (0, 0.030), (1760, 0.070)),
}

# Never retrigger the same sound faster than this. "near" in particular is driven by a
# score that dithers around the threshold, and without a floor it machine-guns.
REPEAT_GUARD = {"near": 1.2, "shot": 0.25}

_files = {}
_player = None
_last = {}
_enabled = True


def _render(path, parts, volume=0.22):
    """Write the tone sequence to ``path`` as 16-bit mono PCM."""
    frames = bytearray()
    for freq, seconds in parts:
        count = int(SAMPLE_RATE * seconds)
        for i in range(count):
            if freq <= 0:
                frames += struct.pack("<h", 0)
                continue
            # Raised-cosine envelope: a square-edged tone clicks, and the click is louder
            # than the note.
            envelope = 0.5 - 0.5 * math.cos(2 * math.pi * min(i, count - i) / count)
            sample = volume * envelope * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)
            frames += struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767))

    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(bytes(frames))


def available():
    """Set the tones up once. False when this machine cannot play them."""
    global _player, _enabled
    if _files:
        return _enabled
    if not _enabled:
        return False

    _player = next((p for p in PLAYERS if shutil.which(p)), None)
    if _player is None:
        _enabled = False
        return False

    try:
        folder = tempfile.mkdtemp(prefix="4d-calib-sound-")
        atexit.register(shutil.rmtree, folder, True)
        for name, parts in TONES.items():
            path = os.path.join(folder, f"{name}.wav")
            _render(path, parts)
            _files[name] = path
    except (OSError, ValueError):
        _enabled = False
        return False
    return True


def play(name):
    """Start ``name`` and return immediately. Never raises, never blocks."""
    if not available() or name not in _files:
        return False

    now = time.monotonic()
    if now - _last.get(name, 0.0) < REPEAT_GUARD.get(name, 0.3):
        return False
    _last[name] = now

    command = [_player, *PLAYER_FLAGS.get(_player, ()), _files[name]]
    try:
        subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return False
    return True


def disable():
    """Turn the sounds off for the rest of the run."""
    global _enabled
    _enabled = False
