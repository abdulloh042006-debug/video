"""Shared constants, easing curves and the beat map."""
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1080, 1920, 30
DUR = 90.0
N_FRAMES = int(DUR * FPS)

# ---- palette (BGR for OpenCV) -------------------------------------------------
def rgb(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


PINK = rgb("#FF4F9A")
MAGENTA = rgb("#D63CFF")
PURPLE = rgb("#7B4DFF")
CYAN = rgb("#22E0E8")
TEAL = rgb("#10B7A8")
ORANGE = rgb("#FF8A1F")
CARROT = rgb("#FF6A13")
YELLOW = rgb("#FFD23F")
GREEN = rgb("#5BE37A")
WHITE = (255, 255, 255)
CREAM = rgb("#FFF4E0")
NAVY = rgb("#161B3D")
PALETTE = [PINK, CYAN, ORANGE, PURPLE, YELLOW, GREEN, MAGENTA]


# ---- math / easing -----------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, u):
    return a + (b - a) * u


def prog(t, t0, t1):
    """0..1 progress of t inside [t0, t1]."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return clamp((t - t0) / (t1 - t0))


def lin(u):
    return u


def in_quad(u):
    return u * u


def out_quad(u):
    return 1 - (1 - u) * (1 - u)


def in_out_quad(u):
    return 2 * u * u if u < 0.5 else 1 - pow(-2 * u + 2, 2) / 2


def in_cubic(u):
    return u ** 3


def out_cubic(u):
    return 1 - (1 - u) ** 3


def in_out_cubic(u):
    return 4 * u ** 3 if u < 0.5 else 1 - pow(-2 * u + 2, 3) / 2


def out_expo(u):
    return 1.0 if u >= 1 else 1 - pow(2, -10 * u)


def in_expo(u):
    return 0.0 if u <= 0 else pow(2, 10 * u - 10)


def out_back(u, s=1.70158):
    u -= 1
    return 1 + (s + 1) * u ** 3 + s * u ** 2


def in_back(u, s=1.70158):
    return (s + 1) * u ** 3 - s * u ** 2


def out_elastic(u):
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    c4 = 2 * math.pi / 3
    return pow(2, -10 * u) * math.sin((u * 10 - 0.75) * c4) + 1


def out_bounce(u):
    n1, d1 = 7.5625, 2.75
    if u < 1 / d1:
        return n1 * u * u
    if u < 2 / d1:
        u -= 1.5 / d1
        return n1 * u * u + 0.75
    if u < 2.5 / d1:
        u -= 2.25 / d1
        return n1 * u * u + 0.9375
    u -= 2.625 / d1
    return n1 * u * u + 0.984375


def smooth(u):
    u = clamp(u)
    return u * u * (3 - 2 * u)


def pulse(dt, tau=0.12):
    """Decaying hit at dt >= 0 (0 before).  Peak 1 at dt=0."""
    return 0.0 if dt < 0 else math.exp(-dt / tau)


def kick(dt, rise=0.04, fall=0.22):
    """Soft attack + decay (nicer than a raw pulse for squash curves)."""
    if dt < 0:
        return 0.0
    if dt < rise:
        return dt / rise
    return math.exp(-(dt - rise) / fall * 2.2)


# ---- perspective of the real room ----------------------------------------------
Y_HORIZON = 830.0
CAM_H = 1.45


def ppm(y):
    """Pixels per metre at floor row y (calibrated against the chairs in the photo)."""
    return max(20.0, (y - Y_HORIZON) / CAM_H)


# ---- beat map ------------------------------------------------------------------
_beats = json.load(open(os.path.join(ROOT, "assets", "audio", "beats.json")))
BEATS = _beats["beats"]
ONSETS = _beats["onsets"]
FEAT = _beats["feat"]
# The tracker's beat times sit ~30 ms after the broadband transients (measured against a 3 ms flux
# curve), and a frame shows for 33 ms, so every visual hit is pulled 40 ms early (measured median error after a 25 ms lead was +17 ms).
VIS_LEAD = 0.040
BA, BB, BC = [[round(b - VIS_LEAD, 4) for b in BEATS[k]] for k in ("A", "B", "C")]
SIL1 = tuple(_beats["sil1"])
SIL2 = tuple(_beats["sil2"])
ALL_BEATS = np.array(BA + BB + BC)


def near_beat(t, window=0.2):
    """(distance since the last beat, index) -> used for groove pulses."""
    i = int(np.searchsorted(ALL_BEATS, t, side="right")) - 1
    if i < 0:
        return 9.0
    return t - ALL_BEATS[i]


def snap(t, tol=0.12):
    """Snap t to the nearest detected beat or strong onset (keeps hits on the music)."""
    best, bd = t, tol
    for b in ALL_BEATS:
        d = abs(b - t)
        if d < bd:
            best, bd = float(b), d
    return best


def feat(name, t):
    i = int(clamp(t * FPS, 0, len(FEAT[name]) - 1))
    return FEAT[name][i]
