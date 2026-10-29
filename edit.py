"""The short cut: hook + the funniest scenes of the long film, audio spliced on the beat.

The long film (director.py) is a pure function of time t, so the edit is just a list of (film time, frames) pieces.
Audio is cut from the same pieces of the 1:30 track, so picture and sound stay locked.

    python edit.py preview <out_dir> <out_frame> [...]     # dump single frames of the edit
"""
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from engine.common import W, H, FPS, clamp, out_cubic  # noqa: E402
import director as D  # noqa: E402
from engine import mograph as mg  # noqa: E402
from engine.common import PINK, CYAN, YELLOW, PURPLE, ORANGE, GREEN, MAGENTA, WHITE  # noqa: E402

PALETTE = [PINK, CYAN, YELLOW, PURPLE, ORANGE, GREEN, MAGENTA]
SR = 44100
HOOK_N = 43                                    # 1.4333 s: exactly one beat before the story starts


def frames(t0, t1):
    return int(round((t1 - t0) * FPS))


# --- the pieces ------------------------------------------------------------------------------------------------
#  kind "hook": three flash-forward shots then a VHS rewind;  kind "film": a straight piece of the long film
def build_timeline():
    A, B, C = D.A, D.B, D.C
    pieces = [
        dict(name="hook", kind="hook", n=HOOK_N, a0=0.0),
        dict(name="entrances", kind="film", t_in=A(6), n=frames(A(6), A(27)), cut="flash"),
        dict(name="hit+freeze", kind="film", t_in=A(38), n=frames(A(38), B(2)), cut="whip"),
        dict(name="carrot", kind="film", t_in=B(15), n=frames(B(15), B(31)), cut="wipe"),
        dict(name="sloth", kind="film", t_in=B(40), n=frames(B(40), B(49)), cut="zoom"),
        dict(name="race", kind="film", t_in=B(80), n=frames(B(80), B(95)), cut="wipe"),
        dict(name="roar", kind="film", t_in=B(98), n=frames(B(98), B(110)), cut="whip"),
        dict(name="inhale+boom", kind="film", t_in=B(124), n=frames(B(124), C(4)), cut="flash"),
        dict(name="finale", kind="film", t_in=C(10), n=frames(C(10), 90.0), cut="wipe"),
    ]
    pos = 0
    for p in pieces:
        p["pos"] = pos
        pos += p["n"]
    return pieces, pos


# --- picture effects ------------------------------------------------------------------------------------------------
def punch(img, s):
    if abs(s - 1) < 1e-3:
        return img
    M = np.array([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def flash_over(img, a, col=255):
    if a <= 0.01:
        return img
    return cv2.addWeighted(img, 1 - a, np.full_like(img, col), a, 0)


def hblur(img, k):
    k = int(k) | 1
    return cv2.blur(img, (k, 1)) if k > 1 else img


def zoom_blur(img, amt, n=5):
    if amt < 0.005:
        return img
    acc = img.astype(np.float32)
    for i in range(1, n):
        acc += punch(img, 1 + amt * i / (n - 1)).astype(np.float32)
    return (acc / n).astype(np.uint8)


def rewind_fx(img, s, seed):
    """VHS rewind: torn horizontal bands, colour split, desaturation, tracking noise and a double-arrow icon."""
    rng = np.random.RandomState(seed)
    out = img.copy()
    for _ in range(int(5 + 9 * s)):
        y0 = rng.randint(0, H - 40)
        hh = rng.randint(14, 90)
        dx = int(rng.uniform(-1, 1) * (40 + 150 * s))
        out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], dx, axis=1)
    sh = int(4 + 9 * s)
    b, g, r = cv2.split(out)
    out = cv2.merge([np.roll(b, -sh, 1), g, np.roll(r, sh, 1)])
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out = cv2.addWeighted(out, 1 - 0.55 * s, cv2.merge([gray, gray, gray]), 0.55 * s, 0)
    out[::4] = (out[::4] * 0.78).astype(np.uint8)
    # tracking noise band
    y0 = int(H * (0.78 + 0.1 * rng.rand()))
    band = rng.randint(70, 230, (70, W, 1)).astype(np.uint8)
    out[y0:y0 + 70] = cv2.addWeighted(out[y0:y0 + 70], 0.25, np.repeat(band, 3, 2), 0.75, 0)
    # ◀◀ icon
    for k in range(2):
        x = 130 + k * 78
        mg.poly(out, [(x + 64, 150), (x, 205), (x + 64, 260)], (255, 255, 255), 0.95)
    return out


def hook_frame(world, k):
    """Out-frame k (0..HOOK_N-1) of the hook."""
    A, B = D.A, D.B
    TR = B(100)
    t_hit = B(28)
    BOOM = D.BOOM
    n1, n2 = 13, 26
    if k < n1:
        j, t, name = k, TR + 0.10 + k / FPS, "roar"
    elif k < n2:
        j, t, name = k - n1, t_hit - 0.14 + (k - n1) / FPS, "hit"
    else:
        j = k - n2
        if j < 8:
            t = BOOM + 0.10 + j / FPS
        else:                                           # rewind: backwards at 2.5x speed
            t = BOOM + 0.10 + 8 / FPS - (j - 7) * 2.5 / FPS
        name = "boom"
    img = world.render(t)
    # every shot lands with a punch-out and a white pop on its first frames
    sc = 1.0 + 0.22 * (1 - out_cubic(clamp(j / 9.0))) ** 1.5 if (name != "boom" or j < 8) else 1.0
    img = punch(img, sc)
    if j < 2:
        img = flash_over(img, 0.55 - 0.25 * j)
    if name == "boom" and j >= 8:
        s = clamp((j - 7) / 5.0)
        img = rewind_fx(img, 0.4 + 0.6 * s, 100 + k)
    return img


def apply_cut(img, style, d, n_before=5, n_after=7):
    """d = out-frame distance from the first frame of the new piece (negative = still the old piece)."""
    if style == "flash":
        if 0 <= d < 5:
            img = flash_over(img, 0.8 * (1 - d / 5.0) ** 1.4)
            img = punch(img, 1.0 + 0.06 * (1 - d / 5.0))
    elif style == "whip":
        if -3 <= d < 0:
            img = hblur(img, 40 + 25 * (-d))
            img = punch(img, 1.0 + 0.03 * (-d))
        elif 0 <= d < 4:
            img = hblur(img, 70 - 17 * d)
            img = flash_over(img, 0.25 * (1 - d / 4.0))
    elif style == "zoom":
        if -4 <= d < 0:
            img = zoom_blur(img, 0.05 * (-d))
            img = flash_over(img, 0.10 * (-d) / 4)
        elif 0 <= d < 5:
            img = zoom_blur(img, 0.16 * (1 - d / 5.0))
            img = flash_over(img, 0.5 * (1 - d / 5.0))
    elif style == "wipe":
        a0 = -n_before
        if a0 <= d < n_after + 1:
            p = (d - a0) / float(n_before + n_after)
            mg.wipe(img, clamp(p), PALETTE, angle=-24)
    return img


# --- frame / audio --------------------------------------------------------------------------------------------------
class Edit:
    def __init__(self):
        self.pieces, self.total = build_timeline()
        self.world = None

    def locate(self, f):
        for i, p in enumerate(self.pieces):
            if p["pos"] <= f < p["pos"] + p["n"]:
                return i, f - p["pos"]
        raise IndexError(f)

    def frame(self, f):
        if self.world is None:
            self.world = D.build()
        i, k = self.locate(f)
        p = self.pieces[i]
        if p["kind"] == "hook":
            img = hook_frame(self.world, k)
        else:
            img = self.world.render(p["t_in"] + k / FPS)
        # transitions: the piece's own entry cut, and the next piece's cut that starts a few frames before it
        if p.get("cut") and i > 0:
            img = apply_cut(img, p["cut"], k)
        if i + 1 < len(self.pieces):
            nxt = self.pieces[i + 1]
            d = f - nxt["pos"]
            if d >= -5 and nxt.get("cut") in ("wipe", "whip", "zoom"):
                img = apply_cut(img, nxt["cut"], d)
        return img

    # audio: slices of the prepared 1:30 track; every non-contiguous join is a 20 ms equal-power crossfade
    def audio(self):
        import soundfile as sf
        trk, sr = sf.read(os.path.join(ROOT, "assets", "audio", "track_90s.wav"), always_2d=True)
        assert sr == SR
        spf = SR // FPS                                      # samples per frame (1470)
        out = np.zeros((self.total * spf, 2), np.float32)
        h = int(0.010 * SR)
        prev_end = None
        for p in self.pieces:
            t0 = 0.0 if p["kind"] == "hook" else p["t_in"]
            s0 = int(round(t0 * SR))
            n = p["n"] * spf
            seg = trk[s0:s0 + n]
            out[p["pos"] * spf:p["pos"] * spf + len(seg)] = seg
            if prev_end is not None and abs(prev_end - s0) > 4:        # a real splice
                pos = p["pos"] * spf
                old = trk[prev_end - h:prev_end + h]
                new = trk[s0 - h:s0 + h]
                ramp = np.linspace(0, math.pi / 2, 2 * h)[:, None]
                out[pos - h:pos + h] = old * np.cos(ramp) + new * np.sin(ramp)
            prev_end = s0 + n
        return out


def main():
    mode = sys.argv[1]
    if mode == "preview":
        out = sys.argv[2]
        os.makedirs(out, exist_ok=True)
        ed = Edit()
        print("pieces:", [(p["name"], p["pos"], p["n"]) for p in ed.pieces], "total", ed.total, "=", ed.total / FPS, "s")
        tiles = []
        for f in [int(x) for x in sys.argv[3:]]:
            img = ed.frame(f)
            cv2.imwrite(os.path.join(out, f"e_{f:05d}.png"), img)
            small = cv2.resize(img, (270, 480), interpolation=cv2.INTER_AREA)
            cv2.putText(small, f"{f} {f / FPS:.2f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            tiles.append(small)
        cols = min(len(tiles), 8)
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.zeros((rows * 480, cols * 270, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * 480:(r + 1) * 480, c * 270:(c + 1) * 270] = t
        cv2.imwrite(os.path.join(out, "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
