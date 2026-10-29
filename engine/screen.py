"""What the projector shows: Ken-Burns crops of the supplied Zootopia stills with beat-synced
punch-ins, chromatic splits and glitch slices."""
import math
import os

import cv2
import numpy as np

from .common import ROOT, clamp, pulse, PALETTE
from .room import SCREEN_W, SCREEN_H

SRC = os.path.join(ROOT, "assets", "src")


def _load(name, rot=None):
    im = cv2.imread(os.path.join(SRC, name))
    if rot == "cw":
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    elif rot == "ccw":
        im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return im


class Screen:
    def __init__(self):
        poster = _load("zoo_poster_rot.jpg", "ccw")      # 1200x675
        lineup = _load("zoo_lineup_rot.jpg", "cw")       # 1308x736
        neon = _load("zoo_neon_street.jpg")
        sunset = _load("zoo_sunset.jpg")
        city = _load("zoo_city.jpg")
        # (image, crop x0, y0, x1, y1) -- 4:3 windows that avoid the logo / watermarks
        self.src = {
            "poster_l": (poster, (0, 150, 700, 675)),
            "poster_r": (poster, (470, 150, 1170, 675)),
            "poster_c": (poster, (300, 120, 900, 570)),
            "neon": (neon, (0, 60, 736, 612)),
            "neon_low": (neon, (0, 552, 736, 1104)),
            "sunset": (sunset, (0, 160, 736, 712)),
            "sunset_low": (sunset, (0, 700, 736, 1252)),
            "city_top": (city, (0, 60, 736, 612)),
            "city_low": (city, (0, 740, 736, 1292)),
            "lineup": (None, None),
        }
        self.lineup = self._build_lineup()
        self.src["lineup"] = (self.lineup, (0, 0, 960, 720))
        self._bars = None

    def _build_lineup(self):
        """Cast lineup on a candy-sky gradient (made from the cut-outs, so no white fog)."""
        from . import sprites as sp
        Wd, Ht = 960, 720
        yy = np.linspace(0, 1, Ht, dtype=np.float32)[:, None]
        top = np.array([255, 120, 190], np.float32)      # BGR-ish pink/purple
        bot = np.array([90, 60, 255], np.float32)
        top = np.array([200, 110, 130], np.float32)
        bot = np.array([130, 70, 255], np.float32)
        g = (top[None, None, :] * (1 - yy[..., None]) + bot[None, None, :] * yy[..., None]) * np.ones((1, Wd, 1), np.float32)
        img = np.clip(g, 0, 255).astype(np.uint8)
        # halftone dots
        for j in range(0, Ht, 28):
            for i in range(0, Wd, 28):
                r = max(0, 5 - 5 * (j / Ht))
                if r > 0.6:
                    cv2.circle(img, (i + (14 if (j // 28) % 2 else 0), j), int(r), (255, 200, 235), -1, cv2.LINE_AA)
        names = ["beaver", "mrbig", "flash", "clawhauser", "judy_badge", "gazelle", "nick_edge"]
        hs = [0.62, 0.50, 0.66, 0.66, 0.62, 0.78, 0.74]
        x = 25
        step = (Wd - 50) / len(names)
        for n, hh in zip(names, hs):
            spr = sp.get(n)
            H_ = int(Ht * hh)
            sc = H_ / spr.body_h
            pm = cv2.resize(spr.pm, (int(spr.w * sc), int(spr.h * sc)), interpolation=cv2.INTER_AREA)
            cx = int(x + step / 2 - pm.shape[1] / 2)
            y0 = Ht + 18 - pm.shape[0]
            x0, x1 = max(cx, 0), min(cx + pm.shape[1], Wd)
            sub = pm[:, x0 - cx:x1 - cx]
            y1 = min(Ht, y0 + sub.shape[0])
            sub = sub[: y1 - y0]
            roi = img[y0:y1, x0:x1].astype(np.float32)
            a = sub[..., 3:4] / 255.0
            img[y0:y1, x0:x1] = np.clip(sub[..., :3] + roi * (1 - a), 0, 255).astype(np.uint8)
            x += step
        return img

    # ---------------------------------------------------------------------------
    def crop(self, name, zoom=1.0, cx=0.5, cy=0.5):
        img, (x0, y0, x1, y1) = self.src[name]
        cw, ch = (x1 - x0), (y1 - y0)
        w = cw / zoom
        h = ch / zoom
        mx = x0 + cx * cw
        my = y0 + cy * ch
        mx = clamp(mx, x0 + w / 2, x1 - w / 2)
        my = clamp(my, y0 + h / 2, y1 - h / 2)
        s = SCREEN_W / w
        M = np.array([[s, 0, -(mx - w / 2) * s], [0, s, -(my - h / 2) * s]], np.float32)
        return cv2.warpAffine(img, M, (SCREEN_W, SCREEN_H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def bars(self, t):
        """Test-card colour bars with a rolling highlight, used when the projector warms up."""
        cols = [(240, 240, 240), (0, 230, 255), (230, 230, 0), (60, 220, 60), (230, 60, 230), (60, 60, 240),
                (230, 70, 40)]
        img = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
        bw = SCREEN_W / len(cols)
        for i, c in enumerate(cols):
            img[:, int(i * bw):int((i + 1) * bw + 1)] = c
        roll = int((t * 900) % SCREEN_H)
        img = np.roll(img, roll // 6, axis=0)
        img[(np.arange(SCREEN_H) % 4) < 1] = (img[(np.arange(SCREEN_H) % 4) < 1] * 0.75).astype(np.uint8)
        return img

    def render(self, t, name, zoom=1.0, cx=0.5, cy=0.5, beats=(), punch=0.08, split=0.0, glitch=0.0,
               tint=None, flash=0.0, scan=True, seed=0, saturate=1.0, invert=0.0):
        z = zoom
        for b in beats:
            if 0 <= t - b < 0.4:
                z *= 1 + punch * pulse(t - b, 0.11)
                if split <= 0:
                    split = 0.0
                break
        img = self.crop(name, z, cx, cy)
        last = None
        for b in beats:
            if b <= t:
                last = b
        sp = split
        if last is not None and t - last < 0.3 and punch > 0:
            sp = max(sp, 7.0 * pulse(t - last, 0.09))
        if sp > 0.3:
            s = int(round(sp))
            b, g, r = cv2.split(img)
            b = np.roll(b, -s, axis=1)
            r = np.roll(r, s, axis=1)
            img = cv2.merge([b, g, r])
        if glitch > 0:
            rng = np.random.RandomState(seed)
            for _ in range(int(2 + glitch * 7)):
                y0 = rng.randint(0, SCREEN_H - 30)
                hh = rng.randint(6, 40)
                dx = int(rng.uniform(-1, 1) * glitch * 90)
                img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], dx, axis=1)
        if saturate != 1.0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.merge([gray, gray, gray])
            img = cv2.addWeighted(img, saturate, gray3, 1 - saturate, 0)
        if tint is not None:
            img = cv2.addWeighted(img, 1 - tint[3], np.full_like(img, tint[:3]), tint[3], 0)
        if scan:
            img[::3] = (img[::3] * 0.9).astype(np.uint8)
        if flash > 0:
            img = cv2.addWeighted(img, 1 - flash, np.full_like(img, 255), flash, 0)
        return img
