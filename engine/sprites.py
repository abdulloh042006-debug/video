"""Cut-out sprites: loading, sticker outline, mip chain, affine blit and contact shadows."""
import math
import os

import cv2
import numpy as np

from .common import ROOT

SPR_DIR = os.path.join(ROOT, "assets", "sprites")
OUTLINE_PX = 2.4                     # sticker outline width in source pixels
OUTLINE_COL = (238, 246, 255)        # BGR, warm white


# sprites whose feet were cut off by the edge of the source picture get a soft fade-out there
FADE_BOTTOM = {"beaver": 0.07, "bellwether": 0.07, "gazelle": 0.04, "nick_edge": 0.0}


class Sprite:
    """Premultiplied BGRA sticker (outline baked in) with a half-size mip chain."""

    def __init__(self, name, outline=True):
        im = cv2.imread(os.path.join(SPR_DIR, name + ".png"), cv2.IMREAD_UNCHANGED)
        if im.shape[2] == 3:
            im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
        a = im[..., 3].astype(np.float32) / 255.0
        # tighten the matte slightly so no background fringe survives
        a = cv2.erode(a, np.ones((2, 2), np.uint8))
        a = cv2.GaussianBlur(a, (0, 0), 0.6)
        fb = FADE_BOTTOM.get(name, 0.0)
        if fb > 0:
            n = max(2, int(a.shape[0] * fb))
            a[-n:] *= np.linspace(1, 0, n, dtype=np.float32)[:, None] ** 1.3
        rgb = im[..., :3].astype(np.float32)
        pad = int(math.ceil(OUTLINE_PX * 2)) + 2
        a = np.pad(a, pad)
        rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        if outline:
            k = int(round(OUTLINE_PX * 2)) | 1
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            o = cv2.dilate((a > 0.35).astype(np.uint8), ker).astype(np.float32)
            o = cv2.GaussianBlur(o, (0, 0), 0.8)
            col = np.array(OUTLINE_COL, np.float32)
            # sprite over outline
            out_a = a + o * (1 - a)
            out_rgb = rgb * a[..., None] + col * (o * (1 - a))[..., None]
        else:
            out_a = a
            out_rgb = rgb * a[..., None]
        self.alpha = out_a
        self.pm = np.dstack([out_rgb, out_a * 255.0]).astype(np.float32)   # premultiplied
        self.sil = a                                                       # silhouette without outline
        self.h, self.w = out_a.shape
        self.pad = pad
        self.body_h = self.h - 2 * pad           # sprite height without outline padding
        self.mips = [self.pm]
        m = self.pm
        for _ in range(3):
            m = cv2.resize(m, (max(1, m.shape[1] // 2), max(1, m.shape[0] // 2)), interpolation=cv2.INTER_AREA)
            self.mips.append(m)

    def pick(self, scale):
        """Return (mip image, factor) for a target on-screen scale (<1 shrinks)."""
        lvl = 0
        s = scale
        while s < 0.55 and lvl < len(self.mips) - 1:
            s *= 2
            lvl += 1
        return self.mips[lvl], 2 ** lvl


_cache = {}


def get(name, outline=True):
    key = (name, outline)
    if key not in _cache:
        _cache[key] = Sprite(name, outline)
    return _cache[key]


def blit(canvas, spr, x, y, height, sx=1.0, sy=1.0, rot=0.0, flip=False, alpha=1.0,
         flash=0.0, tint=None, anchor=(0.5, 1.0), clip=None, bright=1.0):
    """Draw `spr` so its body (sans outline pad) is `height` px tall at scale 1.

    (x, y) is where `anchor` (fractions of the body box) lands.  sx/sy squash and stretch about
    the anchor, rot is degrees clockwise.  `clip` is an optional full-frame uint8 mask.
    """
    if alpha <= 0.002 or height < 1:
        return
    s = height / spr.body_h
    img, fac = spr.pick(s * min(abs(sx), abs(sy)) if min(abs(sx), abs(sy)) > 0.05 else s)
    k = s * fac                                  # remaining scale from mip image to screen
    ax = spr.pad + anchor[0] * (spr.w - 2 * spr.pad)
    ay = spr.pad + anchor[1] * (spr.body_h)
    ax /= fac
    ay /= fac
    fx = -1.0 if flip else 1.0
    th = math.radians(rot)
    c, sn = math.cos(th), math.sin(th)
    # M = T(x,y) R S T(-a)
    a11, a12 = c * k * sx * fx, -sn * k * sy
    a21, a22 = sn * k * sx * fx, c * k * sy
    M = np.array([[a11, a12, x - (a11 * ax + a12 * ay)],
                  [a21, a22, y - (a21 * ax + a22 * ay)]], np.float32)
    h, w = img.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float32) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int) - 1
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 1
    H_, W_ = canvas.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W_), min(y1, H_)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    M2 = M.copy()
    M2[0, 2] -= cx0
    M2[1, 2] -= cy0
    warped = cv2.warpAffine(img, M2, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4] / 255.0
    rgb = warped[..., :3]
    if bright != 1.0:
        rgb = rgb * bright
    if flash > 0:
        rgb = rgb * (1 - flash) + 255.0 * a * flash
    if tint is not None:
        t_ = np.array(tint[:3], np.float32)
        amt = tint[3]
        rgb = rgb * (1 - amt) + t_ * a * amt
    if clip is not None:
        cm = clip[cy0:cy1, cx0:cx1, None].astype(np.float32) / 255.0
        a = a * cm
        rgb = rgb * cm
    if alpha < 1.0:
        a = a * alpha
        rgb = rgb * alpha
    roi = canvas[cy0:cy1, cx0:cx1]
    out = rgb + roi.astype(np.float32) * (1 - a)
    roi[:] = np.clip(out, 0, 255).astype(np.uint8)


# ---- contact shadow ---------------------------------------------------------------
_blob = None


def _make_blob():
    n = 96
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    d = np.sqrt(((xx - n / 2) / (n / 2)) ** 2 + ((yy - n / 2) / (n / 2)) ** 2)
    return np.clip(1 - d, 0, 1) ** 1.4


def shadow(canvas, x, y, width, strength=0.42, squash=0.22, lift=0.0):
    """Soft elliptical contact shadow centred at (x, y). `lift` (0..1) shrinks/lightens it."""
    global _blob
    if _blob is None:
        _blob = _make_blob()
    wid = max(8, int(width * (1.0 - 0.45 * lift)))
    hei = max(4, int(wid * squash))
    st = strength * (1 - 0.6 * lift)
    x0, y0 = int(x - wid / 2), int(y - hei / 2)
    x1, y1 = x0 + wid, y0 + hei
    H_, W_ = canvas.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W_), min(y1, H_)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    b = cv2.resize(_blob, (wid, hei), interpolation=cv2.INTER_LINEAR)[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    roi = canvas[cy0:cy1, cx0:cx1]
    roi[:] = (roi.astype(np.float32) * (1 - b[..., None] * st)).astype(np.uint8)
