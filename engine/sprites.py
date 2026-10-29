"""Cut-out characters that behave like volumes instead of stickers.

Each sprite is a premultiplied BGRA image with a mip chain, plus per-level static maps (ambient
occlusion gradient, inner edge band, outward normals).  At draw time a sprite is

  1. deformed by a tiny "rig" (breathing, body bend, head nod, ear / tail / skirt / mane secondary motion),
  2. lit (room exposure, vertical shading, haze, coloured rim lights that follow the scene),
  3. warped with a real perspective (yaw / pitch) + affine, and
  4. composited with optional blur (depth of field / motion), sharpening and grain.

Shadows are a cast silhouette on the floor plus contact blobs, and a faint mirrored reflection.
"""
import math
import os

import cv2
import numpy as np

from .common import ROOT

SPR_DIR = os.path.join(ROOT, "assets", "sprites")

# sprites whose feet were cut off by the edge of the source picture get a soft fade-out there
FADE_BOTTOM = {"beaver": 0.07, "bellwether": 0.07, "gazelle": 0.04}

# rig geometry in normalised body coordinates (u: 0 left..1 right, v: 0 top..1 bottom)
RIG = {
    "judy_full": dict(neck=0.41, head=0.14, ears=0.27, eyes=(0.285, 0.28, 0.72)),
    "judy_arms": dict(neck=0.50, head=0.14, ears=0.35),
    "judy_badge": dict(neck=0.46, head=0.14, ears=0.36, eyes=(0.355, 0.30, 0.62)),
    "nick_full": dict(neck=0.27, head=0.12, ears=0.10, tail=(0.0, 0.47, 0.60, 0.99), eyes=(0.165, 0.50, 0.80)),
    "nick_arms": dict(neck=0.34, head=0.12, ears=0.08, eyes=(0.222, 0.50, 0.82)),
    "nick_walk": dict(neck=0.30, head=0.12, ears=0.08),
    "nick_edge": dict(neck=0.38, head=0.12, ears=0.08),
    "lion_full": dict(neck=0.32, head=0.12, mane=0.30, eyes=(0.168, 0.45, 0.68)),
    "gazelle": dict(neck=0.28, head=0.20, skirt=(0.52, 0.80)),
    "flash": dict(neck=0.47, head=0.12, arm=(0.78, 0.50, 0.58, 1.0, 0.0, 0.55)),
    "clawhauser": dict(neck=0.42, head=0.12),
    "mrbig": dict(neck=0.45, head=0.12),
    "beaver": dict(neck=0.32, head=0.12),
    "turtle": dict(neck=0.30, head=0.10, head_h=True, legs=0.72),
    "mouse_car": dict(),
    "carrot_b": dict(leaves=0.22),
    "carrot_a": dict(leaves=0.22),
}


def _s01(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Sprite:
    """Premultiplied float32 BGRA with mip chain and static shading maps."""

    def __init__(self, name):
        im = cv2.imread(os.path.join(SPR_DIR, name + ".png"), cv2.IMREAD_UNCHANGED)
        if im.shape[2] == 3:
            im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
        self.name = name
        self.rig = RIG.get(name, {})
        a = im[..., 3].astype(np.float32) / 255.0
        a = cv2.erode(a, np.ones((2, 2), np.uint8))
        a = cv2.GaussianBlur(a, (0, 0), 0.7)
        fb = FADE_BOTTOM.get(name, 0.0)
        if fb > 0:
            n = max(2, int(a.shape[0] * fb))
            a[-n:] *= np.linspace(1, 0, n, dtype=np.float32)[:, None] ** 1.3
        rgb = im[..., :3].astype(np.float32)
        pad = 10
        a = np.pad(a, pad)
        rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        rgb = self._decontaminate(rgb, a)
        self.alpha = a
        self.pm = np.dstack([rgb * a[..., None], a * 255.0]).astype(np.float32)
        self.h, self.w = a.shape
        self.pad = pad
        self.body_h = self.h - 2 * pad
        self.body_w = self.w - 2 * pad
        self.mips = [self.pm]
        m = self.pm
        for _ in range(3):
            m = cv2.resize(m, (max(1, m.shape[1] // 2), max(1, m.shape[0] // 2)), interpolation=cv2.INTER_AREA)
            self.mips.append(m)
        self._static = {}
        self._grid = {}

    # -- colour bleed from the old background is replaced by colour from inside the character ---
    @staticmethod
    def _decontaminate(rgb, a):
        solid = (a > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
        core = (dist > 3.0).astype(np.float32)
        num = cv2.GaussianBlur(rgb * core[..., None], (0, 0), 3.0)
        den = cv2.GaussianBlur(core, (0, 0), 3.0)[..., None]
        est = num / np.maximum(den, 1e-3)
        w_est = _s01((3.2 - dist) / 2.6)[..., None] * (den > 0.03)
        return rgb * (1 - w_est) + est * w_est

    # -- per level helpers -------------------------------------------------------------------
    def level_info(self, lvl):
        fac = 2 ** lvl
        img = self.mips[lvl]
        return img, fac

    def grid(self, lvl):
        if lvl not in self._grid:
            img = self.mips[lvl]
            h, w = img.shape[:2]
            fac = 2 ** lvl
            X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            pad, wb, hb = self.pad / fac, self.body_w / fac, self.body_h / fac
            self._grid[lvl] = dict(X=X, Y=Y, U=(X - pad) / wb, V=(Y - pad) / hb, wb=wb, hb=hb, pad=pad)
        return self._grid[lvl]

    def static(self, lvl):
        if lvl in self._static:
            return self._static[lvl]
        img = self.mips[lvl]
        a = img[..., 3] / 255.0
        h, w = a.shape
        fac = 2 ** lvl
        g = self.grid(lvl)
        solid = (a > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
        band = np.clip(1 - dist / max(2.0, 5.5 / fac), 0, 1) ** 1.8 * (a > 0.3)
        ga = cv2.GaussianBlur(a.astype(np.float32), (0, 0), max(1.0, 3.0 / fac))
        gx = cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy) + 1e-6
        nx, ny = -gx / mag, -gy / mag                      # outward normal
        nx[mag < 2e-3] = 0
        ny[mag < 2e-3] = 0
        V = g["V"]
        ao = 1.06 - 0.22 * _s01((V - 0.30) / 0.70)
        ao = ao * (1 - 0.10 * band)
        s = dict(ao=ao.astype(np.float32), band=band.astype(np.float32), nx=nx.astype(np.float32),
                 ny=ny.astype(np.float32),
                 refl=np.clip(1 - np.clip(1 - V, 0, 2.0) * 1.9, 0, 1).astype(np.float32) ** 1.4)
        self._static[lvl] = s
        return s

    def pick(self, scale):
        lvl, s = 0, scale
        while s < 0.55 and lvl < len(self.mips) - 1:
            s *= 2
            lvl += 1
        return lvl

    # -- rig ---------------------------------------------------------------------------------------
    def deform(self, lvl, p):
        """Return the level image warped by the rig parameters in `p` (all optional)."""
        img = self.mips[lvl]
        if not p:
            return img
        g = self.grid(lvl)
        U, V, wb, hb = g["U"], g["V"], g["wb"], g["hb"]
        cfg = self.rig
        dx = np.zeros_like(U)
        dy = np.zeros_like(U)
        b = p.get("bend", 0.0)
        if b:
            dx += b * wb * np.clip(1 - V, 0, 1.2) ** 1.6
        br = p.get("breath", 0.0)
        if br:
            bump = np.exp(-(((V - 0.46) / 0.22) ** 2))
            dx += (U - 0.5) * wb * br * bump
            dy += (V - 0.46) * hb * br * bump * 0.8
        nod = p.get("nod", 0.0)
        if nod and "neck" in cfg:
            nk, hw = cfg["neck"], cfg.get("head", 0.12)
            if cfg.get("head_h"):
                Wh = 1 - _s01((U - nk) / hw)
                pu, pv = nk, 0.5
            else:
                Wh = 1 - _s01((V - (nk - hw * 0.5)) / hw)
                pu, pv = cfg.get("neck_u", 0.5), nk
            th = np.radians(nod) * Wh
            c, s_ = np.cos(th), np.sin(th)
            rx, ry = (U - pu) * wb, (V - pv) * hb
            dx += (c - 1) * rx - s_ * ry
            dy += s_ * rx + (c - 1) * ry
        ev = cfg.get("ears")
        if ev:
            e, e2 = p.get("ear", 0.0), p.get("ear2", 0.0)
            if e or e2:
                t_ = np.clip((ev - V) / ev, 0, 1) ** 2
                sg = np.where(U < 0.5, -1.0, 1.0)
                dx += (e + e2 * sg) * wb * t_
        tb = cfg.get("tail")
        tl = p.get("tail", 0.0)
        if tb and tl:
            u0, u1, v0, v1 = tb
            mask = _s01((U - u0 + 0.04) / 0.1) * (1 - _s01((U - u1) / 0.08)) * _s01((V - v0) / 0.08) * (1 - _s01((V - v1) / 0.04))
            ramp = np.clip((u1 - U) / (u1 - u0), 0, 1) ** 1.3
            dx += tl * wb * mask * ramp
            dy += -0.25 * tl * hb * mask * ramp
        sk = cfg.get("skirt")
        sv = p.get("skirt", 0.0)
        if sk and sv:
            v0, v1 = sk
            m = _s01((V - v0) / 0.08) * (1 - _s01((V - v1) / 0.08))
            dx += sv * wb * m * np.clip((V - v0) / (v1 - v0), 0, 1)
        fl = p.get("flutter", 0.0)
        top = cfg.get("mane", cfg.get("leaves"))
        if fl and top:
            m = np.clip((top - V) / top, 0, 1)
            ph = p.get("phase", 0.0)
            dx += fl * wb * m * np.sin(V * 16 + U * 8 + ph)
            dy += 0.5 * fl * hb * m * np.cos(U * 11 + ph * 1.3)
        arm = cfg.get("arm")
        av = p.get("arm", 0.0)
        if arm and av:
            pu, pv, u0, u1, v0, v1 = arm
            mask = _s01((U - u0) / 0.12) * (1 - _s01((V - v1) / 0.10))
            th = np.radians(av) * mask
            c, s_ = np.cos(th), np.sin(th)
            rx, ry = (U - pu) * wb, (V - pv) * hb
            dx += (c - 1) * rx - s_ * ry
            dy += s_ * rx + (c - 1) * ry
        ey = cfg.get("eyes")
        bl = p.get("blink", 0.0)
        if ey and bl:
            ve, u0, u1 = ey
            mu = _s01((U - u0) / 0.04) * (1 - _s01((U - u1) / 0.04))
            gz = np.exp(-(((V - ve) / 0.026) ** 2))
            dy += -(V - ve) * hb * 0.92 * bl * gz * mu
        lg = cfg.get("legs")
        ls = p.get("legs", 0.0)
        if lg and ls:
            m = _s01((V - lg) / 0.1)
            dy += ls * hb * m * np.sin(U * 13 + p.get("phase", 0.0) * 2)
        return cv2.remap(img, g["X"] - dx, g["Y"] - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0, 0))

    # -- lighting ------------------------------------------------------------------------------------
    def lit(self, img, lvl, L):
        st = self.static(lvl)
        out = img.copy()
        rgb = out[..., :3]
        rgb *= (st["ao"] * L["amb"])[..., None]
        hz = L.get("haze", 0.0)
        if hz > 0.003:
            a = out[..., 3:4] / 255.0
            rgb[:] = rgb * (1 - hz) + np.array(L["haze_col"], np.float32) * a * hz * L["amb"]
        for (lx, ly, col, k) in L.get("rims", ()):
            if k < 0.02:
                continue
            d = np.clip(0.02 + 1.1 * (st["nx"] * lx + st["ny"] * ly), 0, 1) ** 1.3
            rim = (st["band"] * d * k * (out[..., 3] / 255.0))[..., None]
            rgb += rim * np.array(col, np.float32)
        return out


_cache = {}


def get(name, outline=False):
    if name not in _cache:
        _cache[name] = Sprite(name)
    return _cache[name]


# ------------------------------------------------------------------------------------------------------
# drawing
# ------------------------------------------------------------------------------------------------------
_noise = None
GRAIN_SEED = 0


def _grain(h, w, amount):
    global _noise
    if _noise is None:
        _noise = np.random.RandomState(11).randn(1100, 1100).astype(np.float32)
    rng = np.random.RandomState(GRAIN_SEED & 0xFFFF)
    oy, ox = rng.randint(0, 1100 - h) if h < 1100 else 0, rng.randint(0, 1100 - w) if w < 1100 else 0
    return _noise[oy:oy + h, ox:ox + w] * amount


def _mat(a11, a12, a13, a21, a22, a23, a31=0.0, a32=0.0, a33=1.0):
    return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]], np.float64)


def blit(canvas, spr, x, y, height, sx=1.0, sy=1.0, rot=0.0, flip=False, alpha=1.0, flash=0.0, tint=None,
         anchor=(0.5, 1.0), clip=None, bright=1.0, rig=None, light=None, yaw=0.0, pitch=0.0, blur=0.0,
         reflect=False, sharpen=True, grain=2.2):
    """Draw `spr` so its body is `height` px tall at scale 1; (x, y) is where `anchor` lands.

    sx / sy squash & stretch (negative = mirror), rot degrees clockwise, yaw / pitch real 3D turns
    about the vertical / horizontal axis through the body centre."""
    if alpha <= 0.002 or height < 1:
        return
    s = height / spr.body_h
    m_ = min(abs(sx), abs(sy)) * max(abs(math.cos(math.radians(yaw))), 0.3)
    lvl = spr.pick(s * m_ if m_ > 0.05 else s)
    fac = 2 ** lvl
    img = spr.deform(lvl, rig) if rig else spr.mips[lvl]
    if light is not None:
        img = spr.lit(img, lvl, light)
    if reflect:
        img = img * spr.static(lvl)["refl"][..., None]
    k = s * fac
    ax = (spr.pad + anchor[0] * spr.body_w) / fac
    ay = (spr.pad + anchor[1] * spr.body_h) / fac
    fx = -1.0 if flip else 1.0
    th = math.radians(rot)
    c, sn = math.cos(th), math.sin(th)
    Tm = _mat(1, 0, -ax, 0, 1, -ay)
    S = _mat(k * sx * fx, 0, 0, 0, k * sy, 0)
    Hp = np.eye(3)
    if yaw or pitch:
        f = 2.6 * height
        ph_, th_ = math.radians(yaw), math.radians(pitch)
        Ry = np.array([[math.cos(ph_), 0, math.sin(ph_)], [0, 1, 0], [-math.sin(ph_), 0, math.cos(ph_)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(th_), -math.sin(th_)], [0, math.sin(th_), math.cos(th_)]])
        R = Ry @ Rx
        Hc = _mat(R[0, 0], R[0, 1], 0, R[1, 0], R[1, 1], 0, R[2, 0] / f, R[2, 1] / f, 1.0)
        piv = -0.5 * height * sy                         # pivot at mid-height above the anchor
        Hp = _mat(1, 0, 0, 0, 1, piv) @ Hc @ _mat(1, 0, 0, 0, 1, -piv)
    M3 = _mat(1, 0, x, 0, 1, y) @ _mat(c, -sn, 0, sn, c, 0) @ Hp @ S @ Tm
    h, w = img.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64) @ M3.T
    if (corners[:, 2] < 0.2).any():
        return
    pts = corners[:, :2] / corners[:, 2:3]
    x0, y0 = np.floor(pts.min(0)).astype(int) - 3
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 3
    H_, W_ = canvas.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W_), min(y1, H_)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    M2 = _mat(1, 0, -cx0, 0, 1, -cy0) @ M3
    if yaw or pitch:
        warped = cv2.warpPerspective(img, M2, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    else:
        warped = cv2.warpAffine(img, M2[:2].astype(np.float32), (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if blur > 0.2:
        warped = cv2.GaussianBlur(warped, (0, 0), blur)
    elif sharpen and s * m_ > 1.15:
        bl = cv2.GaussianBlur(warped, (0, 0), 1.3)
        warped = warped + 0.55 * (warped - bl)
        np.maximum(warped, 0, out=warped)
        warped[..., :3] = np.minimum(warped[..., :3], warped[..., 3:4])
    a = warped[..., 3:4] / 255.0
    rgb = warped[..., :3]
    if bright != 1.0:
        rgb = rgb * bright
    if grain and rgb.shape[0] < 1100 and rgb.shape[1] < 1100:
        rgb = rgb + _grain(rgb.shape[0], rgb.shape[1], grain)[..., None] * a
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


# ---- shadows -------------------------------------------------------------------------------------------
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


def cast_shadow(canvas, spr, x, y, height, sx=1.0, flip=False, strength=0.40, lift=0.0, dir_x=0.32, depth=0.17):
    """Silhouette of the character thrown onto the floor behind/beside its feet (soft, flattened)."""
    if height < 20:
        return
    s = height / spr.body_h
    lvl = max(1, spr.pick(s * 0.5))
    fac = 2 ** lvl
    a = spr.mips[lvl][..., 3] / 255.0
    h, w = a.shape
    k = s * fac
    fx = -1.0 if flip else 1.0
    ax = (spr.pad + 0.5 * spr.body_w) / fac
    ay = (spr.pad + spr.body_h) / fac
    # point at height eta above the feet lands dir_x*eta to the side and depth*eta further up the floor
    # source (u, v): eta = (ay - v) * k ;  x' = x + (u - ax)*k*sx*fx + dir_x*eta ;  y' = y - depth*eta
    M = np.array([[k * sx * fx, -dir_x * k, x - ax * k * sx * fx + dir_x * k * ay],
                  [0.0, depth * k, y - depth * k * ay]], np.float32)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float32) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int) - 8
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 8
    H_, W_ = canvas.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W_), min(y1, H_)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    M2 = M.copy()
    M2[0, 2] -= cx0
    M2[1, 2] -= cy0
    sh = cv2.warpAffine(a.astype(np.float32), M2, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR)
    sh = cv2.GaussianBlur(sh, (0, 0), max(2.0, 0.02 * height))
    st = strength * (1 - 0.7 * lift)
    roi = canvas[cy0:cy1, cx0:cx1]
    roi[:] = (roi.astype(np.float32) * (1 - np.clip(sh * st, 0, 0.9)[..., None])).astype(np.uint8)
