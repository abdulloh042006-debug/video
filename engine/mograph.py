"""Motion-graphics primitives drawn straight onto a BGR canvas (sub-pixel, anti-aliased)."""
import math

import cv2
import numpy as np

from .common import (PALETTE, PINK, CYAN, ORANGE, PURPLE, YELLOW, GREEN, MAGENTA, CARROT, WHITE, W, H,
                     clamp, out_cubic, in_out_cubic, out_back, smooth)

SH = 4                       # sub-pixel shift bits
K = 1 << SH


def _p(pts):
    return (np.asarray(pts, np.float64) * K).round().astype(np.int32)


def _bbox(pts, pad=3):
    pts = np.asarray(pts, np.float64)
    x0, y0 = np.floor(pts.min(0)).astype(int) - pad
    x1, y1 = np.ceil(pts.max(0)).astype(int) + pad
    return max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)


def _paint(canvas, bbox, alpha, fn, mode="blend"):
    """Run fn(tmp_roi, ox, oy) on a copy of the ROI and mix it back with `alpha`."""
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or alpha <= 0.003:
        return
    roi = canvas[y0:y1, x0:x1]
    if alpha >= 0.995 and mode == "blend":
        fn(roi, x0, y0)
        return
    tmp = roi.copy() if mode == "blend" else np.zeros_like(roi)
    fn(tmp, x0, y0)
    if mode == "add":
        roi[:] = cv2.add(roi, (tmp * alpha).astype(np.uint8))
    else:
        cv2.addWeighted(tmp, alpha, roi, 1 - alpha, 0, roi)


def poly(canvas, pts, color, alpha=1.0, mode="blend"):
    pts = np.asarray(pts, np.float64)
    bb = _bbox(pts)

    def fn(roi, ox, oy):
        cv2.fillPoly(roi, [_p(pts - [ox, oy])], color, cv2.LINE_AA, SH)
    _paint(canvas, bb, alpha, fn, mode)


def circle(canvas, c, r, color, alpha=1.0, thick=-1, mode="blend"):
    if r < 0.5:
        return
    bb = _bbox([(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)], 4 + (thick if thick > 0 else 0))

    def fn(roi, ox, oy):
        cv2.circle(roi, (int(round((c[0] - ox) * K)), int(round((c[1] - oy) * K))), int(round(r * K)), color,
                   thick if thick < 0 else max(1, int(round(thick))), cv2.LINE_AA, SH)
    _paint(canvas, bb, alpha, fn, mode)


def ring(canvas, c, r, thick, color, alpha=1.0, mode="blend"):
    circle(canvas, c, r, color, alpha, thick=max(1, thick), mode=mode)


_glow_cache = {}


def glow(canvas, c, r, color, k=1.0):
    """Additive radial glow."""
    if r < 2 or k <= 0.003:
        return
    n = 64
    if "g" not in _glow_cache:
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        d = np.sqrt(((xx - n / 2 + .5) / (n / 2)) ** 2 + ((yy - n / 2 + .5) / (n / 2)) ** 2)
        _glow_cache["g"] = np.clip(1 - d, 0, 1) ** 2.2
    g = _glow_cache["g"]
    d = int(r * 2)
    x0, y0 = int(c[0] - r), int(c[1] - r)
    x1, y1 = x0 + d, y0 + d
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    gg = cv2.resize(g, (d, d), interpolation=cv2.INTER_LINEAR)[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    roi += gg[..., None] * (np.array(color, np.float32) * k)
    canvas[cy0:cy1, cx0:cx1] = np.clip(roi, 0, 255).astype(np.uint8)


def star(canvas, c, r_out, r_in, n, rot, color, alpha=1.0, mode="blend"):
    pts = []
    for i in range(n * 2):
        a = math.radians(rot) + i * math.pi / n - math.pi / 2
        r = r_out if i % 2 == 0 else r_in
        pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    poly(canvas, pts, color, alpha, mode)


def sparkle(canvas, c, size, rot=0.0, color=WHITE, alpha=1.0):
    """Four-point twinkle with concave sides."""
    pts = []
    for i in range(8):
        a = math.radians(rot) + i * math.pi / 4 - math.pi / 2
        r = size if i % 2 == 0 else size * 0.22
        pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    poly(canvas, pts, color, alpha)
    glow(canvas, c, size * 1.6, color, 0.35 * alpha)


def rect_rot(canvas, c, w, h, rot, color, alpha=1.0):
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)
    pts = [(c[0] + dx * ca - dy * sa, c[1] + dx * sa + dy * ca)
           for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
    poly(canvas, pts, color, alpha)


def carrot(canvas, c, size, rot=0.0, alpha=1.0):
    """Little flat vector carrot (tip down), `size` = total length."""
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)

    def T(x, y):
        return (c[0] + x * ca - y * sa, c[1] + x * sa + y * ca)
    L, Wd = size, size * 0.34
    body = [T(-Wd / 2, -L * 0.38), T(Wd / 2, -L * 0.38), T(Wd * 0.28, L * 0.1), T(0, L * 0.5),
            T(-Wd * 0.28, L * 0.1)]
    poly(canvas, body, CARROT, alpha)
    for dy in (-0.12, 0.0, 0.14):          # little ridges
        poly(canvas, [T(-Wd * 0.32 + (0.02 * L), L * dy), T(-Wd * 0.05, L * dy + 0.03 * L),
                      T(-Wd * 0.32 + (0.02 * L), L * dy + 0.02 * L)], (30, 90, 220), alpha * 0.9)
    for ang, ln in ((-28, 0.34), (0, 0.42), (28, 0.34)):
        b = math.radians(ang)
        tip = T(math.sin(b) * L * ln, -L * 0.38 - math.cos(b) * L * ln)
        poly(canvas, [T(-Wd * 0.14, -L * 0.36), tip, T(Wd * 0.14, -L * 0.36)], (60, 200, 70), alpha)


def paw(canvas, c, size, rot=0.0, color=WHITE, alpha=1.0):
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)

    def T(x, y):
        return (c[0] + (x * ca - y * sa) * size, c[1] + (x * sa + y * ca) * size)
    for (x, y, r) in ((0, 0.2, 0.34), (-0.42, -0.2, 0.15), (-0.14, -0.46, 0.15), (0.14, -0.46, 0.15),
                      (0.42, -0.2, 0.15)):
        p = T(x, y)
        circle(canvas, p, r * size, color, alpha)


def sunburst(canvas, c, n, rot, r, color, alpha=1.0, duty=0.5, mode="blend"):
    """n alternating rays reaching radius r."""
    step = 2 * math.pi / n
    for i in range(n):
        a0 = math.radians(rot) + i * step
        a1 = a0 + step * duty
        pts = [c, (c[0] + r * math.cos(a0), c[1] + r * math.sin(a0)),
               (c[0] + r * math.cos((a0 + a1) / 2) * 1.02, c[1] + r * math.sin((a0 + a1) / 2) * 1.02),
               (c[0] + r * math.cos(a1), c[1] + r * math.sin(a1))]
        poly(canvas, pts, color, alpha, mode)


def speedlines(canvas, c, n, r_in, r_out, color=WHITE, alpha=0.7, seed=0, width=2.2, jitter=0.35):
    rng = np.random.RandomState(seed)
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        r0 = r_in * rng.uniform(1.0, 1.0 + jitter * 3)
        r1 = r_out * rng.uniform(0.8, 1.15)
        w = width * rng.uniform(0.5, 1.6)
        tip = (c[0] + r1 * math.cos(a), c[1] + r1 * math.sin(a))
        b0 = (c[0] + r0 * math.cos(a + 0.012), c[1] + r0 * math.sin(a + 0.012))
        b1 = (c[0] + r0 * math.cos(a - 0.012), c[1] + r0 * math.sin(a - 0.012))
        poly(canvas, [b0, tip, b1], color, alpha)


def streaks(canvas, x0, x1, y, length, color, alpha, seed, n=10, spread=90, width=3, direction=1):
    """Horizontal motion streaks trailing behind something moving along x."""
    rng = np.random.RandomState(seed)
    for _ in range(n):
        yy = y + rng.uniform(-spread, spread)
        ln = length * rng.uniform(0.4, 1.0)
        xs = x0 + rng.uniform(0, 1) * (x1 - x0)
        w = width * rng.uniform(0.6, 1.5)
        poly(canvas, [(xs, yy - w / 2), (xs - direction * ln, yy), (xs, yy + w / 2)], color, alpha * rng.uniform(0.5, 1))


def halftone(canvas, c, r, spacing, color, alpha=1.0, power=1.4):
    """Comic halftone disc: dots shrink toward the rim."""
    n = int(r / spacing) + 1
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            x = c[0] + (i + 0.5 * (j % 2)) * spacing
            y = c[1] + j * spacing * 0.866
            d = math.hypot(x - c[0], y - c[1]) / r
            if d >= 1:
                continue
            rad = spacing * 0.5 * (1 - d) ** power * 1.15
            if rad > 0.6:
                circle(canvas, (x, y), rad, color, alpha)


def impact_burst(canvas, c, r, rot, color, outline=WHITE, alpha=1.0, spikes=11, seed=1):
    """Comic 'POW' shape without any text."""
    rng = np.random.RandomState(seed)
    pts = []
    for i in range(spikes * 2):
        a = math.radians(rot) + i * math.pi / spikes
        rr = r * (1.0 if i % 2 == 0 else 0.55) * (1 + 0.12 * rng.uniform(-1, 1))
        pts.append((c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)))
    big = [(c[0] + (p[0] - c[0]) * 1.12, c[1] + (p[1] - c[1]) * 1.12) for p in pts]
    poly(canvas, big, outline, alpha)
    poly(canvas, pts, color, alpha)


# ------------------------------------------------------------------------------------
class Confetti:
    """Deterministic confetti / carrot shower: every particle is an analytic function of time."""

    def __init__(self, n, origin, seed=3, speed=(500, 1100), spread=(-math.pi, 0.0), gravity=1500,
                 size=(12, 26), life=(1.6, 2.6), colors=None, kinds=("rect", "circle", "star", "carrot", "tri"),
                 area=0.0, drag=0.9):
        rng = np.random.RandomState(seed)
        self.p = []
        self.g = gravity
        self.drag = drag
        for _ in range(n):
            a = rng.uniform(*spread)
            sp = rng.uniform(*speed)
            ox = origin[0] + rng.uniform(-area, area)
            self.p.append(dict(
                x=ox, y=origin[1], vx=math.cos(a) * sp, vy=math.sin(a) * sp,
                size=rng.uniform(*size), rot=rng.uniform(0, 360), spin=rng.uniform(-540, 540),
                life=rng.uniform(*life), col=(colors or PALETTE)[rng.randint(len(colors or PALETTE))],
                kind=kinds[rng.randint(len(kinds))], flut=rng.uniform(0, 6.28), delay=rng.uniform(0, 0.06)))

    def draw(self, canvas, u, alpha=1.0):
        """u = seconds since burst."""
        for q in self.p:
            tt = u - q["delay"]
            if tt < 0 or tt > q["life"]:
                continue
            d = self.drag
            # drag-limited ballistic motion
            k = 1.8 * (1 - d) + 0.3
            ex = (1 - math.exp(-k * tt)) / k
            x = q["x"] + q["vx"] * ex + 14 * math.sin(tt * 6 + q["flut"])
            y = q["y"] + q["vy"] * ex + 0.5 * self.g * tt * tt
            fade = clamp((q["life"] - tt) / 0.45)
            pop = clamp(tt / 0.08)
            s = q["size"] * (0.4 + 0.6 * pop) * (0.55 + 0.45 * fade)
            rot = q["rot"] + q["spin"] * tt
            flip = abs(math.cos(tt * 7 + q["flut"]))        # paper flipping in the air
            a = alpha * fade
            kd = q["kind"]
            if kd == "rect":
                rect_rot(canvas, (x, y), s * 1.6, s * 0.7 * (0.25 + 0.75 * flip), rot, q["col"], a)
            elif kd == "circle":
                circle(canvas, (x, y), s * 0.5, q["col"], a)
            elif kd == "star":
                star(canvas, (x, y), s * 0.8, s * 0.34, 5, rot, q["col"], a)
            elif kd == "carrot":
                carrot(canvas, (x, y), s * 2.4, rot, a)
            else:
                a_ = math.radians(rot)
                pts = [(x + s * math.cos(a_ + i * 2.094), y + s * math.sin(a_ + i * 2.094)) for i in range(3)]
                poly(canvas, pts, q["col"], a)


def wipe(canvas, p, colors, angle=-25.0, bands=5, mode_out=False):
    """Colour-block stagger wipe.  p: 0..1.  Blocks fly in (first half), the frame is fully covered for ~4
    frames around p=0.45, then they leave on the far side."""
    if p <= 0 or p >= 1:
        return
    a = math.radians(angle)
    d = np.array([math.cos(a), math.sin(a)])         # travel direction
    n = np.array([-d[1], d[0]])
    L = 1210.0                                       # just over the half-diagonal of the frame
    c = np.array([W / 2, H / 2])
    bw = 2 * L / bands
    for i in range(bands):
        dl = 0.08 * i / max(1, bands - 1)
        pin = clamp((p - dl) / 0.36)
        pout = clamp((p - 0.5 - dl) / 0.36)
        front = -L + 2 * L * out_cubic(pin)
        back = -L + 2 * L * in_out_cubic(pout)
        if front <= back + 1:
            continue
        o0, o1 = -L + i * bw - 2, -L + (i + 1) * bw + 2
        pts = [c + d * back + n * o0, c + d * front + n * o0, c + d * front + n * o1, c + d * back + n * o1]
        poly(canvas, pts, colors[i % len(colors)], 1.0)


def eq_bars(canvas, x0, x1, y, h, n, levels, color, alpha=0.9, gap=0.35, round_=True):
    bw = (x1 - x0) / n
    for i in range(n):
        lv = clamp(levels[i % len(levels)])
        bh = max(3.0, h * lv)
        xa = x0 + i * bw + bw * gap / 2
        xb = x0 + (i + 1) * bw - bw * gap / 2
        poly(canvas, [(xa, y), (xb, y), (xb, y - bh), (xa, y - bh)], color, alpha)


def shockwave(canvas, c, u, r_max, color=WHITE, thick=18, alpha=0.9):
    """Expanding ring that thins as it grows. u = 0..1."""
    if u <= 0 or u >= 1:
        return
    r = r_max * out_cubic(u)
    th = max(1.0, thick * (1 - u) ** 1.2)
    ring(canvas, c, r, int(round(th)), color, alpha * (1 - u) ** 0.6)


def ring_e(canvas, c, rx, thick, color, alpha=1.0, squash=0.28, mode="blend"):
    """Ellipse ring lying on the floor (perspective-squashed circle)."""
    if rx < 1:
        return
    ry = rx * squash
    bb = _bbox([(c[0] - rx, c[1] - ry), (c[0] + rx, c[1] + ry)], 6 + thick)

    def fn(roi, ox, oy):
        cv2.ellipse(roi, (int(round((c[0] - ox) * K)), int(round((c[1] - oy) * K))),
                    (int(round(rx * K)), int(round(ry * K))), 0, 0, 360, color, max(1, int(thick)), cv2.LINE_AA, SH)
    _paint(canvas, bb, alpha, fn, mode)


def disc_e(canvas, c, rx, color, alpha=1.0, squash=0.28, mode="blend"):
    if rx < 1:
        return
    ry = rx * squash
    bb = _bbox([(c[0] - rx, c[1] - ry), (c[0] + rx, c[1] + ry)], 6)

    def fn(roi, ox, oy):
        cv2.ellipse(roi, (int(round((c[0] - ox) * K)), int(round((c[1] - oy) * K))),
                    (int(round(rx * K)), int(round(ry * K))), 0, 0, 360, color, -1, cv2.LINE_AA, SH)
    _paint(canvas, bb, alpha, fn, mode)


_gr = {}


def godrays(canvas, c, rot, color, alpha, n=14, width=0.42, radius=1900, blur=7, falloff=1500):
    """Soft additive light shafts fanning out from c (half-res layer, blurred, radial falloff)."""
    h2, w2 = H // 2, W // 2
    layer = np.zeros((h2, w2), np.uint8)
    cc = (c[0] / 2, c[1] / 2)
    step = 2 * math.pi / n
    for i in range(n):
        a0 = math.radians(rot) + i * step
        a1 = a0 + step * width
        pts = np.array([cc, (cc[0] + radius / 2 * math.cos(a0), cc[1] + radius / 2 * math.sin(a0)),
                        (cc[0] + radius / 2 * math.cos((a0 + a1) / 2) * 1.05, cc[1] + radius / 2 * math.sin((a0 + a1) / 2) * 1.05),
                        (cc[0] + radius / 2 * math.cos(a1), cc[1] + radius / 2 * math.sin(a1))], np.float64)
        cv2.fillPoly(layer, [_p(pts)], 255, cv2.LINE_AA, SH)
    layer = cv2.GaussianBlur(layer, (0, 0), blur)
    key = (int(c[0]), int(c[1]), falloff)
    if key not in _gr:
        yy, xx = np.mgrid[0:h2, 0:w2].astype(np.float32)
        d = np.sqrt((xx - cc[0]) ** 2 + (yy - cc[1]) ** 2) * 2
        _gr[key] = np.clip(1 - d / falloff, 0, 1) ** 1.3
    lay = layer.astype(np.float32) / 255.0 * _gr[key]
    lay = cv2.resize(lay, (W, H), interpolation=cv2.INTER_LINEAR)
    col = np.array(color, np.float32) * alpha
    # only touch rows that can have light
    canvas[:] = np.clip(canvas.astype(np.float32) + lay[..., None] * col, 0, 255).astype(np.uint8)


def teardrop(canvas, c, r, color=(255, 200, 80), alpha=1.0):
    """Sweat drop: pointy top, round bottom (c = centre of the round part)."""
    pts = [(c[0], c[1] - r * 2.3), (c[0] - r * 0.9, c[1] - r * 0.4), (c[0] + r * 0.9, c[1] - r * 0.4)]
    poly(canvas, pts, color, alpha)
    circle(canvas, c, r, color, alpha)
    circle(canvas, (c[0] - r * 0.3, c[1] - r * 0.25), r * 0.28, (255, 255, 255), alpha * 0.85)


def clock(canvas, c, r, hour_deg, min_deg, alpha=1.0, rim=CYAN, hand=CARROT):
    """Cartoon wall clock (angles in degrees clockwise from 12)."""
    circle(canvas, c, r * 1.1, WHITE, alpha)
    circle(canvas, c, r * 1.0, rim, alpha)
    circle(canvas, c, r * 0.86, (250, 250, 255), alpha)
    for i in range(12):
        a = math.radians(i * 30 - 90)
        major = i % 3 == 0
        p0 = (c[0] + math.cos(a) * r * (0.62 if major else 0.72), c[1] + math.sin(a) * r * (0.62 if major else 0.72))
        p1 = (c[0] + math.cos(a) * r * 0.8, c[1] + math.sin(a) * r * 0.8)
        rect_rot(canvas, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2), math.hypot(p1[0] - p0[0], p1[1] - p0[1]),
                 r * (0.07 if major else 0.04), math.degrees(a), (70, 60, 60), alpha)
    for deg, ln, wd, col in ((hour_deg, 0.45, 0.1, (70, 60, 60)), (min_deg, 0.72, 0.07, hand)):
        a = math.radians(deg - 90)
        mid = (c[0] + math.cos(a) * r * ln / 2, c[1] + math.sin(a) * r * ln / 2)
        rect_rot(canvas, mid, r * ln, r * wd, math.degrees(a), col, alpha)
    circle(canvas, c, r * 0.08, (70, 60, 60), alpha)


def paste(canvas, img, center, scale, rot=0.0, border=12, bottom=36, alpha=1.0, shadow=True):
    """Paste `img` (BGR) like a polaroid: white frame, rotation about its centre, soft drop shadow."""
    h, w = img.shape[:2]
    sw, sh = w * scale, h * scale
    bd, bt = sw * 0.06, sw * 0.2
    fw, fh = sw + 2 * bd, sh + bd + bt
    a = math.radians(rot)
    ca, sa = math.cos(a), math.sin(a)
    # frame centre is offset downward so the picture sits high inside the frame
    cx, cy = center
    def T(dx, dy):
        return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)
    if shadow:
        poly(canvas, [T(-fw / 2 + 10, -fh / 2 + 14), T(fw / 2 + 10, -fh / 2 + 14), T(fw / 2 + 10, fh / 2 + 14), T(-fw / 2 + 10, fh / 2 + 14)], (0, 0, 0), 0.28 * alpha)
    poly(canvas, [T(-fw / 2, -fh / 2), T(fw / 2, -fh / 2), T(fw / 2, fh / 2), T(-fw / 2, fh / 2)], (250, 248, 245), alpha)
    # picture: top-left of picture in frame coords is (-sw/2, -fh/2 + bd)
    pic_c = T(0, -fh / 2 + bd + sh / 2)
    M = np.array([[ca * scale, -sa * scale, pic_c[0] - (ca * scale * w / 2 - sa * scale * h / 2)],
                  [sa * scale, ca * scale, pic_c[1] - (sa * scale * w / 2 + ca * scale * h / 2)]], np.float32)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float32) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int) - 1
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 1
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    M2 = M.copy()
    M2[0, 2] -= cx0
    M2[1, 2] -= cy0
    warped = cv2.warpAffine(img, M2, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    msk = cv2.warpAffine(np.full((h, w), 255, np.uint8), M2, (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_LINEAR)
    m = (msk.astype(np.float32) / 255.0)[..., None] * alpha
    roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    canvas[cy0:cy1, cx0:cx1] = np.clip(roi * (1 - m) + warped.astype(np.float32) * m, 0, 255).astype(np.uint8)


def crown(canvas, c, size, alpha=1.0):
    """Little golden crown centred at c (flat-bottomed)."""
    w, h = size, size * 0.75
    x0, y0 = c[0] - w / 2, c[1] - h / 2
    pts = [(x0, y0 + h), (x0, y0 + h * 0.25), (x0 + w * 0.25, y0 + h * 0.6), (x0 + w * 0.5, y0), (x0 + w * 0.75, y0 + h * 0.6),
           (x0 + w, y0 + h * 0.25), (x0 + w, y0 + h)]
    poly(canvas, [(p[0] - 3, p[1] + 3) for p in pts], (20, 120, 200), alpha)
    poly(canvas, pts, (40, 200, 255), alpha)
    poly(canvas, [(x0, y0 + h * 0.8), (x0 + w, y0 + h * 0.8), (x0 + w, y0 + h), (x0, y0 + h)], (30, 160, 235), alpha)
    for fx_, fy_, col in ((0.0, 0.25, (60, 60, 240)), (0.5, 0.0, (240, 200, 60)), (1.0, 0.25, (60, 60, 240))):
        circle(canvas, (x0 + w * fx_, y0 + h * fy_), size * 0.07, col, alpha)


def stars_ring(canvas, c, rx, ry, t, n=4, size=14, alpha=1.0, cols=(YELLOW, WHITE, PINK, CYAN)):
    for i in range(n):
        a = t * 5.0 + i * 2 * math.pi / n
        star(canvas, (c[0] + rx * math.cos(a), c[1] + ry * math.sin(a)), size, size * 0.45, 5, a * 40, cols[i % len(cols)], alpha)


_spot = {}


def spotlight(canvas, center, r, dark=0.45, soft=0.55, tint=None):
    """Darken everything outside a soft circle (theatre spotlight)."""
    key = (int(r), int(soft * 100))
    if key not in _spot:
        n = int(r * 2 * 2.2)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        d = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / r
        _spot[key] = np.clip((d - (1 - soft)) / (soft + 0.001), 0, 1)
    m = _spot[key]
    n = m.shape[0]
    x0, y0 = int(center[0] - n / 2), int(center[1] - n / 2)
    mask = np.ones((H, W), np.float32)            # 1 = fully dark outside the circle
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + n, W), min(y0 + n, H)
    mask[:] = 1.0
    if cx1 > cx0 and cy1 > cy0:
        mask[cy0:cy1, cx0:cx1] = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    gain = 1 - (1 - dark) * mask
    out = canvas.astype(np.float32) * gain[..., None]
    if tint is not None:
        out += (mask[..., None] * np.array(tint, np.float32) * 0.12)
    canvas[:] = np.clip(out, 0, 255).astype(np.uint8)


def exclaim(canvas, c, size, alpha=1.0, col=(60, 60, 235)):
    """Comic exclamation mark centred at c: tall tapered bar + dot, bold white outline."""
    def shape(k, colr, a_):
        y0, y1 = c[1] - size * 0.5, c[1] + size * 0.10
        hw_t, hw_b = size * 0.17 * k, size * 0.075 * k
        y0k, y1k = y0 - (k - 1) * size * 0.25, y1 + (k - 1) * size * 0.25
        poly(canvas, [(c[0] - hw_t, y0k + hw_t), (c[0] - hw_b, y1k), (c[0] + hw_b, y1k), (c[0] + hw_t, y0k + hw_t)], colr, a_)
        circle(canvas, (c[0], y0k + hw_t), hw_t, colr, a_)
        circle(canvas, (c[0], c[1] + size * 0.36), size * 0.105 * k, colr, a_)
    shape(1.32, WHITE, alpha)
    shape(1.0, col, alpha)
    circle(canvas, (c[0] - size * 0.045, c[1] - size * 0.40), size * 0.03, (255, 255, 255), alpha * 0.75)
