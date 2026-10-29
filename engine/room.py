"""The real room photo, made controllable: ceiling panels, projector beam, screen, dust."""
import math
import os

import cv2
import numpy as np

from .common import ROOT, W, H, clamp

# screen corner quad measured on the plate (TL, TR, BR, BL)
SCREEN_QUAD = np.array([[268, 557], [668, 557], [669, 862], [272, 862]], np.float32)
SCREEN_C = SCREEN_QUAD.mean(0)
LENS = (448.0, 404.0)                       # projector lens position on the plate
SCREEN_W, SCREEN_H = 800, 600               # content canvas (4:3)


class Room:
    def __init__(self):
        self.plate = cv2.imread(os.path.join(ROOT, "assets", "plate.png"))
        g = cv2.cvtColor(self.plate, cv2.COLOR_BGR2GRAY)
        pan = (g > 205).astype(np.uint8)
        pan[540:] = 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(pan)
        boxes = [tuple(st[i, :4]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 200]
        order = sorted(boxes, key=lambda b: (b[1], b[0]))
        # ids: 0 top, 1 L-upper, 2 R-upper, 3 L-mid, 4 R-mid, 5 centre, 6 L-far, 7 R-far
        want = [(232, 0), (0, 233), (691, 233), (37, 396), (653, 395), (362, 448), (145, 503), (604, 501)]
        self.panels = []
        for wx, wy in want:
            b = min(order, key=lambda o: abs(o[0] - wx) + abs(o[1] - wy))
            x, y, w, h = [int(v) for v in b]
            pad = 70
            x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
            m = np.zeros((y1 - y0, x1 - x0), np.float32)
            roi_mask = (g[y0:y1, x0:x1] > 205).astype(np.float32)
            roi_mask[:, :] *= 1.0
            # only this panel's own pixels
            own = np.zeros_like(roi_mask)
            own[y - y0:y - y0 + h, x - x0:x - x0 + w] = roi_mask[y - y0:y - y0 + h, x - x0:x - x0 + w]
            own = cv2.GaussianBlur(own, (0, 0), 1.2)
            glow = cv2.GaussianBlur(own, (0, 0), 26) * 1.6 + cv2.GaussianBlur(own, (0, 0), 7) * 0.9
            self.panels.append(dict(box=(x0, y0, x1, y1), own=own, glow=np.clip(glow, 0, 1.6),
                                    center=(x + w / 2, y + h / 2), size=(w, h)))
        # screen mask & homography
        self.sx0, self.sy0 = 262, 548
        self.sx1, self.sy1 = 680, 872
        m = np.zeros((self.sy1 - self.sy0, self.sx1 - self.sx0), np.uint8)
        q = (SCREEN_QUAD - [self.sx0, self.sy0]).astype(np.int32)
        cv2.fillConvexPoly(m, q, 255, lineType=cv2.LINE_AA)
        self.screen_mask = (cv2.GaussianBlur(m, (0, 0), 0.6).astype(np.float32) / 255.0)[..., None]
        src = np.array([[0, 0], [SCREEN_W, 0], [SCREEN_W, SCREEN_H], [0, SCREEN_H]], np.float32)
        self.Hs = cv2.getPerspectiveTransform(src, (SCREEN_QUAD - [self.sx0, self.sy0]).astype(np.float32))
        # full-frame screen mask for clipping sprites that emerge from the screen
        full = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(full, SCREEN_QUAD.astype(np.int32), 255, lineType=cv2.LINE_AA)
        self.screen_clip = full
        # screen halo (light the screen throws on the wall)
        halo = np.zeros((H, W), np.float32)
        cv2.fillConvexPoly(halo, SCREEN_QUAD.astype(np.int32), 1.0)
        self.halo = cv2.GaussianBlur(halo, (0, 0), 55)[470:980, 120:820]
        self.halo_off = (120, 470)
        self.halo_wide = cv2.resize(cv2.GaussianBlur(halo, (0, 0), 140)[300:1500, 0:1080], (540, 600))
        # beam: pyramid from the lens to the screen quad
        self.bx0, self.by0, self.bx1, self.by1 = 250, 395, 690, 870
        bw, bh = self.bx1 - self.bx0, self.by1 - self.by0
        hull = np.zeros((bh, bw), np.uint8)
        pts = (np.array([LENS, *SCREEN_QUAD.tolist()], np.float32) - [self.bx0, self.by0]).astype(np.float32)
        cv2.fillConvexPoly(hull, cv2.convexHull(pts.astype(np.int32)), 255, lineType=cv2.LINE_AA)
        hull = cv2.GaussianBlur(hull, (0, 0), 6).astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
        d = np.sqrt((xx + self.bx0 - LENS[0]) ** 2 + (yy + self.by0 - LENS[1]) ** 2)
        fall = np.clip(1.0 - d / 520.0, 0, 1) ** 1.6
        edge = np.clip(1.0 - (np.abs(xx + self.bx0 - LENS[0]) / (d * 0.62 + 40.0)), 0, 1)
        self.beam = hull * (0.35 * fall + 0.65 * fall * edge)
        # dust motes inside the beam
        rng = np.random.RandomState(7)
        self.dust = []
        while len(self.dust) < 110:
            px, py = rng.uniform(self.bx0, self.bx1), rng.uniform(self.by0 + 10, self.by1 - 80)
            ix, iy = int(px - self.bx0), int(py - self.by0)
            if hull[iy, ix] > 0.8:
                self.dust.append((px, py, rng.uniform(0, 6.28), rng.uniform(0.3, 1.0),
                                  rng.uniform(-6, 6), rng.uniform(-10, -2), rng.uniform(0, 6.28)))
        # ambient floating dust over the whole room (very faint)
        self.air = [(rng.uniform(0, W), rng.uniform(300, H), rng.uniform(0, 6.28), rng.uniform(0.4, 1.0),
                     rng.uniform(-8, 8), rng.uniform(-14, -3)) for _ in range(70)]
        # vignette
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W * 0.72)) ** 2 + ((yy - H * 0.48) / (H * 0.66)) ** 2)
        self.vig = np.clip(1.0 - 0.55 * np.clip(r - 0.45, 0, 1) ** 1.6, 0.35, 1).astype(np.float32)
        self.vig3 = np.dstack([self.vig] * 3)

    # ------------------------------------------------------------------------------
    def draw(self, expo=1.0, panels=None, proj=0.0, screen=None, screen_gain=1.0, tint=None, t=0.0,
             beam_col=(255, 240, 225), wide_glow=0.0):
        """Return the lit room (BGR uint8)."""
        panels = panels if panels is not None else [1.0] * 8
        base = cv2.convertScaleAbs(self.plate, alpha=expo)
        for p, w in zip(self.panels, panels):
            x0, y0, x1, y1 = p["box"]
            roi = base[y0:y1, x0:x1].astype(np.float32)
            own = p["own"][..., None]
            lit = self.plate[y0:y1, x0:x1].astype(np.float32)
            dim = lit * (expo * 0.45)
            roi = roi * (1 - own) + (dim * (1 - w) + lit * w) * own
            if w > 0.01:
                roi += p["glow"][..., None] * np.array([255, 246, 232], np.float32) * (0.42 * w)
            base[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)

        if screen is not None and proj > 0.01:
            x0, y0, x1, y1 = self.sx0, self.sy0, self.sx1, self.sy1
            warped = cv2.warpPerspective(screen, self.Hs, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            roi = base[y0:y1, x0:x1].astype(np.float32)
            m = self.screen_mask * clamp(proj)
            roi = roi * (1 - m) + warped.astype(np.float32) * screen_gain * m
            base[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
            mean = np.array(screen.reshape(-1, 3).mean(0), np.float32)
            # screen light spilling on the wall around it
            hx, hy = self.halo_off
            h_ = self.halo
            r2 = base[hy:hy + h_.shape[0], hx:hx + h_.shape[1]].astype(np.float32)
            r2 += h_[..., None] * (mean * 0.55 * proj)[None, None, :]
            base[hy:hy + h_.shape[0], hx:hx + h_.shape[1]] = np.clip(r2, 0, 255).astype(np.uint8)
            if wide_glow > 0:
                wg = cv2.resize(self.halo_wide, (W, 1200))
                r3 = base[300:1500].astype(np.float32)
                r3 += wg[..., None] * (mean * wide_glow * proj)[None, None, :]
                base[300:1500] = np.clip(r3, 0, 255).astype(np.uint8)

        if proj > 0.01:
            x0, y0, x1, y1 = self.bx0, self.by0, self.bx1, self.by1
            roi = base[y0:y1, x0:x1].astype(np.float32)
            col = np.array(beam_col, np.float32)
            if screen is not None:
                col = col * 0.55 + np.array(screen.reshape(-1, 3).mean(0), np.float32) * 0.9
            roi += self.beam[..., None] * col * (0.26 * proj)
            # dust motes catching the light
            layer = np.zeros(roi.shape, np.float32)
            for (px, py, ph, sz, vx, vy, ph2) in self.dust:
                x = px + vx * math.sin(t * 0.35 + ph) * 3 + vx * t * 0.8
                y = py + vy * t * 0.9
                span_y = (self.by1 - 80) - (self.by0 + 10)
                y = (y - (self.by0 + 10)) % span_y + (self.by0 + 10)
                x = (x - self.bx0) % (x1 - x0)
                iy = int(y - self.by0)
                ix = int(x)
                if 0 <= iy < roi.shape[0] and 0 <= ix < roi.shape[1]:
                    tw = 0.5 + 0.5 * math.sin(t * 2.2 + ph2)
                    a = float(self.beam[min(iy, self.beam.shape[0] - 1), ix]) * (0.4 + 1.1 * sz) * (0.4 + tw) * proj
                    v = float(min(255.0, 330.0 * a))
                    cv2.circle(layer, (ix, iy), 1 + int(sz > 0.75), (v, v, v), -1, cv2.LINE_AA)
            roi += layer
            base[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)

        # faint air dust everywhere (depth cue)
        if expo > 0.3:
            layer = np.zeros(base.shape, np.uint8)
            for (px, py, ph, sz, vx, vy) in self.air:
                x = (px + vx * t * 0.6 + 14 * math.sin(t * 0.4 + ph)) % W
                y = (py + vy * t * 0.6) % H
                tw = 0.5 + 0.5 * math.sin(t * 1.7 + ph * 2)
                v = int(255 * 0.30 * tw * sz * min(1.0, expo))
                cv2.circle(layer, (int(x), int(y)), 1, (v, v, v), -1, cv2.LINE_AA)
            base = cv2.add(base, layer)

        if tint is not None:
            c, amt = tint[:3], tint[3]
            base = cv2.addWeighted(base, 1 - amt, np.full_like(base, c), amt, 0) if amt > 0 else base
        return base
