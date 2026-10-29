"""World: owns the room, the actors, the camera and the post chain; renders one frame for any t."""
import math

import cv2
import numpy as np

from .common import W, H, FPS, clamp, lerp, ppm, smooth, PINK, CYAN, WHITE
from . import sprites as sp
from .actors import Actor, Track
from .room import Room
from .screen import Screen


class World:
    def __init__(self):
        self.room = Room()
        self.screen = Screen()
        self.actors = {}
        self.order = []
        # room tracks
        self.expo = Track(1.0)
        self.panels = [Track(0.0) for _ in range(8)]
        self.proj = Track(0.0)
        self.screen_gain = Track(1.0)
        self.wide_glow = Track(0.0)
        self.screen_fn = lambda t: None            # t -> kwargs for Screen.render | "bars" | None
        # camera
        self.zoom = Track(1.0)
        self.cx = Track(0.0)
        self.cy = Track(0.0)
        self.rot = Track(0.0)
        self.shakes = []                            # (t, amp_px, dur, freq)
        self.punches = []                           # (t, amount, dur)  zoom punches
        # post
        self.sat = Track(1.0)
        self.bright = Track(1.0)
        self.contrast = Track(1.0)
        self.flash_a = Track(0.0)
        self.flash_col = (255, 255, 255)
        self.flashes = []                           # (t, dur, color, amount)
        self.vig = Track(1.0)
        self.warm = Track(0.0)                      # -1 cool .. +1 warm
        self.fx_back = []
        self.fx_front = []
        self.fx_screen = []
        self.inverts = []                           # (t0, t1) colour-negative flashes (YTP style)
        self.thaw = None                            # (t0, dur, angle_deg): colour floods back across the frame
        self.captures = {}                          # polaroid captures {t: small frame}
        self.states = {}
        self._snap_cache = {}
        self.vig_u8 = np.dstack([(self.room.vig * 255).astype(np.uint8)] * 3)

    # ---- authoring helpers ---------------------------------------------------------
    def add(self, actor):
        self.actors[actor.key] = actor
        self.order.append(actor.key)
        return actor

    def back(self, t0, t1, fn):
        self.fx_back.append((t0, t1, fn))

    def front(self, t0, t1, fn):
        self.fx_front.append((t0, t1, fn))

    def over(self, t0, t1, fn):
        self.fx_screen.append((t0, t1, fn))

    def shake(self, t, amp=10, dur=0.35, freq=22):
        self.shakes.append((t, amp, dur, freq))

    def punch(self, t, amt=0.04, dur=0.28):
        self.punches.append((t, amt, dur))

    def flash(self, t, dur=0.12, col=WHITE, amt=1.0):
        self.flashes.append((t, dur, col, amt))

    # ---- frame --------------------------------------------------------------------
    def camera_matrix(self, t):
        z = self.zoom(t)
        for (tp, a, d) in self.punches:
            dt = t - tp
            if 0 <= dt < d * 3:
                z *= 1 + a * math.exp(-dt / d * 3.0) * (1 if dt > 0.015 else dt / 0.015)
        dx, dy, rot = self.cx(t), self.cy(t), self.rot(t)
        for (ts, amp, dur, freq) in self.shakes:
            dt = t - ts
            if 0 <= dt < dur:
                f = (1 - dt / dur) ** 2
                dx += amp * f * math.sin(dt * freq * 6.2832) * 0.9
                dy += amp * f * math.cos(dt * freq * 5.3 + 1.0)
                rot += amp * 0.025 * f * math.sin(dt * freq * 4.1)
        a = math.radians(rot)
        # never look outside the photo: raise the zoom just enough for this pan/rotation
        ca_, sa_ = abs(math.cos(a)), abs(math.sin(a))
        pxc, pyc = clamp(W / 2 + dx, 1, W - 1), clamp(H / 2 + dy, 1, H - 1)
        z = max(z, (W * ca_ + H * sa_) / (2 * min(pxc, W - pxc)), (W * sa_ + H * ca_) / (2 * min(pyc, H - pyc)))
        c, s = math.cos(a) * z, math.sin(a) * z
        cx, cy = W / 2, H / 2
        # world point p -> screen: z*R*(p - (c + pan)) + c
        px, py = cx + dx, cy + dy
        M = np.array([[c, -s, cx - (c * px - s * py)], [s, c, cy - (s * px + c * py)]], np.float32)
        return M

    def render(self, t, with_post=True):
        room = self.room
        # --- screen content
        proj = self.proj(t)
        spec = self.screen_fn(t) if proj > 0.01 else None
        screen_img = None
        if spec == "bars":
            screen_img = self.screen.bars(t)
        elif isinstance(spec, dict):
            kw = dict(spec)
            screen_img = self.screen.render(t, **kw)
        panels = [p(t) for p in self.panels]
        canvas = room.draw(expo=self.expo(t), panels=panels, proj=proj if screen_img is not None else 0.0,
                           screen=screen_img, screen_gain=self.screen_gain(t), t=t, wide_glow=self.wide_glow(t))
        # --- back fx
        for (t0, t1, fn) in self.fx_back:
            if t0 <= t < t1:
                fn(canvas, t)
        # --- actors
        states = {}
        self.states = states
        for k in self.order:                      # leaders first, then actors that ride on them
            if not self.actors[k].follows:
                st = self.actors[k].state(t, self)
                if st is not None:
                    states[k] = st
        for k in self.order:
            if self.actors[k].follows:
                st = self.actors[k].state(t, self)
                if st is not None:
                    states[k] = st
        for k, st in sorted(states.items(), key=lambda kv: kv[1]["y"] + kv[1]["actor"].layer):
            a = st["actor"]
            if st["alpha"] <= 0.003:
                continue
            spr = sp.get(st["sprite"], a.outline)
            hpx = st["hpx"]
            yb = st["y"] - st["z"]
            if a.shadow and st["sc"] > 0.02:
                lift = clamp(st["z"] / max(hpx * 0.9, 1))
                sw = hpx * a.shadow_w
                sp.shadow(canvas, st["x"] + st["dx"], st["y"], sw * (1 if not a.flat else 1), 0.5 * st["alpha"],
                          squash=0.26, lift=lift)
                sp.shadow(canvas, st["x"] + st["dx"], st["y"], sw * 0.55, 0.45 * st["alpha"], squash=0.22, lift=lift)
            sp.blit(canvas, spr, st["x"] + st["dx"], yb + st["dy"], hpx, sx=st["sx"], sy=st["sy"], rot=st["rot"],
                    flip=st["flip"], alpha=st["alpha"], flash=st["flash"], anchor=a.anchor,
                    clip=room.screen_clip if st["clip"] else None, bright=a.bright)
            for (tt0, tt1, n, dt) in a.trails:
                if tt0 <= t < tt1:
                    for i in range(1, n + 1):
                        s2 = a.state(t - dt * i, self)
                        if s2 is None:
                            continue
                        sp.blit(canvas, sp.get(s2["sprite"], a.outline), s2["x"] + s2["dx"],
                                s2["y"] - s2["z"] + s2["dy"], s2["hpx"], sx=s2["sx"], sy=s2["sy"], rot=s2["rot"],
                                flip=s2["flip"], alpha=0.34 / i, anchor=a.anchor, bright=a.bright)
        for (t0, t1, fn) in self.fx_front:
            if t0 <= t < t1:
                fn(canvas, t)
        # --- camera
        M = self.camera_matrix(t)
        if abs(M[0, 0] - 1) > 1e-4 or abs(M[0, 1]) > 1e-4 or abs(M[0, 2]) > 0.01 or abs(M[1, 2]) > 0.01:
            canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        for (t0, t1, fn) in self.fx_screen:
            if t0 <= t < t1:
                fn(canvas, t)
        if not with_post:
            return canvas
        return self.post(canvas, t)

    def capture(self, tc, shrink=0.5):
        """A frame of the film at time tc (for the polaroids), cached per process."""
        key = round(tc * 1000)
        if key not in self.captures:
            img = self.render(tc, with_post=True)
            self.captures[key] = cv2.resize(img, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        return self.captures[key]

    # ---- post ---------------------------------------------------------------------
    def post(self, img, t):
        sat, br, ct, wm = self.sat(t), self.bright(t), self.contrast(t), self.warm(t)
        if abs(sat - 1) > 0.01 or abs(br - 1) > 0.01 or abs(ct - 1) > 0.01 or abs(wm) > 0.01:
            wgt = np.array([0.114, 0.587, 0.299], np.float32)
            Mx = np.eye(3, dtype=np.float32) * sat + (1 - sat) * np.tile(wgt, (3, 1))
            gain = np.array([1 - 0.16 * wm, 1.0 - 0.02 * abs(wm), 1 + 0.12 * wm], np.float32)
            Mx = (Mx * (br * ct)) * gain[:, None]
            off = 128 * (1 - ct) * br
            img = cv2.transform(img, Mx, ).astype(np.uint8) if off == 0 else \
                np.clip(cv2.transform(img.astype(np.float32), Mx) + off, 0, 255).astype(np.uint8)
        if self.thaw is not None and self.thaw[0] <= t < self.thaw[0] + self.thaw[1] + 0.05:
            t0, dur, ang = self.thaw
            u = clamp((t - t0) / dur)
            a = math.radians(ang)
            if not hasattr(self, "_thaw_p"):
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                p = xx * math.cos(a) + yy * math.sin(a)
                self._thaw_p = (p - p.min()) / (p.max() - p.min())
            m = np.clip((u * 1.35 - self._thaw_p) / 0.35, 0, 1)
            m = (m * m * (3 - 2 * m))[..., None]
            wgt = np.array([0.114, 0.587, 0.299], np.float32)
            g = (img.astype(np.float32) @ wgt)[..., None]
            gray = np.repeat(g, 3, axis=2)
            img = np.clip(gray * (1 - m) + img.astype(np.float32) * m, 0, 255).astype(np.uint8)
        img = cv2.multiply(img, self.vig_u8, scale=1 / 255.0)
        for (ti0, ti1) in self.inverts:
            if ti0 <= t < ti1:
                img = cv2.bitwise_not(img)
        fa = self.flash_a(t)
        col = self.flash_col
        for (tf, dur, c, amt) in self.flashes:
            dt = t - tf
            if 0 <= dt < dur:
                k = amt * (1 - dt / dur) ** 1.5
                if k > fa:
                    fa, col = k, c
        if fa > 0.004:
            img = cv2.addWeighted(img, 1 - fa, np.full_like(img, col), fa, 0)
        return img
