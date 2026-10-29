"""Actors: position segments + additive modifiers (squash, groove, wobble ...) evaluated per frame."""
import math
import bisect
import zlib

import numpy as np

from .common import (clamp, lerp, ppm, smooth, in_out_quad, out_quad, lin, out_back, out_expo, out_elastic,
                     in_quad, out_cubic, in_out_cubic, ALL_BEATS)


class Track:
    """Scalar animation curve built from chronological `to(t0, t1, v, ease)` calls."""

    def __init__(self, init=0.0):
        self.segs = []       # (t0, t1, v0, v1, ease)
        self.init = init

    def value(self, t):
        v = self.init
        i = bisect.bisect_right([s[0] for s in self.segs], t) - 1
        if i < 0:
            return v
        t0, t1, v0, v1, e = self.segs[i]
        if t >= t1:
            return v1
        return lerp(v0, v1, e(clamp((t - t0) / (t1 - t0))))

    __call__ = value

    def to(self, t0, t1, v, ease=smooth):
        v0 = self.value(t0)
        self.segs.append((t0, max(t1, t0 + 1e-6), v0, v, ease))
        return self

    def set(self, t, v):
        self.segs.append((t, t + 1e-6, v, v, lin))
        return self

    def pulse(self, t0, up, down, peak, base=None, ease_up=out_quad, ease_down=in_out_quad):
        """Rise to `peak` over `up`, fall back over `down`."""
        b = self.value(t0) if base is None else base
        self.to(t0, t0 + up, peak, ease_up)
        self.to(t0 + up, t0 + up + down, b, ease_down)
        return self


class Actor:
    def __init__(self, key, sprite, hm, flip=False, kind="floor", shadow=True, layer=0.0, anchor=(0.5, 1.0),
                 outline=True):
        self.key, self.sprite0, self.hm, self.kind = key, sprite, hm, kind
        self.flip0 = flip
        self.shadow = shadow
        self.layer = layer
        self.anchor = anchor
        self.outline = outline
        self.segs = []         # (t0, t1, fn(t)->(x,y,z))
        self.mods = []         # (t0, t1, fn(t, hpx)->dict)
        self.poses = [(-1e9, sprite, flip)]
        self.vis = []          # (t, bool)
        self.clips = []        # (t0, t1, 'screen')
        self.follows = []
        self.trails = []       # (t0, t1, n, dt) ghost copies behind a fast mover
        self.motions = []      # (t0, t1, fn(t) -> {channel: value}) skeleton motions
        self.shadow_w = 0.5    # contact shadow width as a fraction of the sprite height
        self.flat = False
        self.bright = 1.0
        self.phase = (zlib.crc32(key.encode()) % 6283) / 1000.0     # per-actor offset for idle motion
        self.dof = 0.0                                              # depth-of-field blur radius (px)
        self.rig_k = 1.0                                            # strength of the secondary motion

    # ---- visibility / pose ---------------------------------------------------
    def show(self, t):
        self.vis.append((t, True))
        return self

    def hide(self, t):
        self.vis.append((t, False))
        return self

    def pose(self, t, sprite=None, flip=None):
        _, s0, f0 = self.poses[-1]
        self.poses.append((t, sprite or s0, f0 if flip is None else flip))
        return self

    # ---- skeleton motion channels ------------------------------------------------------------------
    def motion(self, t0, t1, fn):
        self.motions.append((t0, t1, fn))
        return self

    def channels(self, t):
        out = {}
        for (t0, t1, fn) in self.motions:
            if t0 <= t <= t1:
                for k, v in fn(t).items():
                    out[k] = out.get(k, 0.0) + v
        return out

    def walk(self, t0, t1, step_hz=2.4, bob=0.025, arms=True, fade=0.12):
        """Walk cycle.  step_hz = steps per second.
        stride 0 -> 1 -> 0 swaps the legs once per step (side-view sprites); legA/legB lift alternately (front-facing)."""
        def env(t):
            return clamp((t - t0) / fade) * clamp((t1 - t) / fade)

        def ch(t):
            e = env(t)
            ph = math.pi * step_hz * (t - t0)
            s = math.sin(ph)
            return dict(stride=(0.5 - 0.5 * math.cos(ph)) * e, legA=s * e, legB=-s * e,
                        armA=(-s if arms else 0.0) * e, armB=(s if arms else 0.0) * e)
        self.motion(t0, t1, ch)
        return self.mod(t0, t1, lambda t, h: dict(zr=bob * abs(math.sin(math.pi * step_hz * (t - t0))) * env(t)))

    def plod(self, t0, t1, hz=1.6, bob=0.012):
        """Heavy four-legged plod (turtle): legs paddle out of phase, head bobs."""
        def ch(t):
            ph = 2 * math.pi * hz * (t - t0)
            return dict(legA=math.sin(ph), legB=math.sin(ph + 2.1), legC=math.sin(ph + 4.2), head=math.sin(ph * 0.5 + 1.0))
        self.motion(t0, t1, ch)
        return self.mod(t0, t1, lambda t, h: dict(zr=bob * abs(math.sin(2 * math.pi * hz * (t - t0) * 0.5))))

    def wave(self, t0, t1, hz=1.4):
        """Waving hand (channel "wave")."""
        return self.motion(t0, t1, lambda t: dict(wave=math.sin(2 * math.pi * hz * (t - t0))))

    def roar(self, t0, t1, peak=1.0):
        """Jaw drops open fast, holds, then closes."""
        def ch(t):
            u = (t - t0) / (t1 - t0)
            v = out_quad(clamp(u / 0.18)) * (1 - clamp((u - 0.7) / 0.3) ** 2)
            return dict(jaw=peak * v + 0.05 * math.sin((t - t0) * 60) * v)
        return self.motion(t0, t1, ch)

    def star_jump(self, t0, t1, clock=None):
        """Legs open while airborne (channel "spread"); `clock` warps time (freeze frames)."""
        c = clock or (lambda x: x)
        a, b = c(t0), c(t1)
        return self.motion(t0, t1, lambda t: dict(spread=math.sin(clamp((c(t) - a) / max(b - a, 1e-6)) * math.pi) ** 0.7))

    def trail(self, t0, t1, n=3, dt=0.035):
        self.trails.append((t0, t1, n, dt))
        return self

    def clip_screen(self, t0, t1):
        self.clips.append((t0, t1, "screen"))
        return self

    # ---- position segments ---------------------------------------------------
    def stay(self, t, x, y, z=0.0):
        self.segs.append((t, t + 1e-6, lambda tt, x=x, y=y, z=z: (x, y, z)))
        return self

    def move(self, t0, t1, p0, p1, ease=in_out_quad, arc=0.0, arc_ease=None):
        """Travel p0->p1 along the floor; `arc` (px) adds a parabolic lift (hop)."""
        def fn(tt):
            u = clamp((tt - t0) / (t1 - t0))
            e = ease(u)
            z = 4 * arc * u * (1 - u) if arc_ease is None else arc * arc_ease(u)
            return (lerp(p0[0], p1[0], e), lerp(p0[1], p1[1], e), z)
        self.segs.append((t0, t1, fn))
        return self

    def hop(self, t0, t1, p0, p1, h, ease=lin, squash=0.16):
        """Jump with anticipation crouch, take-off stretch and landing squash."""
        self.move(t0, t1, p0, p1, ease=ease, arc=h)
        self.crouch(t0 - 0.11, t0 + 0.02, squash)
        self.stretch(t0, t1, 0.10)
        self.land(t1, squash)
        return self

    def follow(self, t0, t1, other, off=(0.0, 0.0, 0.0), rot=0.0):
        """Ride on another actor (offset in metres of the other's height)."""
        self.follows.append((t0, t1, other, off, rot))
        return self

    # ---- modifiers -------------------------------------------------------------
    def mod(self, t0, t1, fn):
        self.mods.append((t0, t1, fn))
        return self

    def crouch(self, t0, t1, amt=0.16):
        def fn(t, h):
            u = clamp((t - t0) / (t1 - t0))
            k = math.sin(u * math.pi * 0.5)
            return dict(sy=1 - amt * k, sx=1 + amt * 0.7 * k)
        return self.mod(t0, t1, fn)

    def stretch(self, t0, t1, amt=0.1):
        def fn(t, h):
            u = clamp((t - t0) / (t1 - t0))
            k = math.sin(u * math.pi)
            return dict(sy=1 + amt * k, sx=1 - amt * 0.6 * k)
        return self.mod(t0, t1, fn)

    def land(self, t, amt=0.16, dur=0.34):
        def fn(tt, h):
            dt = tt - t
            k = math.exp(-dt * 11) * math.cos(dt * 26)
            return dict(sy=1 - amt * k, sx=1 + amt * 0.75 * k)
        return self.mod(t, t + dur, fn)

    def impact(self, t, amt=0.2, dur=0.4, flash=0.0):
        return self.land(t, amt, dur) if not flash else self.land(t, amt, dur).flash(t, flash)

    def flash(self, t, dur=0.10, amt=1.0):
        def fn(tt, h):
            return dict(flash=amt * clamp(1 - (tt - t) / dur))
        return self.mod(t, t + dur, fn)

    def groove(self, t0, t1, beats=None, amp=0.07, every=1, hop=0.05, phase=0, swing=0.0, damp=7.5):
        """Squash/rebound on every beat (the idle bounce that keeps the cast on the music)."""
        beats = np.asarray(ALL_BEATS if beats is None else beats)
        sel = beats[(beats >= t0 - 0.01) & (beats < t1)][phase::every]

        def fn(t, h):
            i = int(np.searchsorted(sel, t, side="right")) - 1
            if i < 0:
                return {}
            dt = t - sel[i]
            if dt > 0.6:
                return {}
            k = math.exp(-dt * damp) * math.cos(dt * 24)
            d = dict(sy=1 - amp * k, sx=1 + amp * 0.7 * k, zr=hop * max(0.0, -k) if hop else 0.0)
            if swing:
                d["rot"] = swing * (1 if (i % 2) else -1) * math.exp(-dt * 6) * math.sin(dt * 20)
            return d
        return self.mod(t0, t1, fn)

    def wobble(self, t0, t1, deg=4.0, freq=2.0, decay=0.0):
        def fn(t, h):
            u = t - t0
            a = deg * (math.exp(-u * decay) if decay else 1.0)
            return dict(rot=a * math.sin(u * freq * 2 * math.pi))
        return self.mod(t0, t1, fn)

    def spin(self, t0, t1, turns=1.0, ease=in_out_quad, clock=None):
        """Turntable spin around the vertical axis (squeezes sx through zero and mirrors).
        `clock` maps real time to a warped clock (used to freeze it through the silence)."""
        c = clock or (lambda x: x)
        a, b = c(t0), c(t1)

        def fn(t, h):
            u = ease(clamp((c(t) - a) / (b - a)))
            return dict(yaw=u * turns * 360.0)
        return self.mod(t0, t1, fn)

    def tumble(self, t0, t1, deg):
        def fn(t, h):
            u = clamp((t - t0) / (t1 - t0))
            return dict(rot=deg * u)
        return self.mod(t0, t1, fn)

    def scale(self, t0, t1, m0, m1, ease=smooth):
        def fn(t, h):
            return dict(sc=lerp(m0, m1, ease(clamp((t - t0) / (t1 - t0)))))
        return self.mod(t0, t1, fn)

    def fade(self, t0, t1, a0, a1):
        def fn(t, h):
            return dict(alpha=lerp(a0, a1, clamp((t - t0) / (t1 - t0))))
        return self.mod(t0, t1, fn)

    def shake(self, t0, t1, amp=6.0, freq=34.0):
        def fn(t, h):
            u = t - t0
            fade = 1.0 - clamp((t - t0) / (t1 - t0)) ** 2
            return dict(dx=amp * fade * math.sin(u * freq * 6.28), rot=amp * 0.12 * fade * math.cos(u * freq * 5.1))
        return self.mod(t0, t1, fn)

    def pop_in(self, t0, dur=0.36, ease=out_back):
        """Scale up from nothing with overshoot (and show)."""
        self.show(t0)

        def fn(t, h):
            u = clamp((t - t0) / dur)
            s = max(0.001, ease(u))
            return dict(sc=s, sy=1.0 + 0.12 * (1 - u) * math.sin(u * 9))
        return self.mod(t0, t0 + dur, fn)

    def pop_out(self, t0, dur=0.22):
        def fn(t, h):
            u = clamp((t - t0) / dur)
            return dict(sc=max(0.001, 1 - in_quad(u)), alpha=1 - u)
        self.mod(t0, t0 + dur, fn)
        self.hide(t0 + dur)
        return self

    # ---- evaluation --------------------------------------------------------------
    def visible(self, t):
        v = False
        for tt, b in self.vis:
            if tt <= t:
                v = b
        return v

    def state(self, t, world=None):
        if not self.visible(t):
            return None
        x = y = z = None
        for (t0, t1, fn) in self.segs:
            if t0 <= t:
                x, y, z = fn(min(t, t1))
        for (t0, t1, other, off, rot) in self.follows:
            if t0 <= t < t1 and world is not None:
                so = world.states.get(other)
                if so is not None:
                    ho = so["hpx"]
                    fl = -1.0 if so["flip"] else 1.0
                    x = so["x"] + off[0] * ho * fl + so.get("dx", 0)
                    y = so["y"] + 0.5
                    z = so["z"] + off[2] * ho
                    return self._finish(t, x, y, z, so_rot=rot, follow_off=off)
        if x is None:
            return None
        return self._finish(t, x, y, z)

    def _finish(self, t, x, y, z, so_rot=0.0, follow_off=None):
        sprite, flip = self.sprite0, self.flip0
        for (tt, s, f) in self.poses:
            if tt <= t:
                sprite, flip = s, f
        st = dict(x=x, y=y, z=z, sx=1.0, sy=1.0, sc=1.0, rot=so_rot, alpha=1.0, flash=0.0, dx=0.0, dy=0.0, zr=0.0, yaw=0.0, pitch=0.0,
                  sprite=sprite, flip=flip)
        hpx0 = self.hm * ppm(y)
        for (t0, t1, fn) in self.mods:
            if t0 <= t <= t1:
                d = fn(t, hpx0)
                for k, v in d.items():
                    if k in ("sx", "sy", "sc", "alpha"):
                        st[k] *= v
                    elif k == "flash":
                        st[k] = max(st[k], v)
                    else:
                        st[k] += v
        st["hpx"] = hpx0 * st["sc"]
        st["z"] += st["zr"] * st["hpx"]
        st["clip"] = any(t0 <= t < t1 for (t0, t1, _) in self.clips)
        st["actor"] = self
        return st
