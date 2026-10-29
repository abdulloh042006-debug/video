"""The film: every actor, light, camera move and graphic, keyed to the music.

Audio map (seconds): A loud 0-20.22 | silence | B quiet groove 20.91-79.2 | silence | C loud 79.78-90
Beat helpers:  A(i) / B(i) / C(i) -> time of the i-th detected beat in that section.
"""
import math

import numpy as np

from engine.common import *
from engine.world import World
from engine.actors import Actor, Track
from engine import mograph as mg

FREEZE = SIL1[0]            # 20.22 - the music drops dead, the world freezes
THAW = SIL1[1]              # 20.91
HOLD2 = SIL2[0]             # 79.20
BOOM = BC[0]                # 79.81


def wt(t):
    """World time with the 0.69 s freeze cut out (everything 'frozen' uses this clock)."""
    if t < FREEZE:
        return t
    if t < THAW:
        return FREEZE
    return t - (THAW - FREEZE)


def A(i):
    return BA[i]


def B(i):
    """i-th beat of the quiet groove; extrapolated past the last tracked beat (~77.3 s) up to the silence."""
    if i < len(BB):
        return BB[i]
    return BB[-1] + (i - len(BB) + 1) * 0.4575


def C(i):
    return BC[i]


# ---------------------------------------------------------------------------------------------
# small effect builders
# ---------------------------------------------------------------------------------------------
def fx_ring(w, t0, pos, rmax=320, col=WHITE, dur=0.55, thick=16, floor=False, layer=None, alpha=0.9,
            hold=None):
    if layer is None:
        layer = "back" if rmax >= 450 else "front"

    def fn(c, t):
        tt = wt(t) if hold else t
        u = (tt - t0) / dur
        if not 0 < u < 1:
            return
        r = rmax * out_cubic(u)
        th = max(1, int(thick * (1 - u) ** 1.2))
        a = alpha * (1 - u) ** 0.6
        if floor:
            mg.ring_e(c, pos, r, th, col, a)
        else:
            mg.ring(c, pos, r, th, col, a)
    (w.back if layer == "back" else w.front)(t0, t0 + dur + 0.02 + (0.7 if hold else 0), fn)


def fx_sparkles(w, t0, pos, n=6, rad=150, dur=0.7, size=26, seed=1, cols=None, hold=None):
    rng = np.random.RandomState(seed)
    items = [(rng.uniform(0, 6.28), rng.uniform(0.45, 1.0), rng.uniform(0, 0.15), rng.uniform(0.7, 1.3),
              rng.uniform(-30, 30)) for _ in range(n)]
    cols = cols or [WHITE, YELLOW, CYAN, PINK]

    def fn(c, t):
        tt = wt(t) if hold else t
        for i, (ang, rr, dl, sz, rot) in enumerate(items):
            u = (tt - t0 - dl) / dur
            if not 0 < u < 1:
                continue
            r = rad * rr * out_cubic(u)
            s = size * sz * math.sin(u * math.pi) ** 0.8
            mg.sparkle(c, (pos[0] + r * math.cos(ang), pos[1] + r * math.sin(ang) * 0.9), s, rot + u * 90,
                       cols[i % len(cols)], 1.0)
    w.front(t0, t0 + dur + 0.3 + (0.7 if hold else 0), fn)


def fx_burst(w, t0, pos, r=240, col=PINK, dur=0.32, rot=0.0, seed=3, outline=WHITE, layer="front"):
    def fn(c, t):
        u = (t - t0) / dur
        if not 0 <= u < 1:
            return
        s = r * (out_back(clamp(u * 2.2)) if u < 0.45 else 1.0) * (1 - 0.35 * clamp((u - 0.45) / 0.55))
        mg.impact_burst(c, pos, s, rot + u * 12, col, outline, alpha=1 - clamp((u - 0.6) / 0.4) ** 2, seed=seed)
    (w.back if layer == "back" else w.front)(t0, t0 + dur, fn)


def fx_halftone(w, t0, pos, r=260, col=PINK, dur=0.5, spacing=22, alpha=0.8, layer="back"):
    def fn(c, t):
        u = (t - t0) / dur
        if not 0 <= u < 1:
            return
        mg.halftone(c, pos, r * (0.6 + 0.4 * out_cubic(u)), spacing, col, alpha * (1 - u) ** 0.8)
    (w.back if layer == "back" else w.front)(t0, t0 + dur, fn)


def fx_confetti(w, t0, origin, n=70, seed=5, hold=None, **kw):
    cf = mg.Confetti(n, origin, seed=seed, **kw)

    def fn(c, t):
        tt = wt(t) if hold else t
        cf.draw(c, tt - t0)
    life = max(q["life"] + q["delay"] for q in cf.p)
    w.front(t0, t0 + life + 0.1 + (THAW - FREEZE if hold else 0), fn)


def fx_speed(w, t0, t1, center, r_in=300, r_out=1300, n=28, col=WHITE, alpha=0.55, seed=2, fade=0.15):
    def fn(c, t):
        a = alpha * min(1.0, (t - t0) / fade, (t1 - t) / fade)
        if a > 0.02:
            mg.speedlines(c, center, n, r_in, r_out, col, a, seed=seed + int(t * 30) % 6)
    w.front(t0, t1, fn)


def flicker(track, t, on=1.0, off=0.0):
    pts = [(0.0, on), (0.045, off + 0.15), (0.09, on), (0.14, off + 0.35), (0.19, on)]
    for (dt, v) in pts:
        track.set(t + dt, v)


# =============================================================================================
# ACT 1 :  0 - 20.22   "the screen wakes up"
# =============================================================================================
def screen_fn(t):
    # --- ACT 1 -----------------------------------------------------------------------------
    if t < A(8):
        return "bars"
    if t < THAW:
        order = ["city_top", "poster_l", "neon", "sunset", "poster_r", "city_low", "neon_low", "sunset_low",
                 "poster_c", "lineup", "sunset", "city_top", "neon", "poster_r", "sunset_low", "poster_l"]
        tw = min(t, FREEZE)
        cut = 2 if tw < A(30) else 1
        beats = [b for b in BA if b >= A(8)]
        # pick the cut index: every `cut` beats from A(8)
        n_cut = int(np.searchsorted(beats, tw, side="right") - 1) // cut
        t_c = beats[n_cut * cut]
        name = order[n_cut % len(order)]
        u = clamp((tw - t_c) / (0.43 * cut))
        build = clamp((tw - A(30)) / (FREEZE - A(30)))
        if t >= FREEZE:                         # the picture freezes with the music and fizzes
            return dict(name=name, zoom=1.0 + 0.12 * u, cx=0.5, cy=0.5, beats=(), glitch=0.35, saturate=0.25,
                        seed=int(t * 12))
        return dict(name=name, zoom=1.0 + 0.12 * u, cx=0.5 + 0.08 * (u - 0.5), cy=0.5, beats=beats, punch=0.07,
                    glitch=0.0 + 0.5 * build ** 2, seed=int(tw * 30))
    return screen_act2(t)


def _cuts(t, t0, every, names, beats=None):
    """Which source is on screen when we cut every `every` beats from beat index t0; returns (name, u)."""
    bt = np.array(beats if beats is not None else BB)
    i0 = int(t0)                                           # t0 is a beat INDEX
    j = int(np.searchsorted(bt, t, side="right") - 1)
    n = max(0, (j - i0) // every)
    tc = bt[min(len(bt) - 1, i0 + n * every)]
    nxt = bt[min(len(bt) - 1, i0 + (n + 1) * every)]
    return names[n % len(names)], clamp((t - tc) / max(0.2, nxt - tc))


def screen_act2(t):
    beats = [b for b in BB]
    if t < B(16):                                           # V1: the cast lineup on its candy sky
        u = clamp((t - THAW) / (B(16) - THAW))
        return dict(name="lineup", zoom=1.0 + 0.1 * u, cx=0.5, cy=0.55, beats=beats, punch=0.03, split=0.0)
    if t < B(32):                                           # V2: city montage, bumped at the carrot hit
        name, u = _cuts(t, 16, 2, ["city_top", "poster_l", "neon", "poster_r", "sunset", "city_low", "neon_low", "poster_c"])
        hit = B(28)
        kw = dict(name=name, zoom=1.0 + 0.1 * u, cx=0.5, cy=0.5, beats=beats, punch=0.05)
        if t >= hit:
            dt = t - hit
            e = math.exp(-dt * 3.2)
            kw.update(zoom=1.32, cx=0.5 + 0.22 * e * math.sin(dt * 22), glitch=min(1.0, 1.4 * e), seed=int(t * 30),
                      flash=0.5 * math.exp(-dt * 12))
        return kw
    if t < B(48):                                           # V3: calm dusk-coloured movie, drifting slowly
        u = clamp((t - B(32)) / (B(48) - B(32)))
        return dict(name="sunset", zoom=1.0 + 0.25 * u, cx=0.5 + 0.1 * u, cy=0.5, beats=[], punch=0.0)
    if t < B(64):                                           # V4: neon, flashing, every beat
        name, u = _cuts(t, 48, 4, ["neon", "sunset", "neon_low", "poster_r"])
        bi = int(np.searchsorted(BB, t, side="right")) - 1
        col = PAL[bi % len(PAL)]
        return dict(name=name, zoom=1.0 + 0.2 * u, cx=0.5, cy=0.5, beats=beats, punch=0.1, split=2.0,
                    tint=(col[0], col[1], col[2], 0.28))
    if t < B(80):                                           # V5: photo booth
        name, u = _cuts(t, 64, 4, ["poster_c", "city_top", "poster_l", "sunset"])
        return dict(name=name, zoom=1.0 + 0.12 * u, cx=0.5, cy=0.5, beats=beats, punch=0.04)
    if t < B(96):                                           # V6: the street
        name, u = _cuts(t, 80, 4, ["city_low", "sunset_low", "neon_low", "city_top"])
        return dict(name=name, zoom=1.0 + 0.1 * u, cx=0.5, cy=0.5, beats=beats, punch=0.05)
    if t < B(112):                                          # V7: solo, then the roar glitches the picture
        TR = B(100)
        if TR <= t < B(104):
            dt = t - TR
            return dict(name="lineup", zoom=1.0 + 0.3 * math.exp(-dt * 2), cx=0.5, cy=0.5, beats=[], glitch=min(1.0, 1.5 * math.exp(-dt * 1.3)),
                        seed=int(t * 30), split=8.0 * math.exp(-dt * 3), saturate=1.0)
        name, u = _cuts(t, 96, 4, ["neon", "lineup", "neon_low", "sunset"])
        return dict(name=name, zoom=1.0 + 0.1 * u, cx=0.5, cy=0.5, beats=beats, punch=0.05,
                    tint=(255, 120, 200, 0.2) if t < TR else None)
    if t < HOLD2 + 0.01:                                    # V8: the picture blooms to white
        u = clamp((t - B(112)) / (HOLD2 - B(112)))
        return dict(name="lineup", zoom=1.0 + 0.4 * u, cx=0.5, cy=0.5, beats=[], glitch=0.15 * u, seed=int(t * 30), flash=0.8 * u ** 2.2 + 0.1)
    if t < BOOM:                                            # the silence: pure white
        return dict(name="lineup", zoom=1.4, cx=0.5, cy=0.5, beats=[], flash=0.95, scan=False)
    # ACT 3: the cast lineup, punching on every beat of the loud section
    bc = [b for b in BC]
    kw = dict(name="lineup", zoom=1.0 + 0.06 * math.sin(t * 2), cx=0.5, cy=0.55, beats=bc, punch=0.09, split=3.0)
    if BOOM <= t < BOOM + 0.5:
        kw.update(flash=0.9 * (1 - (t - BOOM) / 0.5), zoom=1.5 - 0.5 * clamp((t - BOOM) / 0.5))
    return kw


# final stations of the cast (x, y on the floor)
ST = dict(judy=(630, 1565), nick=(370, 1615), lion=(545, 1175), gaz=(195, 1385), big=(755, 1180),
          beav=(860, 1480), claw=(905, 1985), flash=(150, 2015))
SCREEN_EXIT = (470, 868)


def act1(w):
    # ---- lights chase --------------------------------------------------------------------
    w.expo.init = 0.30
    chase = [(A(0), [0], 0.42), (A(1), [1, 2], 0.55), (A(2), [3, 4], 0.70), (A(3), [5], 0.85), (A(4), [6, 7], 1.0)]
    for (tb, ids, ex) in chase:
        for i in ids:
            flicker(w.panels[i], tb)
        w.expo.to(tb, tb + 0.22, ex, out_quad)
        for (px, py) in [w.room.panels[i]["center"] for i in ids]:
            fx_ring(w, tb, (px, py), rmax=170 + 40 * len(ids), col=(255, 250, 235), dur=0.5, thick=7, alpha=0.8)
            fx_sparkles(w, tb + 0.02, (px, py), n=3, rad=70, dur=0.45, size=22, seed=int(tb * 10) + int(px))
    w.punch(A(2), 0.03)
    w.punch(A(4), 0.05)
    w.shake(A(4), amp=6, dur=0.3)
    # opening camera: crane down/out from the ceiling lights to the aisle
    w.zoom.init = 1.15
    w.cy.init = -112
    w.zoom.to(0.0, A(7), 1.0, in_out_cubic)
    w.cy.to(0.0, A(7), 0.0, in_out_cubic)
    w.zoom.to(A(8), FREEZE, 1.05, in_out_quad)

    # ---- projector ----------------------------------------------------------------------
    w.proj.to(A(7), A(7) + 0.18, 1.0, out_quad)
    w.flash(A(7), 0.22, (255, 245, 230), 0.6)
    w.punch(A(7), 0.05)
    w.wide_glow.to(A(7), A(8), 0.25)
    fx_ring(w, A(7), (448, 404), rmax=420, col=(255, 250, 240), dur=0.7, thick=20)
    fx_sparkles(w, A(7), (448, 404), n=8, rad=170, dur=0.8, size=30, seed=11)
    w.screen_fn = screen_fn
    w.flash(A(8), 0.12, (255, 255, 255), 0.35)

    # ---- the carrot rolls in (the unexpected event) --------------------------------------
    carrot = w.add(Actor("carrot", "carrot_b", 0.42, anchor=(0.5, 0.5)))
    c0, c1 = A(4), A(6)
    cpos = (705, 1545)
    carrot.show(c0)
    carrot.move(c0, c1, (1260, cpos[1]), cpos, ease=out_quad)
    carrot.shadow_w = 0.9
    carrot.mod(c0, c1, lambda t, h: dict(rot=-90 + 9 * math.sin((t - c0) * 13), zr=0.17 + 0.09 * abs(math.sin((t - c0) * 7.5))))

    def stand(t, h):
        u = clamp((t - c1) / 0.5)
        e = out_back(u)
        return dict(rot=-90 * (1 - e), zr=lerp(0.17, 0.5, e))
    carrot.mod(c1, c1 + 0.5, stand)
    carrot.mod(c1 + 0.5, A(11), lambda t, h: dict(rot=3.5 * math.sin((t - c1) * 7), zr=0.5))
    carrot.land(c1 + 0.45, 0.12)
    fx_ring(w, c1, cpos, rmax=260, col=ORANGE, dur=0.6, thick=12, floor=True)
    fx_sparkles(w, c1 + 0.05, (cpos[0], cpos[1] - 100), n=5, rad=110, dur=0.6, size=22, seed=21,
                cols=[YELLOW, ORANGE, WHITE])
    w.punch(c1, 0.035)

    # ---- JUDY pops out of the screen and hops to the carrot ---------------------------------
    judy = w.add(Actor("judy", "judy_full", 0.95))
    j0 = A(8)
    judy.show(j0)
    judy.hop(j0, A(9), (470, 858), (505, 1030), 230)
    judy.hop(A(9), A(10), (505, 1030), (500, 1260), 210)
    judy.hop(A(10), A(11), (500, 1260), ST["judy"], 280)
    judy.mod(j0, j0 + 2.4, lambda t, h: dict(sc=1 + 6.5 * math.exp(-(t - j0) / 0.45)))
    judy.trail(j0, A(10), 3, 0.04)
    fx_ring(w, j0, (470, 850), rmax=300, col=PINK, dur=0.5, thick=14)
    fx_burst(w, j0, (470, 830), r=170, col=PINK, dur=0.3, layer="back")
    fx_sparkles(w, j0, (470, 850), n=7, rad=160, dur=0.6, size=26, seed=31)
    w.flash(j0, 0.1, (255, 190, 230), 0.4)
    for (tt, pos) in ((A(9), (505, 1030)), (A(10), (500, 1260))):
        fx_ring(w, tt, pos, rmax=210, col=WHITE, dur=0.45, thick=9, floor=True, alpha=0.7)
    t_pick = A(11)
    carrot.follow(t_pick, 28.2, "judy", off=(0.30, 0, 0.5), rot=18)
    carrot.mod(t_pick, t_pick + 0.3, lambda t, h: dict(sc=1 + 0.35 * math.sin((t - t_pick) / 0.3 * math.pi)))
    carrot.scale(t_pick, t_pick + 0.25, 1.0, 0.74)
    fx_ring(w, t_pick, ST["judy"], rmax=300, col=ORANGE, dur=0.55, thick=14)
    fx_sparkles(w, t_pick, (ST["judy"][0] + 30, ST["judy"][1] - 220), n=8, rad=170, dur=0.7, size=30, seed=41,
                cols=[YELLOW, ORANGE, WHITE, PINK])
    judy.flash(t_pick, 0.1, 0.8)
    w.punch(t_pick, 0.04)

    # ---- NICK slides in from the left and stops dead ------------------------------------------
    nick = w.add(Actor("nick", "nick_walk", 1.3, flip=True))
    n0, n1 = A(12), A(14)
    nick.show(n0)
    nick.move(n0, n1, (-320, 1628), ST["nick"], ease=out_expo)
    nick.mod(n0, n1, lambda t, h: dict(sx=1 + 0.16 * (1 - clamp((t - n0) / (n1 - n0))) ** 1.5,
                                       rot=-4 * (1 - clamp((t - n0) / (n1 - n0)))))
    nick.trail(n0, n1 - 0.2, 4, 0.04)
    nick.walk(n0 + 0.02, n1 - 0.05, step_hz=7.0, bob=0.035, fade=0.05)
    nick.pose(n1, "nick_full", False)
    nick.impact(n1, 0.2, 0.5, flash=0.12)
    nick.mod(n1, n1 + 0.5, lambda t, h: dict(rot=-7 * math.exp(-(t - n1) * 9) * math.cos((t - n1) * 22)))

    def nick_streaks(c, t):
        u = clamp((t - n0) / (n1 - n0))
        a = 0.6 * (1 - u) ** 0.8 * clamp((t - n0) / 0.1)
        x = nick.state(t, w)
        if x and a > 0.03:
            mg.streaks(c, x["x"] - 380, x["x"] + 20, x["y"] - 170, 420, WHITE, a, seed=int(t * 30), n=11,
                       spread=210, width=5)
    w.front(n0, n1, nick_streaks)
    fx_ring(w, n1, ST["nick"], rmax=330, col=ORANGE, dur=0.55, thick=14, floor=True)
    fx_burst(w, n1, (ST["nick"][0] + 10, ST["nick"][1] - 300), r=330, col=ORANGE, dur=0.34, rot=10, seed=8,
             layer="back")
    fx_halftone(w, n1, (ST["nick"][0] + 10, ST["nick"][1] - 300), r=380, col=YELLOW, dur=0.5, spacing=24)
    w.shake(n1, amp=11, dur=0.35)
    w.punch(n1, 0.05)

    # ---- LIONHEART strides out of the screen -----------------------------------------------------
    lion = w.add(Actor("lion", "lion_full", 1.85))
    l0, l1 = A(14), A(19)
    lion.show(l0)
    lion.move(l0, l1, SCREEN_EXIT, ST["lion"], ease=in_out_quad)
    lion.mod(l0, l1, lambda t, h: dict(sc=1 + 2.6 * math.exp(-(t - l0) / 0.9),
                                       zr=0.03 * abs(math.sin((t - l0) / 0.43 * math.pi)),
                                       rot=2.6 * math.sin((t - l0) / 0.43 * math.pi)))
    lion.land(l1, 0.12)
    lion.walk(l0, l1, step_hz=1.0 / 0.4325, bob=0.02, fade=0.2)
    fx_ring(w, l0, (470, 866), rmax=260, col=CYAN, dur=0.5, thick=12)
    fx_sparkles(w, l0, (470, 860), n=6, rad=130, dur=0.6, size=24, seed=51, cols=[CYAN, WHITE, YELLOW])
    for k in range(15, 19):                       # stomps on the way
        fx_ring(w, A(k), (lerp(470, ST["lion"][0], (A(k) - l0) / (l1 - l0)),
                          lerp(868, ST["lion"][1], in_out_quad((A(k) - l0) / (l1 - l0)))), rmax=120 + 30 * (k - 14),
                col=CYAN, dur=0.4, thick=8, floor=True, alpha=0.6)
    fx_ring(w, l1, ST["lion"], rmax=300, col=CYAN, dur=0.55, thick=12, floor=True)
    w.punch(l1, 0.04)

    # ---- CLAWHAUSER photobombs from the bottom edge ---------------------------------------------------
    claw = w.add(Actor("claw", "clawhauser", 0.62, kind="bust", shadow=False))
    k0 = A(19)
    claw.show(k0)
    claw.wave(k0 + 0.4, k0 + 3.0, hz=1.7)
    claw.move(k0, k0 + 0.42, (ST["claw"][0], 2420), ST["claw"], ease=out_back)
    claw.stay(k0 + 0.42, *ST["claw"])
    fx_ring(w, k0 + 0.15, (ST["claw"][0], 1700), rmax=380, col=YELLOW, dur=0.55, thick=14)
    fx_sparkles(w, k0 + 0.2, (ST["claw"][0], 1620), n=7, rad=200, dur=0.6, size=30, seed=61,
                cols=[YELLOW, WHITE, ORANGE])

    # ---- GAZELLE spins in --------------------------------------------------------------------------
    gaz = w.add(Actor("gaz", "gazelle", 1.8))
    g0, g1 = A(20), A(22)
    gaz.show(g0)
    gaz.move(g0, g1, (-200, ST["gaz"][1]), ST["gaz"], ease=out_cubic, arc=70)
    gaz.spin(g0, g1, 2.5, ease=out_cubic)
    gaz.land(g1, 0.14)
    gaz.trail(g0, g1, 3, 0.05)
    fx_ring(w, g1, ST["gaz"], rmax=300, col=MAGENTA, dur=0.55, thick=12, floor=True)
    fx_sparkles(w, g1, (ST["gaz"][0], ST["gaz"][1] - 350), n=9, rad=210, dur=0.8, size=28, seed=71,
                cols=[PINK, WHITE, MAGENTA])

    # ---- MR. BIG poofs in ----------------------------------------------------------------------------
    big = w.add(Actor("big", "mrbig", 0.95))
    b0 = A(23)
    big.pop_in(b0, 0.4)
    big.stay(b0, *ST["big"])
    fx_halftone(w, b0, (ST["big"][0], ST["big"][1] - 100), r=230, col=WHITE, dur=0.55, spacing=20, layer="front")
    fx_ring(w, b0, ST["big"], rmax=250, col=PURPLE, dur=0.55, thick=12, floor=True)

    # ---- BEAVER falls from the ceiling ---------------------------------------------------------------
    beav = w.add(Actor("beav", "beaver", 0.85))
    d0, d1 = A(24) - 0.5, A(24)
    beav.show(d0)
    beav.move(d0, d1, ST["beav"], ST["beav"], arc=2000, arc_ease=lambda u: 1 - in_quad(u))
    beav.land(d1, 0.2, 0.5)
    beav.wave(d1 + 0.3, d1 + 3.0, hz=1.9)
    fx_ring(w, d1, ST["beav"], rmax=330, col=ORANGE, dur=0.55, thick=14, floor=True)
    fx_burst(w, d1, (ST["beav"][0], ST["beav"][1] - 200), r=240, col=YELLOW, dur=0.3, seed=9, layer="back")
    w.shake(d1, amp=9, dur=0.3)
    w.punch(d1, 0.04)

    # ---- the slow ones: turtle crosses the foreground, car zips past, Flash starts to rise ------------
    turt = w.add(Actor("turt", "turtle", 0.4, flip=False))
    turt.show(A(26))
    turt.move(A(26), FREEZE + 2.2, (1260, 1770), (-200, 1770), ease=lin)
    turt.hide(FREEZE + 2.4)
    turt.plod(A(26), FREEZE + 2.2, hz=1.5, bob=0.012)
    turt.mod(A(26), FREEZE + 2.2, lambda t, h: dict(rot=1.6 * math.sin((t - A(26)) * 6.5)))
    car = w.add(Actor("car", "mouse_car", 0.45))
    car.show(A(34))
    car.move(A(34), A(35) + 0.1, (-150, 1060), (1250, 1100), ease=in_quad)
    car.trail(A(34), A(35) + 0.1, 4, 0.03)
    car.hide(A(35) + 0.12)
    flash = w.add(Actor("flash", "flash", 0.62, kind="bust", shadow=False))
    flash.show(A(30))
    flash.move(A(30), FREEZE + 1.5, (ST["flash"][0], 2260), ST["flash"], ease=lin)
    flash.layer = 500

    # ---- group groove (everybody on the beat) ---------------------------------------------------------------
    def dance(actor, start, amp=0.06):
        actor.groove(start, FREEZE, beats=BA, amp=amp, hop=0.05, swing=1.8)
    dance(judy, A(12), 0.07)
    dance(nick, A(15), 0.055)
    dance(lion, A(19), 0.05)
    dance(claw, A(21), 0.07)
    dance(gaz, A(23), 0.06)
    dance(big, A(25), 0.07)
    dance(beav, A(26), 0.07)
    carrot.mod(A(14), FREEZE, lambda t, h: dict(rot=4 * math.sin((t - A(14)) * 13)))
    # side-step flips on the strong beats
    for k, ts in enumerate([A(i) for i in range(14, 47, 4)]):
        if ts > FREEZE - 0.3:
            break
        judy.pose(ts, flip=bool(k % 2))
        gaz.pose(ts, flip=bool((k + 1) % 2))
        beav.pose(ts, flip=bool(k % 2 == 0))
        lion.pose(ts, flip=bool(k % 2))

    # ---- BUILD: soft god rays from the screen + disco pools ------------------------------------------------
    def rays_back(c, t):
        tt = wt(t)
        u = clamp((tt - A(30)) / 1.2)
        col = [PINK, PURPLE, CYAN][int((tt - A(30)) / 0.43 / 2) % 3]
        mg.godrays(c, (468, 700), tt * 16, col, 0.42 * u, n=14, width=0.45)
    w.back(A(30), THAW, rays_back)

    def disco_pools(c, t):
        tt = wt(t)
        u = clamp((tt - A(24)) / 1.5)
        for i, (cx, cy, col) in enumerate(((300, 1480, PINK), (780, 1420, CYAN), (540, 1640, YELLOW))):
            ph = (tt - A(24)) / 0.43 + i * 1.0
            r = 230 + 60 * math.sin(ph * math.pi)
            mg.disc_e(c, (cx, cy), r, col, 0.16 * u, squash=0.3, mode="blend")
    w.back(A(24), THAW, disco_pools)

    # big hit at A(42)
    hit = A(42)
    w.flash(hit, 0.18, WHITE, 0.7)
    w.punch(hit, 0.08, 0.3)
    w.shake(hit, amp=16, dur=0.45)
    fx_ring(w, hit, (540, 1300), rmax=1300, col=WHITE, dur=0.8, thick=36)
    fx_ring(w, hit + 0.06, (540, 1300), rmax=1000, col=PINK, dur=0.8, thick=24)
    fx_confetti(w, hit, (60, 1850), n=60, seed=71, spread=(-1.45, -0.7), speed=(800, 1500), gravity=1700, hold=FREEZE)
    fx_confetti(w, hit, (1020, 1850), n=60, seed=72, spread=(-2.45, -1.7), speed=(800, 1500), gravity=1700, hold=FREEZE)
    for k, tt in enumerate((A(44), A(45), A(46))):
        w.flash(tt, 0.1, [(255, 255, 255), (255, 200, 240), (200, 255, 255)][k], 0.45)
        w.punch(tt, 0.05 + 0.02 * k, 0.2)
        w.shake(tt, amp=9, dur=0.2)
        fx_ring(w, tt, (540, 1300), rmax=700 + 150 * k, col=[CYAN, PINK, YELLOW][k], dur=0.45, thick=22)
    # judy leaps on the last hit and is caught mid-air by the freeze
    jl0, jl1 = 19.62, THAW + 0.6

    def judy_leap(tt):
        u = clamp((wt(tt) - jl0) / (2 * (FREEZE - jl0)))
        return (ST["judy"][0], ST["judy"][1], 230 * 4 * u * (1 - u))
    judy.segs.append((jl0, jl1, judy_leap))
    judy.crouch(jl0 - 0.12, jl0 + 0.02, 0.16)
    judy.star_jump(jl0, THAW + 0.6, clock=wt)
    judy.land(jl1, 0.2)
    # gazelle is caught mid-pirouette (paper-thin) by the freeze
    gz0 = 19.3
    gaz.spin(gz0, 21.30, 1.0, ease=lin, clock=wt)

    # ---- FREEZE ----------------------------------------------------------------------------------------------
    w.sat.to(FREEZE - 0.02, FREEZE + 0.12, 0.18, out_quad)
    w.contrast.to(FREEZE - 0.02, FREEZE + 0.12, 1.12, out_quad)
    w.zoom.to(FREEZE, FREEZE + 0.7, 1.09, out_cubic)


# =============================================================================================
# ACT 2 :  20.91 - 79.2   "the groove"  (8 vignettes of 4 bars)
# =============================================================================================
HM = dict(judy=0.95, nick=1.3, lion=1.85, gaz=1.8, big=0.95, beav=0.85)
FACE = dict(judy=1, nick=1, lion=-1, gaz=1, big=1, beav=1)       # which way each sprite looks by default
PAL = [PINK, CYAN, ORANGE, PURPLE, YELLOW, GREEN, MAGENTA]


def hpx_of(k, y=None):
    x, yy = ST[k]
    return HM[k] * ppm(yy if y is None else y)


def face(actor, key, t, direction):
    """Make `key` look toward +1 (right) / -1 (left) from time t."""
    actor.pose(t, flip=(FACE[key] != direction))


def flight(actor, t0, t1, p0, p1, arc, ease=lin):
    """Carrot-style flight between two (x, y, z) points with a parabolic arc."""
    def fn(tt):
        u = clamp((tt - t0) / (t1 - t0))
        e = ease(u)
        return (lerp(p0[0], p1[0], e), lerp(p0[1], p1[1], e), lerp(p0[2], p1[2], e) + 4 * arc * u * (1 - u))
    actor.segs.append((t0, t1, fn))


def fx_rays(w, t0, t1, center=(468, 700), cols=(PINK, PURPLE, CYAN), alpha=0.3, spin=16.0, n=14, step=2.0,
            fade=0.8):
    def fn(c, t):
        tt = wt(t)
        u = min(clamp((t - t0) / fade), clamp((t1 - t) / fade))
        col = cols[int((tt - t0) / 0.4575 / step) % len(cols)]
        mg.godrays(c, center, tt * spin, col, alpha * u, n=n, width=0.45)
    w.back(t0, t1, fn)


def act2_v1(w):
    a = w.actors
    judy, nick, lion, gaz, big, beav, claw = (a[k] for k in ("judy", "nick", "lion", "gaz", "big", "beav", "claw"))
    # ---- the thaw: colour floods back across the frozen room ----------------------------------------------------
    w.sat.set(THAW, 1.0)
    w.contrast.set(THAW, 1.0)
    w.thaw = (THAW, 0.85, 8)
    w.flash(THAW, 0.18, WHITE, 0.3)
    w.punch(THAW, 0.05)
    w.shake(THAW, amp=8, dur=0.3)
    w.zoom.to(THAW, B(10), 1.0, in_out_cubic)
    fx_ring(w, THAW, (540, 1300), rmax=1200, col=WHITE, dur=0.9, thick=26)
    fx_ring(w, THAW + 0.08, (540, 1300), rmax=900, col=CYAN, dur=0.9, thick=18)
    fx_rays(w, THAW, B(16), alpha=0.22, spin=7)

    # ---- conga ripple: a wave of hops runs along the line, direction alternates every bar ------------------
    order = ["gaz", "nick", "lion", "judy", "big", "beav", "claw"]
    stations = {k: ST[k] for k in order}
    stations["judy"] = ST["judy"]
    hh = {k: (HM.get(k, 0.62) * ppm(ST[k][1])) for k in order}
    hh["claw"] = 0.62 * ppm(ST["claw"][1])

    def ripple(tb, reverse, k_arc, col_shift=0):
        seq = order[::-1] if reverse else order
        for i, k in enumerate(seq):
            t0 = tb + 0.085 * i
            ac = a[k]
            x, y = stations[k]
            ac.hop(t0, t0 + 0.42, (x, y), (x, y), k_arc * hh[k], squash=0.13)
            if k != "claw":
                fx_ring(w, t0 + 0.42, (x, y), rmax=0.42 * hh[k], col=PAL[(i + col_shift) % len(PAL)], dur=0.45,
                        thick=7, floor=True, alpha=0.8)
            fx_sparkles(w, t0, (x, y - 0.55 * hh[k]), n=3, rad=0.25 * hh[k], dur=0.5, size=18, seed=300 + i + int(tb))
    ripple(B(2), False, 0.12)
    ripple(B(6), True, 0.16, 2)
    ripple(B(10), False, 0.2, 4)
    ripple(B(14), True, 0.24, 1)
    for k in ("gaz", "nick", "lion", "big", "beav", "claw"):
        a[k].groove(B(1), B(16), beats=BB, amp=0.035, hop=0.0)
    judy.groove(B(2), B(16), beats=BB, amp=0.035, hop=0.0)
    # judy lands from the frozen leap at THAW+0.6 (see act 1); gentle sway camera
    w.cx.to(THAW, B(8), 30, in_out_quad)
    w.cx.to(B(8), B(16), -30, in_out_quad)
    w.punch(B(6), 0.02)
    w.punch(B(10), 0.025)
    w.punch(B(14), 0.03)
    w.shake(B(14), amp=6, dur=0.3)
    # flash keeps inching up, the turtle leaves the frame
    a["flash"].move(21.72, B(16), (ST["flash"][0], ST["flash"][1]), (ST["flash"][0], ST["flash"][1] - 40), ease=lin)


def act2_v2(w):
    """CARROT PASS: round 1 one pass per beat, round 2 per half beat, then the throw that bumps the projector."""
    a = w.actors
    car = a["carrot"]
    holders = ["judy", "nick", "gaz", "lion", "big", "beav"]

    def hand(k):
        x, y = ST[k]
        h = hpx_of(k)
        return (x, y + 0.5, 0.52 * h)

    def reaction(k, tc, sender):
        ac = a[k]
        ac.impact(tc, 0.2, 0.4)
        d = 1 if ST[sender][0] > ST[k][0] else -1
        face(ac, k, tc, d)
        x, y, z = hand(k)
        fx_sparkles(w, tc, (x, y - z), n=4, rad=100, dur=0.45, size=22, seed=int(tc * 10) % 97,
                    cols=[YELLOW, ORANGE, WHITE])
        fx_ring(w, tc, ST[k], rmax=0.35 * hpx_of(k), col=ORANGE, dur=0.4, thick=7, floor=True, alpha=0.8)

    def throw_kick(k, tt, toward):
        a[k].mod(tt - 0.02, tt + 0.28, lambda t, h, tt=tt, toward=toward: dict(
            rot=toward * 9 * math.sin((t - tt) / 0.28 * math.pi) * (1 if t >= tt else 0.0)))

    car.hide(B(16) - 0.5)
    car.show(B(16) - 0.5)
    # the camera chases the carrot: a gentle push-in, panning toward each catcher
    w.zoom.to(B(16) - 0.4, B(17), 1.14, in_out_cubic)
    for i in range(6):
        rcv = holders[(i + 1) % 6]
        w.cx.to(B(16 + i), B(17 + i), clamp((ST[rcv][0] - 540) * 0.22, -60, 60), in_out_cubic)
        w.cy.to(B(16 + i), B(17 + i), clamp((ST[rcv][1] - 1350) * 0.12, -60, 60), in_out_cubic)
    for i in range(6):
        rcv = holders[(i + 1) % 6]
        h2 = (B(23) - B(22)) / 2
        w.cx.to(B(22) + h2 * i, B(22) + h2 * (i + 1), clamp((ST[rcv][0] - 540) * 0.22, -60, 60), in_out_quad)
    w.cx.to(B(25), B(26), 0, in_out_cubic)
    w.cy.to(B(25), B(26), 20, in_out_cubic)
    w.zoom.to(B(26), B(27), 1.2, in_out_cubic)
    # ---- round 1 : one pass per beat --------------------------------------------------------------------------
    t_prev = B(16)
    for i in range(6):
        snd, rcv = holders[i], holders[(i + 1) % 6]
        t0, t1 = B(16 + i) + 0.10, B(17 + i)
        p0, p1 = hand(snd), hand(rcv)
        flight(car, t0, t1, p0, p1, 240 + 20 * i)
        car.follow(B(17 + i), B(17 + i) + 0.1, rcv, off=(0.28, 0, 0.5), rot=18)
        car.mod(t0, t1, lambda t, h, t0=t0, t1=t1: dict(rot=360 * clamp((t - t0) / (t1 - t0)) * 1.0))
        car.trail(t0, t1, 3, 0.03)
        throw_kick(snd, B(16 + i), 1 if ST[rcv][0] > ST[snd][0] else -1)
        reaction(rcv, B(17 + i), snd)
        if i % 2 == 0:
            w.punch(B(17 + i), 0.02, 0.2)
    # first pass leaves judy's hand
    car.follow(B(16) - 0.5, B(16) + 0.1, "judy", off=(0.30, 0, 0.5), rot=18)
    # ---- round 2 : double time ------------------------------------------------------------------------------------
    half = (B(23) - B(22)) / 2
    for i in range(6):
        snd, rcv = holders[i], holders[(i + 1) % 6]
        t0, t1 = B(22) + half * i, B(22) + half * (i + 1)
        flight(car, t0, t1, hand(snd), hand(rcv), 200)
        car.mod(t0, t1, lambda t, h, t0=t0, t1=t1: dict(rot=360 * clamp((t - t0) / (t1 - t0))))
        car.trail(t0, t1, 3, 0.02)
        reaction(rcv, t1, snd)
        w.punch(t1, 0.025, 0.14)
    # after round 2 the carrot lands back with judy
    t_r2 = B(22) + half * 6
    car.follow(t_r2, B(27), "judy", off=(0.30, 0, 0.5), rot=18)
    # ---- the pause ---------------------------------------------------------------------------------------------------
    for k in holders:
        d = 1 if ST["judy"][0] > ST[k][0] else -1
        if k != "judy":
            face(a[k], k, B(25), d)
    for k in ("nick", "lion"):
        a[k].wobble(B(25), B(26), 2.0, 5.0)
    # ---- winding up and the throw at the projector ------------------------------------------------------------------
    j = a["judy"]
    j.crouch(B(26), B(27) - 0.02, 0.22)
    j.stretch(B(27), B(27) + 0.2, 0.16)
    hand_j = hand("judy")
    target = (448, 905, 501)
    t_th, t_hit = B(27), B(28)
    flight(car, t_th, t_hit, hand_j, target, 120, ease=in_quad)
    car.scale(t_th, t_hit, 1.0, 3.2, ease=in_quad)
    car.mod(t_th, t_hit, lambda t, h: dict(rot=720 * clamp((t - t_th) / (t_hit - t_th))))
    car.trail(t_th, t_hit, 4, 0.03)
    car.follow(B(26) - 0.0, t_th, "judy", off=(0.30, 0, 0.5), rot=18)
    # impact at the projector
    w.flash(t_hit, 0.2, WHITE, 0.8)
    w.punch(t_hit, 0.09, 0.3)
    w.shake(t_hit, amp=18, dur=0.5)
    fx_ring(w, t_hit, (448, 404), rmax=700, col=WHITE, dur=0.8, thick=30, alpha=0.95)
    fx_ring(w, t_hit + 0.05, (448, 404), rmax=520, col=ORANGE, dur=0.7, thick=20)
    fx_burst(w, t_hit, (448, 404), r=170, col=YELLOW, dur=0.35, seed=5)
    fx_sparkles(w, t_hit, (448, 404), n=12, rad=260, dur=0.9, size=34, seed=91)
    for i in range(6):                         # lights stutter
        flicker(w.panels[i], t_hit + 0.02 * i, 1.0, 0.0)
    # carrot ricochets down the aisle and bounces to rest
    t_b = [t_hit, B(29), B(29) + 0.28, B(29) + 0.5]
    land = (560, 1470 + 70, 0)
    flight(car, t_hit, B(29), (448, 905, 501), (540, 1520, 0), 230, ease=lin)
    car.scale(t_hit, B(29), 3.2, 1.0, ease=out_quad)
    car.mod(t_hit, B(29), lambda t, h: dict(rot=360 * clamp((t - t_hit) / (B(29) - t_hit))))
    flight(car, B(29), B(29) + 0.3, (540, 1520, 0), (590, 1520, 0), 110)
    flight(car, B(29) + 0.3, B(29) + 0.52, (590, 1520, 0), (620, 1520, 0), 50)
    car.mod(B(29), B(29) + 0.52, lambda t, h: dict(rot=0))
    car.mod(B(29) + 0.52, B(31), lambda t, h: dict(zr=0.17, rot=-90 + 4 * math.sin((t - B(29)) * 9) * math.exp(-(t - B(29) - 0.5))))
    # everyone startles
    for i, k in enumerate(holders):
        fx_exclaim(w, t_hit + 0.06 + 0.035 * i, (ST[k][0] + 20, ST[k][1] - 1.12 * hpx_of(k)), size=0.26 * hpx_of(k) + 70)
    for k in holders:
        ac = a[k]
        ac.mod(t_hit, t_hit + 0.35, lambda t, h, t_hit=t_hit: dict(
            sy=1 + 0.13 * math.sin(clamp((t - t_hit) / 0.35) * math.pi), sx=1 - 0.09 * math.sin(clamp((t - t_hit) / 0.35) * math.pi)))
        ac.shake(t_hit, t_hit + 0.4, 5)
    # the carrot wiggles its way out and everybody exhales
    for k in holders:
        a[k].groove(B(29) + 0.6, B(32), beats=BB, amp=0.03, hop=0.0)
    w.rot.to(t_hit, t_hit + 0.2, 2.5, out_quad)
    w.rot.to(t_hit + 0.2, B(30), 0.0, in_out_cubic)


def fx_exclaim(w, t0, pos, size=120, dur=0.9, col=None):
    """Comic '!' that pops over a startled head, wobbles and pops away."""
    colr = col or (40, 200, 255)

    def fn(c, t):
        u = (t - t0) / dur
        if not 0 <= u < 1:
            return
        sc = out_back(clamp(u / 0.18)) * (1 - in_quad(clamp((u - 0.78) / 0.22)))
        mg.exclaim(c, (pos[0] + 6 * math.sin(u * 34), pos[1] - 16 * math.sin(clamp(u / 0.25) * 1.57)), size * sc, 1.0, colr)
    w.front(t0, t0 + dur, fn)


def fx_sweat(w, t0, pos, size=16, dur=0.7):
    def fn(c, t):
        u = (t - t0) / dur
        if not 0 <= u < 1:
            return
        mg.teardrop(c, (pos[0], pos[1] + 70 * in_quad(u)), size * (1 - 0.3 * u), (255, 215, 120), alpha=1 - u ** 2)
    w.front(t0, t0 + dur, fn)


def act2_v3(w):
    """SLOTH TIME: everybody waits for Flash while the room lives through a day/night time-lapse; his greeting is
    so slow that the pay-off lands exactly on the downbeat of the next section."""
    a = w.actors
    fl, car = a["flash"], a["carrot"]
    t0, t1 = B(32), B(48)
    # camera: dive to the bottom-left where the sloth is, then drift across the waiting crowd and come back
    w.zoom.to(t0 - 0.2, t0 + 0.55, 1.3, in_out_cubic)
    w.cx.to(t0 - 0.2, t0 + 0.55, -125, in_out_cubic)
    w.cy.to(t0 - 0.2, t0 + 0.55, 200, in_out_cubic)
    w.cx.to(B(36), B(44), 90, in_out_quad)
    w.cy.to(B(36), B(44), 60, in_out_quad)
    w.zoom.to(B(36), B(44), 1.2, in_out_quad)
    w.zoom.to(B(44), B(47), 1.38, in_out_cubic)
    w.cx.to(B(44), B(47), -125, in_out_cubic)
    w.cy.to(B(44), B(47), 215, in_out_cubic)
    w.rot.to(t0, B(40), 0, lin)
    # flash climbs
    fl.move(B(32), B(47), (ST["flash"][0], ST["flash"][1] - 40), (ST["flash"][0] + 10, 1925), ease=lin)
    fl.groove(B(32), B(44), beats=BB, amp=0.012, hop=0.0, damp=4)
    fl.mod(B(44), B(47), lambda t, h: dict(rot=14 * smooth(clamp((t - B(44)) / (B(47) - B(44)))) * 1.0,
                                           sy=1 + 0.03 * smooth(clamp((t - B(44)) / (B(47) - B(44))))))
    fl.mod(B(47), B(48), lambda t, h: dict(rot=14 - 40 * clamp((t - B(47)) / 0.2) if t < B(47) + 0.2 else -26 + 26 * clamp((t - B(47) - 0.2) / 0.25) ** 0.5 if t < B(47) + 0.45 else 0,
                                           sc=1 + 0.05 * math.sin(clamp((t - B(47)) / 0.4) * math.pi)))
    # the carrot lies where it fell
    car.mod(B(31), B(48), lambda t, h: dict(zr=0.17, rot=-90))
    # impatience: a different fidget for each character, double time
    sub = []
    for i in range(32, 48):
        sub += [B(i), (B(i) + B(i + 1)) / 2]
    sub = np.array(sub)
    fid = dict(nick=dict(amp=0.03, swing=3.5), lion=dict(amp=0.02, swing=1.2), beav=dict(amp=0.08, swing=2.0),
               big=dict(amp=0.05, swing=4.0), gaz=dict(amp=0.025, swing=2.5), judy=dict(amp=0.04, swing=3.0))
    for k, f in fid.items():
        a[k].groove(B(32), B(47), beats=sub, amp=f["amp"], hop=0.0, swing=f["swing"], damp=9)
    for i in range(32, 47):
        face(a["judy"], "judy", B(i), 1 if i % 2 else -1)           # looks left, right, left ...
    face(a["nick"], "nick", B(32), -1)
    face(a["gaz"], "gaz", B(32), -1)
    face(a["lion"], "lion", B(32), 1)
    for k in ("beav", "big"):
        face(a[k], k, B(32), -1)
    # sweat drops
    heads = dict(nick=(ST["nick"][0], ST["nick"][1] - 1.15 * hpx_of("nick")), lion=(ST["lion"][0] + 90, ST["lion"][1] - 0.95 * hpx_of("lion")),
                 beav=(ST["beav"][0] + 60, ST["beav"][1] - 0.85 * hpx_of("beav")), big=(ST["big"][0] + 60, ST["big"][1] - 0.8 * hpx_of("big")),
                 gaz=(ST["gaz"][0] + 110, ST["gaz"][1] - 1.7 * hpx_of("gaz")), judy=(ST["judy"][0] + 50, ST["judy"][1] - 0.95 * hpx_of("judy")))
    names = list(heads)
    for i in range(34, 46):
        k = names[i % len(names)]
        fx_sweat(w, B(i), heads[k], size=15)

    # the clock + day/night time-lapse
    def clock_fx(c, t):
        u = clamp((t - t0) / 0.5)
        v = clamp((B(47) - t) / 0.3)
        al = min(u, v)
        if al <= 0.01:
            return
        k = (t - t0) / (B(47) - t0)
        sc = out_back(clamp((t - t0) / 0.45)) * (1 - 0.35 * (1 - al))
        r = 150 * sc
        beat_i = int(np.searchsorted(BB, t, side="right")) - 1
        dt = t - BB[beat_i]
        mg.clock(c, (540, 250), r * (1 + 0.04 * math.exp(-dt * 9)), k * 360 * 1.0, k * 360 * 12, alpha=al)
        # little ticking ring on each beat
        if dt < 0.4:
            mg.ring(c, (540, 250), r * (1.1 + 0.9 * dt / 0.4), 4, CYAN, 0.7 * (1 - dt / 0.4))
    w.over(t0, B(47) + 0.1, clock_fx)
    # day -> dusk -> night -> dawn
    w.expo.to(t0, B(36), 0.78, in_out_quad)
    w.warm.to(t0, B(36), 1.0, in_out_quad)
    w.expo.to(B(36), B(40), 0.55, in_out_quad)
    w.warm.to(B(36), B(40), -1.0, in_out_quad)
    w.expo.to(B(40), B(44), 0.9, in_out_quad)
    w.warm.to(B(40), B(44), 0.6, in_out_quad)
    w.expo.to(B(44), B(47), 1.0, in_out_quad)
    w.warm.to(B(44), B(47), 0.0, in_out_quad)
    # the greeting lands: ding
    tg = B(47)
    fl.wave(tg - 0.1, tg + 2.2, hz=1.5)
    w.punch(tg, 0.04)
    fx_ring(w, tg, (150, 1700), rmax=420, col=YELLOW, dur=0.6, thick=16)
    fx_sparkles(w, tg, (150, 1560), n=10, rad=260, dur=0.8, size=34, seed=611, cols=[YELLOW, WHITE, ORANGE])
    # ...then total stillness for one beat before the party
    for k in ("nick", "lion", "beav", "big", "gaz", "judy"):
        face(a[k], k, B(47), -1 if k != "lion" else 1)


def mirror_ball(c, center, r, ang, t):
    """Disco ball made of drawn facets (spins with `ang`)."""
    cv2_ = mg.cv2
    mg.poly(c, [(center[0] - 3, 0), (center[0] + 3, 0), (center[0] + 3, center[1] - r), (center[0] - 3, center[1] - r)],
            (200, 200, 210), 0.9)
    mg.circle(c, center, r, (190, 195, 205))
    # facets: longitudes squeeze as the ball turns
    for i in range(6):
        ph = ang + i * 30
        rx = abs(math.cos(math.radians(ph))) * r
        if rx > 2:
            cv2_.ellipse(c, (int(center[0]), int(center[1])), (int(rx), int(r)), 0, 0, 360, (120, 125, 140), 2, cv2_.LINE_AA)
    for j in (-0.55, -0.2, 0.2, 0.55):
        yy = center[1] + j * r
        half = math.sqrt(max(0, 1 - j * j)) * r
        cv2_.line(c, (int(center[0] - half), int(yy)), (int(center[0] + half), int(yy)), (120, 125, 140), 2, cv2_.LINE_AA)
    mg.circle(c, (center[0] - r * 0.3, center[1] - r * 0.3), r * 0.22, (255, 255, 255), 0.55)
    for i in range(7):
        a_ = ang * 0.05 + i * 0.9 + t * 3
        mg.sparkle(c, (center[0] + r * 0.8 * math.cos(a_) * math.cos(i), center[1] + r * 0.8 * math.sin(a_ * 1.3)),
                   9 + 5 * math.sin(t * 9 + i), i * 20, WHITE, 0.9 if math.sin(t * 6 + i * 2) > 0 else 0.0)


def act2_v4(w):
    """DISCO: lights down, mirror ball up; Judy & Nick side-step in mirror, everybody gets a move per bar."""
    a = w.actors
    judy, nick, lion, gaz, big, beav, claw, car = (a[k] for k in ("judy", "nick", "lion", "gaz", "big", "beav", "claw", "carrot"))
    t0, t1 = B(48), B(64)
    # transition wipe + party lights
    w.over(t0 - 0.45, t0 + 0.5, lambda c, t: mg.wipe(c, (t - (t0 - 0.45)) / 0.95, [PINK, CYAN, PURPLE, ORANGE, YELLOW], angle=-22))
    w.expo.to(t0 - 0.05, t0 + 0.2, 0.62, out_quad)
    w.warm.to(t0 - 0.05, t0 + 0.2, -0.45, out_quad)
    w.zoom.to(t0 - 0.3, t0 + 0.4, 1.0, in_out_cubic)
    w.cx.to(t0 - 0.3, t0 + 0.4, 0, in_out_cubic)
    w.cy.to(t0 - 0.3, t0 + 0.4, 0, in_out_cubic)
    w.rot.to(t0, t0 + 0.2, 0, lin)
    w.flash(t0 + 0.02, 0.25, WHITE, 0.6)
    w.punch(t0 + 0.02, 0.07, 0.3)
    w.shake(t0 + 0.02, amp=14, dur=0.4)
    # camera sway on the beat
    for i in range(48, 64):
        w.rot.to(B(i), B(i + 1), (1.6 if i % 2 else -1.6), in_out_quad)
    w.zoom.to(t0 + 0.4, t1 - 0.4, 1.07, in_out_quad)
    for i in range(48, 64, 2):
        w.punch(B(i), 0.015, 0.2)

    # party lighting: sweeping spotlights from the two ceiling panels, mirror ball with an EQ ring
    def party_back(c, t):
        u = clamp((t - t0) / 0.5) * clamp((t1 + 0.2 - t) / 0.4)
        if u <= 0:
            return
        tt = t - t0
        for k, (cx, cy, col) in enumerate(((150, 405, PINK), (780, 402, CYAN), (460, 455, YELLOW))):
            sweep = 25 * math.sin(tt * 1.7 + k * 2.1)
            mg.godrays(c, (cx, cy), 90 + sweep + (-30 if k == 0 else 30 if k == 1 else 0), col, 0.55 * u, n=1,
                       width=0.09, falloff=1700, blur=9)
        # floor pools follow the beat
        bi = int(np.searchsorted(BB, t, side="right")) - 1
        for i, (cx, cy) in enumerate(((300, 1480), (780, 1420), (540, 1660), (200, 1560), (880, 1640))):
            col = PAL[(bi + i) % len(PAL)]
            dt = t - BB[bi]
            r = 200 + 70 * math.exp(-dt * 7)
            mg.disc_e(c, (cx, cy), r, col, 0.18 * u, squash=0.3)
    w.back(t0 - 0.2, t1 + 0.4, party_back)

    def ball_fx(c, t):
        u = clamp((t - t0) / 0.4) * clamp((t1 + 0.1 - t) / 0.3)
        if u <= 0:
            return
        yy = 255 * out_back(clamp((t - t0) / 0.5))
        ctr = (540, yy)
        tt = t - t0
        # EQ ring
        n = 40
        for i in range(n):
            ang = i * 2 * math.pi / n + tt * 0.4
            v = (0.55 * feat("low", t) + 0.3 * feat("mid", t) + 0.2 * feat("high", t)) * (0.55 + 0.45 * math.sin(tt * 9 + i * 1.7)) ** 2
            ln = 18 + 78 * clamp(v)
            r0 = 92
            p0 = (ctr[0] + r0 * math.cos(ang), ctr[1] + r0 * math.sin(ang))
            p1 = (ctr[0] + (r0 + ln) * math.cos(ang), ctr[1] + (r0 + ln) * math.sin(ang))
            mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            mg.rect_rot(c, mid, ln, 9, math.degrees(ang), PAL[i % len(PAL)], 0.95 * u)
        mirror_ball(c, ctr, 72, tt * 90, t)
        # room dots from the ball
        for i in range(26):
            ang = i * 0.62 + tt * 0.9
            rr = 320 + 520 * ((i * 37) % 11) / 11
            px, py = 540 + rr * math.cos(ang), 1000 + 0.9 * rr * math.sin(ang * 1.0) * 0.8 + 300
            if 0 < px < W and 400 < py < 1900:
                s_ = 7 + 4 * math.sin(tt * 7 + i)
                mg.circle(c, (px, py), s_, WHITE, 0.55 * u)
    w.front(t0 - 0.1, t1 + 0.3, ball_fx)

    # screen: neon colours flashing
    # ---- choreography ------------------------------------------------------------------------------------------------
    jx, jy = ST["judy"]
    nx, ny = ST["nick"]
    # bar 1: mirrored side-steps
    pj, pn = (jx, jy), (nx, ny)
    for i in range(48, 52):
        sgn = 1 if i % 2 == 0 else -1
        qj, qn = (jx + sgn * 45, jy), (nx - sgn * 45, ny)
        judy.hop(B(i), B(i + 1), pj, qj, 0.14 * hpx_of("judy"), squash=0.1)
        nick.hop(B(i), B(i + 1), pn, qn, 0.1 * hpx_of("nick"), squash=0.08)
        face(judy, "judy", B(i), 1 if sgn > 0 else -1)
        face(nick, "nick", B(i), -1 if sgn > 0 else 1)
        pj, pn = qj, qn
    for k in ("lion", "gaz", "big", "beav", "claw"):
        a[k].groove(B(48), B(52), beats=BB, amp=0.08, hop=0.04, swing=3.0, damp=6)
    lion.wobble(B(48), B(52), 4.0, 0.55)
    # bar 2: spins
    judy.hop(B(52), B(54), pj, (jx, jy), 0.2 * hpx_of("judy"), squash=0.1)
    judy.spin(B(52), B(54), 2.0, ease=in_out_quad)
    nick.hop(B(52), B(54), pn, (nx, ny), 0.14 * hpx_of("nick"), squash=0.1)
    nick.spin(B(52) + 0.2, B(54) + 0.2, -2.0, ease=in_out_quad)
    gaz.spin(B(52), B(55), 3.0, ease=in_out_cubic)
    gaz.hop(B(52), B(55), ST["gaz"], ST["gaz"], 0.1 * hpx_of("gaz"))
    for k, (hh, dl) in {"beav": (0.2, 0.0), "big": (0.16, 0.1)}.items():
        a[k].hop(B(53) + dl, B(54) + dl, ST[k], ST[k], hh * hpx_of(k), squash=0.14)
        a[k].hop(B(54) + dl, B(55) + dl, ST[k], ST[k], hh * hpx_of(k), squash=0.14)
    claw.groove(B(52), B(56), beats=BB, amp=0.08, hop=0.05, swing=4.0)
    lion.groove(B(52), B(56), beats=BB, amp=0.06, hop=0.03, swing=2.0)
    # bar 3: pose-swaps on the beat + a tilt wave
    for i in range(56, 60):
        if i % 2 == 0:
            nick.pose(B(i), "nick_full", False)
        else:
            nick.pose(B(i), "nick_walk", True)
        judy.pose(B(i), flip=bool(i % 2))
        nick.impact(B(i), 0.12, 0.3)
        judy.impact(B(i), 0.12, 0.3)
    nick.pose(B(60), "nick_full", False)
    for j, k in enumerate(("gaz", "lion", "big", "beav", "claw", "judy", "nick")):
        ph = j * 0.7
        a[k].mod(B(56), B(60), lambda t, h, ph=ph: dict(rot=7 * math.sin((t - B(56)) / (2 * 0.4575) * 2 * math.pi - ph)))
    for i in (56, 58):
        fx_ring(w, B(i), (540, 1300), rmax=900, col=PAL[i % 7], dur=0.6, thick=22)
    # bar 4: rising hops, everyone together
    for j, i in enumerate(range(60, 64)):
        for k in ("judy", "nick", "lion", "gaz", "big", "beav"):
            x, y = ST[k]
            a[k].hop(B(i), B(i + 1), (x, y), (x, y), (0.12 + 0.07 * j) * hpx_of(k), squash=0.12 + 0.02 * j)
        a["claw"].hop(B(i), B(i + 1), ST["claw"], ST["claw"], (0.05 + 0.03 * j) * 480, squash=0.1)
        fx_ring(w, B(i + 1), (540, 1380), rmax=700 + 150 * j, col=PAL[(i + 2) % 7], dur=0.5, thick=18)
        w.punch(B(i + 1), 0.02 + 0.015 * j, 0.18)
    w.flash(B(63), 0.1, WHITE, 0.25)


def viewfinder(c, t, t0, t1, flashes, locked_col=GREEN):
    """Camera viewfinder brackets + focus box in screen space."""
    u = clamp((t - t0) / 0.3) * clamp((t1 - t) / 0.3)
    if u <= 0:
        return
    bi = int(np.searchsorted(BB, t, side="right")) - 1
    dt = t - BB[bi]
    pad = 70 - 16 * (1 - u) + 6 * math.exp(-dt * 9)
    ln, th = 110, 9
    col = (250, 250, 250)
    for (sx, sy) in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        x0 = pad if sx > 0 else W - pad
        y0 = pad if sy > 0 else H - pad
        mg.poly(c, [(x0, y0), (x0 + sx * ln, y0), (x0 + sx * ln, y0 + sy * th), (x0 + sx * th, y0 + sy * th),
                    (x0 + sx * th, y0 + sy * ln), (x0, y0 + sy * ln)], col, 0.9 * u)
    # focus box that locks on at each countdown beat
    near = min([abs(t - f) for f in flashes])
    box = 190 + 60 * clamp(near / 0.9)
    lock = near < 0.5
    bc = locked_col if lock else (250, 250, 250)
    cx, cy = 540, 1060
    for (sx, sy) in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        x0, y0 = cx + sx * box / 2, cy + sy * box / 2
        mg.poly(c, [(x0, y0), (x0 - sx * 40, y0), (x0 - sx * 40, y0 - sy * 5), (x0 - sx * 5, y0 - sy * 5),
                    (x0 - sx * 5, y0 - sy * 40), (x0, y0 - sy * 40)], bc, 0.85 * u)


def act2_v5(w):
    """PHOTO BOOTH: countdown, flash, polaroid flies to the shelf along the top. Four different poses."""
    a = w.actors
    judy, nick, lion, gaz, big, beav, claw, car = (a[k] for k in ("judy", "nick", "lion", "gaz", "big", "beav", "claw", "carrot"))
    t0, t1 = B(64), B(80)
    F = [B(67), B(71), B(75), B(79)]
    # back to the bright room
    w.expo.to(t0 - 0.05, t0 + 0.2, 1.0, out_quad)
    w.warm.to(t0 - 0.05, t0 + 0.2, 0.0, out_quad)
    w.rot.to(t0 - 0.05, t0 + 0.15, 0.0, out_quad)
    w.zoom.to(t0 - 0.05, t0 + 0.3, 1.0, out_cubic)
    w.flash(t0, 0.3, WHITE, 0.7)
    w.punch(t0, 0.05)
    w.over(t0, t1, lambda c, t: viewfinder(c, t, t0, t1, F))

    # judy takes the carrot again
    car.follow(B(63), B(72), "judy", off=(0.30, 0, 0.5), rot=18)
    car.follow(B(76), B(84), "judy", off=(0.30, 0, 0.5), rot=18)
    car.hide(B(72))
    car.show(B(76))

    # bust close-ups for photo 3
    jb = w.add(Actor("judy_b", "judy_arms", 0.78, kind="bust", shadow=False))
    nb = w.add(Actor("nick_b", "nick_arms", 0.8, kind="bust", shadow=False))
    jb.layer = 400
    nb.layer = 400
    jb.show(B(72))
    nb.show(B(72))
    jb.move(B(72), B(72) + 0.45, (520, 2650), (520, 2000), ease=out_back)
    nb.move(B(72), B(72) + 0.45, (780, 2750), (780, 2030), ease=out_back)
    jb.stay(B(72) + 0.45, 520, 2000)
    nb.stay(B(72) + 0.45, 780, 2030)
    jb.groove(B(72) + 0.5, B(76), beats=BB, amp=0.03, hop=0.0, swing=2.0)
    nb.groove(B(72) + 0.5, B(76), beats=BB, amp=0.03, hop=0.0, swing=2.0)
    jb.hide(B(76))
    nb.hide(B(76))
    for ac in (judy, nick):
        ac.hide(B(72))
        ac.show(B(76))
    fx_halftone(w, B(72), (ST["judy"][0], ST["judy"][1] - 250), r=300, col=PINK, dur=0.5, layer="front")
    fx_halftone(w, B(72), (ST["nick"][0], ST["nick"][1] - 350), r=360, col=CYAN, dur=0.5, layer="front")
    fx_halftone(w, B(76), (300, 1700), r=300, col=YELLOW, dur=0.5, layer="front")
    fx_halftone(w, B(76), (780, 1650), r=330, col=PURPLE, dur=0.5, layer="front")

    cast = ["judy", "nick", "lion", "gaz", "big", "beav"]
    # photo 1: proud group pose (nothing special, they pop to attention)
    for k in cast:
        a[k].impact(B(64) + 0.05, 0.14, 0.4)
        a[k].groove(B(65), B(68), beats=BB, amp=0.02, hop=0.0, damp=5)
    # photo 2: everybody in the air at the flash
    rots = dict(judy=-8, nick=6, lion=-5, gaz=9, big=-7, beav=8)
    for k in cast:
        x, y = ST[k]
        tj = F[1] - 0.26
        a[k].crouch(tj - 0.14, tj, 0.18)
        a[k].hop(tj, F[1] + 0.26, (x, y), (x, y), 0.34 * hpx_of(k), squash=0.1)
        a[k].mod(tj, F[1] + 0.26, lambda t, h, k=k, tj=tj: dict(rot=rots[k] * math.sin(clamp((t - tj) / 0.52) * math.pi)))
    claw.hop(F[1] - 0.26, F[1] + 0.26, ST["claw"], ST["claw"], 150, squash=0.1)
    # photo 3: the others stand behind the busts
    for k in ("lion", "gaz", "big", "beav"):
        a[k].groove(B(72), B(76), beats=BB, amp=0.04, hop=0.0, swing=2.0)
    face(lion, "lion", B(72), -1)
    # photo 4: chaos - everybody tilted, flipping
    rots4 = dict(judy=-14, nick=12, lion=-10, gaz=16, big=-12, beav=14)
    for k in cast:
        a[k].mod(B(77), F[3] + 0.3, lambda t, h, k=k: dict(rot=rots4[k] * smooth(clamp((t - B(77)) / 0.25))))
        a[k].impact(B(77), 0.14, 0.4)
        face(a[k], k, B(77), 1 if k in ("judy", "gaz", "beav") else -1)

    # countdown + flash + polaroid
    slots = [(165, 250, -5.0), (385, 235, 3.5), (605, 250, -3.0), (825, 238, 5.5)]
    scale_final = 0.37
    for k, f in enumerate(F):
        for j in range(3):                                   # 3, 2, 1 dots
            tb = B(64 + 4 * k + j)
            def dots(c, t, tb=tb, j=j, k=k):
                dt = t - tb
                if not 0 <= dt < (B(65 + 4 * k + j) - tb):
                    return
                n_lit = 3 - j
                for m in range(3):
                    cx = 540 + (m - 1) * 120
                    lit = m < n_lit
                    r = 34 if lit else 20
                    pop = 1 + 0.35 * math.exp(-dt * 9) if (m == n_lit - 1) else 1.0
                    mg.circle(c, (cx, 640), r * pop + 7, WHITE, 0.95)
                    mg.circle(c, (cx, 640), r * pop, (60, 60, 235) if lit else (170, 170, 170), 1.0)
            w.over(tb, B(65 + 4 * k + j), dots)
        w.flash(f, 0.4, WHITE, 1.0)
        w.punch(f, 0.05, 0.25)
        fx_ring(w, f, (540, 1000), rmax=1100, col=WHITE, dur=0.5, thick=30)
        fx_sparkles(w, f, (540, 1000), n=10, rad=480, dur=0.7, size=44, seed=700 + k, cols=[WHITE, YELLOW, CYAN, PINK])
        sx, sy, sr = slots[k]

        def polaroid(c, t, f=f, sx=sx, sy=sy, sr=sr, k=k):
            ts = f + 0.06
            u = (t - ts) / 0.55
            if u < 0:
                return
            ex = clamp(u)
            e = in_out_cubic(ex) if ex < 1 else 1.0
            e2 = out_back(clamp((u - 0.0) / 1.0), 1.15)
            # leave on the way out
            leave = clamp((t - (B(79) + 0.5)) / 0.35) if k < 4 else 0
            scale = lerp(1.9, scale_final, e2 if e2 < 1.2 else 1.0) if False else lerp(1.85, scale_final, e)
            cx = lerp(540, sx, e)
            cy = lerp(930, sy, e) - 700 * out_cubic(leave) * 0 - 520 * in_quad(leave)
            rot = lerp(0, sr, e) + 25 * in_quad(leave) * (1 if k % 2 else -1)
            wob = 1.0 + 0.03 * math.exp(-(t - ts - 0.5) * 6) * math.sin((t - ts) * 22) if t > ts + 0.5 else 1.0
            mg.paste(c, w.capture(f - 0.04), (cx, cy), scale * wob, rot)
        w.over(f + 0.06, t1, polaroid)
    # leaving: transition into the race
    w.flash(t1, 0.2, WHITE, 0.4)


def floor_strip(c, x_near, y_far, y_near, w_near, cols=((255, 255, 255), (30, 30, 30)), cells=8, alpha=0.9, vp=(460, 830)):
    """Chequered line lying on the floor, running toward the vanishing point."""
    def xs(y, xn, wn):
        k = (y - vp[1]) / (y_near - vp[1])
        return vp[0] + (xn - vp[0]) * k, wn * k
    for i in range(cells):
        ya = lerp(y_far, y_near, i / cells)
        yb = lerp(y_far, y_near, (i + 1) / cells)
        xa, wa = xs(ya, x_near, w_near)
        xb, wb = xs(yb, x_near, w_near)
        for j in range(2):
            col = cols[(i + j) % 2]
            ax0, ax1 = xa - wa / 2 + wa * j / 2, xa - wa / 2 + wa * (j + 1) / 2
            bx0, bx1 = xb - wb / 2 + wb * j / 2, xb - wb / 2 + wb * (j + 1) / 2
            mg.poly(c, [(ax0, ya), (ax1, ya), (bx1, yb), (bx0, yb)], col, alpha)


def act2_v6(w):
    """RACE: the mouse's car rockets to the finish and overshoots; the turtle plods and wins."""
    a = w.actors
    car, turt, claw, flash = a["car"], a["turt"], a["claw"], a["flash"]
    t0, t1 = B(80), B(96)
    X0, XF = 150, 905
    Y_CAR, Y_TUR = 1690, 1800
    # transition: polaroids and flash out, busts duck away to free the lane
    w.flash(t0, 0.25, WHITE, 0.5)
    w.punch(t0, 0.05)
    for ac, x_ in ((claw, ST["claw"][0]), (flash, ST["flash"][0] + 10)):
        pass
    claw.move(t0, t0 + 0.4, ST["claw"], (ST["claw"][0], 2500), ease=in_back)
    flash.move(t0, t0 + 0.4, (ST["flash"][0] + 10, 1925), (ST["flash"][0] + 10, 2500), ease=in_back)
    claw.move(B(95), B(95) + 0.45, (ST["claw"][0], 2500), ST["claw"], ease=out_back)
    flash.move(B(95), B(95) + 0.45, (ST["flash"][0] + 10, 2500), (ST["flash"][0] + 10, 2010), ease=out_back)
    claw.groove(B(96), B(96) + 3, beats=BB, amp=0.05, hop=0.03)
    # cast heads: spectators flip with the action
    spect = ["judy", "nick", "lion", "gaz", "big", "beav"]
    for k in spect:
        a[k].groove(B(80), B(96), beats=BB, amp=0.035, hop=0.0, damp=6)
    # race lights
    L0 = B(80)

    def lights(c, t):
        u = clamp((t - (L0 - 0.1)) / 0.25) * clamp((B(84) - t) / 0.3)
        if u <= 0:
            return
        yy = 330 - 60 * (1 - out_back(u))
        mg.poly(c, [(300, yy - 60), (780, yy - 60), (780, yy + 60), (300, yy + 60)], (40, 30, 30), 0.92 * u)
        mg.poly(c, [(300, yy - 60), (780, yy - 60), (780, yy - 54), (300, yy - 54)], WHITE, 0.4 * u)
        for m in range(3):
            cx = 540 + (m - 1) * 150
            on_red = (t >= B(80) and m == 0) or (t >= B(81) and m == 1)
            on_green = t >= B(82) and m == 2
            col = (60, 60, 245) if on_red else (90, 240, 100) if on_green else (70, 70, 70)
            if (m == 2 and t < B(82)):
                col = (70, 70, 70)
            if t >= B(82):
                col = (90, 240, 100)
            r = 40
            mg.circle(c, (cx, yy), r + 6, (20, 20, 20), u)
            mg.circle(c, (cx, yy), r, col, u)
            if col != (70, 70, 70):
                mg.glow(c, (cx, yy), 110, col, 0.7 * u)
    w.over(L0 - 0.1, B(84) + 0.1, lights)
    for tt in (B(80), B(81)):
        w.punch(tt, 0.02, 0.2)

    # lines on the floor
    def lines(c, t):
        u = clamp((t - (t0 - 0.0)) / 0.4) * clamp((t1 + 0.4 - t) / 0.5)
        if u <= 0:
            return
        floor_strip(c, 205, 1640, 1830, 56, alpha=0.9 * u)
        floor_strip(c, 905, 1640, 1830, 56, alpha=0.9 * u)
    w.back(t0 - 0.1, t1 + 0.5, lines)

    # the racers
    car.show(t0)
    turt.show(t0)
    car.stay(t0, X0, Y_CAR)
    turt.stay(t0, X0 + 10, Y_TUR)
    car.hm, turt.hm = 0.5, 0.42
    turt.pose(t0, "turtle", True)
    car.pose(t0, "mouse_car", False)
    car.hm = 0.5
    car.pop_in(t0, 0.35)
    turt.pop_in(t0 + 0.12, 0.35)
    car.groove(t0 + 0.4, B(82), beats=BB, amp=0.06, hop=0.0, swing=3.0)       # engine revving
    turt.groove(t0 + 0.4, B(82), beats=BB, amp=0.03, hop=0.0)
    car.mod(B(81), B(82), lambda t, h: dict(rot=2.5 * math.sin(t * 90)))
    # GO!
    tg = B(82)
    fx_burst(w, tg, (540, 1000), r=220, col=GREEN, dur=0.36, seed=12, layer="back")
    fx_ring(w, tg, (X0, Y_CAR), rmax=330, col=WHITE, dur=0.5, thick=12, floor=True)
    w.flash(tg, 0.12, (200, 255, 200), 0.35)
    w.punch(tg, 0.06, 0.25)
    w.shake(tg, amp=12, dur=0.4)
    dash = 0.78
    car.move(tg, tg + dash, (X0, Y_CAR), (1300, Y_CAR), ease=in_quad)
    car.mod(tg, tg + dash, lambda t, h: dict(sx=1 + 0.25 * clamp((t - tg) / dash), sy=1 - 0.08 * clamp((t - tg) / dash)))
    car.trail(tg, tg + dash, 5, 0.03)

    def dash_fx(c, t):
        st = car.state(t, w)
        if st is None:
            return
        u = clamp((t - tg) / dash)
        if 0 < u < 1:
            mg.streaks(c, st["x"] - 520, st["x"], st["y"] - 70, 520, WHITE, 0.7, seed=int(t * 40), n=14, spread=120, width=5)
            for i in range(5):
                rr = 26 + 30 * ((t * 7 + i * 0.37) % 1)
                mg.circle(c, (X0 + 40 + 50 * i - 70 * clamp((t - tg) * 2), Y_CAR - 10 - 10 * i), rr, (240, 240, 245), 0.5 * (1 - ((t * 7 + i * 0.37) % 1)))
    w.front(tg, tg + dash + 0.2, dash_fx)
    # spectators follow the car with their heads
    for k in spect:
        face(a[k], k, tg, 1)
        face(a[k], k, B(81), -1)
    # camera whip to the right and back to follow the turtle
    w.zoom.to(tg - 0.05, tg + 0.3, 1.12, out_cubic)
    w.cx.to(tg - 0.05, tg + 0.5, 55, out_cubic)
    w.cx.to(B(84), B(92), -40, in_out_quad)
    w.cx.to(B(92), B(93) + 0.3, 55, in_out_cubic)
    w.cy.to(tg, B(92), 90, in_out_quad)
    # the plod
    tw = B(93)
    turt.move(tg, tw, (X0 + 10, Y_TUR), (XF - 10, Y_TUR), ease=lin)
    turt.plod(tg, tw, hz=2.0, bob=0.018)
    turt.mod(tg, tw, lambda t, h: dict(rot=2.4 * math.sin((t - tg) * 6.5)))
    for i in range(84, 93, 2):
        face(a["judy"], "judy", B(i), 1 if i % 4 == 0 else -1)
    # sweat drops while everybody waits
    for i in range(85, 92):
        k = spect[i % len(spect)]
        x, y = ST[k]
        fx_sweat(w, B(i), (x + 40, y - 0.9 * hpx_of(k)), size=14)
    # the car comes back, dizzy
    tr0, tr1 = B(88), B(94)
    car.move(tr0, tr1, (1320, Y_CAR), (XF + 70, Y_CAR), ease=out_quad)
    car.mod(tr0, tr1, lambda t, h: dict(rot=14 * math.sin((t - tr0) * 5.5) * (1 - 0.7 * clamp((t - tr0) / (tr1 - tr0))),
                                        sx=-1.0, sy=1 + 0.05 * math.sin((t - tr0) * 11)))
    car.trail(tr0, tr0, 0, 0.0)

    def dizzy(c, t):
        st = car.state(t, w)
        if st is None:
            return
        u = clamp((t - tr0) / (tr1 - tr0 + 0.8))
        mg.stars_ring(c, (st["x"], st["y"] - st["hpx"] - 34), 60, 16, t, n=4, size=15, alpha=1 - 0.6 * u)
    w.front(tr0 + 0.1, tr1 + 0.9, dizzy)
    # victory
    fx_ring(w, tw, (XF, Y_TUR), rmax=380, col=YELLOW, dur=0.6, thick=14, floor=True)
    fx_ring(w, tw, (540, 1300), rmax=1000, col=PINK, dur=0.8, thick=26)
    w.flash(tw, 0.18, WHITE, 0.5)
    w.punch(tw, 0.07, 0.3)
    w.shake(tw, amp=12, dur=0.4)
    fx_confetti(w, tw, (60, 1900), n=50, seed=801, spread=(-1.4, -0.75), speed=(800, 1400), gravity=1600)
    fx_confetti(w, tw, (1020, 1900), n=50, seed=802, spread=(-2.4, -1.75), speed=(800, 1400), gravity=1600)
    fx_sparkles(w, tw, (XF, Y_TUR - 160), n=10, rad=240, dur=0.9, size=32, seed=803)
    turt.hop(tw, tw + 0.42, (XF - 10, Y_TUR), (XF - 10, Y_TUR), 0.5 * hpx_of("judy") * 0.6, squash=0.14)
    turt.spin(tw + 0.45, tw + 1.25, 1.0, ease=in_out_quad)
    turt.groove(tw + 0.9, t1, beats=BB, amp=0.05, hop=0.04)

    def crown_fx(c, t):
        st = turt.state(t, w)
        if st is None or t < tw:
            return
        u = out_back(clamp((t - tw) / 0.35))
        sg = 1 if st["flip"] else -1                    # sprite faces left by default: the head is on that side
        mg.crown(c, (st["x"] + sg * 0.33 * st["hpx"], st["y"] - st["z"] - 0.84 * st["hpx"] - 40 * (1 - u)), 74 * u, 1.0)
    w.front(tw, t1 + 0.5, crown_fx)
    for k in spect:
        x, y = ST[k]
        a[k].hop(tw, tw + 0.45, (x, y), (x, y), 0.18 * hpx_of(k), squash=0.14)
        a[k].hop(B(94), B(95), (x, y), (x, y), 0.22 * hpx_of(k), squash=0.14)
        face(a[k], k, tw, 1 if x < XF else -1)
    # the car slumps at the end
    car.mod(tr1, t1, lambda t, h: dict(sy=0.88, sx=-1.04))
    car.impact(tr1, 0.22, 0.5)


def act2_v7(w):
    """GAZELLE'S SOLO under a spotlight, then the LION'S ROAR knocks everybody over like dominoes."""
    a = w.actors
    judy, nick, lion, gaz, big, beav, claw, flash, car = (a[k] for k in ("judy", "nick", "lion", "gaz", "big", "beav", "claw", "flash", "carrot"))
    t0, t1 = B(96), B(112)
    TR = B(100)                                     # the roar
    # transition + dim room for the solo
    w.flash(t0, 0.2, (255, 230, 255), 0.4)
    w.punch(t0, 0.05)
    w.expo.to(t0 - 0.05, t0 + 0.3, 0.55, out_quad)
    w.warm.to(t0 - 0.05, t0 + 0.3, -0.6, out_quad)
    w.expo.to(TR - 0.2, TR + 0.05, 1.0, out_quad)
    w.warm.to(TR - 0.2, TR + 0.05, 0.0, out_quad)
    w.cx.to(t0, B(100) - 0.1, 0, lin)
    w.cy.to(t0 - 0.05, B(100), 0, in_out_cubic)
    w.zoom.to(t0 - 0.05, B(100), 1.0, in_out_cubic)
    car.hide(B(99))
    for k_ in ("turt", "car"):
        a[k_].pop_out(B(96) - 0.1)
    gx, gy = ST["gaz"]

    # spotlight on gazelle
    def solo(c, t):
        u = clamp((t - t0) / 0.35) * clamp((TR - 0.05 - t) / 0.18)
        if u <= 0:
            return
        gs = gaz.state(t, w)
        if gs is None:
            return
        mg.spotlight(c, (gs["x"], gs["y"] - 0.5 * gs["hpx"]), 430, dark=1 - 0.55 * u, soft=0.55, tint=(255, 120, 200))
        mg.godrays(c, (gs["x"] + 40, 0), 90, (255, 235, 255), 0.0, n=1) if False else None
    w.front(t0, TR, solo)
    # her pirouette + poses
    gaz.hop(B(96), B(98), (gx, gy), (gx, gy), 0.12 * hpx_of("gaz"), squash=0.1)
    gaz.spin(B(96), B(98), 2.0, ease=in_out_cubic)
    gaz.hop(B(98), B(99), (gx, gy), (gx + 40, gy), 0.08 * hpx_of("gaz"), squash=0.1)
    gaz.hop(B(99), B(100), (gx + 40, gy), (gx, gy), 0.08 * hpx_of("gaz"), squash=0.1)
    gaz.wobble(B(98), B(100), 6.0, 1.1)
    for i in range(96, 100):
        fx_sparkles(w, B(i), (gx + 20, gy - 0.9 * hpx_of("gaz")), n=5, rad=0.35 * hpx_of("gaz"), dur=0.6, size=26, seed=910 + i,
                    cols=[PINK, WHITE, MAGENTA, YELLOW])
    fx_ring(w, B(98), (gx, gy), rmax=330, col=MAGENTA, dur=0.55, thick=12, floor=True)
    for k in ("judy", "nick", "beav", "big"):
        a[k].groove(B(96), TR, beats=BB, amp=0.05, hop=0.03, swing=4.0)
    lion.groove(B(96), B(99), beats=BB, amp=0.03, hop=0.0)
    claw.groove(B(96), TR, beats=BB, amp=0.05, hop=0.03, swing=3.0)

    # ---- the roar ---------------------------------------------------------------------------------------------
    lx, ly = ST["lion"]
    mouth = (lx, ly - 0.62 * hpx_of("lion"))
    lion.mod(B(99) - 0.1, TR, lambda t, h: dict(sc=1 + 0.07 * smooth(clamp((t - (B(99) - 0.1)) / (TR - B(99) + 0.1))),
                                                  sy=1 + 0.04 * smooth(clamp((t - (B(99) - 0.1)) / (TR - B(99) + 0.1)))))
    lion.mod(TR, TR + 0.7, lambda t, h: dict(sc=1.07 + 0.3 * math.exp(-(t - TR) * 5) * (1 if t > TR + 0.02 else (t - TR) / 0.02) - 0.07 * clamp((t - TR) / 0.7),
                                               rot=2.5 * math.sin((t - TR) * 60) * math.exp(-(t - TR) * 6)))
    lion.roar(TR - 0.1, TR + 0.9)
    w.flash(TR, 0.25, WHITE, 0.85)
    w.punch(TR, 0.1, 0.35)
    w.shake(TR, amp=26, dur=0.7, freq=26)
    for k, (rm, th, col) in enumerate(((1500, 30, WHITE), (1200, 22, ORANGE), (900, 16, YELLOW))):
        fx_ring(w, TR + 0.07 * k, mouth, rmax=rm, col=col, dur=0.85, thick=th)
    fx_speed(w, TR, TR + 0.7, mouth, r_in=180, r_out=1700, n=44, alpha=0.7, seed=17)
    for i in range(6):
        flicker(w.panels[i], TR + 0.015 * i, 1.0, 0.0)

    order = [("big", 0.0, "B101"), ("gaz", 0.0, "B101"), ("judy", 0.0, "B102"), ("beav", 0.0, "B103"), ("nick", 0.0, "B104")]
    fall_at = dict(big=B(101), gaz=B(101) + 0.25, judy=B(102), beav=B(103), nick=B(104))
    rise_at = dict(nick=B(108), beav=B(108) + 0.3, judy=B(109), gaz=B(109) + 0.3, big=B(110))
    for k, _, _ in order:
        x, y = ST[k]
        d = math.hypot(x - mouth[0], y - mouth[1])
        tb = TR + d / 3400.0
        sign = 1 if x >= lx else -1
        tf, tr_ = fall_at[k], rise_at[k]
        ac = a[k]
        # blown back
        ac.mod(tb, tf, lambda t, h, sign=sign, tb=tb, tf=tf: dict(
            rot=sign * (30 * smooth(clamp((t - tb) / 0.18)) + 8 * math.sin((t - tb) * 30) * math.exp(-(t - tb) * 4)),
            sx=1 + 0.08 * math.exp(-(t - tb) * 8), dx=sign * 55 * out_cubic(clamp((t - tb) / 0.4))))
        # pressed flat to the floor (squash & stretch, stays inside the frame)
        ac.mod(tf, tr_, lambda t, h, sign=sign, tf=tf: (lambda e: dict(
            sy=1 - 0.66 * e, sx=1 + 0.38 * e, rot=sign * 9 * e, dx=sign * (55 + 25 * e)))(out_bounce(clamp((t - tf) / 0.3))))
        fx_ring(w, tf + 0.22, (x + sign * 90, y), rmax=0.55 * hpx_of(k), col=WHITE, dur=0.5, thick=10, floor=True, alpha=0.8)
        fx_halftone(w, tf + 0.2, (x + sign * 90, y - 40), r=0.35 * hpx_of(k), col=(235, 235, 235), dur=0.5, spacing=14, layer="front")
        w.shake(tf + 0.22, amp=8, dur=0.25)
        w.punch(tf + 0.22, 0.02, 0.18)
        # rise (reverse domino): springs back with an elastic overshoot
        ac.mod(tr_, tr_ + 0.7, lambda t, h, sign=sign, tr_=tr_: (lambda f: dict(
            sy=1 - 0.66 * f, sx=1 + 0.38 * f, rot=sign * 9 * f, dx=sign * (55 + 25 * f) * f))(1 - out_elastic(clamp((t - tr_) / 0.6))))
        ac.hop(tr_, tr_ + 0.45, (x, y), (x, y), 0.15 * hpx_of(k), squash=0.14)
        fx_sparkles(w, tr_ + 0.3, (x, y - 0.8 * hpx_of(k)), n=5, rad=0.3 * hpx_of(k), dur=0.5, size=22, seed=950 + int(tr_))

        def dizzy(c, t, ac=ac, tf=tf, tr_=tr_, sign=sign, k=k):
            if tf + 0.35 < t < tr_:
                x_, y_ = ST[k]
                mg.stars_ring(c, (x_ + sign * 80, y_ - 0.42 * hpx_of(k)), 60, 14, t + hash(k) % 5, n=3, size=13)
        w.front(tf + 0.35, tr_, dizzy)
    # busts get blown out of frame and come back
    claw.move(TR + 0.1, TR + 0.5, ST["claw"], (1500, ST["claw"][1] + 200), ease=out_cubic)
    flash.move(TR + 0.15, TR + 0.55, (ST["flash"][0] + 10, 2010), (-500, 2200), ease=out_cubic)
    claw.move(B(110), B(110) + 0.45, (1500, ST["claw"][1] + 200), ST["claw"], ease=out_back)
    flash.move(B(110) + 0.1, B(110) + 0.55, (-500, 2200), (ST["flash"][0] + 10, 2010), ease=out_back)
    # the lion's embarrassment
    for i, f_ in zip(range(104, 108), (1, -1, 1, -1)):
        face(lion, "lion", B(i), f_)
    fx_sweat(w, B(105), (lx + 110, ly - 0.95 * hpx_of("lion")), size=20)
    fx_sweat(w, B(106) + 0.2, (lx - 110, ly - 0.9 * hpx_of("lion")), size=17)
    lion.mod(B(104), B(108), lambda t, h: dict(sc=1 - 0.02 * math.sin((t - B(104)) * 6)))
    # everybody pops up together
    for k in ("judy", "nick", "lion", "gaz", "big", "beav"):
        x, y = ST[k]
        a[k].hop(B(111), B(112), (x, y), (x, y), 0.22 * hpx_of(k), squash=0.15)
    w.punch(B(111), 0.03, 0.2)
    w.flash(B(112), 0.15, WHITE, 0.35)
    w.rot.to(TR, B(104), 2.5, in_out_cubic)
    w.rot.to(B(104), B(108), -2.0, in_out_quad)
    w.rot.to(B(108), B(111), 0.0, in_out_cubic)


def act2_v8(w):
    """BUILD-UP: the carrot becomes a star, pulses speed up, the room strobes ... and then the music dies."""
    a = w.actors
    car = a["carrot"]
    t0, tE = B(112), HOLD2
    cast = ["judy", "nick", "lion", "gaz", "big", "beav"]
    # lights fall away, only the screen and the carrot glow
    w.expo.to(t0, B(118), 0.42, in_out_quad)
    w.warm.to(t0, B(118), -0.5, in_out_quad)
    w.flash(t0, 0.15, WHITE, 0.3)
    # carrot rises from judy's hand to the centre
    car.show(t0 + 0.2)
    car.follow(t0 + 0.2, B(114), "judy", off=(0.30, 0, 0.5), rot=18)
    cx0 = (ST["judy"][0] + 0.3 * hpx_of("judy"), ST["judy"][1] + 0.5, 0.5 * hpx_of("judy"))
    ctr = (540, 1330, 690)
    flight(car, B(114), B(116), cx0, ctr, 160, ease=in_out_cubic)
    car.scale(B(114), B(117), 1.0, 2.4, ease=in_out_quad)
    car.mod(B(114), B(116), lambda t, h: dict(rot=360 * smooth(clamp((t - B(114)) / (B(116) - B(114))))))
    car.mod(B(116), tE + 0.6, lambda t, h: dict(sc=1.0, rot=3 * math.sin(t * 20) * clamp((t - B(116)) / 4.0),
                                                  zr=0.0))
    car.stay(B(116), *ctr)
    car.shadow_w = 1.4
    car.trail(B(114), B(116), 4, 0.03)
    car.hm = 0.42
    fx_ring(w, B(114), (540, 1300), rmax=900, col=ORANGE, dur=0.7, thick=22)

    # pulses that fall INTO the carrot: ever faster
    plist = []
    for i in range(116, 120):
        plist.append(B(i))
    for i in range(120, 124):
        plist += [B(i), (B(i) + B(i + 1)) / 2]
    for i in range(124, 128):
        d = (B(i + 1) - B(i)) / 4
        plist += [B(i) + d * m for m in range(4)]
    plist = [p for p in plist if p < tE - 0.05]

    def inhale(c, t):
        cpos = (540, 1330 - 520 * 0.0)
        # centre of the carrot on screen
        for tp in plist:
            u = (t - tp) / 0.34
            if 0 <= u < 1:
                r = 900 * (1 - out_cubic(u)) + 60
                mg.ring(c, (540, 640), r, max(2, int(16 * (1 - u))), [CYAN, PINK, YELLOW, WHITE][plist.index(tp) % 4], 0.8 * (1 - u))
        # growing halo
        k = clamp((t - B(116)) / (tE - B(116)))
        mg.glow(c, (540, 640), 240 + 520 * k * k, (180, 230, 255), 0.5 + 1.0 * k)
        mg.glow(c, (540, 640), 120 + 300 * k, (255, 255, 255), 0.8 * k)
    w.front(B(116) - 0.1, tE + 0.65, inhale)
    # floating motes drawn toward the carrot
    def motes(c, t):
        k = clamp((t - B(116)) / (tE - B(116)))
        rng = np.random.RandomState(5)
        for i in range(60):
            ang = rng.uniform(0, 6.28)
            rr0 = rng.uniform(300, 1100)
            ph = (t * (0.3 + 0.5 * rng.uniform()) + rng.uniform()) % 1.0
            rr = rr0 * (1 - ph) + 30
            x = 540 + rr * math.cos(ang + 4 * ph)
            y = 640 + rr * 0.8 * math.sin(ang + 4 * ph)
            mg.circle(c, (x, y), 3 + 4 * ph * k, (255, 245, 220), 0.7 * k * min(1, (1 - ph) * 3))
    w.front(B(116), tE + 0.65, motes)

    # light strobe that speeds up
    for i in range(116, 128):
        rate = 1 if i < 120 else 2 if i < 124 else 4
        for m in range(rate):
            tt = B(i) + (B(i + 1) - B(i)) * m / rate
            if tt < tE - 0.05:
                flicker(w.panels[(i + m) % 8], tt, 1.0, 0.0)
    # characters tremble more and more, then crouch
    for k in cast:
        a[k].mod(B(116), tE, lambda t, h, k=k: dict(
            dx=(0.5 + 5.5 * clamp((t - B(116)) / (tE - B(116))) ** 2) * math.sin(t * 61 + hash(k) % 7)))
        x, y = ST[k]
        a[k].crouch(B(126), B(126) + 0.3, 0.2)
        a[k].mod(B(126) + 0.3, tE + 0.6, lambda t, h: dict(sy=0.8, sx=1.14))
    for i in range(116, 126, 2):
        w.punch(B(i), 0.012, 0.2)
    # camera creeps in and shakes harder
    w.zoom.to(B(112), tE, 1.18, in_quad)
    w.cy.to(B(112), tE, -90, in_out_quad)
    w.cx.to(B(112), tE, 0, in_out_quad)
    w.rot.to(B(112), B(116), 0, lin)
    for i in range(116, 127):
        amp = 2 + 9 * (i - 116) / 11
        w.shake(B(i), amp=amp, dur=0.45, freq=18 + i - 116)
    # the screen goes white
    w.sat.to(B(122), tE, 1.2, in_quad)


# =============================================================================================
# ACT 3 :  79.81 - 90.0   "the finale"
# =============================================================================================
def fx_sunburst(w, t0, t1, center, cols=(PINK, YELLOW), alpha=0.5, spin=40.0, n=18, fade=0.25, r=2300):
    def fn(c, t):
        u = min(clamp((t - t0) / fade), clamp((t1 - t) / fade))
        if u <= 0:
            return
        mg.sunburst(c, center, n, (t - t0) * spin, r, cols[0], alpha * u, duty=0.5)
        mg.sunburst(c, center, n, (t - t0) * spin + 180 / n, r, cols[1], alpha * u * 0.8, duty=0.5)
    w.back(t0, t1, fn)


def fx_eq_ring(w, t0, t1, center, r0=290, n=64, maxlen=170, alpha=0.9):
    def fn(c, t):
        u = min(clamp((t - t0) / 0.4), clamp((t1 - t) / 0.5))
        if u <= 0:
            return
        tt = t - t0
        base = 0.45 * feat("low", t) + 0.35 * feat("mid", t) + 0.2 * feat("high", t)
        for i in range(n):
            ang = i * 2 * math.pi / n + tt * 0.25
            v = base * (0.35 + 0.65 * abs(math.sin(tt * 7 + i * 0.9) * math.cos(tt * 3.1 + i * 0.37)))
            ln = 14 + maxlen * clamp(v)
            p0 = (center[0] + r0 * math.cos(ang), center[1] + r0 * math.sin(ang))
            mid = (center[0] + (r0 + ln / 2) * math.cos(ang), center[1] + (r0 + ln / 2) * math.sin(ang))
            mg.rect_rot(c, mid, ln, 11, math.degrees(ang), PAL[i % len(PAL)], alpha * u)
    w.back(t0, t1, fn)


def fx_float_carrots(w, t0, t1, n=16, seed=4):
    rng = np.random.RandomState(seed)
    items = [(rng.uniform(40, 1040), rng.uniform(0, 1), rng.uniform(5.5, 9.0), rng.uniform(44, 96), rng.uniform(0, 6.28),
              rng.uniform(-25, 25)) for _ in range(n)]

    def fn(c, t):
        u0 = clamp((t - t0) / 0.5) * clamp((t1 - t) / 0.6)
        if u0 <= 0:
            return
        for (x0, ph, dur, sz, sw, rot0) in items:
            u = ((t - t0) / dur + ph) % 1.0
            y = 2050 - u * 2300
            x = x0 + 34 * math.sin(u * 9 + sw)
            mg.carrot(c, (x, y), sz, rot0 + 14 * math.sin(u * 7 + sw), 0.92 * u0)
    w.back(t0, t1, fn)


def act3(w):
    a = w.actors
    car, turt, carcar = a["carrot"], a["turt"], a["car"]
    cast = ["judy", "nick", "lion", "gaz", "big", "beav"]
    t0 = BOOM
    # ---- the silence: everything inhales -------------------------------------------------------------------------------
    w.bright.to(HOLD2, t0, 0.82, in_quad)
    w.bright.set(t0, 1.0)
    w.vig.set(t0, 1.0)
    w.zoom.to(HOLD2, t0, 1.23, lin)
    # ---- BOOM ----------------------------------------------------------------------------------------------------------------------
    w.expo.to(t0, t0 + 0.1, 1.0, out_quad)
    w.warm.to(t0, t0 + 0.1, 0.0, out_quad)
    w.sat.to(t0, t0 + 0.3, 1.18, out_quad)
    w.flash(t0, 0.55, WHITE, 1.0)
    w.punch(t0, 0.1, 0.35)
    w.shake(t0, amp=28, dur=0.8, freq=24)
    w.zoom.to(t0, C(2), 1.0, out_cubic)
    w.cy.to(t0, C(2), 0, out_cubic)
    w.rot.to(t0, C(1), 0, out_cubic)
    car.hide(t0)
    w.thaw = None
    for k, (rm, th, col) in enumerate(((2000, 46, WHITE), (1700, 34, PINK), (1400, 28, CYAN), (1100, 22, YELLOW), (800, 16, ORANGE))):
        fx_ring(w, t0 + 0.05 * k, (540, 640), rmax=rm, col=col, dur=0.9, thick=th, layer="back")
    fx_sunburst(w, t0, C(3), (540, 640), cols=(PINK, YELLOW), alpha=0.55, spin=30)
    fx_confetti(w, t0, (540, 640), n=110, seed=1001, spread=(0, 6.283), speed=(650, 1900), gravity=1300, size=(16, 34),
                life=(1.8, 3.0), kinds=("carrot", "star", "carrot", "circle", "rect"))
    fx_confetti(w, t0 + 0.05, (0, 1900), n=50, seed=1002, spread=(-1.5, -0.6), speed=(900, 1700), gravity=1700, size=(14, 30))
    fx_confetti(w, t0 + 0.05, (1080, 1900), n=50, seed=1003, spread=(-2.55, -1.65), speed=(900, 1700), gravity=1700, size=(14, 30))
    fx_sparkles(w, t0 + 0.02, (540, 640), n=14, rad=520, dur=0.9, size=46, seed=1004)
    # the whole cast leaps
    spins = dict(judy=1.0, gaz=2.0, nick=-1.0, beav=1.0, big=-1.0, lion=0.0)
    for k in cast:
        x, y = ST[k]
        a[k].crouch(t0 - 0.0, t0 + 0.0001, 0.0)
        a[k].mod(t0 - 0.0, t0 + 0.03, lambda t, h: dict(sy=1.0))
        a[k].segs.append((t0, C(2), lambda tt, x=x, y=y, k=k: (x, y, 0.85 * hpx_of(k) * 4 * clamp((tt - t0) / (C(2) - t0)) * (1 - clamp((tt - t0) / (C(2) - t0))))))
        a[k].land(C(2), 0.2, 0.5)
        a[k].trail(t0, C(2), 4, 0.04)
        if spins[k]:
            a[k].spin(t0 + 0.08, C(2) - 0.05, spins[k], ease=in_out_quad)
        a[k].mod(t0, C(2), lambda t, h, k=k: dict(rot=(10 if k in ("judy", "beav", "big") else -10) * math.sin(clamp((t - t0) / (C(2) - t0)) * math.pi)))
        fx_ring(w, C(2) + 0.03, (x, y), rmax=0.55 * hpx_of(k), col=PAL[cast.index(k) % 7], dur=0.5, thick=10, floor=True)
    a["claw"].move(t0, C(2), ST["claw"], ST["claw"], arc=220)
    a["claw"].land(C(2), 0.18, 0.5)
    a["flash"].move(t0, C(2), (ST["flash"][0] + 10, 2010), (ST["flash"][0] + 10, 2010), arc=170)
    a["flash"].land(C(2), 0.12, 0.5)
    # landing: group pose #1
    w.flash(C(2), 0.2, WHITE, 0.4)
    w.punch(C(2), 0.06, 0.3)
    w.shake(C(2), amp=14, dur=0.4)
    fx_ring(w, C(2), (540, 1400), rmax=1500, col=WHITE, dur=0.8, thick=34, layer="back")
    fx_sparkles(w, C(2), (540, 1100), n=14, rad=560, dur=0.9, size=40, seed=1010)
    # ---- the party ---------------------------------------------------------------------------------------------------------------------
    PA = C(2)
    PB = C(15)
    fx_eq_ring(w, C(3), C(20), (468, 710), r0=300)
    fx_sunburst(w, C(3), C(18), (468, 710), cols=(PURPLE, CYAN), alpha=0.22, spin=14, n=20)
    fx_float_carrots(w, C(3), C(20), n=16)
    for k in cast + ["claw"]:
        a[k].groove(C(2) + 0.2, C(15), beats=BC, amp=0.075, hop=0.06, swing=3.5, damp=6.5)
    a["flash"].groove(C(2) + 0.2, C(15), beats=BC, amp=0.04, hop=0.03, swing=2.0)
    for i in range(3, 15):
        for k in ("judy", "gaz", "beav"):
            face(a[k], k, C(i), 1 if i % 2 else -1)
        for k in ("nick", "lion", "big"):
            face(a[k], k, C(i), -1 if i % 2 else 1)
    # beat-by-beat camera sway + pulse
    for i in range(3, 15):
        w.rot.to(C(i), C(i + 1), (2.2 if i % 2 else -2.2), in_out_quad)
        if i % 2 == 0:
            w.punch(C(i), 0.02, 0.18)
    for i in (6, 8):
        fx_ring(w, C(i), (540, 1300), rmax=1100, col=PAL[i % 7], dur=0.6, thick=26, layer="back")
    # floor pools
    def pools(c, t):
        u = clamp((t - C(3)) / 0.5) * clamp((C(20) - t) / 0.6)
        if u <= 0:
            return
        bi = int(np.searchsorted(BC, t, side="right")) - 1
        dt = t - BC[max(bi, 0)]
        for i, (cx, cy) in enumerate(((300, 1480), (780, 1420), (540, 1660), (200, 1560), (880, 1640))):
            mg.disc_e(c, (cx, cy), 210 + 70 * math.exp(-dt * 7), PAL[(bi + i) % 7], 0.2 * u, squash=0.3)
    w.back(C(3), C(20) + 0.2, pools)
    # screen: bright cast lineup, punching on every beat
    # ---- the musical highlight: 3-hit combo + whip -----------------------------------------------------------------------------
    for i in (8,):
        w.inverts.append((C(i), C(i) + 0.07))
        w.flash(C(i), 0.15, WHITE, 0.35)
        w.punch(C(i), 0.07, 0.25)
    left, right = ["gaz", "nick", "lion"], ["judy", "beav", "big"]
    for grp, tb, col, side in ((left, C(10), PINK, 1), (right, C(11), CYAN, -1)):
        for k in grp:
            x, y = ST[k]
            a[k].crouch(tb - 0.16, tb, 0.18)
            a[k].hop(tb, tb + 0.4, (x, y), (x, y), 0.55 * hpx_of(k), squash=0.14)
            a[k].spin(tb + 0.04, tb + 0.38, 1.0 * side, ease=in_out_quad)
            a[k].trail(tb, tb + 0.4, 3, 0.04)
        cx = 250 if side > 0 else 830
        fx_halftone(w, tb, (cx, 1250), r=460, col=col, dur=0.55, spacing=26, alpha=0.9, layer="back")
        fx_ring(w, tb + 0.38, (cx, 1500), rmax=700, col=col, dur=0.6, thick=22, floor=True)
        w.rot.to(tb, tb + 0.4, -4.5 * side, out_cubic)
        w.cx.to(tb, tb + 0.4, 60 * side * -1 * -1, out_cubic) if False else None
        w.punch(tb, 0.04, 0.2)
    # C12: everybody + the camera whips (zoom + dutch tilt), then settles
    tw_ = C(12)
    w.zoom.to(tw_ - 0.05, tw_ + 0.22, 1.28, out_cubic)
    w.rot.to(tw_ - 0.05, tw_ + 0.22, 7.0, out_cubic)
    w.cy.to(tw_ - 0.05, tw_ + 0.22, -120, out_cubic)
    w.zoom.to(tw_ + 0.22, C(14), 1.0, in_out_cubic)
    w.rot.to(tw_ + 0.22, C(14), 0.0, in_out_cubic)
    w.cy.to(tw_ + 0.22, C(14), 0.0, in_out_cubic)
    w.inverts.append((tw_, tw_ + 0.06))
    w.flash(tw_, 0.2, (255, 240, 200), 0.6)
    w.over(tw_ - 0.1, tw_ + 0.6, lambda c, t: mg.wipe(c, (t - (tw_ - 0.1)) / 0.7, [PINK, CYAN, YELLOW, PURPLE, ORANGE], angle=18))
    for k in cast:
        x, y = ST[k]
        a[k].hop(tw_, tw_ + 0.42, (x, y), (x, y), 0.4 * hpx_of(k), squash=0.14)
    fx_sparkles(w, tw_, (540, 1000), n=16, rad=600, dur=0.8, size=42, seed=1020)
    # C14: the big hit - highest jump, 4 confetti cannons
    th_ = C(14)
    w.flash(th_, 0.3, WHITE, 0.8)
    w.punch(th_, 0.1, 0.3)
    w.shake(th_, amp=20, dur=0.6, freq=24)
    for k in cast:
        x, y = ST[k]
        a[k].crouch(th_ - 0.16, th_, 0.2)
        a[k].hop(th_, th_ + 0.5, (x, y), (x, y), 0.9 * hpx_of(k), squash=0.18)
        a[k].trail(th_, th_ + 0.5, 4, 0.04)
        if k in ("judy", "gaz"):
            a[k].spin(th_ + 0.05, th_ + 0.45, 1.0, ease=in_out_quad)
    a["claw"].hop(th_, th_ + 0.5, ST["claw"], ST["claw"], 300, squash=0.14)
    a["flash"].hop(th_, th_ + 0.5, (ST["flash"][0] + 10, 2010), (ST["flash"][0] + 10, 2010), 240, squash=0.1)
    for k, (rm, th, col) in enumerate(((1900, 40, WHITE), (1500, 28, PINK), (1100, 20, CYAN))):
        fx_ring(w, th_ + 0.06 * k, (540, 1300), rmax=rm, col=col, dur=0.8, thick=th, layer="back")
    for cx, cy, sd, sp in ((0, 1900, 1031, (-1.5, -0.6)), (1080, 1900, 1032, (-2.55, -1.65)), (0, 700, 1033, (-0.3, 0.7)), (1080, 700, 1034, (2.4, 3.4))):
        fx_confetti(w, th_, (cx, cy), n=45, seed=sd, spread=sp, speed=(800, 1500), gravity=1500, size=(14, 30))
    fx_sunburst(w, th_, th_ + 1.0, (540, 1000), cols=(YELLOW, ORANGE), alpha=0.4, spin=-50)
    w.inverts.append((th_, th_ + 0.05))
    # ---- settle into the final tableau --------------------------------------------------------------------------------------------
    w.rot.to(C(14) + 0.2, C(17), 0.0, in_out_cubic)
    # turtle with the crown rolls in along the front
    turt.hm = 0.42
    carcar.hide(C(15))
    turt.show(C(16))
    turt.pose(C(16), "turtle", False)
    turt.stay(C(16), 1260, 1820)
    turt.move(C(16), C(19), (1260, 1820), (590, 1820), ease=out_cubic)
    turt.plod(C(16), C(19), hz=2.6, bob=0.02)
    turt.land(C(19), 0.14)
    turt.groove(C(19) + 0.3, 90.0, beats=BC, amp=0.03, hop=0.0)

    def crown_final(c, t):
        st = turt.state(t, w)
        if st is None or t < C(16):
            return
        u = out_back(clamp((t - C(18)) / 0.3))
        if u <= 0.01:
            return
        sg = 1 if st["flip"] else -1
        mg.crown(c, (st["x"] + sg * 0.33 * st["hpx"], st["y"] - st["z"] - 0.84 * st["hpx"]), 78 * u * st["sc"], 1.0)
    w.front(C(18), 91, crown_final)
    # everyone returns to a clean, camera-facing pose
    for k in cast:
        face(a[k], k, C(19), 1 if k in ("nick", "gaz", "beav", "big") else -1)
        a[k].groove(C(15) + 0.4, C(20), beats=BC, amp=0.035, hop=0.02, swing=1.5)
    face(a["lion"], "lion", C(19), -1)
    # screen goes calm: the cast lineup
    w.zoom.to(C(15), C(20), 1.0, in_out_quad)
    # ---- FINAL POSE ------------------------------------------------------------------------------------------------------------------
    tf_ = C(20)
    w.flash(tf_, 0.3, WHITE, 0.65)
    w.punch(tf_, 0.07, 0.3)
    w.shake(tf_, amp=12, dur=0.45)
    w.sat.to(tf_, tf_ + 0.8, 1.1, out_cubic)
    for k in cast:
        x, y = ST[k]
        a[k].hop(tf_ - 0.28, tf_, (x, y), (x, y), 0.3 * hpx_of(k), squash=0.18)
        a[k].land(tf_, 0.2, 0.5)
        fx_ring(w, tf_, (x, y), rmax=0.6 * hpx_of(k), col=PAL[cast.index(k) % 7], dur=0.6, thick=10, floor=True)
    a["claw"].hop(tf_ - 0.28, tf_, ST["claw"], ST["claw"], 150, squash=0.1)
    a["flash"].hop(tf_ - 0.28, tf_, (ST["flash"][0] + 10, 2010), (ST["flash"][0] + 10, 2010), 120, squash=0.1)
    a["flash"].land(tf_, 0.12, 0.5)
    fx_ring(w, tf_, (540, 1350), rmax=1500, col=WHITE, dur=0.9, thick=34, layer="back")
    fx_ring(w, tf_ + 0.06, (540, 1350), rmax=1100, col=PINK, dur=0.9, thick=22, layer="back")
    fx_sparkles(w, tf_, (540, 1050), n=18, rad=520, dur=1.1, size=40, seed=1050)
    fx_confetti(w, tf_, (0, 1900), n=40, seed=1061, spread=(-1.5, -0.7), speed=(700, 1300), gravity=1500, size=(14, 28))
    fx_confetti(w, tf_, (1080, 1900), n=40, seed=1062, spread=(-2.45, -1.65), speed=(700, 1300), gravity=1500, size=(14, 28))
    # judy holds the carrot again
    car.show(tf_)
    car.follow(tf_, 91, "judy", off=(0.30, 0, 0.5), rot=18)
    car.mod(tf_, tf_ + 0.4, lambda t, h: dict(sc=out_back(clamp((t - tf_) / 0.35)) * 1.0))
    car.hm = 0.34
    # ---- hold: a living last frame --------------------------------------------------------------------------------------------------
    for k in cast + ["claw", "flash"]:
        a[k].groove(tf_ + 0.5, 90.5, beats=BC, amp=0.012, hop=0.0, damp=3, swing=0.0)
    fx_sparkles(w, C(22), (540, 1000), n=10, rad=500, dur=0.9, size=30, seed=1070)
    fx_ring(w, C(22), (540, 1350), rmax=900, col=YELLOW, dur=0.8, thick=14, layer="back")
    w.punch(C(22), 0.025, 0.3)
    w.zoom.to(tf_ + 0.5, 90.0, 1.025, in_out_quad)


def add_rims(w):
    """Coloured rim lights: the cast catches the colour of whatever lights the room (rays, disco, carrot glow ...)."""
    cv = lambda c: np.array(c, np.float32)

    def build_rays(t):
        if t < A(30) or t >= THAW:
            return []
        tt = wt(t)
        u = clamp((tt - A(30)) / 1.2)
        col = [PINK, PURPLE, CYAN][int((tt - A(30)) / 0.43 / 2) % 3]
        return [(0.0, -1.0, cv(col), 0.55 * u)]
    w.rim_fx.append((A(30), THAW, build_rays))

    def calm_rays(t):
        u = clamp((t - THAW) / 0.8) * clamp((B(16) - t) / 0.8)
        col = [PINK, PURPLE, CYAN][int((wt(t) - THAW) / 0.4575 / 2) % 3]
        return [(0.0, -1.0, cv(col), 0.4 * u)]
    w.rim_fx.append((THAW, B(16), calm_rays))

    def disco(t):
        bi = int(np.searchsorted(BB, t, side="right")) - 1
        dt = t - BB[bi]
        k = 0.38 + 0.35 * math.exp(-dt * 6)
        return [(-0.9, -0.4, cv(PAL[bi % 7]), k), (0.9, -0.4, cv(PAL[(bi + 3) % 7]), k), (0.0, -1.0, cv(YELLOW), 0.18)]
    w.rim_fx.append((B(48), B(64) + 0.3, disco))

    def roar(t):
        dt = t - B(100)
        return [(0.0, -1.0, cv(WHITE), 0.9 * math.exp(-dt * 4.5)), (-1.0, 0.0, cv(ORANGE), 0.6 * math.exp(-dt * 4.5))]
    w.rim_fx.append((B(100), B(100) + 1.2, roar))

    def buildup(t):
        k = clamp((t - B(116)) / (HOLD2 - B(116)))
        return [(0.0, -1.0, cv((255, 245, 225)), 0.15 + 0.95 * k * k), (0.0, 1.0, cv((230, 235, 255)), 0.35 * k)]
    w.rim_fx.append((B(112), BOOM, buildup))

    def party(t):
        bi = int(np.searchsorted(BC, t, side="right")) - 1
        dt = t - BC[max(bi, 0)]
        k = 0.36 + 0.32 * math.exp(-dt * 6)
        if t < C(3):
            k *= 1.2
        return [(-0.85, -0.5, cv(PAL[bi % 7]), k), (0.85, -0.5, cv(PAL[(bi + 2) % 7]), k), (0.0, -1.0, cv(WHITE), 0.3)]
    w.rim_fx.append((BOOM, C(19), party))

    def final(t):
        u = clamp((t - C(18)) / 0.8)
        dt = t - C(20)
        pop = math.exp(-dt * 5) if dt > 0 else 0.0
        return [(0.0, -1.0, cv((255, 238, 215)), 0.22 * u + 0.6 * pop), (-0.8, -0.3, cv(PINK), 0.18 * u), (0.8, -0.3, cv(CYAN), 0.18 * u)]
    w.rim_fx.append((C(18), 91, final))


def build():
    w = World()
    act1(w)
    act2_v1(w)
    act2_v2(w)
    act2_v3(w)
    act2_v4(w)
    act2_v5(w)
    act2_v6(w)
    act2_v7(w)
    act2_v8(w)
    act3(w)
    add_rims(w)
    for k_ in ("claw", "flash", "judy_b", "nick_b"):
        if k_ in w.actors:
            w.actors[k_].dof = 1.1
    return w
