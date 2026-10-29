"""Repairs for the cut-outs: run after cutout.py / fix_sprites.py.

Most defects came from the source pictures themselves (a character partly hidden behind another one, white
fog or a white crosswalk stripe that matches the fur, shadows) and from my first, too blunt clean-ups.
Each repair below re-derives the affected region from the original picture or patches the PNG in place:

    lion_full   right leg was hidden behind the sloth -> sloth removed, leg + shoe re-drawn
    clawhauser  left body side swallowed by the sloth's arm -> colour-keyed back in
    gazelle     elbow chopped by the Fru Fru exclusion line -> line moved, hair lock removed
    turtle      hind foot box-cropped, road shadow -> re-cut with a colour based shadow remover
    nick_walk   cheek tuft lost on the white crosswalk stripe -> tuft restored
    + generic   holes in eyes / windshield filled, halos removed, thin slivers and floating bits deleted

    U2NET_HOME=<models> python tools/repair_sprites.py [name ...]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cutout as C  # noqa: E402

SPR = C.OUT


# ---------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------
def read(name):
    return cv2.imread(os.path.join(SPR, name + ".png"), cv2.IMREAD_UNCHANGED)


def write(name, bgra, trim=True):
    a = bgra[..., 3]
    if trim:
        ys, xs = np.where(a > 8)
        bgra = bgra[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    cv2.imwrite(os.path.join(SPR, name + ".png"), bgra)
    print(f"{name}: {bgra.shape[1]}x{bgra.shape[0]}")


def poly_mask(shape, pts, ss=4):
    m = np.zeros(shape[:2], np.uint8)
    p = (np.asarray(pts, np.float64) * ss).round().astype(np.int32)
    big = np.zeros((shape[0] * ss, shape[1] * ss), np.uint8)
    cv2.fillPoly(big, [p], 255)
    return cv2.resize(big, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def largest(mask, min_area=0):
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    keep = np.zeros(n, bool)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    for i in range(1, n):
        keep[i] = i == big or st[i, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[lab], mask, 0).astype(mask.dtype)


def soften(alpha_u8, sigma=0.8):
    """Edge anti-aliasing for a hard mask."""
    return cv2.GaussianBlur(alpha_u8, (0, 0), sigma)


def inpaint_rgb(rgb, hole, radius=6):
    h = cv2.dilate((hole > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
    return cv2.inpaint(rgb, h, radius, cv2.INPAINT_TELEA)


def defringe(bgra, width=2.5):
    """Replace the colour of edge pixels by colour from just inside the silhouette (kills halos from the old background)."""
    a = bgra[..., 3].astype(np.float32) / 255.0
    solid = (a > 0.6).astype(np.uint8)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
    core = (dist > width).astype(np.float32)
    rgb = bgra[..., :3].astype(np.float32)
    num = cv2.GaussianBlur(rgb * core[..., None], (0, 0), 2.5)
    den = cv2.GaussianBlur(core, (0, 0), 2.5)[..., None]
    est = num / np.maximum(den, 1e-3)
    w = np.clip((width + 0.5 - dist) / (width + 0.5), 0, 1)[..., None] * (den > 0.02)
    out = rgb * (1 - w) + est * w
    bgra = bgra.copy()
    bgra[..., :3] = np.clip(out, 0, 255).astype(np.uint8)
    return bgra


def fill_holes(bgra, boxes):
    """Opaque-fill transparent holes inside the given (u0, v0, u1, v1) fractional boxes (eyes, windshield ...)."""
    a = bgra[..., 3].copy()
    h, w = a.shape
    for (u0, v0, u1, v1) in boxes:
        x0, y0, x1, y1 = int(u0 * w), int(v0 * h), int(u1 * w), int(v1 * h)
        sub = (a[y0:y1, x0:x1] > 128).astype(np.uint8)
        closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        # only fill holes completely enclosed by the character (flood from the box border)
        inv = (1 - closed).astype(np.uint8)
        n, lab = cv2.connectedComponents(inv, connectivity=4)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        enclosed = np.isin(lab, [i for i in range(1, n) if i not in border])
        a[y0:y1, x0:x1][enclosed] = 255
    out = bgra.copy()
    out[..., 3] = a
    return out


def drop_thin(bgra, open_k=5, grow_k=5):
    """Delete hairline slivers / strings left by the matting (opening on the silhouette)."""
    a = bgra[..., 3].astype(np.float32) / 255.0
    solid = (a > 0.5).astype(np.uint8)
    op = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    keep = cv2.dilate(op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow_k, grow_k)))
    out = bgra.copy()
    out[..., 3] = (a * cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 0.8) * 255).astype(np.uint8)
    return out


def drop_islands(bgra, min_frac=0.02):
    a = bgra[..., 3]
    m = (a > 40).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return bgra
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros(n, bool)
    for i in range(1, n):
        keep[i] = st[i, cv2.CC_STAT_AREA] >= big * min_frac
    grown = cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = bgra.copy()
    out[..., 3] = np.where(grown, a, 0)
    return out


# ---------------------------------------------------------------------------------------------------------
# per-character repairs
# ---------------------------------------------------------------------------------------------------------
def lion():
    """The sloth in front hid the lion's right leg: cut the sloth away, then draw the missing part of the leg."""
    src = "zoo_neon_street.jpg"
    box = (470, 580, 730, 950)
    full = C.whole_cut(src, "isnet-general-use")
    arr = full[box[1]:box[3], box[0]:box[2]].copy()
    rgb = C.load(src).crop(box)
    rgb = np.array(rgb)
    ox, oy = box[0], box[1]

    def P(pts):
        return [(x - ox, y - oy) for x, y in pts]
    # everything right of the sloth's face outline goes
    sloth = poly_mask(arr.shape, P([(662, 840), (640, 852), (628, 868), (621, 886), (620, 902), (626, 916), (638, 926),
                                    (650, 940), (656, 952), (656, 1000), (740, 1000), (740, 840)]))
    alpha = np.minimum(arr[..., 3], 255 - sloth)
    # the white road reflection between the legs is not part of the lion
    lum = rgb.astype(np.float32).mean(2)
    gap = poly_mask(arr.shape, P([(592, 862), (610, 862), (610, 946), (592, 946)])) > 0
    alpha = np.where(gap & (lum > 52), 0, alpha).astype(np.uint8)
    # the leg that the sloth covered: a trouser column and the shoe, painted from the visible navy / brown pixels
    leg = poly_mask(arr.shape, P([(610, 806), (651, 806), (651, 912), (653, 926), (652, 936), (646, 943), (638, 945), (612, 945),
                                  (608, 930), (608, 922), (610, 880)]))
    sloth_d = cv2.dilate(sloth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))   # include its pale fur fringe
    valid = ((alpha > 200) & (sloth_d < 30)).astype(np.float32)
    ys = np.arange(arr.shape[0])[:, None] + oy
    trouser_zone = (ys < 921).astype(np.float32) * np.ones_like(valid)
    rgb_f = rgb.astype(np.float32)
    out = rgb_f.copy()
    for zone, sig in ((trouser_zone, 11.0), (1 - trouser_zone, 5.0)):
        v = valid * zone
        v *= ((poly_mask(arr.shape, P([(560, 760), (651, 760), (651, 950), (560, 950)]))) > 0)
        num = cv2.GaussianBlur(rgb_f * v[..., None], (0, 0), sig)
        den = cv2.GaussianBlur(v, (0, 0), sig)[..., None]
        est = num / np.maximum(den, 1e-4)
        m = ((leg > 0) & (valid < 0.5) & (zone > 0)).astype(np.float32)[..., None]
        out = out * (1 - m) + est * m
    # the new trouser gets a soft shadow toward its outer edge so it does not look flat
    xs = np.arange(arr.shape[1])[None, :] + ox
    shade = 1.0 - 0.28 * np.clip((xs - 636) / 16.0, 0, 1) * (ys < 921)
    new_px = ((leg > 0) & (valid < 0.5))[..., None]
    out = np.where(new_px, out * shade[..., None], out)
    rgb_f = np.clip(out, 0, 255).astype(np.uint8)
    alpha = np.maximum(alpha, leg)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bgra = np.dstack([rgb_f[..., ::-1], alpha]).astype(np.uint8)
    bgra = drop_islands(bgra, 0.05)
    bgra[..., 3] = soften(largest(bgra[..., 3] > 40).astype(np.uint8) * 0 + bgra[..., 3], 0.7)
    write("lion_full", bgra)


def _cut2x(src, box, model="isnet-general-use", scale=2):
    """rembg on an upscaled crop; returns (BGR image, RGBA from rembg)."""
    from rembg import remove
    im = C.load(src).crop(box)
    im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    out = np.array(remove(im, session=C.session(model), post_process_mask=True))
    return np.array(im)[..., ::-1].copy(), out


def clawhauser():
    """The sloth's arm (left) overlapped the lineup; rembg took a bite out of his uniform. Key the blue cloth back in."""
    bgr, out = _cut2x("zoo_lineup_rot.jpg", (720, 505, 945, 736))
    arr = out.copy()
    arr = C.dehaze(arr)
    alpha = arr[..., 3].copy()
    h, w = alpha.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, Ss, Vv = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    cloth = ((Hh >= 96) & (Hh <= 128) & (Ss > 28) & (Vv > 45) & (Vv < 215)).astype(np.uint8)
    dark = ((Vv < 70)).astype(np.uint8)                       # belt, pouch, tie
    key = cloth | dark
    zone = poly_mask(alpha.shape, [(0.0 * w, 0.30 * h), (0.55 * w, 0.30 * h), (0.55 * w, h), (0.0, h)]) > 0
    # the sloth arm is on the far left above the shoulder: keep the key out of it
    arm = poly_mask(alpha.shape, [(0, 0), (0.035 * w, 0), (0.035 * w, 0.42 * h), (0, 0.42 * h)]) > 0
    key = (key > 0) & zone & ~arm
    key = cv2.morphologyEx(key.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    key = cv2.morphologyEx(key, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    alpha = np.maximum(alpha, (key * 255).astype(np.uint8))
    # smooth the silhouette: close the remaining dents of the left side
    m = (alpha > 128).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = largest(m)
    alpha = np.where(m > 0, np.maximum(alpha, 255 * (m > 0) * (poly_mask(alpha.shape, [(0, 0.42 * h), (0.6 * w, 0.42 * h), (0.6 * w, h), (0, h)]) > 0)), 0).astype(np.uint8)
    alpha = soften(alpha, 1.0)
    bgra = np.dstack([bgr, alpha])
    bgra = drop_thin(bgra, 7, 7)
    bgra = defringe(bgra, 3.0)
    write("clawhauser", bgra)


def turtle():
    """Re-cut with a wider box (the hind foot touched the old crop edge) and remove the road shadow by colour."""
    bgr, out = _cut2x("zoo_city.jpg", (305, 1172, 492, 1306))
    alpha = out[..., 3].copy()
    h, w = alpha.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    hue = hsv[..., 0].astype(int)
    # road shadow is blue/violet-grey; the turtle is green (hue 25-70), orange or tan
    # isnet drops the legs standing on the white stripe: key the saturated green / orange body colours back in
    body = ((sat > 55) & (hue >= 24) & (hue <= 75) & (val > 35)).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = np.maximum(alpha, body * 255)
    shadow = (sat < 90) & (val < 150) & ((hue > 80) | (hue < 8))
    ys = np.arange(h)[:, None]
    shadow |= (sat < 55) & (val > 150)                  # white crosswalk stripe between the legs
    shadow &= ys > 0.45 * h
    shadow = cv2.dilate(shadow.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    alpha[shadow] = 0
    m = largest(alpha > 100)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = largest(m)
    inv = (1 - m).astype(np.uint8)                        # fill only small enclosed holes (not the gap between the legs)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, ww, hh, ar = st[i]
        touches = x == 0 or y == 0 or x + ww == m.shape[1] or y + hh == m.shape[0]
        if not touches and ar < 150:
            m[lab == i] = 1
    alpha = np.where(m > 0, np.maximum(alpha, 160), 0).astype(np.uint8)
    alpha = soften(alpha, 1.0)
    bgra = defringe(np.dstack([bgr, alpha]), 3.0)
    write("turtle", bgra)


def fill_from_source(name, boxes):
    """Opaque-fill holes that are completely enclosed by the character (white eyes, windshield glass, shirt gaps)
    inside the given fractional boxes, using the real pixels of the source picture."""
    im = read(name)
    src = C.source_crop(name)
    h, w = im.shape[:2]
    src = src[:h, :w]
    if src.shape[:2] != (h, w):
        src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    a = im[..., 3].copy()
    total = 0
    solid = (a > 128).astype(np.uint8)
    closed = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    inv = (1 - closed).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(inv, connectivity=4)
    for i in range(1, n):
        x, y, ww, hh, ar = st[i]
        if x == 0 or y == 0 or x + ww == w or y + hh == h:
            continue                                         # open to the outside: a real gap
        cu, cv_ = cen[i][0] / w, cen[i][1] / h
        if any(u0 <= cu <= u1 and v0 <= cv_ <= v1 for (u0, v0, u1, v1) in boxes) and ar < 0.02 * w * h:
            m = cv2.dilate((lab == i).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            im[m, :3] = src[m]
            a[m] = 255
            total += int(m.sum())
    im[..., 3] = a
    print(f"  {name}: filled {total} px from the source")
    return im


def generic():
    """Light-touch clean-up for everything else: enclosed holes (eyes, glass), floating islands, hairline slivers, halos."""
    jobs = {
        "judy_arms": dict(holes=[(0.0, 0.18, 0.55, 0.40)]),
        "judy_full": dict(holes=[(0.15, 0.22, 0.85, 0.34)]),
        "nick_arms": dict(holes=[(0.45, 0.20, 0.85, 0.36)]),
        "nick_full": dict(holes=[(0.50, 0.20, 0.85, 0.30)]),
        "mouse_car": dict(holes=[(0.0, 0.0, 1.0, 0.40)]),
        "beaver": dict(holes=[(0.40, 0.55, 0.85, 0.72)], thin=True),
        "judy_badge": dict(thin=True, erase=[(0.0, 0.55, 0.245, 0.70)]),      # shards of a neighbour's sleeve
        "flash": dict(thin=True),
        "mrbig": dict(),
        "nick_edge": dict(),
        "carrot_b": dict(),
        "carrot_a": dict(),
        "nick_walk": dict(),
    }
    for name, o in jobs.items():
        if not os.path.exists(os.path.join(SPR, name + ".png")):
            continue
        im = read(name)
        if o.get("holes"):
            im = fill_from_source(name, o["holes"])
        for (u0, v0, u1, v1) in o.get("erase", []):
            h_, w_ = im.shape[:2]
            im[int(v0 * h_):int(v1 * h_), int(u0 * w_):int(u1 * w_), 3] = 0
        im = drop_islands(im, 0.04)
        if o.get("thin"):
            im = drop_thin(im, 5, 5)
        im = defringe(im, 2.5)
        write(name, im, trim=False)


# ---------------------------------------------------------------------------------------------------------
REPAIRS = {"lion_full": lion, "clawhauser": clawhauser, "turtle": turtle, "generic": generic}

if __name__ == "__main__":
    names = sys.argv[1:] or list(REPAIRS)
    for n in names:
        REPAIRS[n]()
