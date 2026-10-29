"""Character cut-out tool.

Crops each character out of the supplied Zootopia reference images, removes the
background with rembg (isnet) and writes trimmed RGBA sprites to assets/sprites.

Run:  U2NET_HOME=<dir with isnet-*.onnx> python tools/cutout.py [name ...]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "src")
OUT = os.path.join(ROOT, "assets", "sprites")
os.makedirs(OUT, exist_ok=True)


def load(name):
    im = Image.open(os.path.join(SRC, name)).convert("RGB")
    if name == "zoo_lineup_rot.jpg":      # sideways in the upload -> rotate upright
        im = im.rotate(-90, expand=True)
    if name == "zoo_poster_rot.jpg":
        im = im.rotate(90, expand=True)
    return im


# name: (source, (x0, y0, x1, y1), model[, options])
#   options: dehaze (strip the white fog left by the lineup poster's fade),
#            erase=[(x0,y0,x1,y1)...] in crop coords (/scale applied), largest (keep one blob only)
JOBS = {
    # img3: neon street
    "nick_full":    ("zoo_neon_street.jpg", (0, 540, 280, 1000), "isnet-general-use", dict(whole=True)),
    "judy_full":    ("zoo_neon_street.jpg", (250, 640, 410, 1000), "isnet-general-use", dict(whole=True)),
    "lion_full":    ("zoo_neon_street.jpg", (470, 580, 730, 950), "isnet-general-use", dict(whole=True, erase=[(150, 270, 270, 370)])),
    # img4: sunset, busts
    "nick_arms":    ("zoo_sunset.jpg", (300, 660, 590, 1308), "isnet-general-use"),
    "judy_arms":    ("zoo_sunset.jpg", (110, 730, 350, 1308), "isnet-general-use"),
    "carrot_a":     ("zoo_sunset.jpg", (130, 230, 195, 350), "isnet-general-use"),
    "carrot_b":     ("zoo_sunset.jpg", (365, 360, 440, 525), "isnet-general-use"),
    # img2: lineup
    "flash":        ("zoo_lineup_rot.jpg", (490, 500, 740, 736), "isnet-general-use", dict(dehaze=True)),
    "clawhauser":   ("zoo_lineup_rot.jpg", (720, 505, 945, 736), "isnet-general-use", dict(dehaze=True)),
    "judy_badge":   ("zoo_lineup_rot.jpg", (895, 520, 1030, 736), "isnet-general-use", dict(dehaze=True)),
    "nick_edge":    ("zoo_lineup_rot.jpg", (1160, 480, 1308, 736), "isnet-general-use", dict(dehaze=True)),
    "mrbig":        ("zoo_lineup_rot.jpg", (300, 575, 450, 736), "isnet-general-use", dict(dehaze=True)),
    "bellwether":   ("zoo_lineup_rot.jpg", (195, 505, 320, 736), "isnet-general-use", dict(dehaze=True)),
    "beaver":       ("zoo_lineup_rot.jpg", (70, 535, 245, 736), "isnet-general-use", dict(dehaze=True)),
    # img6: city
    "nick_walk":    ("zoo_city.jpg", (455, 1060, 680, 1295), "isnet-general-use"),
    "turtle":       ("zoo_city.jpg", (315, 1180, 480, 1300), "isnet-general-use"),
    "mouse_car":    ("zoo_city.jpg", (205, 1172, 308, 1300), "isnet-general-use", dict(largest=True)),
    # img5: poster
}

_sessions = {}


def session(model):
    if model not in _sessions:
        _sessions[model] = new_session(model)
    return _sessions[model]


def keep_main(alpha, min_frac=0.02):
    """Drop small detached specks, keep components that are a real part of the subject."""
    a = (alpha > 40).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(a, connectivity=8)
    if n <= 1:
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas.max()
    keep = np.zeros(n, bool)
    for i, ar in enumerate(areas, start=1):
        if ar >= big * min_frac * 5 or ar == big:
            keep[i] = True
    m = keep[lab]
    # slightly grow so soft edge pixels of kept parts survive
    m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return np.where(m, alpha, 0).astype(np.uint8)


def dehaze(arr):
    """Remove the near-white fog the lineup poster leaves around characters.

    Flood-fills from transparent pixels through whitish, low-saturation pixels so only
    haze that is connected to the outside is removed (white parts of a character
    that are enclosed by dark outlines survive).
    """
    rgb = arr[..., :3].astype(np.int16)
    mx, mn = rgb.max(2), rgb.min(2)
    whitish = ((mn > 190) & ((mx - mn) < 40)).astype(np.uint8)
    outside = (arr[..., 3] < 30).astype(np.uint8)
    reach = outside.copy()
    for _ in range(600):
        grown = cv2.dilate(reach, np.ones((3, 3), np.uint8)) & (whitish | outside)
        if (grown == reach).all():
            break
        reach = grown
    arr[..., 3] = np.where((reach > 0) & (whitish > 0), 0, arr[..., 3])
    # soften the edge that is left
    a = cv2.GaussianBlur(arr[..., 3], (0, 0), 0.8)
    arr[..., 3] = np.minimum(arr[..., 3], np.maximum(a, 0)) if False else a
    return arr


_whole = {}


def whole_cut(src, model):
    key = (src, model)
    if key not in _whole:
        _whole[key] = np.array(remove(load(src), session=session(model), post_process_mask=True))
    return _whole[key]


def cut(name):
    job = JOBS[name]
    src, box, model = job[:3]
    opt = job[3] if len(job) > 3 else {}
    scale = 1
    if opt.get("whole"):
        arr = whole_cut(src, model)[box[1]:box[3], box[0]:box[2]].copy()
    else:
        im = load(src).crop(box)
        # upscale small crops 2x before matting for cleaner edges
        scale = 2 if max(im.size) < 400 else 1
        if scale != 1:
            im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
        arr = np.array(remove(im, session=session(model), post_process_mask=True))
    for (x0, y0, x1, y1) in opt.get("erase", []):
        arr[y0:y1, x0:x1, 3] = 0
    if opt.get("dehaze"):
        arr = dehaze(arr)
    arr[..., 3] = keep_main(arr[..., 3], min_frac=1.0 if opt.get("largest") else 0.02)
    ys, xs = np.where(arr[..., 3] > 8)
    if len(xs) == 0:
        print("EMPTY", name)
        return
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    Image.fromarray(arr).save(os.path.join(OUT, name + ".png"))
    print(name, arr.shape[1], "x", arr.shape[0])


def gazelle():
    """Gazelle is cream-coloured on a white fog, which isnet drops (head vanishes).
    Use isnet for the body and a colour key for head/horns/neck, then clean the fog."""
    sc = 4
    im = load("zoo_lineup_rot.jpg").crop((1010, 490, 1130, 736))
    im = im.resize((im.width * sc, im.height * sc), Image.LANCZOS)
    rgb = np.array(im)
    rb = np.array(remove(im, session=session("isnet-general-use"), post_process_mask=True))[..., 3]
    H, W = rgb.shape[:2]
    r, g, b = [rgb[..., i].astype(int) for i in range(3)]
    lum, warm = (r + g + b) / 3, r - b
    reg = np.zeros((H, W), bool)
    reg[0:95 * sc, 50 * sc:120 * sc] = True
    cand = reg & (((warm > 16) & (lum < 238)) | (lum < 120))
    reg2 = np.zeros((H, W), bool)
    reg2[70 * sc:165 * sc, 52 * sc:100 * sc] = True
    cand |= reg2 & (warm > 7) & (lum < 246)
    head = (cand * 255).astype(np.uint8)
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    head = cv2.morphologyEx(head, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    alpha = np.maximum(rb, head)
    alpha[75 * sc:, 86 * sc:] = 0              # Fru Fru's hair
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, lab, st, _ = cv2.connectedComponentsWithStats((alpha > 100).astype(np.uint8))
    keep = np.zeros_like(alpha)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 4000:
            keep[lab == i] = 255
    alpha = cv2.GaussianBlur(keep, (0, 0), 1.5)
    out = np.dstack([rgb, alpha])
    ys, xs = np.where(alpha > 8)
    out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    o = Image.fromarray(out)
    a = np.array(o.resize((o.width // 2, o.height // 2), Image.LANCZOS))
    rgb2 = a[..., :3].astype(int)
    mn, mx = rgb2.min(2), rgb2.max(2)
    w = cv2.morphologyEx(((mn > 222) & ((mx - mn) < 20)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    a[..., 3] = np.where(w > 0, 0, a[..., 3])
    n, lab, st, _ = cv2.connectedComponentsWithStats((a[..., 3] > 60).astype(np.uint8))
    ok = np.zeros(n, bool)
    for i in range(1, n):
        ok[i] = st[i, cv2.CC_STAT_AREA] > 1500
    a[..., 3] = np.where(ok[lab], cv2.GaussianBlur(a[..., 3], (0, 0), 0.8), 0)
    ys, xs = np.where(a[..., 3] > 8)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    Image.fromarray(a).save(os.path.join(OUT, "gazelle.png"))
    print("gazelle", a.shape[1], "x", a.shape[0])


if __name__ == "__main__":
    names = sys.argv[1:] or list(JOBS) + ["gazelle"]
    for n in names:
        gazelle() if n == "gazelle" else cut(n)
