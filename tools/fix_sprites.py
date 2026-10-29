"""Repair ragged notches left by the lineup poster's overlapping characters (idempotent).

Fills concave holes in the lower part of a sprite: alpha is closed with a large kernel and the
new pixels get colour from their neighbours (Telea inpaint)."""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPR = os.path.join(ROOT, "assets", "sprites")

# name: (first row fraction to repair, closing kernel, x range fraction)
FIX = {
    "clawhauser": (0.40, 61, (0.0, 0.62)),
    "beaver": (0.55, 35, (0.0, 0.75)),
}


def fix(name, y0f, k, xr):
    p = os.path.join(SPR, name + ".png")
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    a = im[..., 3]
    h, w = a.shape
    solid = (a > 128).astype(np.uint8)
    pad = k
    sp = cv2.copyMakeBorder(solid, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    # keep the bottom/side edges solid so closing does not eat inward from the image border
    sp[pad + h - 1: pad + h + pad, :] = 0
    closed = cv2.morphologyEx(sp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    closed = closed[pad:pad + h, pad:pad + w]
    region = np.zeros_like(solid)
    region[int(h * y0f):, int(w * xr[0]):int(w * xr[1])] = 1
    new = ((closed > 0) & (solid == 0) & (region > 0)).astype(np.uint8)
    if new.sum() == 0:
        print(name, "nothing to fix")
        return
    bgr = im[..., :3].copy()
    mask = cv2.dilate(new, np.ones((3, 3), np.uint8))
    bgr = cv2.inpaint(bgr, mask * 255, 9, cv2.INPAINT_TELEA)
    a2 = np.maximum(a, (new * 255).astype(np.uint8))
    a2 = cv2.GaussianBlur(a2, (0, 0), 0.9)
    a2 = np.maximum(a2, a)
    cv2.imwrite(p, np.dstack([bgr, a2]))
    print(name, "filled", int(new.sum()), "px")


if __name__ == "__main__":
    for n in (sys.argv[1:] or FIX):
        fix(n, *FIX[n])
