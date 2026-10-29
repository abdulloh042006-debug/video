"""Render a few frames of the film for review:  python tools/peek.py out_dir t1 t2 ...  (sheet.jpg + frames)."""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import director


def main():
    out = sys.argv[1]
    times = [float(x) for x in sys.argv[2:]]
    os.makedirs(out, exist_ok=True)
    w = director.build()
    tiles = []
    for t in times:
        t0 = time.time()
        img = w.render(t)
        dt = time.time() - t0
        cv2.imwrite(os.path.join(out, f"f_{t:06.2f}.png"), img)
        small = cv2.resize(img, (360, 640), interpolation=cv2.INTER_AREA)
        cv2.putText(small, f"{t:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(small)
        print(f"t={t:.2f} {dt:.2f}s", flush=True)
    cols = min(len(tiles), 5)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * 640, cols * 360, 3), np.uint8)
    for i, tl in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * 640:(r + 1) * 640, c * 360:(c + 1) * 360] = tl
    cv2.imwrite(os.path.join(out, "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
