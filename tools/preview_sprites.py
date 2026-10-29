"""Contact sheet of all sprites on a checkerboard -> assets/sprites_preview.png"""
import glob
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fs = sorted(glob.glob(os.path.join(ROOT, "assets", "sprites", "*.png")))
H, pad, items = 330, 14, []
for f in fs:
    im = Image.open(f)
    r = H / im.height
    w = int(im.width * r)
    if w > 300:
        r = 300 / im.width
        w, h = 300, int(im.height * r)
    else:
        h = H
    items.append((os.path.basename(f)[:-4], im.resize((w, h), Image.LANCZOS)))
rows, cur, cw = [], [], 0
for it in items:
    if cw + it[1].width + pad > 1600:
        rows.append(cur)
        cur, cw = [], 0
    cur.append(it)
    cw += it[1].width + pad
rows.append(cur)
sheet = Image.new("RGB", (1600, len(rows) * (H + 34) + 10), (34, 38, 60))
d = ImageDraw.Draw(sheet)
y = 8
for row in rows:
    x = 10
    for n, im in row:
        chk = Image.new("RGB", im.size, (70, 76, 104))
        cd = ImageDraw.Draw(chk)
        for yy in range(0, im.height, 16):
            for xx in range(0, im.width, 16):
                if (xx // 16 + yy // 16) % 2:
                    cd.rectangle([xx, yy, xx + 15, yy + 15], fill=(90, 98, 130))
        sheet.paste(chk, (x, y + 22))
        sheet.paste(im, (x, y + 22), im)
        d.text((x, y + 4), n, fill=(255, 255, 255))
        x += im.width + pad
    y += H + 34
sheet.save(os.path.join(ROOT, "assets", "sprites_preview.png"))
print(sheet.size, len(fs), "sprites")
