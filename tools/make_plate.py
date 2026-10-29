"""Crop the supplied room screenshot (black bars top/bottom) to the 9:16 plate used by the film."""
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
im = Image.open(os.path.join(ROOT, "assets", "src", "room.png")).convert("RGB")
a = np.asarray(im).astype(int).sum(2)
rows = np.where(a.mean(1) > 30)[0]
plate = im.crop((0, rows.min(), im.width, rows.max() + 1)).resize((1080, 1920), Image.LANCZOS)
plate.save(os.path.join(ROOT, "assets", "plate.png"))
print("plate", plate.size, "from rows", rows.min(), rows.max())
