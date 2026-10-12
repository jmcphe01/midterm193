"""Generate the placeholder pixel-art assets in assets/.

    python make_assets.py

assets/grass.png       32x32 tileable pixel-art grass for the ground (regenerate freely).
assets/background.png  placeholder backdrop. Replace it with any PNG you like; the game keeps the image's
                       aspect ratio. This script will not overwrite an existing background.png.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)


def make_grass(path, size=32, seed=7):
    rnd = random.Random(seed)
    greens = [(58, 128, 52), (64, 138, 56), (52, 118, 48), (72, 148, 60)]
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = rnd.choices(greens, weights=[5, 4, 3, 1])[0]
    for _ in range(size):                                   # little darker/lighter blades
        x, y = rnd.randrange(size), rnd.randrange(size - 2)
        px[x, y] = (88, 168, 70)
        px[x, y + 1] = (40, 96, 40)
    img.save(path)


def make_background(path, w=640, h=200):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):                                      # sky gradient, top to horizon
        t = y / (h - 1)
        d.line([(0, y), (w, y)], fill=(int(70 + 150 * t), int(120 + 100 * t), int(210 + 30 * t)))
    rnd = random.Random(3)
    for _ in range(9):                                      # blocky clouds
        cx, cy = rnd.randrange(w), rnd.randrange(10, h // 2)
        for i in range(rnd.randrange(3, 6)):
            d.rectangle([cx + i * 14, cy, cx + i * 14 + 18, cy + 8], fill=(248, 250, 255))
    d.rectangle([w - 120, 22, w - 90, 52], fill=(255, 236, 150))   # sun
    for base, col in ((h - 38, (44, 92, 70)), (h - 20, (34, 74, 56))):   # hills, stepped for a pixel look
        x = 0
        while x < w:
            step = rnd.randrange(16, 40)
            top = base - rnd.randrange(0, 22)
            d.rectangle([x, top, x + step, h], fill=col)
            x += step
    img.save(path)


make_grass(ASSETS / "grass.png")
bg = ASSETS / "background.png"
if not bg.exists():
    make_background(bg)
print("assets ready in", ASSETS)
