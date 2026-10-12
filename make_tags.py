"""Generate the printable difficulty tags into assets/tags/.

    python make_tags.py

  print_1_easy.png, print_2_medium.png, print_3_hard.png
                              one full landscape-letter page per tag, 300 dpi, with a label: print these
  difficulty_tags.pdf         the same three pages as one PDF
  difficulty_tags_sheet.png   all three on one page, for a quick look
  tag_easy_id1.png ...        the bare tag only (white border included), for reuse

Print at 100% scale (no 'fit to page' shrinking) on white paper. Bigger is easier for the camera to detect.
Every tag is drawn at a whole number of pixels per cell so the printed edges are perfectly sharp.
"""
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from config import DIFFICULTIES
from tags import make_tag_image

OUT = Path(__file__).parent / "assets" / "tags"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
PAGE_W, PAGE_H = int(11 * DPI), int(8.5 * DPI)       # US letter, landscape (3300 x 2550)


def font(size):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "/Library/Fonts/Arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def centred(draw, text, cx, y, fnt):
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, fill=0, font=fnt)


pages = []
for d in DIFFICULTIES:
    Image.fromarray(make_tag_image(d.tag_id, px=800)).convert("L").save(OUT / f"tag_{d.name.lower()}_id{d.tag_id}.png")

    tag = Image.fromarray(make_tag_image(d.tag_id, px=1280)).convert("L")   # 160 px per cell -> 1920 px with border
    page = Image.new("L", (PAGE_W, PAGE_H), 255)
    page.paste(tag, ((PAGE_W - tag.width) // 2, 40))
    draw = ImageDraw.Draw(page)
    centred(draw, f"{d.name}  (tag {d.tag_id})", PAGE_W / 2, 40 + tag.height + 70, font(150))
    page.save(OUT / f"print_{d.tag_id}_{d.name.lower()}.png", dpi=(DPI, DPI))
    pages.append(page)

pages[0].save(OUT / "difficulty_tags.pdf", save_all=True, append_images=pages[1:], resolution=DPI)

sheet = Image.new("L", (PAGE_W, PAGE_H), 255)
sd = ImageDraw.Draw(sheet)
col_w = PAGE_W // 3
for i, d in enumerate(DIFFICULTIES):
    tag = Image.fromarray(make_tag_image(d.tag_id, px=672)).convert("L")    # 84 px per cell -> 1008 px with border
    sheet.paste(tag, (i * col_w + (col_w - tag.width) // 2, 560))
    centred(sd, d.name, i * col_w + col_w / 2, 340, font(130))
    centred(sd, f"tag {d.tag_id}", i * col_w + col_w / 2, 560 + tag.height + 30, font(80))
sheet.save(OUT / "difficulty_tags_sheet.png", dpi=(DPI, DPI))
print("wrote", *sorted(p.name for p in OUT.iterdir()), sep="\n  ")
