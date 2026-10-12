"""Screens drawn on top of the scene: title, choose-difficulty, countdown, and the in-game HUD."""
from PIL import Image as PILImage
from panda3d.core import SamplerState, TransparencyAttrib

from render import C, flat          # render first: it configures Ursina for macOS before Ursina is used
from ursina import Entity, Text, Texture, camera, color

import config as cfg
from tags import make_tag_image

TEXT_SHARPNESS = 5      # glyph-texture resolution multiplier (blur fix for large text)
TEXT = C(255, 255, 255)
SHADOW = C(10, 14, 30)
ACCENT = C(255, 214, 90)
CARD = C(18, 30, 62, 190)
CARD_ON = C(46, 150, 96, 230)


class Label:
    """Text with a drop shadow so it stays readable over any part of the scene."""

    def __init__(self, text, y, scale, x=0, col=TEXT, origin=(0, 0)):
        off = 0.0015 + 0.0004 * scale
        self._shadow = Text(text, parent=camera.ui, origin=origin, x=x + off, y=y - off, z=0.02,
                            scale=scale, color=SHADOW)
        self._main = Text(text, parent=camera.ui, origin=origin, x=x, y=y, z=0, scale=scale, color=col)

    @property
    def text(self):
        return self._main.text

    @text.setter
    def text(self, value):
        if value != self._main.text:
            self._main.text = value
            self._shadow.text = value

    @property
    def enabled(self):
        return self._main.enabled

    @enabled.setter
    def enabled(self, value):
        self._main.enabled = self._shadow.enabled = value

    def color(self, col):
        self._main.color = col

    def scale(self, s):
        self._main.scale = self._shadow.scale = s


_TEX = {}


def solid(col):
    """A 1-colour texture. Alpha has to live in the texture: an alpha set via Entity.color is ignored by the
    flat (shader-less) pipeline here, which is why a semi-transparent panel would draw fully opaque."""
    key = tuple(round(c * 255) for c in col)
    if key not in _TEX:
        _TEX[key] = Texture(PILImage.new("RGBA", (8, 8), key))
    return _TEX[key]


def _panel(w, h, x, y, col, z=1):
    p = flat(Entity(parent=camera.ui, model="quad", scale=(w, h), x=x, y=y, z=z, texture=solid(col),
                    color=color.white))
    p.model.setTransparency(TransparencyAttrib.M_alpha)   # on the model node: setting it on the Entity has no effect
    return p


class UI:
    def __init__(self):
        # Ursina builds each text from a small glyph texture and then scales it, so big text (title,
        # countdown numbers) looks blurry at the default resolution. Panda3D shares one font between all
        # Text objects, so raise the default once, before any text is created.
        Text.default_resolution *= TEXT_SHARPNESS
        main, sub = cfg.TITLE.split(" (", 1)
        sub = "(" + sub

        self.dim = _panel(4, 2, 0, 0, C(6, 10, 26, 120), z=3)
        self.title = Label(main.upper(), 0.12, 7.5, col=ACCENT)
        self.subtitle = Label(sub, 0.0, 2.6)
        self.skip = Label("press SPACE to skip", -0.40, 1.3, col=C(220, 228, 245))

        self.choose_head = Label("CHOOSE DIFFICULTY", 0.36, 4.2, col=ACCENT)
        self.choose_sub = Label("hold a tag up to the camera to start", 0.27, 1.9)
        self.cards = []
        for i, d in enumerate(cfg.DIFFICULTIES):
            x = (i - 1) * 0.50
            card = _panel(0.42, 0.50, x, -0.03, CARD, z=2)
            tag = Texture(PILImage.fromarray(make_tag_image(d.tag_id, px=240)).convert("RGBA"))
            tag._texture.set_magfilter(SamplerState.FT_nearest)
            icon = flat(Entity(parent=camera.ui, model="quad", texture=tag, scale=0.30, x=x, y=0.02, z=1.5,
                               color=color.white))
            name = Label(d.name, x=x, y=-0.17, scale=2.6)
            note = Label(f"tag {d.tag_id}", x=x, y=-0.225, scale=1.3, col=C(200, 212, 235))
            self.cards.append((card, icon, name, note))
        self.bar_bg = _panel(0.90, 0.028, 0, -0.34, C(0, 0, 0, 170), z=1)
        self.bar = flat(Entity(parent=camera.ui, model="quad", origin=(-.5, 0), scale=(0.0001, 0.028),
                               x=-0.45, y=-0.34, z=0.5, color=C(90, 225, 130)))
        self.status = Label("", -0.41, 1.4, col=C(220, 228, 245))

        self.count_label = Label("", 0.0, 16, col=ACCENT)
        self.count_diff = Label("", 0.30, 3.2)
        self.count_hint = Label("get ready!", -0.28, 2.4)

        self.hits = Label("0", 0.43, 4.2)
        self.diff_tag = Label("", 0.47, 1.8, x=-0.72, origin=(-0.5, 0))
        self.hint = Label("demo rally  -  R: back to title   Q: quit", -0.47, 1.1, x=-0.74, origin=(-0.5, 0))

        card_items = [self.bar_bg, self.bar, self.choose_head, self.choose_sub, self.status]
        for c in self.cards:
            card_items += list(c)
        self._groups = {
            "title": [self.dim, self.title, self.subtitle, self.skip],
            "choose": [self.dim] + card_items,
            "countdown": [self.count_label, self.count_diff, self.count_hint],
            "play": [self.hits, self.diff_tag, self.hint],
        }
        self._all = {id(i): i for g in self._groups.values() for i in g}.values()
        self.show("title")

    def show(self, state):
        on = {id(i) for i in self._groups[state]}
        for item in self._all:
            item.enabled = id(item) in on

    # ---- per-state updates ----
    def update_choose(self, candidate_tag, progress, status):
        for (card, _, name, note), d in zip(self.cards, cfg.DIFFICULTIES):
            lit = candidate_tag == d.tag_id
            card.texture = solid(CARD_ON if lit else CARD)
            name.color(ACCENT if lit else TEXT)
        self.bar.scale_x = max(0.0001, 0.90 * progress)
        if candidate_tag is not None:
            self.status.text = f"selecting {cfg.BY_TAG[candidate_tag].name}..."
        else:
            self.status.text = status

    def set_countdown(self, text, difficulty_name, pop):
        self.count_label.text = text
        self.count_label.scale(16 * (1 + 0.45 * pop))
        self.count_diff.text = difficulty_name

    def set_play(self, difficulty_name, hits):
        self.diff_tag.text = difficulty_name
        self.hits.text = str(hits)
