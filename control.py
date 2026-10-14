"""Turning a tracked hand into a racket position. Pure maths: no camera, no Ursina, so it is easy to test.

The hand is measured by where it is in the camera image: (ax, ay) as fractions of the frame, mirrored like a mirror
and with up as positive:
    ax: 0 = the left edge of the mirrored picture, 1 = the right edge  (your own right is larger)
    ay: 0 = the bottom edge, 1 = the top edge
The camera is assumed to stay put after calibration, so no correction is applied for where you stand.
Calibration records the four corners of the region you can comfortably reach. It is treated as a plain rectangle
(left edge = the average of your two left corners, and so on); there is no perspective warp. Any (ax, ay) is then
mapped linearly onto the playing area (metres).
"""
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

# The rectangle on the racket plane that the calibrated reach region maps onto (metres).
RACKET_X = (-1.0, 1.0)
RACKET_Y = (0.85, 1.55)
CORNER_NAMES = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT")
CALIBRATION_PATH = Path(__file__).parent / "calibration.json"
SPACE = "image"          # saved with the calibration; older (shoulder-relative) calibrations are not compatible

# the hand must be steadier than this (fraction of the frame width) for a corner to be taken, and must have travelled
# at least MIN_TRAVEL from the previous corner before the next one counts
STEADY_RADIUS = 0.025
MIN_TRAVEL = 0.07


class OneEuro:
    """Adaptive low-pass filter (the '1 euro' filter): smooth when the hand is slow, responsive when it is fast,
    so a trembling hand is steady but a swing is not delayed. `beta` is tuned for image-fraction coordinates."""

    def __init__(self, min_cutoff=1.2, beta=4.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = self._dx = self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t is None:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        if t <= self._t:                    # the same camera frame seen again: nothing new to filter
            return self._x
        dt = t - self._t
        dx = (x - self._x) / dt
        self._dx += self._alpha(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        self._x += self._alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x


class Calibration:
    """The mapping from hand (ax, ay) to racket-plane (x, y): a plain rectangle, no perspective warp."""

    def __init__(self, corners):
        self.corners = [tuple(map(float, c)) for c in corners]          # TL, TR, BR, BL
        tl, tr, br, bl = self.corners
        self.x0, self.x1 = (tl[0] + bl[0]) / 2, (tr[0] + br[0]) / 2      # left / right edge of the reach rectangle
        self.y1, self.y0 = (tl[1] + tr[1]) / 2, (bl[1] + br[1]) / 2      # top / bottom edge

    def map(self, ax, ay):
        x = RACKET_X[0] + (ax - self.x0) / (self.x1 - self.x0) * (RACKET_X[1] - RACKET_X[0])
        y = RACKET_Y[0] + (ay - self.y0) / (self.y1 - self.y0) * (RACKET_Y[1] - RACKET_Y[0])
        return float(x), float(y)

    @property
    def box(self):
        """The reach rectangle as four (ax, ay) points, for drawing it on the camera preview."""
        return [(self.x0, self.y1), (self.x1, self.y1), (self.x1, self.y0), (self.x0, self.y0)]

    @staticmethod
    def check(corners):
        """None if the four corners make a usable region, otherwise a short reason."""
        pts = np.float32(corners)
        tl, tr, br, bl = pts
        left, right = (tl[0] + bl[0]) / 2, (tr[0] + br[0]) / 2
        top, bottom = (tl[1] + tr[1]) / 2, (bl[1] + br[1]) / 2
        if right - left < 0.10:
            return "the left and right corners are too close together (or in the wrong order) - reach wider"
        if top - bottom < 0.06:
            return "the top and bottom corners are too close together (or in the wrong order) - reach higher and lower"
        return None

    def save(self, path=CALIBRATION_PATH):
        Path(path).write_text(json.dumps({"space": SPACE, "corners": self.corners}, indent=2))

    @classmethod
    def load(cls, path=CALIBRATION_PATH):
        try:
            blob = json.loads(Path(path).read_text())
            if blob.get("space") != SPACE:                 # an older, shoulder-relative calibration: redo it
                return None
            corners = blob["corners"]
            return cls(corners) if len(corners) == 4 and Calibration.check(corners) is None else None
        except (OSError, ValueError, KeyError, TypeError):
            return None


class CornerCalibrator:
    """Collects the four corners. A corner is taken once the hand holds still for `hold` seconds (or accept_now()).
    After each corner the hand must travel away before the next one can be taken."""

    def __init__(self, hold=1.0, steady_radius=STEADY_RADIUS, min_travel=MIN_TRAVEL):
        self.hold, self.radius, self.min_travel = hold, steady_radius, min_travel
        self.corners = []
        self._anchor = None          # (ax, ay, t_start) of the current steady stretch
        self._samples = deque(maxlen=120)

    @property
    def index(self):
        return len(self.corners)

    @property
    def done(self):
        return len(self.corners) == 4

    def reset(self):
        self.corners.clear()
        self._anchor = None
        self._samples.clear()

    def _far_from_last(self, ax, ay):
        if not self.corners:
            return True
        lx, ly = self.corners[-1]
        return math.hypot(ax - lx, ay - ly) >= self.min_travel

    def update(self, pos, now):
        """pos is the current hand (ax, ay) or None. Returns hold progress 0..1 for the corner being captured."""
        if self.done:
            return 1.0
        if pos is None or not self._far_from_last(*pos):
            self._anchor = None
            self._samples.clear()
            return 0.0
        x, y = pos
        if self._anchor is None or math.hypot(x - self._anchor[0], y - self._anchor[1]) > self.radius:
            self._anchor = (x, y, now)
            self._samples.clear()
        self._samples.append((x, y))
        progress = min(1.0, (now - self._anchor[2]) / self.hold)
        if progress >= 1.0:
            self.accept_now(pos)
            return 0.0
        return progress

    def accept_now(self, pos):
        """Take the current hand position as the next corner (used for the Space key)."""
        if pos is None or self.done:
            return False
        if self._samples:
            xs, ys = zip(*self._samples)
            pos = (sum(xs) / len(xs), sum(ys) / len(ys))
        self.corners.append((float(pos[0]), float(pos[1])))
        self._anchor = None
        self._samples.clear()
        return True
