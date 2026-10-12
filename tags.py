"""AprilTag (36h11) generation, detection, and the 'hold it steady to select' logic."""
import cv2
import numpy as np

from config import BY_TAG, TAG_GRACE_SECONDS, TAG_HOLD_SECONDS

_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def make_tag_image(tag_id, px=800, quiet_cells=2):
    """White-bordered AprilTag as a uint8 grayscale image. 36h11 is 6x6 data cells plus a 1-cell black border."""
    marker = cv2.aruco.generateImageMarker(_DICT, tag_id, px)
    margin = (px // 8) * quiet_cells          # an AprilTag is 8 cells wide including its black border
    return cv2.copyMakeBorder(marker, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


class TagReader:
    """Finds AprilTags in a frame. Returns {tag_id: pixel_area}, largest = closest/most prominent."""

    def __init__(self, only_ids=None):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(_DICT, params)
        self._only = set(only_ids) if only_ids is not None else None

    def detect(self, frame):
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        found = {}
        if ids is not None:
            for quad, tag_id in zip(corners, ids.ravel()):
                tag_id = int(tag_id)
                if self._only is None or tag_id in self._only:
                    found[tag_id] = float(cv2.contourArea(quad.reshape(4, 2).astype(np.float32)))
        return found


class HoldSelector:
    """Turns noisy per-frame detections into one decision: a tag must stay in view for `hold` seconds.

    update() returns (candidate_tag_id | None, progress 0..1, confirmed_tag_id | None).
    Brief dropouts (<= grace seconds) don't reset the hold; showing a different tag restarts it.
    """

    def __init__(self, hold=TAG_HOLD_SECONDS, grace=TAG_GRACE_SECONDS, valid_ids=None):
        self.hold, self.grace = hold, grace
        self.valid = set(valid_ids if valid_ids is not None else BY_TAG)
        self.reset()

    def reset(self):
        self._cand = None
        self._since = 0.0
        self._last_seen = 0.0

    def update(self, detections, now):
        seen = {i: a for i, a in detections.items() if i in self.valid}
        if seen:
            best = max(seen, key=seen.get)                 # biggest tag wins if several are visible
            if best != self._cand:
                self._cand, self._since = best, now
            self._last_seen = now
        elif self._cand is not None and now - self._last_seen > self.grace:
            self._cand = None
        if self._cand is None:
            return None, 0.0, None
        progress = min(1.0, (now - self._since) / self.hold)
        return self._cand, progress, (self._cand if progress >= 1.0 else None)
