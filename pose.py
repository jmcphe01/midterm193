"""Pose tracking: finds the racket hand in a camera frame with MediaPipe.

The hand is reported as its position in the camera image, (ax, ay) as fractions of the frame, mirrored like a mirror
and with up as positive (see control.py). Only the wrist has to be visible; the shoulders and elbow are used just for
drawing the arm on the preview.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

MODEL_PATH = Path(__file__).parent / "assets" / "pose_landmarker_lite.task"

# MediaPipe landmark numbers. These are anatomical: "right" is the PERSON's right, however the image is mirrored.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
WRIST_MIN_VIS = 0.2             # the wrist is often half-hidden (arm across the body, paddle in hand); a low-
                                # visibility estimate is still far more useful than losing the hand
DRAW_MIN_VIS = 0.3              # shoulders / elbow below this are not drawn
TRACK_CONFIDENCE = 0.3          # MediaPipe's own thresholds (default 0.5): lower keeps tracking through occlusion


@dataclass
class PoseSample:
    ax: float                    # hand position across the mirrored picture, 0..1: larger = more to YOUR right
    ay: float                    # hand position up the picture, 0..1: larger = higher
    stamp: float                 # time.time() when the frame was captured
    points: dict = field(default_factory=dict)   # name -> (x_px, y_px) in the ORIGINAL (un-mirrored) frame
    confidence: float = 1.0      # wrist visibility 0..1 (low = half hidden, so the position is less certain)


def to_pixel(ax, ay, width, height):
    """Where the image position (ax, ay) lands in the ORIGINAL (un-mirrored) frame, for drawing overlays."""
    return ((1.0 - ax) * width, (1.0 - ay) * height)


class PoseTracker:
    """Wraps MediaPipe's PoseLandmarker. Create it and call process() from ONE thread (it is not thread-safe)."""

    def __init__(self, hand="right"):
        if hand not in ("right", "left"):
            raise ValueError("hand must be 'right' or 'left'")
        self.hand = hand
        self._wrist = R_WRIST if hand == "right" else L_WRIST
        self._elbow = R_ELBOW if hand == "right" else L_ELBOW
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"missing pose model: {MODEL_PATH}")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=TRACK_CONFIDENCE, min_pose_presence_confidence=TRACK_CONFIDENCE,
            min_tracking_confidence=TRACK_CONFIDENCE))
        self._last_ms = 0
        self.reason = ""              # why the last frame gave no sample (for the UI)

    def process(self, bgr):
        """Returns a PoseSample, or None if the racket hand isn't visible in this frame."""
        h, w = bgr.shape[:2]
        ms = max(int(time.time() * 1000), self._last_ms + 1)      # VIDEO mode needs strictly increasing times
        self._last_ms = ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(image, ms)
        self.reason = ""
        if not result.pose_landmarks:
            self.reason = "no person in view"
            return None
        lm = result.pose_landmarks[0]
        if lm[self._wrist].visibility < WRIST_MIN_VIS:
            self.reason = "your racket hand is hidden"
            return None
        px = lambda i: (lm[i].x * w, lm[i].y * h)                 # normalised -> pixels
        wrist = px(self._wrist)
        points = {"wrist": wrist}
        for name, idx in (("l_shoulder", L_SHOULDER), ("r_shoulder", R_SHOULDER), ("elbow", self._elbow)):
            if lm[idx].visibility >= DRAW_MIN_VIS:
                points[name] = px(idx)
        # mirrored, like the preview: your right hand is on the right, so flip x; y is flipped so up is positive
        return PoseSample(1.0 - wrist[0] / w, 1.0 - wrist[1] / h, time.time(), points,
                          confidence=lm[self._wrist].visibility)

    def close(self):
        self._landmarker.close()


def draw_overlay(frame, sample, marks=None, box=None):
    """Draw the tracked arm (and, if given, calibration marks / the reach rectangle) onto `frame` in place. `marks`
    and `box` are (ax, ay) points. No text, so the picture can be mirrored afterwards."""
    h, w = frame.shape[:2]
    if box:
        poly = [tuple(int(c) for c in to_pixel(ax, ay, w, h)) for ax, ay in box]
        for a, b in zip(poly, poly[1:] + poly[:1]):
            cv2.line(frame, a, b, (255, 214, 90), 2)
    for ax, ay in (marks or []):
        cv2.circle(frame, tuple(int(c) for c in to_pixel(ax, ay, w, h)), 9, (255, 214, 90), -1)
    if sample is None:
        return frame
    pts = {k: tuple(int(c) for c in v) for k, v in sample.points.items()}
    if "l_shoulder" in pts and "r_shoulder" in pts:
        cv2.line(frame, pts["l_shoulder"], pts["r_shoulder"], (255, 255, 255), 3)
    if "elbow" in pts:
        shoulders = [pts[k] for k in ("l_shoulder", "r_shoulder") if k in pts]
        if shoulders:
            near = min(shoulders, key=lambda p: (p[0] - pts["elbow"][0]) ** 2 + (p[1] - pts["elbow"][1]) ** 2)
            cv2.line(frame, near, pts["elbow"], (90, 220, 120), 4)
        cv2.line(frame, pts["elbow"], pts["wrist"], (90, 220, 120), 4)
    shaky = sample.confidence < 0.5                                    # half hidden: show it in orange
    cv2.circle(frame, pts["wrist"], 14, (40, 150, 255) if shaky else (60, 60, 230), -1)
    cv2.circle(frame, pts["wrist"], 14, (255, 255, 255), 2)
    return frame
