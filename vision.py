"""Camera thread: grabs frames, finds AprilTags and the racket hand, and builds a preview picture, all without
ever blocking the game's render loop.

Frame sources: a webcam (default), a still image looped as a fake camera (`image_path`, for testing), or a fully
synthetic scripted hand (`fake_pose`, for testing the pose/calibration flow without a person).
"""
import math
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from pose import PoseSample, draw_overlay
from tags import TagReader

VIEW_SIZE = (480, 270)          # preview picture sent to the UI (16:9)


@dataclass
class Snapshot:
    tags: dict = field(default_factory=dict)   # {tag_id: pixel_area}
    pose: object = None                         # latest PoseSample, or None when nobody is visible
    view: object = None                         # mirrored, annotated preview (BGR, VIEW_SIZE)
    stamp: float = 0.0
    ok: bool = False                            # True once frames are arriving
    error: str = ""
    pose_error: str = ""
    pose_reason: str = ""                      # why there is no pose right now (for the on-screen hint)


class FakeHand:
    """A scripted hand for testing: visits the four reach corners (holding still at each), then circles."""

    CORNERS = [(0.15, 0.95), (1.55, 1.0), (1.60, -0.35), (0.10, -0.30)]    # u, v in shoulder widths
    HOME = (0.85, 0.30)
    MOVE, HOLD = 0.7, 1.5

    def uv(self, t):
        t -= 0.6                                              # a moment at home first
        if t < 0:
            return self.HOME
        pts = [self.HOME] + self.CORNERS
        seg = self.MOVE + self.HOLD
        k = int(t / seg)
        if k < 4:
            a, b, s = pts[k], pts[k + 1], t - k * seg
            f = min(1.0, s / self.MOVE)
            f = f * f * (3 - 2 * f)                           # ease in/out
            return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
        s = t - 4 * seg                                       # then circle around the middle of the region
        return 0.85 + 0.65 * math.cos(s * 2.2), 0.30 + 0.55 * math.sin(s * 2.2)

    def frame_and_sample(self, t, size=VIEW_SIZE):
        w, h = size
        frame = np.full((h, w, 3), (58, 52, 48), np.uint8)
        cv2.rectangle(frame, (0, int(h * 0.82)), (w, h), (70, 64, 58), -1)
        sw = 0.23 * w                                       # scale of the drawn person (shoulder width, pixels)
        centre = (w / 2, h * 0.58)
        cv2.circle(frame, (int(centre[0]), int(centre[1] - sw * 0.95)), int(sw * 0.42), (150, 170, 190), -1)   # head
        cv2.rectangle(frame, (int(centre[0] - sw * 0.5), int(centre[1])), (int(centre[0] + sw * 0.5), h), (96, 80, 70), -1)
        u, v = self.uv(t)                                   # the scripted hand, in shoulder widths from the body
        wrist = (centre[0] - u * sw, centre[1] - v * sw)
        right_shoulder = (centre[0] - sw / 2, centre[1])                        # the person's right is image-left
        elbow = ((right_shoulder[0] + wrist[0]) / 2, (right_shoulder[1] + wrist[1]) / 2 + sw * 0.25)
        points = {"l_shoulder": (centre[0] + sw / 2, centre[1]), "r_shoulder": right_shoulder, "wrist": wrist,
                  "elbow": elbow}
        return frame, PoseSample(1.0 - wrist[0] / w, 1.0 - wrist[1] / h, time.time(), points)


class Vision(threading.Thread):
    def __init__(self, camera_index=0, image_path=None, width=1280, height=720, pose=False, hand="right",
                 fake_pose=False):
        super().__init__(daemon=True)
        self.camera_index, self.image_path = camera_index, image_path
        self.size = (width, height)
        self.want_pose, self.hand, self.fake_pose = pose, hand, fake_pose
        self.want_tags = True                       # the game turns this off outside the choose screen (saves CPU)
        self.overlay = ([], None)                   # (corner marks, reach rectangle), as (ax, ay) points, drawn on the preview
        self._reader = TagReader()
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self._halt = threading.Event()
        self._cap, self._still = None, None

    def open(self):
        """Open the frame source. Call from the MAIN thread before start(): macOS shows its camera
        permission prompt from the thread that first opens the camera, and it must be the main one."""
        if self.fake_pose:
            return True
        if self.image_path:
            self._still = cv2.imread(str(self.image_path))
            if self._still is None:
                self._publish(error=f"cannot read {self.image_path}")
            return self._still is not None
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.camera_index, backend)
        if not self._cap.isOpened():
            self._publish(error=f"cannot open camera {self.camera_index}")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        return True

    def get(self):
        with self._lock:
            return self._snap

    def stop(self):
        self._halt.set()

    def _publish(self, **kw):
        with self._lock:
            self._snap = Snapshot(**{**self._snap.__dict__, **kw})

    def _make_view(self, frame, sample):
        """Small, mirrored picture with the tracked arm (and calibration marks) drawn on it."""
        h, w = frame.shape[:2]
        view = cv2.resize(frame, VIEW_SIZE, interpolation=cv2.INTER_AREA)
        k = VIEW_SIZE[0] / w
        scaled = None
        if sample is not None:
            scaled = PoseSample(sample.ax, sample.ay, sample.stamp,
                                {n: (x * k, y * k) for n, (x, y) in sample.points.items()}, sample.confidence)
        marks, box = self.overlay
        draw_overlay(view, scaled, marks, box)
        return cv2.flip(view, 1)                    # mirror, like a mirror: your right hand is on the right

    def run(self):
        tracker, hand = None, FakeHand() if self.fake_pose else None
        if self.want_pose and not self.fake_pose:
            try:
                from pose import PoseTracker
                tracker = PoseTracker(self.hand)    # created here: MediaPipe objects must stay in one thread
            except Exception as exc:                # noqa: BLE001 - report anything (missing model, bad install)
                self._publish(pose_error=f"pose tracking unavailable: {exc}")
        cap, still = self._cap, self._still
        if cap is None and still is None and hand is None:
            return                                   # open() failed; the error is already published
        t0 = time.time()
        try:
            while not self._halt.is_set():
                sample = None
                if hand is not None:
                    frame, sample = hand.frame_and_sample(time.time() - t0)
                    time.sleep(1 / 30)
                elif still is not None:
                    frame = still.copy()
                    time.sleep(1 / 30)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        self._publish(ok=False, error="camera is not delivering frames")
                        time.sleep(0.1)
                        continue
                tags = self._reader.detect(frame) if (self.want_tags and hand is None) else {}
                if tracker is not None:
                    try:
                        sample = tracker.process(frame)
                    except Exception as exc:        # noqa: BLE001
                        self._publish(pose_error=f"pose tracking failed: {exc}")
                        tracker = None
                self._publish(tags=tags, pose=sample, view=self._make_view(frame, sample), stamp=time.time(),
                              ok=True, error="", pose_reason=(tracker.reason if tracker is not None else ""))
        finally:
            if cap is not None:
                cap.release()
            if tracker is not None:
                tracker.close()
