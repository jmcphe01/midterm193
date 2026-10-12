"""Camera thread: grabs frames and finds AprilTags without ever blocking the game's render loop.

Frame sources: a webcam (default), or a still image looped as a fake camera (`image_path`, for testing).
Pose estimation will be added to this same thread later.
"""
import threading
import time
from dataclasses import dataclass, field

import cv2

from tags import TagReader


@dataclass
class Snapshot:
    tags: dict = field(default_factory=dict)   # {tag_id: pixel_area}
    frame: object = None                        # latest BGR frame (not flipped)
    stamp: float = 0.0
    ok: bool = False                            # True once frames are arriving
    error: str = ""


class Vision(threading.Thread):
    def __init__(self, camera_index=0, image_path=None, width=1280, height=720):
        super().__init__(daemon=True)
        self.camera_index, self.image_path = camera_index, image_path
        self.size = (width, height)
        self._reader = TagReader()
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self._halt = threading.Event()
        self._cap, self._still = None, None

    def open(self):
        """Open the frame source. Call from the MAIN thread before start(): macOS shows its camera
        permission prompt from the thread that first opens the camera, and it must be the main one."""
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

    def run(self):
        cap, still = self._cap, self._still
        if cap is None and still is None:
            return                                   # open() failed; the error is already published
        try:
            while not self._halt.is_set():
                if still is not None:
                    frame = still.copy()
                    time.sleep(1 / 30)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        self._publish(ok=False, error="camera is not delivering frames")
                        time.sleep(0.1)
                        continue
                self._publish(tags=self._reader.detect(frame), frame=frame, stamp=time.time(), ok=True, error="")
        finally:
            if cap is not None:
                cap.release()
