"""Swing detection from the hub's IMU: spot the START of a swing, then classify it as 'right' or 'left' (or not a
swing at all) from the first ~0.1 s of motion with a small neural network.

Why classify early: the ball is only returnable for a moment around the racket. Waiting for the whole swing to
finish would deliver the answer after the ball has already gone past. A swing's direction is already visible in its
first 100 ms, so we decide then.

collect_swings.py (recording), train_swings.py (training) and the game all share the feature code in this file.
If it ever differs between them, the model still loads and still predicts, just nonsense, with no error.
"""
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATASET_PATH = DATA_DIR / "swing_data.joblib"
MODEL_PATH = DATA_DIR / "swing_model.joblib"


def paths(fake=False):
    """(dataset path, model path). Simulated-hub data lives in data/fake/ so it can never mix with real recordings."""
    base = DATA_DIR / "fake" if fake else DATA_DIR
    return base / "swing_data.joblib", base / "swing_model.joblib"


PRE, POST = 0.15, 0.03          # classification window around the onset (s). A swing is a ~100 ms flick, so the
                                # decision must come within ~30 ms of the trigger or it arrives after the contact
N_POINTS = 24                   # the window is resampled to this many points per channel (any hub sample rate)
RECORD_PRE, RECORD_POST = 0.5, 0.7   # raw seconds around each onset kept when recording (for augmentation)
COOLDOWN = 0.35                 # s after an event before another can start
REARM_FRACTION = 0.5            # energy must fall below this fraction of the threshold before re-arming
MIN_SWING_PROB = 0.5            # P(right) + P(left) must reach this for the motion to count as a swing at all
MIN_DIRECTION_CONFIDENCE = 0.7  # ... and one direction must hold this share of it, else it is a swing of unknown direction
SWING_LABELS = ("right", "left")   # the two kinds of swing. There is no "not a swing" class: a motion counts as a
ALL_LABELS = SWING_LABELS          # swing when it crosses the trigger, and the network only decides WHICH WAY


def decide(classes, proba, min_swing_prob=MIN_SWING_PROB):
    """The decision rule shared by the game and the training report. Returns (label, confidence) or (None, p):
    'right' / 'left', or 'swing' when it is clearly a swing but the direction is unsure (it then aims straight).
    A lower `min_swing_prob` makes the game more willing to call a motion a swing (more sensitive)."""
    p = dict(zip(map(str, classes), map(float, proba)))
    p_swing = p.get("right", 0.0) + p.get("left", 0.0)
    if p_swing < min_swing_prob:
        return None, p_swing
    direction = "right" if p.get("right", 0.0) >= p.get("left", 0.0) else "left"
    if max(p.get("right", 0.0), p.get("left", 0.0)) / p_swing < MIN_DIRECTION_CONFIDENCE:
        return "swing", p_swing
    return direction, p_swing


@dataclass
class SwingEvent:
    label: str            # 'right' | 'left', or 'swing' when no trained model is available (direction unknown)
    confidence: float
    t: float              # when the swing STARTED (host clock), so the game can account for the detection delay
    strength: float       # 1.0 = a typical swing from training; the game scales the return pace with it


def gyro_energy(values, bias):
    """Rotation speed (raw hub units) with the resting gyro bias removed. values: (n, 6) or (6,)."""
    v = np.atleast_2d(values)
    return np.linalg.norm(v[:, 3:6] - np.asarray(bias), axis=1)


def window_features(times, values, onset, bias, pre=PRE, post=POST, n=N_POINTS):
    """Feature vector for the window [onset - pre, onset + post], and its peak gyro energy.

    Resampling by timestamp makes this independent of the hub's (variable) Bluetooth rate. The gyro bias and the
    resting accelerometer reading (gravity, taken at the start of the window) are removed, so what is left is the
    motion itself, not how the hub happens to be sitting."""
    grid = np.linspace(onset - pre, onset + post, n)
    w = np.stack([np.interp(grid, times, values[:, k]) for k in range(6)], axis=1)
    w[:, 3:] -= np.asarray(bias)
    w[:, :3] -= w[0, :3]
    peak = float(np.linalg.norm(w[:, 3:6], axis=1).max())
    return np.concatenate([w.ravel(), w.max(axis=0), w.min(axis=0)]), peak


def estimate_rest(values):
    """(gyro bias, noise level of the rotation energy) from a stretch where the hub sat still."""
    bias = values[:, 3:6].mean(axis=0)
    e = gyro_energy(values, bias)
    return bias, float(e.mean()), float(e.std())


def find_onsets(t, v, bias, threshold, first_only=False):
    """Times where the rotation energy crosses `threshold`, one per separate burst of motion.

    The same arming rule as the live detector (it must calm down between bursts, plus a cooldown), so recordings
    are aligned exactly the way the game will align live swings."""
    energy = gyro_energy(v, bias)
    onsets, armed, quiet, last = [], True, None, -99.0
    for ti, ei in zip(t, energy):
        if armed:
            if ei > threshold and ti - last > COOLDOWN:
                onsets.append(float(ti))
                armed, quiet, last = False, None, ti + POST
                if first_only:
                    break
        elif ei < REARM_FRACTION * threshold:
            quiet = quiet if quiet is not None else ti
            if ti - quiet >= 0.08:
                armed = True
        else:
            quiet = None
    return onsets


def default_threshold(noise_mean, noise_std):
    """Rotation energy that counts as 'the paddle started moving': well above the sensor noise, and (floor) high
    enough that casual handling does not set it off. The floor suits LEGO hub readings, where a swing peaks at
    roughly 1000-2000."""
    return max(noise_mean + 10 * noise_std, 250.0)


def event_from_samples(samples, bias, label, seconds=1.6, min_peak=100.0):
    """Turn the hub's most recent `seconds` of readings into one labelled training example (the same format
    collect_swings.py stores), or None if nothing moved. Used by the game's teach mode: you press a key right
    after a swing the game got wrong and that motion becomes an example of what you meant."""
    if not samples:
        return None
    end = samples[-1].t
    recent = [x for x in samples if x.t >= end - seconds]
    if len(recent) < 8:
        return None
    t = np.array([x.t for x in recent])
    v = np.array([[x.ax, x.ay, x.az, x.gx, x.gy, x.gz] for x in recent])
    energy = gyro_energy(v, bias)
    peak_i = int(np.argmax(energy))
    if energy[peak_i] < min_peak:
        return None
    return {"label": label, "t": t - t[peak_i], "v": v, "peak": float(energy[peak_i]), "stamp": float(t[peak_i])}


def append_event(dataset_path, event, bias, threshold, replace_last=False):
    """Add `event` to the recordings on disk (creating the file if needed). With replace_last, the most recently
    added example is dropped first (used to correct a mislabelled key press). Returns how many of its class exist."""
    import joblib
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    data = joblib.load(dataset_path) if dataset_path.exists() else {"events": [], "sessions": []}
    if not data["sessions"]:
        data["sessions"].append({"bias": np.asarray(bias), "threshold": float(threshold), "noise_mean": 0.0,
                                 "noise_std": 0.0, "rate": 0.0, "time": 0.0, "fake": False, "manual": False})
    if replace_last and data["events"]:
        data["events"].pop()
    data["events"].append(event)
    joblib.dump(data, dataset_path)
    return sum(1 for e in data["events"] if e["label"] == event["label"])


def load_bundle(path=MODEL_PATH):
    """The trained model and everything the live detector needs, or None if it hasn't been trained yet."""
    import joblib
    try:
        return joblib.load(path)
    except (OSError, ValueError, EOFError):
        return None


class SwingDetector:
    """Feed it IMU samples; it returns SwingEvents.

    With a trained bundle: onset -> wait POST seconds -> decide right / left.
    Without one (not trained yet): learns the resting level from the first second, then reports every swing as
    label 'swing' (timing and strength only, no direction)."""

    REST_SECONDS = 1.2

    def __init__(self, bundle=None):
        self.bundle = bundle
        self.model = bundle["model"] if bundle else None
        self.threshold = bundle["threshold"] if bundle else None
        self.bias = np.asarray(bundle["bias"]) if bundle else None
        self.ref_peak = bundle.get("ref_peak", 600.0) if bundle else 600.0
        self.min_swing_prob = bundle.get("min_swing_prob", MIN_SWING_PROB) if bundle else MIN_SWING_PROB
        self.on_trigger = None            # optional callback(info dict) for every trigger, to see what the hub does
        self.pre, self.post = (bundle["pre"], bundle["post"]) if bundle else (PRE, POST)
        self._buf = deque()
        self._onset = None
        self._last_event = -99.0
        self._armed = True
        self._quiet_since = None
        self._rest = []                   # samples used to learn the resting level when there is no model
        self._weak_peak = 0.0             # strongest recent motion that stayed below the trigger (diagnostics only)
        self.last_energy = 0.0

    @property
    def ready(self):
        return self.threshold is not None

    def recent_samples(self):
        """The hub readings of the last ~2 seconds (used by the game's teach mode)."""
        return list(self._buf)

    def status(self):
        if self.model is not None:
            return "swing model loaded"
        return "no trained model: any swing counts" if self.ready else "hold the paddle still..."

    def _learn_rest(self, s):
        self._rest.append(s)
        if self._rest[-1].t - self._rest[0].t >= self.REST_SECONDS:
            vals = np.array([[x.ax, x.ay, x.az, x.gx, x.gy, x.gz] for x in self._rest])
            self.bias, mean, std = estimate_rest(vals)
            self.threshold = default_threshold(mean, std)

    def feed(self, samples):
        events = []
        for s in samples:
            self._buf.append(s)
            while self._buf and s.t - self._buf[0].t > 2.0:
                self._buf.popleft()
            if not self.ready:
                self._learn_rest(s)
                continue
            e = float(gyro_energy(np.array([[s.ax, s.ay, s.az, s.gx, s.gy, s.gz]]), self.bias)[0])
            self.last_energy = e
            if self.on_trigger is not None and self._onset is None and self._armed:   # a motion that never reached the trigger
                if self.threshold * 0.3 < e < self.threshold:
                    self._weak_peak = max(self._weak_peak, e)
                elif e < self.threshold * 0.2 and self._weak_peak:
                    if self._weak_peak >= 0.5 * self.threshold:
                        self.on_trigger({"label": "below_trigger", "peak": self._weak_peak, "threshold": self.threshold})
                    self._weak_peak = 0.0
                elif e >= self.threshold:
                    self._weak_peak = 0.0
            if self._onset is None:
                if not self._armed:                                    # wait for the previous motion to die down
                    if e < REARM_FRACTION * self.threshold:
                        self._quiet_since = self._quiet_since or s.t
                        if s.t - self._quiet_since >= 0.08 and s.t - self._last_event > COOLDOWN:
                            self._armed = True
                    else:
                        self._quiet_since = None
                elif e > self.threshold:
                    self._onset, self._armed = s.t, False
            elif s.t >= self._onset + self.post:
                event = self._classify(self._onset)
                self._last_event = s.t
                self._onset, self._quiet_since = None, None
                if event is not None:
                    events.append(event)
        return events

    def _classify(self, onset):
        window = [x for x in self._buf if onset - self.pre - 0.05 <= x.t <= onset + self.post + 0.05]
        if len(window) < 4:
            return None
        t = np.array([x.t for x in window])
        v = np.array([[x.ax, x.ay, x.az, x.gx, x.gy, x.gz] for x in window])
        feats, peak = window_features(t, v, onset, self.bias, self.pre, self.post)
        strength = float(np.clip(peak / self.ref_peak, 0.0, 2.5))
        if self.model is None:
            if self.on_trigger is not None:
                self.on_trigger({"onset": onset, "peak": peak, "label": "swing", "p": {}})
            return SwingEvent("swing", 1.0, onset, strength)
        proba = self.model.predict_proba([feats])[0]
        label, conf = decide(self.model.classes_, proba, self.min_swing_prob)
        if self.on_trigger is not None:
            self.on_trigger({"onset": onset, "peak": peak, "label": label, "p": dict(zip(map(str, self.model.classes_), map(float, proba)))})
        if label is None:
            return None
        return SwingEvent(str(label), conf, onset, strength)
