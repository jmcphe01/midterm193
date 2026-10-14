"""Reads the LEGO Education Double Motor's IMU (accelerometer + gyroscope) on its own thread.

The hub talks over Bluetooth Low Energy through the `legoeducation` library, which keeps the latest reading in
`robot.imu_device`. We poll that, timestamp each NEW reading, and queue it for the swing detector. Connecting can
take seconds (a Bluetooth scan), so it happens on this thread, never on the game's.

`FakeHub` produces a believable synthetic signal so everything downstream can be tested without a hub. Its
waveforms are an invention for testing: the real accuracy has to be measured on your own recordings.
"""
import math
import random
import threading
import time
from collections import deque, namedtuple

import numpy as np

Sample = namedtuple("Sample", "t ax ay az gx gy gz")      # t = host receipt time (s); raw hub units
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")


def parse_card(text):
    """'red:1129' -> (le.LEGO_COLOR_RED, '1129'). The Connection Card picks which hub to use when several are on."""
    import legoeducation as le
    color, _, serial = text.partition(":")
    value = getattr(le, "LEGO_COLOR_" + color.strip().upper(), None)
    if value is None or not serial:
        raise ValueError(f"--card must look like red:1129 (colour:serial), got {text!r}")
    return value, serial.strip()


class FakeHub:
    """A simulated hub. Idle = sensor noise + a gravity vector + a gyro bias. swing('right'|'left'|'bump', strength)
    schedules a motion; call sample(t) at ~60 Hz to read it."""

    GYRO_BIAS = (6.0, -4.0, 3.0)
    GRAVITY = (30.0, -20.0, 1000.0)       # accelerometer at rest (1000 ~ 1 g)

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._motions = []                # (t_start, kind, strength, jitter params)

    def swing(self, kind, strength=1.0, t=None):
        """kind: 'right' | 'left' (a real swing) or 'bump' (something that is NOT a swing: a knock or a wobble)."""
        self._motions.append((time.time() if t is None else t, kind, strength,
                              self.rng.uniform(0.85, 1.2), self.rng.uniform(-0.15, 0.15), self.rng.uniform(0.8, 1.25)))

    def sample(self, t):
        g = list(self.GYRO_BIAS)
        a = list(self.GRAVITY)
        for i in range(3):
            g[i] += self.rng.gauss(0, 4.0)
            a[i] += self.rng.gauss(0, 6.0)
        for (t0, kind, strength, amp_j, tilt_j, dur_j) in list(self._motions):
            s = t - t0
            if kind in ("right", "left"):
                d = 1.0 if kind == "right" else -1.0
                dur = 0.34 * dur_j
                if 0 <= s <= dur:
                    x = s / dur
                    env = math.sin(math.pi * x) ** 2 * strength * amp_j
                    g[2] += d * 760 * env                       # the wrist rotates about z...
                    g[1] += d * 190 * env * (1 + tilt_j)        # ...with some roll
                    g[0] += tilt_j * 120 * env
                    a[0] += d * 520 * math.sin(2 * math.pi * x) * strength * amp_j      # accelerate, then brake
                    a[2] += 140 * env
            else:                                               # a bump: short, mixed axes, no left/right pattern
                dur = 0.18 * dur_j
                if 0 <= s <= dur:
                    env = math.sin(math.pi * s / dur) ** 2 * strength * amp_j
                    g[0] += 380 * env * (1 if tilt_j > 0 else -1)
                    g[1] += 260 * env * (-1 if tilt_j > 0 else 1)
                    a[1] += 300 * env
            if s > 1.5:
                self._motions.remove((t0, kind, strength, amp_j, tilt_j, dur_j))
        return Sample(t, *a, *g)


class IMUReader(threading.Thread):
    """Connects to the hub (or a FakeHub) and queues every new IMU reading. Use drain() from the game thread."""

    def __init__(self, card=None, delay_ms=15, fake=None):
        super().__init__(daemon=True)
        self.card, self.delay_ms, self.fake = card, delay_ms, fake
        self.status = "connecting to the hub..."
        self.connected = False
        self.rate = 0.0                                  # measured samples per second
        self._queue = deque(maxlen=4000)
        self._halt = threading.Event()

    def drain(self):
        out = []
        while self._queue:
            out.append(self._queue.popleft())
        return out

    def stop(self):
        self._halt.set()

    def close(self, timeout=4.0):
        """Stop and WAIT for the hub to disconnect. Exiting while the Bluetooth thread is still running can crash
        the process at shutdown (macOS reports 'Killed: 9'), so scripts call this before they end."""
        self._halt.set()
        if self.is_alive():
            self.join(timeout)

    def _push(self, sample, stamps):
        self._queue.append(sample)
        stamps.append(sample.t)
        while stamps and sample.t - stamps[0] > 2.0:
            stamps.popleft()
        if len(stamps) > 5:
            self.rate = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-3)

    def run(self):
        stamps = deque()
        if self.fake is not None:
            self.connected, self.status = True, "simulated hub"
            while not self._halt.is_set():
                self._push(self.fake.sample(time.time()), stamps)
                time.sleep(1 / 60)
            return
        robot = None
        try:
            import legoeducation as le
            robot = le.DoubleMotor()
            kwargs = {"device_notification_delay": self.delay_ms}
            if self.card:
                kwargs.update(card_color=self.card[0], card_serial=self.card[1])
            robot.connect(**kwargs)
            if not robot.connected:
                self.status = "could not connect to a Double Motor (is it on and in range?)"
                return
            self.connected, self.status = True, "hub connected"
            last = None
            while not self._halt.is_set():
                imu = robot.imu_device
                if imu is not last:                          # a new notification replaces the object
                    last = imu
                    vals = (imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ,
                            imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ)
                    if not any(math.isnan(v) for v in vals):
                        self._push(Sample(time.time(), *vals), stamps)
                time.sleep(0.002)
        except Exception as exc:                             # noqa: BLE001 - show any failure on screen
            self.status = f"hub error: {exc}"
        finally:
            self.connected = False
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception:                            # noqa: BLE001
                    pass


def samples_to_array(samples):
    """list[Sample] -> (times, values) with values shaped (n, 6)."""
    t = np.array([s.t for s in samples], dtype=float)
    v = np.array([[s.ax, s.ay, s.az, s.gx, s.gy, s.gz] for s in samples], dtype=float)
    return t, v
