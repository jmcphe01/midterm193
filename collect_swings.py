"""Record labelled swings from the LEGO hub, to train the swing classifier.

    python collect_swings.py --manual              # YOU mark the start and end of each swing (most reliable)
    python collect_swings.py                       # swings are detected automatically
    python collect_swings.py --per-class 40        # more examples = a better model
    python collect_swings.py --manual --classes right   # record only the RIGHT swings (or left)
    python collect_swings.py --manual --fresh      # start over: old recordings are archived, not deleted
    python collect_swings.py --card red:1129       # a different hub (default: the orange 1129 card in config.py)
    python collect_swings.py --threshold 150       # override the motion trigger if it's too jumpy / too deaf
    python collect_swings.py --fake [--manual]     # a simulated hub, unattended (to test the pipeline)

Mount the hub on the paddle EXACTLY as it will be during the game (the model learns that orientation; if you
re-mount it, record again). You'll be asked to:

  1. hold the paddle still for 3 seconds   (measures the sensor's resting level, which sets the trigger)
  2. swing it to your RIGHT, over and over (like a forehand sweeping to your right)
  3. swing it to your LEFT

--manual: press ENTER to start, make ONE swing, press ENTER to end it. Type u + ENTER instead to undo the last
one, or q + ENTER to finish the class early. The game's own trigger rule is applied afterwards to find where each swing begins, so
the recording lines up with what the game sees live; a swing too weak to cross the trigger is reported and not kept.

Automatic mode: after each swing bring the paddle back SLOWLY and pause, so the return stroke isn't recorded as a
second swing. Either way, each class is saved as soon as it is finished, to data/swing_data.joblib, and recordings
accumulate across runs. Ctrl-C ends the current class early (what you have so far is kept).
"""
import argparse
import random
import sys
import time
from collections import deque

import joblib
import numpy as np

import config as cfg
from imu import FakeHub, IMUReader, parse_card
from swing import (ALL_LABELS, RECORD_POST, RECORD_PRE, REARM_FRACTION, default_threshold, estimate_rest, find_onsets,
                   gyro_energy, paths)

INSTRUCTIONS = {
    "right": "Swing the paddle to your RIGHT (a forehand sweeping right).",
    "left": "Swing the paddle to your LEFT.",
}


def slice_event(label, samples, onset, bias):
    """One training example: the raw readings from RECORD_PRE before `onset` to RECORD_POST after it."""
    win = [x for x in samples if onset - RECORD_PRE <= x.t <= onset + RECORD_POST]
    if len(win) < 8:
        return None
    t = np.array([x.t for x in win]) - onset
    v = np.array([[x.ax, x.ay, x.az, x.gx, x.gy, x.gz] for x in win])
    return {"label": label, "t": t, "v": v, "peak": float(gyro_energy(v, bias).max())}


class EventRecorder:
    """Cuts a continuous IMU stream into motion events (onset detected by rotation energy). Automatic mode."""

    def __init__(self, bias, threshold):
        self.bias, self.threshold = np.asarray(bias), threshold
        self.buf = deque()
        self._onset, self._armed, self._quiet = None, True, None

    def feed(self, samples):
        """Returns completed events: dicts with relative times 't', values 'v' (n, 6) and the peak energy."""
        done = []
        for s in samples:
            self.buf.append(s)
            while self.buf and s.t - self.buf[0].t > RECORD_PRE + RECORD_POST + 1.5:
                self.buf.popleft()
            e = float(gyro_energy(np.array([[s.ax, s.ay, s.az, s.gx, s.gy, s.gz]]), self.bias)[0])
            if self._onset is None:
                if not self._armed:
                    if e < REARM_FRACTION * self.threshold:
                        self._quiet = self._quiet or s.t
                        if s.t - self._quiet >= 0.3:
                            self._armed = True
                    else:
                        self._quiet = None
                elif e > self.threshold:
                    self._onset, self._armed, self._quiet = s.t, False, None
            elif s.t >= self._onset + RECORD_POST:
                onset, self._onset = self._onset, None
                ev = slice_event("", list(self.buf), onset, self.bias)
                if ev is not None:
                    done.append(ev)
        return done


def wait_for_hub(reader, timeout=40):
    print("Connecting to the hub (turn it on and keep it close)...")
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout:
        if reader.status != last:
            last = reader.status
            print("  ", last)
        if reader.connected:
            return True
        if not reader.is_alive():
            return False
        time.sleep(0.2)
    return False


def record_auto(label, reader, bias, threshold, per_class, fake, rng):
    reader.drain()
    rec = EventRecorder(bias, threshold)
    got, next_fake, t_start = [], time.time() + 0.5, time.time()
    try:
        while len(got) < per_class:
            if fake and time.time() >= next_fake:                    # an unattended 'user'
                fake.swing(label, rng.uniform(0.6, 1.4))
                next_fake = time.time() + 1.5
            for ev in rec.feed(reader.drain()):
                ev["label"] = label
                got.append(ev)
                print(f"     captured {label} {len(got)}/{per_class}   (peak {ev['peak']:.0f}, trigger {threshold:.0f})")
            if time.time() - t_start > 600:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("     (stopped early)")
    return got


def record_manual(label, reader, bias, threshold, per_class, fake, rng):
    """You press ENTER to start and ENTER to end each swing."""
    got, buf, last_end = [], [], 0.0

    def ask(prompt):
        if fake:
            return ""
        return input(prompt).strip().lower()

    try:
        while len(got) < per_class:
            cmd = ask(f"     [{len(got)}/{per_class}] ENTER = start a {label} swing   (u = undo last, q = finish) > ")
            if cmd == "q":
                break
            if cmd == "u":
                if got:
                    got.pop()
                    print("     undone.")
                continue
            buf += reader.drain()
            t_start = time.time()
            if fake:                                                  # an unattended 'user'
                fake.swing(label, rng.uniform(0.6, 1.4), t=t_start + rng.uniform(0.3, 0.9))
                time.sleep(1.6)
            else:
                print("        recording...  make the move, then press ENTER when it is finished")
                input("")
            t_end = time.time()
            time.sleep(0.35)                                          # let the tail of the motion arrive
            buf += reader.drain()
            # Never reach back into the previous recording (its swing tail would be found as this swing), and look
            # for the motion only from just before ENTER was pressed; the earlier samples are kept as pre-roll.
            floor = max(t_start - 1.0, last_end)
            seg = [s for s in buf if floor <= s.t <= t_end + 0.6]
            last_end = t_end + 0.6
            buf = [s for s in buf if s.t > t_end - 2.0]
            search = [s for s in seg if s.t >= max(t_start - 0.2, floor)]
            if len(search) < 12:
                print("        almost no data arrived for that one; try again.")
                continue
            t = np.array([x.t for x in search])
            v = np.array([[x.ax, x.ay, x.az, x.gx, x.gy, x.gz] for x in search])
            energy = gyro_energy(v, bias)
            peak = float(energy.max())
            if peak < threshold:
                print(f"        no real motion in that one (peak {peak:.0f}; the game reacts above about {threshold:.0f}): not kept."
                      f" Make a quicker, bigger swing.")
                continue
            # Keep the WHOLE recording. The trainer finds where the swing starts (the climb into its peak), so a
            # little hand movement or the key press before the swing can't be mistaken for the swing.
            tp = float(t[int(np.argmax(energy))])
            events = [{"label": label, "t": np.array([x.t for x in seg]) - tp,
                       "v": np.array([[x.ax, x.ay, x.az, x.gx, x.gy, x.gz] for x in seg]), "peak": peak}]
            if len(find_onsets(t, v, bias, 0.5 * peak)) > 1:
                print("        (two strong movements in that recording; the bigger one is used. If that wasn't"
                      " your swing, type u and redo it.)")
            got += events
            print(f"        kept: {len(events)} example{'s' if len(events) != 1 else ''}   (peak {peak:.0f})")
    except KeyboardInterrupt:
        print("     (stopped early)")
    for e in got:
        e["label"] = label
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", default=cfg.HUB_CARD, help="Connection Card colour:serial (default %(default)s)")
    ap.add_argument("--per-class", type=int, default=30, help="examples to record per class (default 30)")
    ap.add_argument("--threshold", type=float, help="rotation energy that counts as a motion (default: from the rest)")
    ap.add_argument("--manual", action="store_true", help="you press ENTER to start and end each swing")
    ap.add_argument("--fresh", action="store_true",
                    help="start a new dataset: the old recordings are moved aside (not deleted) first")
    ap.add_argument("--classes", default="right,left",
                    help="which swings to record, e.g. --classes right to redo just the right swings (default: right,left)")
    ap.add_argument("--fake", action="store_true", help="simulated hub, unattended")
    args = ap.parse_args()

    fake = FakeHub(seed=1) if args.fake else None
    reader = IMUReader(card=parse_card(args.card), fake=fake)
    reader.start()
    if not wait_for_hub(reader):
        reader.close()
        raise SystemExit(f"Could not connect: {reader.status}")
    time.sleep(0.5)
    print(f"Connected. Receiving about {reader.rate:.0f} samples per second.")

    wanted = [c.strip() for c in args.classes.split(",") if c.strip()]
    labels = [c for c in ALL_LABELS if c in wanted]
    total = len(labels) + 1

    # 1. rest
    print(f"\n1/{total}  Rest the paddle on the table and keep it STILL for 3 seconds...")
    if not fake:
        input("     Press Enter when it is resting, then don't touch it. ")
    reader.drain()
    time.sleep(3.0)
    rest = reader.drain()
    if len(rest) < 20:
        raise SystemExit("Almost no data arrived. Check the connection and try again.")
    vals = np.array([[s.ax, s.ay, s.az, s.gx, s.gy, s.gz] for s in rest])
    bias, noise_mean, noise_std = estimate_rest(vals)
    threshold = args.threshold or default_threshold(noise_mean, noise_std)
    print(f"     resting rotation noise: mean {noise_mean:.1f}, std {noise_std:.1f}  ->  motion trigger {threshold:.0f}")

    dataset_path, _ = paths(fake=bool(fake))
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and dataset_path.exists():
        archive = dataset_path.with_name(f"swing_data_old_{time.strftime('%Y%m%d_%H%M%S')}.joblib")
        dataset_path.rename(archive)
        print(f"Old recordings moved to {archive.name} (kept, but no longer used for training).")
    data = joblib.load(dataset_path) if dataset_path.exists() else {"events": [], "sessions": []}
    data["sessions"].append({"bias": bias, "threshold": threshold, "noise_mean": noise_mean, "noise_std": noise_std,
                             "rate": reader.rate, "time": time.time(), "fake": bool(fake), "manual": args.manual})
    record = record_manual if args.manual else record_auto
    rng = random.Random(3)
    new_events = []
    for step, label in enumerate(labels, start=2):
        print(f"\n{step}/{total}  {label.upper()}: {INSTRUCTIONS[label]}")
        if not fake and not args.manual:
            input(f"     Press Enter to start, then record {args.per_class} examples. ")
        got = record(label, reader, bias, threshold, args.per_class, fake, rng)
        for e in got:
            e["label"] = label
        new_events += got
        data["events"] += got                                         # saved after EVERY class, so a stop loses nothing
        joblib.dump(data, dataset_path)
        print(f"     saved {len(got)} {label} examples to {dataset_path}")

    counts = {lab: sum(1 for e in data["events"] if e["label"] == lab) for lab in ALL_LABELS}
    print(f"\nThe dataset now holds: {counts}")
    peaks = [e["peak"] for e in new_events if e["label"] in ("right", "left")]
    if peaks:
        print(f"Swing peak strength this session: min {min(peaks):.0f}, median {np.median(peaks):.0f}, max {max(peaks):.0f}"
              f"  (trigger {threshold:.0f})")
    print("Next: python train_swings.py" + (" --fake" if fake else ""))
    reader.close()


if __name__ == "__main__":
    sys.exit(main())
