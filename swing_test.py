"""See what the hub detects, live, before playing.

    python swing_test.py                 # the hub on the orange 1129 card (see config.py)
    python swing_test.py --card red:1129
    python swing_test.py --raw           # also print the raw readings (to check axes / units)
    python swing_test.py --verbose       # also explain every motion that did NOT count (too weak / non-swing)
    python swing_test.py --fake          # a simulated hub that swings by itself (to see the output format)

Swing the paddle left and right. Each recognised swing prints its direction, confidence and strength. Without a
trained model (data/swing_model.joblib) every swing is reported as 'swing' with no direction. Ctrl-C to quit.
"""
import argparse
import random
import time

import config as cfg
from imu import FakeHub, IMUReader, parse_card
from swing import SwingDetector, load_bundle, paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", default=cfg.HUB_CARD)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="also explain every motion that did NOT count as a swing")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    fake = FakeHub(seed=2) if args.fake else None
    reader = IMUReader(card=parse_card(args.card), fake=fake)
    reader.start()
    detector = SwingDetector(load_bundle(paths(fake=bool(fake))[1]))
    if args.verbose:
        def explain(info):
            if info["label"] == "below_trigger":
                print(f"   ignored  too weak: peak {info['peak']:.0f}, the trigger is {info['threshold']:.0f}")
            elif info["label"] is None:
                p = info["p"]
                print(f"   ignored  looked like a non-swing: peak {info['peak']:.0f}, probabilities "
                      + ", ".join(f"{k} {v:.2f}" for k, v in p.items()))
        detector.on_trigger = explain
    print("Connecting to the hub (turn it on and keep it close)...")
    last_status, last_print, next_fake = "", 0.0, time.time() + 2.0
    rng = random.Random(1)
    try:
        while True:
            if reader.status != last_status:
                last_status = reader.status
                print("  ", last_status)
            if fake is not None and time.time() >= next_fake:
                fake.swing(rng.choice(["right", "left", "bump"]), rng.uniform(0.7, 1.3))
                next_fake = time.time() + 1.6
            samples = reader.drain()
            if samples and args.raw and time.time() - last_print > 0.25:
                s = samples[-1]
                print(f"   acc {s.ax:7.0f} {s.ay:7.0f} {s.az:7.0f}   gyro {s.gx:7.0f} {s.gy:7.0f} {s.gz:7.0f}")
                last_print = time.time()
            for ev in detector.feed(samples):
                print(f"   SWING  {ev.label:>6}   confidence {ev.confidence:.2f}   strength {ev.strength:.2f}"
                      f"   (reported {1000 * (time.time() - ev.t):.0f} ms after it started)")
            if samples == [] and not reader.connected and not reader.is_alive():
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        print(f"\n{detector.status()}; measured rate {reader.rate:.0f} samples/s")


if __name__ == "__main__":
    main()
