"""Train the swing classifier (a small neural network) from the recordings in data/swing_data.joblib.

    python train_swings.py                          # train with the settings below, report cross-validated scores
    python train_swings.py --sweep                  # try several trigger levels / window lengths and compare
    python train_swings.py --threshold 300 --post 0.05    # train with specific settings

The game decides THAT a swing happened by a motion trigger (the rotation rate crossing `--threshold`). This network
only decides WHICH WAY: it sees a short window of the motion (`--pre` seconds before the trigger to `--post` after it)
and says 'right' or 'left'. If it can't tell, the swing still counts and aims straight.

A swing the trigger never fires on can't be recognised live, so the scores count those as missed. Each recorded
example is expanded into many augmented copies (shifted in time, scaled in strength, a little noise) for training;
scores come from cross-validation, i.e. from examples the network never saw.
"""
import argparse
import time

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from swing import ALL_LABELS, N_POINTS, POST, PRE, SWING_LABELS, decide, gyro_energy, paths, window_features


def swing_onset(t, energy, threshold):
    """Where a swing starts: the point where the rotation rate climbs through `threshold` on the way to its peak.
    (Not the first twitch: a hand often moves a little, or presses a key, long before the actual swing.)"""
    ip = int(np.argmax(energy))
    if energy[ip] <= threshold:
        return None
    i = ip
    while i > 0 and energy[i - 1] > threshold:
        i -= 1
    return float(t[i])


def prepare(events, bias, threshold, post=POST):
    """Attach `onset0` (where the trigger fires on this recording; None = unusable) and `why` for the unusable ones:
    'weak' (never reaches the trigger) or 'cut' (the recording ends before the decision window would)."""
    out = []
    for e in events:
        energy = gyro_energy(e["v"], bias)
        onset = swing_onset(e["t"], energy, threshold)
        why = "" if onset is not None else "weak"
        if onset is not None and e["t"][-1] < onset + post + 0.03:
            onset, why = None, "cut"
        peak_t = float(e["t"][int(np.argmax(energy))])
        out.append({**e, "onset0": onset, "why": why, "peak_t": peak_t})
    return out


def features_for(event, bias, pre, post, shift=0.0, scale=1.0, rng=None, noise=0.0):
    f, _ = window_features(event["t"], event["v"], event["onset0"] + shift, bias, pre, post)
    f = f * scale
    if noise and rng is not None:
        f = f + rng.normal(0, noise, f.shape) * (np.abs(f).mean() + 1e-6)
    return f


def build_train(events, bias, pre, post, rng, copies=14):
    X, y = [], []
    for e in events:
        if e["onset0"] is None:
            continue
        X.append(features_for(e, bias, pre, post))
        y.append(e["label"])
        for _ in range(copies):
            b = bias + rng.normal(0, 4.0, 3)                                 # a slightly different gyro bias
            X.append(features_for(e, b, pre, post, shift=rng.uniform(-0.04, 0.04), scale=rng.uniform(0.8, 1.25),
                                  rng=rng, noise=0.03))
            y.append(e["label"])
    return np.array(X), np.array(y)


def new_model(seed):
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(48, 24), alpha=1e-3, max_iter=800, early_stopping=True,
                                       n_iter_no_change=25, random_state=seed))


def decide_all(model, events, bias, pre, post, min_swing_prob=0.5):
    """What the game would do with each recording: 'right' / 'left' / 'swing' (direction unsure) / None (never
    triggers, so the game doesn't see it)."""
    out = []
    for e in events:
        if e["onset0"] is None:
            out.append(None)
        else:
            f = features_for(e, bias, pre, post)
            out.append(decide(model.classes_, model.predict_proba([f])[0], min_swing_prob)[0])
    return out


def cross_validate(events, bias, pre, post, seed, folds=5, min_swing_prob=0.5):
    """(true label, decision) for every usable recording, each decided by a model that never saw it in training."""
    events = [e for e in events if e["why"] != "cut"]                        # cut-off recordings can't be judged
    labels = np.array([e["label"] for e in events])
    pairs = []
    for k, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(labels, labels)):
        rng = np.random.default_rng(seed + k)
        X, y = build_train([events[i] for i in tr], bias, pre, post, rng)
        model = new_model(seed + k).fit(X, y)
        test = [events[i] for i in te]
        pairs += list(zip(labels[te], decide_all(model, test, bias, pre, post, min_swing_prob)))
    return pairs


def summarize(pairs, leads=()):
    n = max(len(pairs), 1)
    return {
        "right_dir": 100 * sum(d == t for t, d in pairs) / n,
        "unsure": 100 * sum(d == "swing" for t, d in pairs) / n,
        "wrong": 100 * sum(d in SWING_LABELS and d != t for t, d in pairs) / n,
        "missed": 100 * sum(d is None for t, d in pairs) / n,
        "lead": 1000 * float(np.median(leads)) if len(leads) else float("nan"),
    }


def print_table(pairs):
    cols = ["right", "left", "swing", None]
    names = ["-> right", "-> left", "-> swing(?)", "-> not seen"]
    print("   what the game would do:      " + "  ".join(f"{n:>11}" for n in names))
    for lab in ALL_LABELS:
        row = [sum(1 for t, d in pairs if t == lab and d == c) for c in cols]
        print(f"   {'a ' + lab + ' swing':>26}   " + "  ".join(f"{v:>11}" for v in row))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true", help="evaluate only")
    ap.add_argument("--sweep", action="store_true", help="compare several trigger levels and window lengths")
    ap.add_argument("--threshold", type=float, help="motion trigger (default: the median of your recording sessions)")
    ap.add_argument("--min-swing-prob", type=float, default=0.35,
                    help="kept for older models; with only right/left every trigger is a swing")
    ap.add_argument("--pre", type=float, default=PRE, help="seconds of motion before the trigger the network sees")
    ap.add_argument("--post", type=float, default=POST, help="seconds after the trigger it waits (more = surer, but later)")
    ap.add_argument("--fake", action="store_true", help="use the simulated-hub data in data/fake/ (for testing)")
    args = ap.parse_args()

    dataset_path, model_path = paths(fake=args.fake)
    if not dataset_path.exists():
        raise SystemExit("No recordings yet. Run: python collect_swings.py" + (" --fake" if args.fake else ""))
    data = joblib.load(dataset_path)
    ignored = sum(1 for e in data["events"] if e["label"] not in ALL_LABELS)
    events = [e for e in data["events"] if e["label"] in ALL_LABELS]
    if ignored:
        print(f"(Ignoring {ignored} old 'not a swing' recordings: that class has been removed.)")
    counts = {lab: sum(1 for e in events if e["label"] == lab) for lab in ALL_LABELS}
    print("Recorded swings:", counts)
    if min(counts.values()) < 6:
        raise SystemExit("Need at least 6 swings of each direction (30+ is better). Run collect_swings.py again.")
    if any(s.get("fake") for s in data["sessions"]) and not all(s.get("fake") for s in data["sessions"]):
        print("WARNING: the dataset mixes simulated and real recordings; delete data/swing_data.joblib and re-record.")

    bias = np.median([s["bias"] for s in data["sessions"]], axis=0)
    session_threshold = float(np.median([s["threshold"] for s in data["sessions"]]))
    swing_peaks = sorted(e["peak"] for e in events)
    print(f"Swing strength (peak rotation rate): lowest {swing_peaks[0]:.0f}, 10th percentile "
          f"{np.percentile(swing_peaks, 10):.0f}, median {np.median(swing_peaks):.0f}.")

    if args.sweep:
        levels = sorted({100.0, 150.0, 250.0, 350.0, 450.0, 600.0})
        print(f"\nSweeping the trigger level and the decision window (5-fold cross-validation, {len(events)} swings).")
        print("'lead' = how many ms BEFORE the swing's peak speed (about when the paddle meets the ball) the decision")
        print("arrives; a negative lead means the answer comes too late to matter. A LOWER trigger catches gentler")
        print("swings but also fires on more non-swing motion, since the trigger is now what tells a swing from a non-swing.")
        print("  trigger   post   | swings: right  unsure  WRONG | not seen (below the trigger or cut off) |  lead")
        results = []
        for thr in levels:
            for post in (0.0, 0.03, 0.05, 0.08, 0.12):
                prepared = prepare(events, bias, thr, post)
                leads = [e["peak_t"] - (e["onset0"] + post) for e in prepared if e["onset0"] is not None]
                m = summarize(cross_validate(prepared, bias, args.pre, post, args.seed), leads)
                unseen = 100 * sum(1 for e in prepared if e["onset0"] is None) / len(prepared)
                # reward correct directions and a decision that arrives before contact; punish wrong direction hard
                score = m["right_dir"] + 0.5 * m["unsure"] - 3 * m["wrong"] - 1.0 * unseen + min(m["lead"], 0) * 0.15 - thr * 0.02
                results.append((score, thr, post, m, unseen))
                print(f"  {thr:6.0f}  {post:5.2f}s  |         {m['right_dir']:4.0f}%   {m['unsure']:4.0f}%  {m['wrong']:4.0f}% |"
                      f"                 {unseen:4.0f}%                    | {m['lead']:5.0f}")
        results.sort(key=lambda r: -r[0])
        print("\nBest balances:")
        for score, thr, post, m, unseen in results[:3]:
            print(f"  --threshold {thr:.0f} --post {post:.2f}   (right {m['right_dir']:.0f}%, wrong {m['wrong']:.0f}%, "
                  f"not seen {unseen:.0f}%, decision {m['lead']:.0f} ms before the peak)")
        return

    threshold = args.threshold or session_threshold
    events = prepare(events, bias, threshold, args.post)
    weak = {lab: sum(1 for e in events if e["label"] == lab and e["why"] == "weak") for lab in ALL_LABELS}
    cut = {lab: sum(1 for e in events if e["label"] == lab and e["why"] == "cut") for lab in ALL_LABELS}
    print(f"\nTrigger {threshold:.0f}, window {args.pre:.2f}s before to {args.post:.2f}s after the trigger.")
    print(f"Swings that never reach the trigger (the game would not see them): {weak}")
    if any(cut.values()):
        print(f"Recordings CUT OFF before the swing is complete (left out; re-record these): {cut}")
    swing_leads = [e["peak_t"] - (e["onset0"] + args.post) for e in events if e["onset0"] is not None]
    if swing_leads:
        print(f"Timing: the decision arrives {1000 * np.median(swing_leads):.0f} ms before the swing's peak speed (median);"
              f" {100 * np.mean(np.array(swing_leads) >= 0):.0f}% of swings are decided before their peak.")
    pairs = cross_validate(events, bias, args.pre, args.post, args.seed, min_swing_prob=args.min_swing_prob)
    m = summarize(pairs)
    print("\nCross-validated (each recording judged by a model that never saw it):")
    print_table(pairs)
    print(f"   of the swings: right direction {m['right_dir']:.0f}%, direction unsure (counts, aims straight) "
          f"{m['unsure']:.0f}%, WRONG direction {m['wrong']:.0f}%, not seen {m['missed']:.0f}%")

    if args.no_save:
        return
    rng = np.random.default_rng(args.seed)
    X, y = build_train(events, bias, args.pre, args.post, rng)
    final = new_model(args.seed).fit(X, y)
    bundle = {"model": final, "labels": list(ALL_LABELS), "bias": bias, "threshold": threshold,
              "ref_peak": float(np.median(swing_peaks)), "pre": args.pre, "post": args.post, "n": N_POINTS,
              "min_swing_prob": args.min_swing_prob, "trained": time.time(), "counts": counts}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"\nSaved the model (trained on all {sum(counts.values())} swings) to {model_path}")
    print(f"Trigger {threshold:.0f}, typical swing strength {np.median(swing_peaks):.0f}.  Next: python game.py --imu")


if __name__ == "__main__":
    main()
