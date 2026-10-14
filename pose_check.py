"""Quick check that pose tracking sees you: opens the webcam and shows what the game will use.

    python pose_check.py                 # right hand
    python pose_check.py --hand left --camera 1

Shows a mirrored picture with your arm and tracked wrist drawn on it, plus the numbers the game gets: ax (how far
across the picture the hand is, 0 = left edge, 1 = right edge) and ay (0 = bottom, 1 = top). Stand where you will
play, with your racket hand in view, and don't move the camera afterwards. Press Q or Esc to quit.
"""
import argparse
import time

import cv2

from pose import PoseTracker, draw_overlay


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    args = ap.parse_args()

    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera} (macOS: allow Camera access for your terminal app)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = PoseTracker(args.hand)

    fps, last = 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        sample = tracker.process(frame)
        view = draw_overlay(frame.copy(), sample)
        view = cv2.flip(view, 1)                                   # mirror, like the game's preview
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
        last = now
        if sample is None:
            text, colour = f"no pose: {tracker.reason or 'step back so your racket arm is visible'}", (60, 60, 230)
        else:
            text, colour = f"ax {sample.ax:.2f}   ay {sample.ay:.2f}   (fractions of the picture)", (90, 220, 120)
        cv2.putText(view, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
        cv2.putText(view, f"{fps:.0f} fps   Q: quit", (20, view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        cv2.imshow("pose check", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
