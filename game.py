"""Ping Pong (a game inferior to Frisbee).

State machine:  title -> choose (hold an AprilTag up to pick a difficulty) -> countdown 3,2,1 -> play

    python game.py                          # webcam; 1/2/3 keys also pick a difficulty
    python game.py --no-camera              # keys only
    python game.py --tag-image some.png     # a still image stands in for the camera (testing)

Run `python make_tags.py` once to generate the printable tags.
"""
import argparse
import math

import render                                  # noqa: F401  must come first (macOS GL setup), before ui / ursina
from render import World, init_app
from ursina import Entity, application, clamp, invoke, lerp, mouse, time

import config as cfg
from physics import Rally
from tags import HoldSelector
from ui import UI
from vision import Vision


def parse_args():
    ap = argparse.ArgumentParser(description=cfg.TITLE)
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--no-camera", action="store_true", help="skip the webcam; pick the difficulty with keys 1/2/3")
    ap.add_argument("--tag-image", help="use a still image as a fake camera (testing)")
    ap.add_argument("--start-state", choices=["title", "choose", "countdown", "play"], default="title")
    ap.add_argument("--difficulty", choices=[d.name.lower() for d in cfg.DIFFICULTIES], default="medium",
                    help="difficulty to use when starting directly in countdown/play (testing)")
    ap.add_argument("--shots", default="", help="comma-separated seconds at which to save shot_<s>.png (testing)")
    ap.add_argument("--quit-after", type=float, default=0, help="quit after this many seconds (testing)")
    ap.add_argument("--wiggle", action="store_true", help="move the racket automatically (testing)")
    return ap.parse_args()


def save_shot(path):
    from panda3d.core import PNMImage
    img = PNMImage()
    base.win.getScreenshot(img)     # `base` is injected into builtins by Panda3D
    img.write(path)


class Game(Entity):
    def __init__(self, args, world, ui, vision):
        super().__init__()
        self.args, self.world, self.ui, self.vision = args, world, ui, vision
        self.selector = HoldSelector()
        self.clock = 0.0                        # total running time (menu camera sway, wiggle mode)
        self.moved = False                      # mouse has been moved at least once
        self.difficulty = next(d for d in cfg.DIFFICULTIES if d.name.lower() == args.difficulty)
        self.go(args.start_state)

    # ---- state changes ----
    def go(self, state, difficulty=None):
        self.state, self.t = state, 0.0
        if difficulty is not None:
            self.difficulty = difficulty
        self.ui.show(state)
        self.world.set_visible(ball=(state == "play"), racket=(state in ("countdown", "play")))
        if state == "choose":
            self.selector.reset()
        elif state == "play":
            self.rally = Rally(self.difficulty.speed)
            self.hits, self.prev_dir = 0, 1
            self.ui.set_play(self.difficulty.name, 0)

    def pick(self, difficulty):
        self.go("countdown", difficulty)

    # ---- helpers ----
    def camera_status(self):
        if self.vision is None:
            return "no camera  -  press 1, 2 or 3 to choose"
        snap = self.vision.get()
        if snap.error:
            return f"{snap.error}  -  press 1, 2 or 3 instead"
        if not snap.ok:
            return "starting camera..."
        return "show a tag to the camera  (or press 1, 2 or 3)"

    def drive_racket(self, dt):
        if mouse.velocity.length() > 0:
            self.moved = True
        mx, my = (mouse.x, mouse.y) if self.moved else (0.0, -0.1)
        if self.args.wiggle:
            mx, my = 0.28 * math.sin(self.clock * 3.1), -0.05 + 0.12 * math.sin(self.clock * 2.3)
        self.world.drive_racket(*self.world.mouse_to_plane(mx, my), dt)

    # ---- per-frame ----
    def update(self):
        dt = time.dt
        self.t += dt
        self.clock += dt
        if self.state in ("title", "choose"):
            self.world.sway(self.clock, 0.12)
        else:
            self.world.center_camera()          # aiming maths assumes camera.x == 0

        if self.state == "title":
            if self.t >= cfg.TITLE_SECONDS:
                self.go("choose")

        elif self.state == "choose":
            snap = self.vision.get() if self.vision is not None else None
            detections = snap.tags if (snap and snap.ok) else {}
            cand, progress, confirmed = self.selector.update(detections, time.time())
            self.ui.update_choose(cand, progress, self.camera_status())
            if confirmed is not None:
                self.pick(cfg.BY_TAG[confirmed])

        elif self.state == "countdown":
            self.drive_racket(dt)
            if self.t < cfg.COUNTDOWN_FROM:
                n = cfg.COUNTDOWN_FROM - int(self.t)
                self.ui.set_countdown(str(n), self.difficulty.name, 1.0 - (self.t % 1.0))
            elif self.t < cfg.COUNTDOWN_FROM + cfg.GO_SECONDS:
                self.ui.set_countdown("GO!", self.difficulty.name, 1.0 - (self.t - cfg.COUNTDOWN_FROM) / cfg.GO_SECONDS)
            else:
                self.go("play")

        elif self.state == "play":
            self.drive_racket(dt)
            x, y, z, d = self.rally.state(self.t)
            self.world.place_ball(x, y, z)
            if d == -1:
                ax, ay, _, frac = self.rally.arrival(self.t)
                self.world.show_ring(ax, ay, 0.07 + 0.38 * clamp(frac, 0, 1))
            else:
                self.world.hide_ring()
            if d != self.prev_dir and d == 1:        # the return arrives: count it (placeholder for real hit tests)
                self.hits += 1
                self.ui.set_play(self.difficulty.name, self.hits)
            self.prev_dir = d

    def input(self, key):
        if key in ("q", "escape"):
            application.quit()
        elif self.state == "title" and key in ("space", "enter"):
            self.go("choose")
        elif self.state == "choose" and key in ("1", "2", "3"):
            self.pick(cfg.DIFFICULTIES[int(key) - 1])
        elif self.state in ("countdown", "play") and key == "r":
            self.go("title")


def main():
    args = parse_args()
    app = init_app(cfg.TITLE)
    world, ui = World(), UI()

    vision = None
    if not args.no_camera:
        vision = Vision(camera_index=args.camera, image_path=args.tag_image)
        if vision.open():
            vision.start()
        else:
            print("camera:", vision.get().error, "- use keys 1/2/3 to choose a difficulty")

    Game(args, world, ui, vision)
    for s in [x for x in args.shots.split(",") if x.strip()]:
        invoke(save_shot, f"shot_{s.strip()}.png", delay=float(s))
    if args.quit_after:
        invoke(application.quit, delay=args.quit_after)
    try:
        app.run()
    finally:
        if vision is not None:
            vision.stop()


if __name__ == "__main__":
    main()
