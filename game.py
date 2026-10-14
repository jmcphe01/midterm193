"""Ping Pong (a game inferior to Frisbee).

State machine:  title -> choose (hold an AprilTag up to pick a difficulty) -> [calibrate, pose control only]
                -> countdown 3,2,1 -> play (a real rally against the opponent) -> game over -> choose ...

    python game.py                          # webcam for the tags; the mouse moves the racket
    python game.py --pose                   # your hand, tracked by the webcam, moves the racket
    python game.py --no-camera              # keys + mouse only
    python game.py --tag-image some.png     # a still image stands in for the camera (testing)
    python game.py --fake-pose              # a scripted fake hand stands in for you (testing)
    python game.py --pose --imu             # hand tracked by the webcam, swings sensed by the hub on the paddle
    python game.py --autoplay               # a bot plays, for demos and testing

Returning a ball needs the racket near the ball AND a swing. A swing comes from the LEGO hub's IMU on the paddle
(--imu: a neural network recognises right / left swings), or, without a hub, from a quick flick of the racket or
the SPACE key.

Run `python make_tags.py` once to generate the printable tags.
"""
import argparse
import math
import random

import render                                  # noqa: F401  must come first (macOS GL setup), before ui / ursina
from render import World, init_app
from ursina import Entity, application, clamp, invoke, mouse, time

import config as cfg
from control import Calibration, CornerCalibrator, OneEuro
from imu import FakeHub, IMUReader, parse_card
from physics import ZONE_FAR, Rally
from scoreboard import Scoreboard
from sound import Sounds
from swing import SwingDetector, append_event, event_from_samples, load_bundle, paths
from tags import HoldSelector
from ui import UI
from vision import Vision

SWING_SPEED = 1.4          # m/s: racket faster than this counts as a swing (stand-in for the IMU)
SWING_COOLDOWN = 0.25      # s between swings
GAME_OVER_SECONDS = 1.3    # the ball is left to fly/fall this long before the game-over screen
GAME_OVER_SCREEN_SECONDS = 6.0
POSE_FRESH_SECONDS = 0.35  # a pose reading older than this counts as "nobody visible"
REASONS = {"miss": "you missed the ball", "net": "your shot hit the net", "out": "your shot went out"}


def parse_args():
    ap = argparse.ArgumentParser(description=cfg.TITLE)
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--no-camera", action="store_true", help="skip the webcam; pick the difficulty with keys 1/2/3")
    ap.add_argument("--pose", action="store_true", help="control the racket with your hand (webcam pose tracking)")
    ap.add_argument("--hand", choices=["right", "left"], default="right", help="the hand that holds the racket")
    ap.add_argument("--recalibrate", action="store_true", help="ignore the saved calibration and calibrate again")
    ap.add_argument("--fake-pose", action="store_true", help="a scripted fake hand stands in for you (testing)")
    ap.add_argument("--imu", action="store_true", help="detect swings with the LEGO hub's IMU on the paddle")
    ap.add_argument("--card", default=cfg.HUB_CARD, help="Connection Card of the hub, colour:serial (default %(default)s)")
    ap.add_argument("--fake-imu", action="store_true", help="a simulated hub stands in for the real one (testing)")
    ap.add_argument("--teach", action="store_true",
                    help="teach mode (needs --imu): after a swing, press the LEFT or RIGHT arrow for the way you swung "
                         "and that motion is saved as a training example")
    ap.add_argument("--no-mqtt", action="store_true", help="don't publish the score")
    ap.add_argument("--mqtt-topic", default=cfg.MQTT_TOPIC, help="topic to publish the score to (default %(default)s)")
    ap.add_argument("--mqtt-mode", choices=["record", "streak"], default="record",
                    help="publish your RECORD streak (default) or the live streak")
    ap.add_argument("--mqtt-retain", action="store_true", help="retain the last score on the broker for late subscribers")
    ap.add_argument("--reset-record", action="store_true", help="forget the saved record streak")
    ap.add_argument("--record-file", default=None, help="where the record streak is saved (default record.json here)")
    ap.add_argument("--mute", action="store_true", help="no sound effects")
    ap.add_argument("--tag-image", help="use a still image as a fake camera (testing)")
    ap.add_argument("--start-state", choices=["title", "choose", "calibrate", "countdown", "play", "gameover"],
                    default="title")
    ap.add_argument("--difficulty", choices=[d.name.lower() for d in cfg.DIFFICULTIES], default="medium",
                    help="difficulty to use when starting directly in countdown/play (testing)")
    ap.add_argument("--autoplay", action="store_true", help="a bot plays the rally (demos / testing)")
    ap.add_argument("--miss-after", type=int, default=0, help="with --autoplay: the bot gives up after this many hits")
    ap.add_argument("--seed", type=int, default=None, help="random seed for the rally (testing)")
    ap.add_argument("--log", action="store_true", help="print rally events to the terminal")
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
    def __init__(self, args, world, ui, vision, imu=None, fake_hub=None, scoreboard=None, sounds=None):
        super().__init__()
        self.args, self.world, self.ui, self.vision = args, world, ui, vision
        self.scoreboard = scoreboard or Scoreboard(enabled=False)
        self.sounds = sounds or Sounds(enabled=False)
        self._last_count = None
        self.imu, self.fake_hub = imu, fake_hub
        self.detector = SwingDetector(load_bundle(paths(fake=bool(fake_hub))[1])) if imu is not None else None
        if self.detector is not None:
            self.detector.on_trigger = self.on_imu_trigger
        self.taught = 0
        self._last_taught_stamp = None
        if imu is not None:
            ui.hint.text = "move the racket onto the ring, then SWING the paddle  -  R: title  Q: quit"
            if args.teach:
                ui.hint.text = "TEACH MODE: right after a swing press <- (you swung left) or -> (you swung right)  -  R: title  Q: quit"
        self.selector = HoldSelector()
        self.rng = random.Random(args.seed)
        self.clock = 0.0                        # total running time (menu camera sway, wiggle mode)
        self.moved = False                      # mouse has been moved at least once
        self.last_swing_clock = -99.0
        self.prev_speed = 0.0
        self.bot_swung = False
        self.hit_clock = -99.0                  # for the ball-pop effect
        self.result = ("", 0, False)            # (reason, streak, new_record) shown on the game-over screen
        self.difficulty = next(d for d in cfg.DIFFICULTIES if d.name.lower() == args.difficulty)

        # pose control: your hand (tracked by the webcam) drives the racket through a calibration
        self.control = "pose" if ((args.pose or args.fake_pose) and vision is not None) else "mouse"
        self.save_calibration = not args.fake_pose          # a scripted fake hand must not overwrite yours
        self.calibration = None
        if self.control == "pose" and not (args.recalibrate or args.fake_pose):
            self.calibration = Calibration.load()
        self.calibrator = CornerCalibrator()
        self.after_calibrate = "countdown"
        self.cal_message, self.cal_message_until = "", 0.0
        self.fu, self.fv = OneEuro(), OneEuro()
        self.pose_xy_last, self._last_stamp = None, 0.0
        self.ui.enable_corner_preview(self.control == "pose")
        if self.control == "pose" and self.calibration is not None:
            self.vision.overlay = ([], self.calibration.box)

        start = args.start_state
        if start in ("countdown", "play", "calibrate") and self.control == "pose" and self.calibration is None:
            self.after_calibrate = "countdown" if start == "calibrate" else start
            start = "calibrate"
        elif start == "calibrate":
            start = "choose"                    # calibration only exists for pose control
        self.go(start)

    @property
    def best(self):
        return self.scoreboard.best             # the record streak (saved between runs)

    # ---- state changes ----
    def go(self, state, difficulty=None):
        self.state, self.t = state, 0.0
        self.log("state ->", state)
        if difficulty is not None:
            self.difficulty = difficulty
        self.ui.show(state)
        self.world.set_visible(ball=(state == "play"), racket=(state in ("countdown", "play", "gameover")),
                               opponent=(state == "play"))
        if self.vision is not None:
            self.vision.want_tags = (state == "choose")        # only read tags when choosing: saves CPU
        if state == "choose":
            self.selector.reset()
        elif state == "calibrate":
            self.calibrator.reset()
            self.cal_message = ""
            if self.vision is not None:
                self.vision.overlay = ([], None)
        elif state == "countdown":
            self._last_count = None
        elif state == "play":
            self.scoreboard.new_game()
            self.rally = Rally(self.difficulty, random.Random(self.rng.random()))
            self.bot_swung = False
            self.ui.set_play(self.difficulty.name, 0, self.best)
        elif state == "gameover":
            reason, streak, record = self.result
            self.ui.set_gameover(REASONS.get(reason, reason), streak, self.best, record)

    def pick(self, difficulty):
        self.difficulty = difficulty
        if self.control == "pose" and self.calibration is None:
            self.after_calibrate = "countdown"
            self.go("calibrate")
        else:
            self.go("countdown")

    def log(self, *a):
        if self.args.log:
            print(f"[{self.clock:6.2f}s]", *a, flush=True)

    # ---- helpers ----
    def camera_status(self):
        if self.vision is None:
            return "no camera  -  press 1, 2 or 3 to choose"
        snap = self.vision.get()
        if snap.error:
            return f"{snap.error}  -  press 1, 2 or 3 instead"
        if not snap.ok:
            return "starting camera..."
        extra = "   C: recalibrate" if self.control == "pose" else ""
        return "show a tag to the camera  (or press 1, 2 or 3)" + extra

    def swing_age(self):
        return self.clock - self.last_swing_clock

    def fresh_pose(self):
        """(snapshot, pose sample or None). None when nobody has been visible for a moment."""
        snap = self.vision.get()
        s = snap.pose
        return snap, (s if s is not None and time.time() - s.stamp < POSE_FRESH_SECONDS else None)

    def pose_xy(self):
        """Racket-plane position of the tracked hand, smoothed. Holds the last spot if you step out of view."""
        _, s = self.fresh_pose()
        if s is not None and s.stamp != self._last_stamp:
            self._last_stamp = s.stamp
            self.pose_xy_last = self.calibration.map(self.fu(s.ax, s.stamp), self.fv(s.ay, s.stamp))
        return self.pose_xy_last

    def drive_racket(self, dt):
        if self.control == "pose" and self.calibration is not None:
            xy = self.pose_xy() or (0.0, 1.1)
            self.world.drive_racket(*xy, dt, self.swing_age())
            return
        if mouse.velocity.length() > 0:
            self.moved = True
        mx, my = (mouse.x, mouse.y) if self.moved else (0.0, -0.1)
        if self.args.wiggle:
            mx, my = 0.28 * math.sin(self.clock * 3.1), -0.05 + 0.12 * math.sin(self.clock * 2.3)
        self.world.drive_racket(*self.world.mouse_to_plane(mx, my), dt, self.swing_age())

    def do_swing(self):
        """The player swung. Hands the racket's velocity to the rally (it aims the return from it)."""
        v = self.world.racket_vel
        self.last_swing_clock = self.clock
        if self.state == "play":
            self.rally.swing(v.x, v.y)

    def update_imu(self):
        """Feed new hub readings to the swing detector and act on any swings it recognises."""
        if self.imu is None:
            return
        for ev in self.detector.feed(self.imu.drain()):
            self.on_imu_swing(ev)
        if self.imu.connected:
            rate = f"{self.imu.rate:.0f} Hz"
            self.ui.set_imu_status(f"hub: {self.imu.status} ({rate})  -  {self.detector.status()}")
        else:
            self.ui.set_imu_status(f"hub: {self.imu.status}")

    def teach(self, label):
        """Save the hub's last ~1.6 s as a training example of `label` ('right' | 'left')."""
        if self.detector is None or not self.detector.ready:
            self.ui.flash_swing("teach: the hub isn't ready yet", time.time(), 1.2, ignored=True)
            return
        event = event_from_samples(self.detector.recent_samples(), self.detector.bias, label)
        if event is None:
            self.ui.flash_swing("nothing to learn from: no motion in the last 1.6 s", time.time(), 1.2, ignored=True)
            return
        # pressing a second key for the SAME motion corrects the first press instead of adding a duplicate
        same = self._last_taught_stamp is not None and abs(event["stamp"] - self._last_taught_stamp) < 0.05
        n = append_event(paths(fake=bool(self.fake_hub))[0], event, self.detector.bias, self.detector.threshold,
                         replace_last=same)
        self._last_taught_stamp = event["stamp"]
        if not same:
            self.taught += 1
        name = {"right": "RIGHT swing", "left": "LEFT swing"}[label]
        self.log("taught:", label, f"peak {event['peak']:.0f}", "(corrected the previous key)" if same else "")
        self.ui.flash_swing(f"{'changed to' if same else 'learned'}: {name}   ({self.taught} this session)", time.time(), 1.5)

    def on_imu_trigger(self, info):
        """Called for every motion the detector looked at. Real swings are announced by on_imu_swing; this explains
        the ones that did NOT count, so you can see why a swing seemed to be ignored."""
        if info["label"] == "below_trigger":
            msg = f"too weak: peak {info['peak']:.0f}, needs {info['threshold']:.0f}"
        elif info["label"] is None:
            p = info["p"]
            msg = f"looked like a non-swing (swing {p.get('right', 0) + p.get('left', 0):.2f})"
        else:
            return
        self.log("imu ignored:", msg)
        self.ui.flash_swing(msg, time.time(), 1.0, ignored=True)

    def on_imu_swing(self, ev):
        """A swing was recognised. It is reported ~0.15 s after it started, so back-date it by that much."""
        age = max(0.0, time.time() - ev.t)
        direction = {"right": 1, "left": -1}.get(ev.label, 0)
        speed = clamp(1.6 + 2.2 * ev.strength, 1.6, 5.5)            # m/s equivalent of how hard the swing was
        self.last_swing_clock = self.clock - age
        self.log("imu swing:", ev.label, f"conf {ev.confidence:.2f}", f"strength {ev.strength:.2f}", f"age {age * 1000:.0f} ms")
        arrow = {"right": ">> RIGHT swing", "left": "LEFT swing <<"}.get(ev.label, "swing")
        self.ui.flash_swing(arrow, time.time())
        if self.state == "play":
            self.rally.swing(direction * speed, 0.0, age, direction)

    def detect_swing(self):
        """Stand-in for the IMU: a quick flick of the racket is a swing."""
        v = self.world.racket_vel
        speed = math.hypot(v.x, v.y)
        if speed > SWING_SPEED >= self.prev_speed and self.swing_age() > SWING_COOLDOWN:
            self.do_swing()
        self.prev_speed = speed

    def bot(self, dt):
        """A player for demos/tests: moves the racket onto the predicted arrival spot, swings as the ball enters
        the zone. With --miss-after N it wanders off after N hits so the game ends."""
        r = self.rally
        pa = r.predict_arrival()
        if self.args.miss_after and r.streak >= self.args.miss_after:
            target = (1.1, 1.9)
        elif pa is not None:
            target = (pa[0], pa[1])
        else:
            target = (r.pos[0], r.pos[1])
        self.world.drive_racket(*target, dt, self.swing_age())
        # With a (simulated) hub the swing is reported ~0.15 s late, so the bot starts it a little earlier.
        lead = abs(r.vel[2]) * 0.2 if self.fake_hub is not None else 0.0
        if r.phase == "toplayer" and r.pos[2] <= ZONE_FAR + 0.10 + lead and not self.bot_swung:
            self.bot_swung = True
            if self.fake_hub is not None:
                self.fake_hub.swing(self.rng.choice(["right", "left"]), self.rng.uniform(0.7, 1.3))
            else:
                self.last_swing_clock = self.clock
                r.swing(self.rng.gauss(0, 1.6), self.rng.gauss(0, 1.0))
        elif r.phase != "toplayer":
            self.bot_swung = False

    # ---- per-frame ----
    def update(self):
        dt = min(time.dt, 0.05)
        self.t += dt
        self.clock += dt
        self.update_imu()
        self.ui.tick(time.time())
        self.ui.set_mqtt_status(f"{self.scoreboard.status}   record {self.best}" if self.scoreboard.enabled
                                else f"record {self.best}")
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

        elif self.state == "calibrate":
            self.update_calibrate()

        elif self.state == "countdown":
            self.drive_racket(dt)
            self.show_preview()
            if self.t < cfg.COUNTDOWN_FROM:
                n = cfg.COUNTDOWN_FROM - int(self.t)
                if n != self._last_count:
                    self._last_count = n
                    self.sounds.play("tick")
                self.ui.set_countdown(str(n), self.difficulty.name, 1.0 - (self.t % 1.0))
            elif self.t < cfg.COUNTDOWN_FROM + cfg.GO_SECONDS:
                if self._last_count != 0:
                    self._last_count = 0
                    self.sounds.play("start")
                self.ui.set_countdown("GO!", self.difficulty.name, 1.0 - (self.t - cfg.COUNTDOWN_FROM) / cfg.GO_SECONDS)
            else:
                self.go("play")

        elif self.state == "play":
            self.update_play(dt)

        elif self.state == "gameover":
            self.drive_racket(dt)
            self.show_preview()
            if self.t >= GAME_OVER_SCREEN_SECONDS:
                self.go("choose")

    def show_preview(self):
        if self.control == "pose":
            self.ui.set_view(self.vision.get().view)

    def update_calibrate(self):
        snap, s = self.fresh_pose()
        before = self.calibrator.index
        progress = self.calibrator.update((s.ax, s.ay) if s else None, time.time())
        if self.calibrator.index != before:
            self.log("corner", self.calibrator.index, "captured at", tuple(round(c, 3) for c in self.calibrator.corners[-1]))
        self.vision.overlay = (list(self.calibrator.corners), None)
        if self.calibrator.done:
            error = Calibration.check(self.calibrator.corners)
            if error is None:
                self.calibration = Calibration(self.calibrator.corners)
                if self.save_calibration:
                    self.calibration.save()
                self.fu, self.fv = OneEuro(), OneEuro()
                self.pose_xy_last = None
                self.vision.overlay = ([], self.calibration.box)
                self.log("calibrated:", [tuple(round(c, 2) for c in p) for p in self.calibration.corners])
                self.go(self.after_calibrate)
                return
            self.cal_message, self.cal_message_until = error + "  -  let's try again", time.time() + 4.0
            self.calibrator.reset()
        if time.time() < self.cal_message_until:
            status = self.cal_message
        elif s is None:
            why = snap.pose_reason or "no person in view"
            status = f"{why}  -  keep your racket hand in view of the camera"
        else:
            status = "hold still to take the corner  (or press SPACE to take it now)"
        self.ui.update_calibrate(self.calibrator.index, progress, status)
        self.ui.set_view(snap.view)

    def update_play(self, dt):
        r = self.rally
        if self.args.autoplay:
            self.bot(dt)
        else:
            self.drive_racket(dt)
            if self.imu is None:
                self.detect_swing()                  # no hub: a quick flick of the racket is the swing
        self.show_preview()

        v = self.world.racket_vel
        events = r.step(dt, self.world.racket.x, self.world.racket.y, v.x, v.y)
        for ev in events:
            self.log(*ev)
            if ev[0] == "hit":
                self.hit_clock = self.clock
                self.scoreboard.on_hit(r.streak)
                self.ui.set_play(self.difficulty.name, r.streak, self.best)
                self.sounds.play("hit", volume=clamp(0.75 + math.hypot(*r.swing_v) / 14, 0.75, 1.0))     # harder swing, louder
            elif ev[0] == "opp_hit":
                self.sounds.play("opp")
            elif ev[0] == "bounce":
                self.sounds.play("bounce")
            elif ev[0] == "net":
                self.sounds.play("net")

        pop = max(0.0, 1.0 - (self.clock - self.hit_clock) / 0.15)
        self.world.place_ball(*r.pos, pop=pop)
        self.world.set_opponent(r.pos[0], r.pos[1], r.t - r.opponent_hit_t, dt)

        pa = r.predict_arrival()
        if pa is not None:                                  # the ball is heading for the player: show where
            x, y, seconds = pa
            self.world.show_ring(x, y, 2 * self.difficulty.hit_radius + 0.30 * clamp(seconds / 0.8, 0, 1))
        else:
            self.world.hide_ring()

        if r.phase == "dead" and r.dead_time >= GAME_OVER_SECONDS:
            record = self.scoreboard.on_game_over(r.streak)       # also publishes the score over MQTT
            self.result = (r.reason, r.streak, record)
            self.log("game over:", r.reason, "streak", r.streak, "record", self.best, "new record" if record else "")
            self.go("gameover")
            self.sounds.play("gameover")
            if record:
                invoke(self.sounds.play, "record", delay=1.6)

    def input(self, key):
        if key in ("q", "escape") and not (key == "escape" and self.state == "calibrate"):
            application.quit()
        elif key == "escape" and self.state == "calibrate":
            self.go("choose")
        elif self.state == "title" and key in ("space", "enter"):
            self.go("choose")
        elif self.state == "choose" and key in ("1", "2", "3"):
            self.pick(cfg.DIFFICULTIES[int(key) - 1])
        elif self.state == "choose" and key == "c" and self.control == "pose":
            self.after_calibrate = "choose"
            self.go("calibrate")
        elif self.state == "calibrate" and key == "space":
            _, s = self.fresh_pose()
            self.calibrator.accept_now((s.ax, s.ay) if s else None)
        elif self.state == "play" and key in ("space", "left mouse down") and not self.args.autoplay:
            self.do_swing()
        elif self.args.teach and key in ("left arrow", "right arrow") and self.state in ("play", "gameover"):
            self.teach({"left arrow": "left", "right arrow": "right"}[key])
        elif self.state == "gameover" and key in ("space", "enter"):
            self.go("choose")
        elif self.state in ("countdown", "play", "gameover") and key == "r":
            self.go("title")
        elif key == "m" and self.vision is not None and (self.args.pose or self.args.fake_pose):
            self.control = "mouse" if self.control == "pose" else "pose"       # switch control on the fly
            self.ui.enable_corner_preview(self.control == "pose")
            self.log("control ->", self.control)


def main():
    args = parse_args()
    app = init_app(cfg.TITLE)
    world, ui = World(), UI()

    vision = None
    if args.fake_pose or not args.no_camera:
        vision = Vision(camera_index=args.camera, image_path=args.tag_image, pose=args.pose, hand=args.hand,
                        fake_pose=args.fake_pose)
        if vision.open():
            vision.start()
        else:
            print("camera:", vision.get().error, "- use keys 1/2/3 to choose a difficulty")

    imu, fake_hub = None, None
    if args.imu or args.fake_imu:
        fake_hub = FakeHub() if args.fake_imu else None
        imu = IMUReader(card=parse_card(args.card), fake=fake_hub)
        imu.start()                                  # connects on its own thread, so the game window opens at once

    extra = {"record_path": args.record_file} if args.record_file else {}
    scoreboard = Scoreboard(enabled=not args.no_mqtt, topic=args.mqtt_topic, retain=args.mqtt_retain, mode=args.mqtt_mode, **extra)
    if args.reset_record:
        scoreboard.reset_record()
    scoreboard.start()
    sounds = Sounds(enabled=not args.mute)

    game = Game(args, world, ui, vision, imu, fake_hub, scoreboard, sounds)
    for s in [x for x in args.shots.split(",") if x.strip()]:
        invoke(save_shot, f"shot_{s.strip()}.png", delay=float(s))
    if args.quit_after:
        invoke(application.quit, delay=args.quit_after)
    try:
        app.run()
    finally:
        if vision is not None:
            vision.stop()
        if imu is not None:
            imu.close()
        scoreboard.close()
        if args.teach and game.taught:
            print(f"\nYou taught {game.taught} examples. Retrain with:  python train_swings.py --threshold 250 --post 0.03")


if __name__ == "__main__":
    main()
