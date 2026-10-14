# midterm193 — virtual ping-pong

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python make_assets.py          # grass tile (+ a placeholder background if assets/background.png is missing)
```

Libraries: OpenCV (`opencv-python`), LEGO Education (`legoeducation`), Ursina (3D engine, Panda3D based).

## Playing

```bash
python make_tags.py            # once: writes the printable difficulty tags to assets/tags/
python make_sounds.py          # once: synthesises the sound effects into assets/sounds/
python game.py                 # webcam; keys 1/2/3 also pick a difficulty
python game.py --no-camera     # keys only
```

Flow: title screen -> **choose difficulty** (hold a printed AprilTag up to the camera for about a second)
-> countdown 3, 2, 1 -> play -> game over -> choose again. In play: `R` returns to the title, `Q` quits.

### How the rally works

The opponent (blue racket, far end) serves; the ball bounces once on your half. A ring on your racket's plane
shows where the ball will arrive and shrinks as it gets close. To return it you need both:

1. **Position:** the racket within the difficulty's hit radius of the ball (0.30 / 0.24 / 0.19 m for
   easy / medium / hard), and
2. **A swing** shortly before: for now a quick flick of the racket (mouse) or `SPACE`. The LEGO hub's IMU
   will replace this.

Your return is aimed from your swing: swing sideways to steer it left or right, swing harder to send it
deeper. It is solved ballistically so it always clears the net and lands on the table. The game ends when
you miss the ball or your shot hits the net / goes out; your score is the streak of returns, and the
session best is kept ("NEW RECORD!"). Physics (`physics.py`) is pure Python: gravity, table bounces,
the net, the floor.

### Pose control (your hand moves the racket)

```bash
python pose_check.py            # first: confirm the webcam sees your shoulders and racket arm
python game.py --pose           # play with your hand (add --hand left if you hold it in your left)
```

The webcam tracks your racket hand with MediaPipe. It is measured relative to your shoulders in units of
shoulder width, so where you stand and how far you are from the camera don't matter. The first time (and
whenever you press `C` on the choose screen, or pass `--recalibrate`) the game asks you to **calibrate**:
reach to the top-left, top-right, bottom-right and bottom-left of the area you can comfortably reach, holding
still for a second at each (or press `SPACE` to take a corner now). That reach region is mapped onto the playing
area, and saved to `calibration.json`. A small camera preview with your reach box sits in the bottom right
during play. `M` switches between pose and mouse control at any time.

### Swing detection with the LEGO hub (IMU)

The hub (Double Motor, the orange **1129** card; change `HUB_CARD` in `config.py` or pass `--card red:1129`) is
mounted on the paddle. Its accelerometer + gyroscope tell the game when you swing and which way. Bluetooth
permission is needed for your terminal app (macOS asks the first time).

```bash
python swing_test.py --raw                    # 1. is the hub streaming? raw readings + every swing it detects
python collect_swings.py --manual             # 2. record ~30 swings to your right and ~30 to your left (ENTER starts/ends each)
python train_swings.py --sweep                # 3. compare trigger levels / decision windows on YOUR recordings
python train_swings.py --threshold 250 --post 0.03    #    train with the setting the sweep recommends
python game.py --pose --imu                   # 4. play
```

`collect_swings.py --manual --classes right` re-records just one direction (right or left).

**Retraining on the motion you actually use (teach mode).** If your in-game swing differs from your recordings
(smaller, because the camera only sees so much of your wrist range), teach the model from real play:

```bash
python game.py --pose --imu --teach     # play normally
```

Right after a swing, press the left arrow (you swung left) or the right arrow (you swung right). The hub's last
1.6 s is saved as a labelled example; pressing the other arrow for the same motion corrects it. Aim for 15-20 per
direction, then quit (it prints the retrain command):

```bash
python train_swings.py --sweep          # optional: re-pick the trigger level, since your swings may be softer now
python train_swings.py --threshold 250 --post 0.03
```

To start the swing dataset over instead, `collect_swings.py --manual --fresh` archives the old recordings
(renamed, never deleted) and records new ones.

**Tuning sensitivity.** There is no "not a swing" class: a motion counts as a swing when it crosses the **trigger**
(its rotation rate rises above a set level), and the network only decides which way. Gentle handling never reaches
the trigger; a hard knock that does will count as a swing, which only matters if your racket is at the ball at that
moment. In the game, a swing that doesn't register says why on screen ("too weak: peak 210, needs 250");
`swing_test.py --verbose` prints the same. If swings are "too weak", lower the trigger
(`train_swings.py --threshold 200`) and record or teach some gentler swings; if too many knocks count, raise it.

Mount the hub on the paddle exactly as it will be used; if you move it, record again. Before you train a
model, `--imu` still works: any swing counts, with no direction (so the return aims straight).

How it works (`swing.py`): the detector spots the *start* of a swing from the gyro's rotation rate, waits a few
hundredths of a second, and a small neural network (`scikit-learn` MLP) reads a short window of motion around the
trigger and decides **right** or **left**. If it can't tell, the swing still counts and aims straight. Timing
matters: on the LEGO hub a swing is a ~100 ms flick (and the hub reports only ~38 times a second), so the decision
must come within ~30 ms of the trigger or it arrives after the paddle has already met the ball. `train_swings.py
--sweep` measures accuracy *and* how long before the swing's peak speed the decision arrives, for several trigger
levels, so you can pick a setting that is both accurate and in time. The event is back-dated to when the swing
started. The racket's speed during the swing sets how hard the return is hit; the hub decides the direction.
`--fake-imu` uses a simulated hub (and `collect_swings.py --fake` / `train_swings.py --fake` use `data/fake/`, kept
separate so simulated data never mixes with yours).

### Score over MQTT

Your **record** (most consecutive returns) is published as a float string such as `12.0` to
`ME193/Rogers/JakeMcPherson` on the course's public broker `test.mosquitto.org` (QoS 0, no retain, the settings the
course MQTT page recommends). It is sent when the game starts, each time you set a new record (so it updates live
as your streak climbs past it), and at game over. The record is saved in `record.json`, so restarting doesn't
reset it. To watch it, subscribe to the topic in the course's MQTT debugger page.

```bash
python game.py --no-mqtt              # don't publish (use this when practising or testing)
python game.py --mqtt-mode streak     # publish the live streak instead of the record
python game.py --mqtt-retain          # have the broker keep the last score for late subscribers
python game.py --reset-record         # forget the saved record
```

The broker is public: anyone can read the topic. Publishing happens on a background thread, so no Wi-Fi or a slow
broker never freezes the game (the status line just says it is retrying). The topic and broker are in `config.py`;
`mqttlib.py` is the course's wrapper, unchanged.

### Sound

`python make_sounds.py` synthesises the effects into `assets/sounds/`: countdown beeps, a jingle at GO!, the ball
off your racket (louder for harder swings), the opponent's racket, the table, the net, game over, and a flourish for
a new record. Replace any of the WAV files with your own (same names); `--mute` silences the game.

Return aiming uses the racket's fastest motion during the swing: sideways steers, speed sets depth and pace,
an upward swing lobs the ball and a downward one drives it flatter and faster.

Testing flags: `--autoplay` (a bot plays), `--miss-after N` (the bot gives up after N hits), `--seed`,
`--log` (print rally events), `--fake-pose` (a scripted hand stands in for you), `--start-state`, `--difficulty`, `--shots`, `--quit-after`.

| Tag | Difficulty | Ball speed |
|-----|------------|-----------|
| 1   | EASY       | 2.8 m/s   |
| 2   | MEDIUM     | 3.6 m/s   |
| 3   | HARD       | 4.8 m/s   |

Print `assets/tags/difficulty_tags.pdf` (one tag per page) at 100% scale on white paper. Hold the tag
roughly 0.5-1.5 m from the webcam, flat toward it, with decent light. Tags smaller than about 10 cm across the
camera frame get unreliable. Tags are the 36h11 AprilTag family, detected with OpenCV's `cv2.aruco`.

Code map: `config.py` (constants, difficulties), `physics.py` (pure game logic), `tags.py` (tag generation,
detection, hold-to-select), `vision.py` (camera thread), `render.py` (the 3D scene, the only file that sets up
Ursina), `ui.py` (title / choose / countdown / HUD), `game.py` (state machine).

## Look-and-feel spike (original single-file prototype)

```bash
python spike_ursina.py         # scripted rally; move the mouse to move/tilt the racket; q quits
```

Test flags: `--wiggle` (drive the racket automatically), `--shot` (save `spike_shot.png`),
`--burst` (save 8 frames).

Assets: `assets/grass.png` (pixel-art ground tile) and `assets/background.png` (any image; tune
`BACKGROUND_KEEP_TOP` / `BACKGROUND_HEIGHT_M` at the top of `spike_ursina.py`).

## macOS / Ursina gotchas (already handled in the spike)

- Panda3D needs `gl-version 3 2` set before the window opens, or Ursina's GLSL 150 shaders fail to compile.
- Ursina's full-screen fade overlay renders opaque black here and hides the scene: `camera.overlay.enabled = False`.
- `color.rgb()` takes 0–1 floats in Ursina 7; the spike uses a 0–255 helper `C(r, g, b)`.
- The UI camera is only sized on a window-aspect event, so HUD text stayed invisible until the window was
  refocused: call `window.update_aspect_ratio()` once at startup.
- Ursina's shadow bias is hard-coded too low (striped legs / speckles): the spike uses a copy of the shader
  with a larger bias and a tight shadow area (`sun.update_bounds`).
- Lit shaders ignore alpha and `unlit_shader` is GLSL 130 (unsupported): transparent objects use `shader=None`.
