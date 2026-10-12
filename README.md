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
python game.py                 # webcam; keys 1/2/3 also pick a difficulty
python game.py --no-camera     # keys only
```

Flow: title screen -> **choose difficulty** (hold a printed AprilTag up to the camera for about a second)
-> countdown 3, 2, 1 -> play. In play: `R` returns to the title, `Q` quits.

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
