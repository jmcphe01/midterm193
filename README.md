# midterm193 — virtual ping-pong

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python make_assets.py          # grass tile (+ a placeholder background if assets/background.png is missing)
```

Libraries: OpenCV (`opencv-python`), LEGO Education (`legoeducation`), Ursina (3D engine, Panda3D based).

## Look-and-feel spike

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
