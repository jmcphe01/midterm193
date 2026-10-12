"""Render-only look-and-feel spike: table, net, scripted ball + shadow, mouse-driven racket.

Run:  python spike_ursina.py          (interactive; move the mouse, q/Esc quits)
      python spike_ursina.py --shot   (saves spike_shot.png after 3 s and quits)
"""
import math
import sys

from pathlib import Path

from panda3d.core import SamplerState, loadPrcFileData

# macOS hands Panda3D a legacy GL 2.1 context by default; Ursina's shaders need GLSL 150.
loadPrcFileData("", "gl-version 3 2")

from ursina import *
from ursina.shaders import lit_with_shadows_shader


def C(r, g, b, a=255):
    """0-255 colour (ursina 7's color.rgb/rgba take 0-1 floats, which silently over-exposes)."""
    return Color(r / 255, g / 255, b / 255, a / 255)


app = Ursina(title="Pong spike", borderless=False, vsync=True)
# Ursina's full-screen fade overlay renders opaque black on this Mac (GL 3.2 core profile),
# hiding the whole 3D scene, so switch it off.
camera.overlay.enabled = False
window.color = C(18, 24, 40)
from ursina.shader import Shader
_acne_fix_shader = Shader(language=Shader.GLSL, name="lit_shadow_bias",
                          vertex=lit_with_shadows_shader.vertex,
                          fragment=lit_with_shadows_shader.fragment.replace("const float bias = 0.001;",
                                                                           "const float bias = 0.006;"),
                          default_input=lit_with_shadows_shader.default_input)
Entity.default_shader = _acne_fix_shader

# --- dimensions (metres, table tennis scale; z runs from player (-) to opponent (+)) ---
TABLE_W, TABLE_L, TABLE_H = 1.525, 2.74, 0.76
HALF_L = TABLE_L / 2
PLAYER_Z = -HALF_L - 0.25       # plane the racket lives on
BALL_R = 0.055   # slightly oversized so it reads on screen

# --- scene ---
ASSETS = Path(__file__).parent / "assets"
PIXEL_BACKGROUND = False        # True keeps a pixel-art background crisp instead of smoothing it

grass_tex = Texture(ASSETS / "grass.png")
grass_tex._texture.set_magfilter(SamplerState.FT_nearest)               # crisp pixels up close...
grass_tex._texture.set_minfilter(SamplerState.FT_linear_mipmap_linear)  # ...smooth in the distance
floor = Entity(model="plane", scale=60, texture=grass_tex, texture_scale=(30, 30), y=0, color=color.white)

# backdrop: any image standing at the far end of the grass.
#   BACKGROUND_KEEP_TOP: fraction of the image height kept from the top (1.0 = the whole image). The
#       current photo is a lake whose bottom half is reflection, so only the mountains/shore are kept.
#   BACKGROUND_HEIGHT_M: how tall that kept part stands in the scene (its width fills the view; a
#       smaller height squashes the picture vertically so more of it fits above the horizon).
BACKGROUND_KEEP_TOP = 0.52
BACKGROUND_HEIGHT_M = 6.5
BG_Z = 22
from PIL import Image as _PILImage
_bg_img = _PILImage.open(ASSETS / "background.png").convert("RGBA")
_bg_img = _bg_img.crop((0, 0, _bg_img.width, int(_bg_img.height * BACKGROUND_KEEP_TOP)))
bg_tex = Texture(_bg_img)
if PIXEL_BACKGROUND:
    bg_tex._texture.set_magfilter(SamplerState.FT_nearest)
backdrop = Entity(model="quad", texture=bg_tex, scale=(40, BACKGROUND_HEIGHT_M), position=(0, BACKGROUND_HEIGHT_M / 2 - 0.05, BG_Z),
                  shader=None, color=color.white)
table = Entity(model="cube", scale=(TABLE_W, 0.05, TABLE_L), y=TABLE_H - 0.025,
               color=C(24, 92, 160))
Entity(model="cube", scale=(TABLE_W + 0.02, 0.052, 0.012), y=TABLE_H - 0.025, color=color.white)  # centre line
for sx in (-1, 1):
    Entity(model="cube", scale=(0.012, 0.052, TABLE_L), x=sx * (TABLE_W / 2 - 0.006),
           y=TABLE_H - 0.025, color=color.white)
for sz in (-1, 1):
    Entity(model="cube", scale=(TABLE_W, 0.052, 0.012), z=sz * (HALF_L - 0.006),
           y=TABLE_H - 0.025, color=color.white)
for sx in (-1, 1):
    for sz in (-1, 1):
        Entity(model="cube", scale=(0.06, TABLE_H - 0.05, 0.06),
               position=(sx * (TABLE_W / 2 - 0.1), (TABLE_H - 0.05) / 2, sz * (HALF_L - 0.2)),
               color=C(50, 55, 70))
NET_W, NET_H = TABLE_W + 0.2, 0.15
net = Entity(model="cube", scale=(NET_W, NET_H, 0.004), y=TABLE_H + NET_H / 2,
             color=C(225, 232, 240, 110))                                    # translucent mesh panel
Entity(model="cube", scale=(NET_W, 0.012, 0.012), y=TABLE_H + NET_H, color=color.white)   # top tape
for sx in (-1, 1):
    Entity(model="cube", scale=(0.015, 0.17, 0.02), x=sx * (TABLE_W / 2 + 0.1), y=TABLE_H + 0.085,
           color=C(200, 200, 200))

ball = Entity(model="sphere", scale=BALL_R * 2, color=C(255, 150, 40))

racket = Entity()
_disc = Cylinder(32, start=0, height=1, direction=(0, 0, 1))     # axis along z: spans z 0..1
Entity(parent=racket, model=_disc, origin_z=.5, scale=(0.2, 0.22, 0.018), color=C(215, 165, 105))   # wooden rim
blade = Entity(parent=racket, model=_disc, origin_z=.5, scale=(0.17, 0.19, 0.026), color=C(205, 32, 48))  # rubber faces
Entity(parent=racket, model="cube", scale=(0.03, 0.1, 0.016), y=-0.15, color=C(150, 110, 70))   # handle
# Flat colour, no lighting: the lit shader brightened the blade as it tilted and self-shadowed it into
# stripes, which read as an orange flash while moving. It still casts its shadow on the grass.
for _part in racket.children:
    _part.shader = None

ring = Entity(model=Circle(32, mode="line", thickness=3), scale=0.2,
              color=C(255, 255, 255, 190), shader=None)

for _e in (net,):
    _e.shader = None                  # the lit shader ignores alpha; unlit_shader is GLSL 130 (unsupported here)
for _e in (net, ring):
    _e.setTransparency(True)
window.fps_counter.enabled = False
window.entity_counter.enabled = False
window.collider_counter.enabled = False
window.exit_button.visible = False
window.cog_button.enabled = False
# Ursina only sizes the UI camera on an 'aspect ratio changed' event, which macOS fires on focus/resize,
# not at startup: without this the HUD text stays invisible until you click away and back.
invoke(window.update_aspect_ratio, delay=0.1)
invoke(window.update_aspect_ratio, delay=0.6)

sun = DirectionalLight(shadows=True, shadow_map_resolution=Vec2(4096, 4096))
sun.look_at(Vec3(0.4, -1, 0.6))
_bounds = Entity(model="cube", scale=(3.6, 2.0, 5.2), y=0.9, visible=False, add_to_scene_entities=True)
invoke(sun.update_bounds, _bounds, delay=0.3)   # fit the shadow map to the play space, not the whole floor
AmbientLight(color=C(110, 120, 150))

camera.position = (0, 1.25, -3.3)
camera.rotation_x = 13
camera.fov = 62

hud = Text(text="0", origin=(0, 0), position=(0, 0.42), scale=3, color=color.white)
Text(text="spike: scripted ball, move the mouse", position=window.bottom_left + Vec2(0.012, 0.04), scale=1, color=color.white)

# --- scripted rally ---
SPEED = 3.5          # m/s along z
t = 0.0
hits = 0
prev_dir = 1
moved = False
racket_vel = Vec3(0, 0, 0)


Z_NEAR = PLAYER_Z + 0.09                      # stop just in front of the racket face (not inside it)
Z_FAR = HALF_L + 0.7
LEG = Z_FAR - Z_NEAR
S_BOUNCE = 0.70                               # fraction of a leg at which it bounces
HIT_H = 0.30                                  # height above the table surface where each shot is struck
ARC_PRE, ARC_POST = 0.30, 0.03                # extra arch before / after the bounce


def ball_state(t):
    """Scripted rally. Each leg starts at HIT_H, arcs down to ONE bounce on the receiving half of the
    table, then rises back to HIT_H where the next shot is struck, so the path is continuous (no jumps).
    Returns (x, y, z, direction)."""
    leg_time = LEG / SPEED
    k = int(t / leg_time)
    s = (t / leg_time) % 1.0
    going_out = (k % 2 == 0)
    z = Z_NEAR + LEG * s if going_out else Z_FAR - LEG * s
    ground = TABLE_H + BALL_R
    if s < S_BOUNCE:
        u = s / S_BOUNCE
        h = HIT_H * (1 - u) + ARC_PRE * 4 * u * (1 - u)
    else:
        v = (s - S_BOUNCE) / (1 - S_BOUNCE)
        h = HIT_H * v + ARC_POST * 4 * v * (1 - v)
    x = 0.35 * math.sin(t * 0.9)
    return x, ground + h, z, (1 if going_out else -1)


def mouse_to_plane(mx, my, plane_z):
    """World (x, y) where the camera ray through the mouse crosses the plane z = plane_z."""
    aspect = window.aspect_ratio
    th = math.tan(math.radians(camera.fov) / 2)          # ursina's fov is horizontal
    px, py = mx / (0.5 * aspect) * th, my / 0.5 * th / aspect
    pitch = math.radians(camera.rotation_x)               # positive = looking down
    dx, dy, dz = px, py * math.cos(pitch) - math.sin(pitch), py * math.sin(pitch) + math.cos(pitch)
    k = (plane_z - camera.z) / dz
    return camera.x + k * dx, camera.y + k * dy


def update():
    global t, hits, prev_dir
    t += time.dt
    x, y, z, d = ball_state(t)
    ball.position = (x, y, z)

    # racket sits exactly under the cursor: cast the mouse ray onto the racket plane
    global moved, racket_vel
    if mouse.velocity.length() > 0:
        moved = True
    mx, my = (mouse.x, mouse.y) if moved else (0.0, -0.1)
    if "--wiggle" in sys.argv:                       # test mode: drive the racket without a mouse
        mx, my = 0.28 * math.sin(t * 3.1), -0.05 + 0.12 * math.sin(t * 2.3)
    tx, ty = mouse_to_plane(mx, my, PLAYER_Z)
    target = Vec3(clamp(tx, -1.3, 1.3), clamp(ty, TABLE_H + 0.05, 2.2), PLAYER_Z)
    prev = Vec3(*racket.position)
    racket.position = lerp(racket.position, target, min(1, 20 * time.dt))
    racket_vel = lerp(racket_vel, (racket.position - prev) / max(time.dt, 1e-3), min(1, 12 * time.dt))
    # lean into the motion: roll with sideways speed, pitch with vertical speed
    racket.rotation = (clamp(-racket_vel.y * 14, -35, 35),
                       clamp(racket_vel.x * 12, -35, 35),
                       clamp(-racket_vel.x * 20, -45, 45))

    # target ring: where the ball WILL arrive on the racket plane, shrinking as it gets close
    if d == -1:
        leg_time = LEG / SPEED
        t_arrive = (int(t / leg_time) + 1) * leg_time
        ax, ay, _, _ = ball_state(t_arrive - 1e-4)
        ring.position = (ax, ay, Z_NEAR)
        ring.scale = 0.07 + 0.38 * clamp((t_arrive - t) / leg_time, 0, 1)
        ring.enabled = True
    else:
        ring.enabled = False

    # a "hit" on every return arrival
    if d != prev_dir and d == 1:
        hits += 1
        hud.text = str(hits)
    prev_dir = d



def input(key):
    if key in ("q", "escape"):
        application.quit()


if "--burst" in sys.argv:                            # test mode: 8 frames, 0.2 s apart -> burst_N.png
    def _burst(i=[0]):
        from panda3d.core import PNMImage
        img = PNMImage()
        base.win.getScreenshot(img)
        img.write(f"burst_{i[0]}.png")
        i[0] += 1
        if i[0] >= 8:
            application.quit()
    for _k in range(8):
        invoke(_burst, delay=2.0 + 0.2 * _k)

if "--shot" in sys.argv:
    def _shot():
        from panda3d.core import PNMImage
        img = PNMImage()
        base.win.getScreenshot(img)
        img.write("spike_shot.png")
        application.quit()
    invoke(_shot, delay=3)

app.run()
