"""The 3D scene (Ursina). IMPORT THIS BEFORE ANYTHING ELSE THAT USES URSINA: it sets the macOS GL profile.

Game logic lives in physics.py; this module only draws and moves objects it is told about.
See README.md ("macOS / Ursina gotchas") for why each workaround below exists.
"""
import math
from pathlib import Path

from panda3d.core import SamplerState, loadPrcFileData

# macOS hands Panda3D a legacy GL 2.1 context by default; Ursina's shaders need GLSL 150.
loadPrcFileData("", "gl-version 3 2")

from PIL import Image as PILImage            # noqa: E402
from ursina import *                         # noqa: E402,F401,F403
from ursina.shader import Shader             # noqa: E402
from ursina.shaders import lit_with_shadows_shader  # noqa: E402

import config as cfg                         # noqa: E402
from physics import Z_OPP_HIT                # noqa: E402

ASSETS = Path(__file__).parent / "assets"

# Backdrop: any image standing at the far end of the grass.
#   BACKGROUND_KEEP_TOP:  fraction of the image height kept from the top (1.0 = whole image)
#   BACKGROUND_HEIGHT_M:  how tall that part stands in the scene (smaller = squashed, more fits in view)
BACKGROUND_KEEP_TOP = 0.52
BACKGROUND_HEIGHT_M = 6.5
PIXEL_BACKGROUND = False       # True keeps a pixel-art background crisp instead of smoothing it
BG_Z = 22


def C(r, g, b, a=255):
    """0-255 colour (ursina 7's color.rgb/rgba take 0-1 floats, which silently over-exposes)."""
    return Color(r / 255, g / 255, b / 255, a / 255)


def init_app(title):
    """Create the Ursina app and apply the macOS workarounds. Call once, before building a World."""
    app = Ursina(title=title, borderless=False, vsync=True)
    camera.overlay.enabled = False        # Ursina's fade overlay renders opaque black here and hides the scene
    window.color = C(18, 24, 40)
    window.fps_counter.enabled = False
    window.entity_counter.enabled = False
    window.collider_counter.enabled = False
    window.exit_button.visible = False
    window.cog_button.enabled = False
    # The UI camera is only sized on an 'aspect ratio changed' event, which macOS fires on focus, not at
    # startup: without this the HUD stays invisible until you click away and back.
    invoke(window.update_aspect_ratio, delay=0.1)
    invoke(window.update_aspect_ratio, delay=0.6)
    # Ursina hard-codes a shadow bias of 0.001, which self-shadows (striped legs, speckled surfaces).
    Entity.default_shader = Shader(
        language=Shader.GLSL, name="lit_shadow_bias",
        vertex=lit_with_shadows_shader.vertex,
        fragment=lit_with_shadows_shader.fragment.replace("const float bias = 0.001;", "const float bias = 0.006;"),
        default_input=lit_with_shadows_shader.default_input)
    return app


def flat(entity):
    """Draw an object with a flat colour: no lighting, no shadow receiving. Lit shaders ignore alpha, and
    tilting a lit racket made it flash lighter and stripe with self-shadow."""
    entity.shader = None
    return entity


class World:
    """Builds the scene and exposes the few things the game needs to move."""

    def __init__(self):
        T = cfg
        self.racket_vel = Vec3(0, 0, 0)

        # --- ground and backdrop ---
        grass = Texture(ASSETS / "grass.png")
        grass._texture.set_magfilter(SamplerState.FT_nearest)               # crisp pixels up close...
        grass._texture.set_minfilter(SamplerState.FT_linear_mipmap_linear)  # ...smooth in the distance
        Entity(model="plane", scale=60, texture=grass, texture_scale=(30, 30), y=0, color=color.white)

        bg = PILImage.open(ASSETS / "background.png").convert("RGBA")     # RGBA: Ursina's Texture wants 4 channels
        bg = bg.crop((0, 0, bg.width, int(bg.height * BACKGROUND_KEEP_TOP)))
        bg_tex = Texture(bg)
        if PIXEL_BACKGROUND:
            bg_tex._texture.set_magfilter(SamplerState.FT_nearest)
        flat(Entity(model="quad", texture=bg_tex, scale=(40, BACKGROUND_HEIGHT_M),
                    position=(0, BACKGROUND_HEIGHT_M / 2 - 0.05, BG_Z), color=color.white))

        # --- table ---
        surface_y = T.TABLE_H - 0.025
        Entity(model="cube", scale=(T.TABLE_W, 0.05, T.TABLE_L), y=surface_y, color=C(24, 92, 160))
        Entity(model="cube", scale=(T.TABLE_W + 0.02, 0.052, 0.012), y=surface_y, color=color.white)   # centre line
        for sx in (-1, 1):
            Entity(model="cube", scale=(0.012, 0.052, T.TABLE_L), x=sx * (T.TABLE_W / 2 - 0.006),
                   y=surface_y, color=color.white)
        for sz in (-1, 1):
            Entity(model="cube", scale=(T.TABLE_W, 0.052, 0.012), z=sz * (T.HALF_L - 0.006),
                   y=surface_y, color=color.white)
        for sx in (-1, 1):
            for sz in (-1, 1):
                Entity(model="cube", scale=(0.06, T.TABLE_H - 0.05, 0.06),
                       position=(sx * (T.TABLE_W / 2 - 0.1), (T.TABLE_H - 0.05) / 2, sz * (T.HALF_L - 0.2)),
                       color=C(50, 55, 70))
        net_w, net_h = T.TABLE_W + 0.2, 0.15
        flat(Entity(model="cube", scale=(net_w, net_h, 0.004), y=T.TABLE_H + net_h / 2,
                    color=C(225, 232, 240, 110))).setTransparency(True)          # translucent panel
        Entity(model="cube", scale=(net_w, 0.012, 0.012), y=T.TABLE_H + net_h, color=color.white)  # top tape
        for sx in (-1, 1):
            Entity(model="cube", scale=(0.015, 0.17, 0.02), x=sx * (T.TABLE_W / 2 + 0.1),
                   y=T.TABLE_H + 0.085, color=C(200, 200, 200))

        # --- ball, racket, target ring ---
        self.ball = Entity(model="sphere", scale=T.BALL_R * 2, color=C(255, 150, 40))
        self.racket = Entity()
        def disc():                                  # one mesh per part: origin_z edits the mesh in place, so sharing breaks it
            return Cylinder(32, start=0, height=1, direction=(0, 0, 1))   # axis along z: spans z 0..1
        Entity(parent=self.racket, model=disc(), origin_z=.5, scale=(0.2, 0.22, 0.018), color=C(215, 165, 105))  # rim
        Entity(parent=self.racket, model=disc(), origin_z=.5, scale=(0.17, 0.19, 0.026), color=C(205, 32, 48))    # faces
        Entity(parent=self.racket, model="cube", scale=(0.03, 0.1, 0.016), y=-0.15, color=C(150, 110, 70))      # handle
        for part in self.racket.children:
            flat(part)                                                   # still casts its shadow on the grass
        self.opp = Entity(position=(0, T.TABLE_H + 0.3, Z_OPP_HIT + 0.12))
        Entity(parent=self.opp, model=disc(), origin_z=.5, scale=(0.2, 0.22, 0.018), color=C(215, 165, 105))
        Entity(parent=self.opp, model=disc(), origin_z=.5, scale=(0.17, 0.19, 0.026), color=C(40, 90, 205))
        Entity(parent=self.opp, model="cube", scale=(0.03, 0.1, 0.016), y=-0.15, color=C(150, 110, 70))
        for part in self.opp.children:
            flat(part)
        self.ring = flat(Entity(model=Circle(32, mode="line", thickness=3), scale=0.2,
                                color=C(255, 255, 255, 190)))
        self.ring.setTransparency(True)

        # --- light and camera ---
        sun = DirectionalLight(shadows=True, shadow_map_resolution=Vec2(4096, 4096))
        sun.look_at(Vec3(0.4, -1, 0.6))
        bounds = Entity(model="cube", scale=(3.6, 2.0, 5.2), y=0.9, visible=False)
        invoke(sun.update_bounds, bounds, delay=0.3)      # fit the shadow map to the play space, not the floor
        AmbientLight(color=C(110, 120, 150))
        camera.position = (0, 1.25, -3.3)
        camera.rotation_x = 13
        camera.fov = 62

    # ---- things the game moves ----
    def set_visible(self, ball=False, racket=False, opponent=None):
        self.ball.enabled = ball
        self.racket.enabled = racket
        self.opp.enabled = ball if opponent is None else opponent
        if not ball:
            self.ring.enabled = False

    def place_ball(self, x, y, z, pop=0.0):
        """pop (0..1) briefly swells the ball, as hit feedback."""
        self.ball.position = (x, y, z)
        self.ball.scale = cfg.BALL_R * 2 * (1 + 0.7 * pop)

    def set_opponent(self, ball_x, ball_y, swing_age, dt):
        """The opponent's racket tracks the ball and swings when it returns it."""
        tx = clamp(ball_x, -0.9, 0.9)
        ty = max(ball_y, cfg.TABLE_H + 0.2)
        self.opp.x = lerp(self.opp.x, tx, min(1, 8 * dt))
        self.opp.y = lerp(self.opp.y, ty, min(1, 8 * dt))
        self.opp.rotation_x = 55 * max(0.0, 1 - swing_age / 0.22)

    def show_ring(self, x, y, scale):
        self.ring.position = (x, y, cfg.PLAYER_Z + 0.04)
        self.ring.scale = scale
        self.ring.enabled = True

    def hide_ring(self):
        self.ring.enabled = False

    def mouse_to_plane(self, mx, my, plane_z=cfg.PLAYER_Z):
        """World (x, y) where the camera ray through the mouse crosses the plane z = plane_z."""
        aspect = window.aspect_ratio
        th = math.tan(math.radians(camera.fov) / 2)          # ursina's fov is horizontal
        px, py = mx / (0.5 * aspect) * th, my / 0.5 * th / aspect
        pitch = math.radians(camera.rotation_x)               # positive = looking down
        dx, dy, dz = px, py * math.cos(pitch) - math.sin(pitch), py * math.sin(pitch) + math.cos(pitch)
        k = (plane_z - camera.z) / dz
        return camera.x + k * dx, camera.y + k * dy

    def drive_racket(self, tx, ty, dt, swing_age=99.0):
        """Glide the racket toward (tx, ty) on the player plane, leaning into its own motion. swing_age is the
        time since the last swing: the blade snaps forward for a moment so the swing is visible."""
        target = Vec3(clamp(tx, -1.3, 1.3), clamp(ty, cfg.TABLE_H + 0.05, 2.2), cfg.PLAYER_Z)
        prev = Vec3(*self.racket.position)
        self.racket.position = lerp(self.racket.position, target, min(1, 20 * dt))
        self.racket_vel = lerp(self.racket_vel, (self.racket.position - prev) / max(dt, 1e-3), min(1, 12 * dt))
        v = self.racket_vel
        self.racket.rotation = (clamp(-v.y * 14, -35, 35) - 50 * max(0.0, 1 - swing_age / 0.2),
                                clamp(v.x * 12, -35, 35), clamp(-v.x * 20, -45, 45))

    def sway(self, t, amount):
        """Slow sideways camera drift for the menu screens (0 during play so mouse aiming stays exact)."""
        camera.x = amount * math.sin(t * 0.4)

    def center_camera(self):
        camera.x = 0
