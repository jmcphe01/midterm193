"""Ball physics and rally rules. Pure Python (no Ursina), so it can be simulated headlessly and tested.

Coordinates: x lateral, y up, z along the table; the player stands at -z, the opponent at +z.

A rally:  the opponent serves -> the ball bounces once on the player's half -> the player returns it
(racket close enough to the ball AND a recent swing) -> the ball must clear the net and land on the
opponent's half -> the opponent returns it -> ... The game ends when the player misses, or their return
hits the net / lands out. Each shot is solved ballistically so it lands where it is aimed.
"""
import math
import random

import config as cfg
from config import BALL_R, HALF_L, PLAYER_Z, TABLE_H, TABLE_L, TABLE_W

G = 4.0                        # gravity m/s^2. Real is 9.81; lower gives flatter, more readable arcs
RESTITUTION = 0.86             # fraction of vertical speed kept after a table bounce
BOUNCE_FRICTION = 0.97         # horizontal speed kept after a table bounce
NET_H = 0.15
NET_W = TABLE_W + 0.2
NET_TOP = TABLE_H + NET_H
NET_CLEARANCE = 0.05           # shots are aimed to pass at least this far above the net
GROUND = TABLE_H + BALL_R      # ball-centre height when resting on the table
Z_OPP_HIT = HALF_L + 0.45      # the opponent returns the ball when it gets this far
ZONE_FAR = PLAYER_Z + 0.28     # the ball can be returned once it is inside [ZONE_NEAR, ZONE_FAR] ...
ZONE_NEAR = PLAYER_Z - 0.12    # ... and counts as missed once it gets past ZONE_NEAR
ARRIVAL_Z = PLAYER_Z + 0.08    # where the target ring predicts the ball (middle of the zone)
ARRIVAL_MIN_H, ARRIVAL_MAX_H = 0.14, 0.55   # opponent shots must arrive within this height above the table
SWING_WINDOW = 0.28            # a swing counts for this long (s) before the ball reaches the racket
SUBSTEP = 1 / 240


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def solve_launch(p0, tx, tz, speed):
    """Launch velocity (vx, vy, vz) from p0 so the ball first touches the table at (tx, tz), travelling about
    `speed` m/s along z, and clears the net. If the flat shot would clip the net it is lobbed (slower, higher)."""
    x0, y0, z0 = p0
    dz = tz - z0
    last = None
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.42):
        vz = math.copysign(speed * scale, dz)
        t = dz / vz
        vx = (tx - x0) / t
        vy = (GROUND - y0 + 0.5 * G * t * t) / t
        last = (vx, vy, vz)
        tn = (0.0 - z0) / vz                       # time to reach the net plane
        if 0.0 < tn < t:
            y_net = y0 + vy * tn - 0.5 * G * tn * tn
            if y_net - BALL_R < NET_TOP + NET_CLEARANCE:
                continue
        return last
    return last


def advance(pos, vel, h):
    """One physics step of length h. Returns (pos, vel, events); events are ('bounce', side), ('net',), ('floor',)."""
    x, y, z = pos
    vx, vy, vz = vel
    events = []
    vy -= G * h
    nx, ny, nz = x + vx * h, y + vy * h, z + vz * h

    # the net: crossing its plane below its top edge
    if ((z < 0.0 <= nz) or (z >= 0.0 > nz)) and abs(nx) < NET_W / 2 and TABLE_H - 0.05 < ny - BALL_R < NET_TOP:
        nz = z
        vz = -vz * 0.25
        vx *= 0.5
        events.append(("net",))

    # the table
    if vy < 0 and y >= GROUND > ny and abs(nx) <= TABLE_W / 2 and abs(nz) <= TABLE_L / 2:
        ny = GROUND
        vy = -vy * RESTITUTION
        vx *= BOUNCE_FRICTION
        vz *= BOUNCE_FRICTION
        events.append(("bounce", "player" if nz < 0 else "opponent"))
    elif vy < 0 and ny <= BALL_R:                  # the floor
        ny = BALL_R
        vy = -vy * 0.5
        vx *= 0.8
        vz *= 0.8
        events.append(("floor",))
    return (nx, ny, nz), (vx, vy, vz), events


def simulate_arrival(pos, vel, h=1 / 120):
    """Run the ball forward until it crosses ARRIVAL_Z. Returns (x, y, seconds)."""
    t = 0.0
    while pos[2] > ARRIVAL_Z and t < 3.0:
        pos, vel, _ = advance(pos, vel, h)
        t += h
    return pos[0], pos[1], t


class Rally:
    """One game's worth of rally state: ball, score, and the rules. Feed it the racket each frame."""

    def __init__(self, difficulty, rng=None):
        self.d = difficulty
        self.rng = rng or random.Random()
        self.t = 0.0
        self.streak = 0
        self.phase = "toplayer"       # toplayer | toopp | dead
        self.reason = ""              # why the game ended: miss | net | out
        self.dead_time = 0.0
        self.last_swing_t = -99.0
        self.swing_v = (0.0, 0.0)
        self.swing_dir = 0
        self.bounces_on_player_side = 0
        self.opp_bounced = False
        self.opponent_hit_t = -99.0   # for animating the opponent's racket
        self.pos, self.vel = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        self._opponent_shot(serve=True)

    # ---- inputs ----
    def swing(self, vx=0.0, vy=0.0, age=0.0, direction=0):
        """The player swung. (vx, vy): the swing's velocity in m/s. age: how long ago it STARTED (the IMU only
        reports a swing ~0.15 s after it begins). direction: +1 / -1 if the IMU knows left/right, else 0, in which
        case the racket's own sideways motion decides."""
        self.last_swing_t = self.t - age
        self.swing_v = (vx, vy)
        self.swing_dir = direction

    # ---- shots ----
    def _opponent_shot(self, serve=False):
        x, y, z = self.pos
        if serve:
            x, y, z = self.rng.uniform(-0.3, 0.3), TABLE_H + 0.35, Z_OPP_HIT
        y = max(y, TABLE_H + 0.2)
        speed = self.d.speed * self.rng.uniform(0.92, 1.08)
        self.pos = (x, y, z)
        # Aim by where the ball should ARRIVE at the player (so it never drifts far outside the table), then
        # work back to the point on the near half where it should bounce.
        arrive_x = self.rng.uniform(-0.65, 0.65)
        for attempt in range(16):                       # re-roll shots that arrive too low/high to be fair
            bounce_z = self.rng.uniform(-1.1, -0.55) if attempt < 15 else -1.0
            bounce_x = x + (arrive_x - x) * (bounce_z - z) / (ARRIVAL_Z - z)
            vel = solve_launch(self.pos, bounce_x, bounce_z, speed)
            height = simulate_arrival(self.pos, vel)[1] - TABLE_H
            if ARRIVAL_MIN_H <= height <= ARRIVAL_MAX_H:
                break
        self.vel = vel
        self.phase = "toplayer"
        self.bounces_on_player_side = 0
        self.opp_bounced = False
        self.opponent_hit_t = self.t

    def _player_return(self):
        """Aim the return from the racket's motion: sideways steers it, swing speed sets depth and pace, and an
        upward swing lobs it while a downward one drives it flatter and faster."""
        sx, sy = self.swing_v
        speed = math.hypot(sx, sy)
        tx = clamp(0.30 * sx, -0.62, 0.62) + self.rng.uniform(-0.04, 0.04)    # swing sideways to aim
        tz = clamp(0.55 + 0.10 * speed, 0.55, 1.05)                           # swing harder to go deeper
        pace = clamp((0.85 + 0.07 * speed) * (1 - 0.07 * clamp(sy, -3, 3)), 0.7, 1.4)
        self.vel = solve_launch(self.pos, tx, tz, self.d.speed * pace)
        self.phase = "toopp"
        self.opp_bounced = False
        self.last_swing_t = -99.0                                              # a swing returns one ball only
        self.streak += 1

    def _die(self, reason):
        self.phase, self.reason = "dead", reason

    # ---- simulation ----
    def step(self, dt, racket_x, racket_y, racket_vx=0.0, racket_vy=0.0):
        """Advance by dt seconds with the racket at (racket_x, racket_y) on the player plane, moving at
        (racket_vx, racket_vy) m/s. Returns a list of events:
        ('hit', streak) ('opp_hit',) ('bounce', side) ('miss',) ('net',) ('out',)."""
        events = []
        # Remember the fastest racket motion during the swing: the return is aimed from it, so it does not matter
        # whether the racket is still moving at the exact instant of contact.
        if self.t - self.last_swing_t <= SWING_WINDOW:
            if self.swing_dir:                                  # the IMU said which way: only the speed comes from the racket
                racket_vx = self.swing_dir * abs(racket_vx)
            if math.hypot(racket_vx, racket_vy) > math.hypot(*self.swing_v):
                self.swing_v = (racket_vx, racket_vy)
        n = max(1, int(math.ceil(dt / SUBSTEP)))
        h = dt / n
        for _ in range(n):
            self.t += h
            self.pos, self.vel, physics_events = advance(self.pos, self.vel, h)
            if self.phase == "dead":
                self.dead_time += h
                continue
            for ev in physics_events:
                if ev[0] == "net":
                    events.append(ev)
                    if self.phase == "toopp":
                        self._die("net")
                elif ev[0] == "bounce":
                    events.append(ev)
                    if ev[1] == "player":
                        self.bounces_on_player_side += 1
                        if self.phase == "toopp":              # returned onto their own side
                            self._die("out")
                    elif self.phase == "toopp":
                        self.opp_bounced = True
                elif ev[0] == "floor" and self.phase == "toopp":
                    self._die("out")
                if self.phase == "dead":
                    if self.reason != "net":              # ('net',) was already reported by the physics event
                        events.append((self.reason,))
                    break
            if self.phase == "dead":
                continue

            x, y, z = self.pos
            if self.phase == "toplayer":
                if ZONE_NEAR <= z <= ZONE_FAR and self.bounces_on_player_side >= 1:
                    near = math.hypot(x - racket_x, y - racket_y) <= self.d.hit_radius
                    swung = self.t - self.last_swing_t <= SWING_WINDOW
                    if near and swung:
                        self._player_return()
                        events.append(("hit", self.streak))
                        continue
                if z < ZONE_NEAR:
                    self._die("miss")
                    events.append(("miss",))
            elif self.phase == "toopp" and z >= Z_OPP_HIT:
                if not self.opp_bounced:
                    self._die("out")
                    events.append(("out",))
                else:
                    self._opponent_shot()
                    events.append(("opp_hit",))
        return events

    def predict_arrival(self):
        """Where and when the incoming ball will cross ARRIVAL_Z: (x, y, seconds). None if it isn't incoming."""
        if self.phase != "toplayer":
            return None
        return simulate_arrival(self.pos, self.vel)
