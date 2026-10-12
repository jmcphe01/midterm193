"""Game logic that must not depend on Ursina. For now: the scripted demo rally, parametrised by speed."""
import math

from config import BALL_R, HALF_L, PLAYER_Z, TABLE_H

Z_NEAR = PLAYER_Z + 0.09          # stop just in front of the racket face (not inside it)
Z_FAR = HALF_L + 0.7
LEG = Z_FAR - Z_NEAR
S_BOUNCE = 0.70                   # fraction of a leg at which it bounces
HIT_H = 0.30                      # height above the table surface where each shot is struck
ARC_PRE, ARC_POST = 0.30, 0.03    # extra arch before / after the bounce


class Rally:
    """Scripted rally. Each leg starts at HIT_H, arcs down to ONE bounce on the receiving half of the
    table, then rises back to HIT_H where the next shot is struck, so the path is continuous."""

    def __init__(self, speed):
        self.speed = speed
        self.leg_time = LEG / speed

    def state(self, t):
        """(x, y, z, direction) at time t; direction is +1 going away from the player, -1 coming back."""
        k = int(t / self.leg_time)
        s = (t / self.leg_time) % 1.0
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

    def arrival(self, t):
        """Where the incoming ball will reach the racket plane: (x, y, seconds_until_arrival, fraction_of_leg_left)."""
        t_arrive = (int(t / self.leg_time) + 1) * self.leg_time
        x, y, _, _ = self.state(t_arrive - 1e-4)
        return x, y, t_arrive - t, (t_arrive - t) / self.leg_time
