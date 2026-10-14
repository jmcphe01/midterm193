"""Shared constants. Pure Python: no Ursina / OpenCV imports, so every module can use it."""
from dataclasses import dataclass

TITLE = "Ping Pong (a game inferior to Frisbee)"

# --- geometry (metres, table-tennis scale; z runs from the player (-) to the opponent (+)) ---
TABLE_W, TABLE_L, TABLE_H = 1.525, 2.74, 0.76
HALF_L = TABLE_L / 2
BALL_R = 0.055            # slightly oversized so it reads on screen
PLAYER_Z = -HALF_L - 0.25  # plane the racket lives on


@dataclass(frozen=True)
class Difficulty:
    name: str
    tag_id: int       # AprilTag (36h11 family) that selects it
    speed: float      # ball speed along the table, m/s
    hit_radius: float # how close (m) the racket must be to the ball to return it


DIFFICULTIES = (
    Difficulty("EASY", 1, 2.8, 0.30),
    Difficulty("MEDIUM", 2, 3.6, 0.24),
    Difficulty("HARD", 3, 4.8, 0.19),
)
BY_TAG = {d.tag_id: d for d in DIFFICULTIES}

# --- start-up flow timing (seconds) ---
TITLE_SECONDS = 3.0       # title screen auto-advances (Space skips)
TAG_HOLD_SECONDS = 0.9    # hold a tag steady this long to select it
TAG_GRACE_SECONDS = 0.35  # tag may drop out of detection this long without resetting the hold
COUNTDOWN_FROM = 3
GO_SECONDS = 0.7

# --- the LEGO hub on the paddle (its Connection Card, as colour:serial) ---
HUB_CARD = "orange:1129"

# --- MQTT: the score goes to this topic on the course's public broker (see the MQTT page on the course Notion) ---
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/Rogers/JakeMcPherson"      # the payload is your record streak as a float, e.g. "12.0"

