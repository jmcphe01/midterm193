"""The player's record (most consecutive returns) and publishing it over MQTT.

The record is kept in record.json, so restarting the game doesn't reset it. Publishing happens on a background
thread: a slow or unreachable broker (or no Wi-Fi) can never freeze or crash the game; it just shows in the status
text and the thread keeps retrying.

Payload: the record as a float string such as "12.0", QoS 0, no retain (the course page: "in robotics you
typically run no retain and QoS = 0"), to the topic in config.py.
"""
import json
import queue
import threading
import time
from pathlib import Path

import config as cfg

RECORD_PATH = Path(__file__).parent / "record.json"


class Scoreboard:
    def __init__(self, enabled=True, topic=cfg.MQTT_TOPIC, broker=cfg.MQTT_BROKER, port=cfg.MQTT_PORT, retain=False,
                 mode="record", record_path=RECORD_PATH):
        self.enabled, self.topic, self.broker, self.port, self.retain = enabled, topic, broker, port, retain
        self.mode = mode                      # 'record' = publish the best streak; 'streak' = the live streak
        self.record_path = Path(record_path)
        self.best = self._load()
        self.status = "MQTT off" if not enabled else "MQTT: connecting..."
        self.last_published = None
        self.published_count = 0
        self._broke_record = False            # did the current game set a new record?
        self._queue = queue.Queue()
        self._halt = threading.Event()
        self._thread = None

    # ---- the record ----
    def _load(self):
        try:
            return int(json.loads(self.record_path.read_text())["best"])
        except (OSError, ValueError, KeyError):
            return 0

    def _save(self):
        try:
            self.record_path.write_text(json.dumps({"best": self.best}))
        except OSError:
            pass

    def reset_record(self):
        self.best = 0
        self._save()

    # ---- game events ----
    def new_game(self):
        self._broke_record = False

    def on_hit(self, streak):
        """A return landed. In 'record' mode only a new record changes what is published."""
        if streak > self.best:
            self.best = streak
            self._broke_record = True
            self._save()
            if self.mode == "record":
                self._publish(float(self.best))
        if self.mode == "streak":
            self._publish(float(streak))

    def on_game_over(self, streak):
        """Publishes the final value. Returns True if this game set a new record (for the 'NEW RECORD!' banner)."""
        broke, self._broke_record = self._broke_record and streak > 0, False
        self._publish(float(self.best if self.mode == "record" else streak))
        return broke

    # ---- MQTT ----
    def start(self):
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def close(self, timeout=2.0):
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _publish(self, value):
        if self.enabled:
            self._queue.put(f"{value:.1f}")

    def _run(self):
        from mqttlib import MQTTClient            # the course's wrapper around paho
        client = None
        payload_waiting = f"{float(self.best):.1f}"      # announce the current record as soon as we connect
        while not self._halt.is_set():
            try:
                if client is None:
                    self.status = "MQTT: connecting..."
                    client = MQTTClient(self.broker, self.port)
                    client.connect(timeout=5)
                    if not client._connected.is_set():
                        raise ConnectionError("no answer from the broker")
                    self.status = "MQTT: connected"
                if payload_waiting is not None:
                    client.publish(self.topic, payload_waiting, qos=0, retain=self.retain)
                    self.last_published, self.published_count = payload_waiting, self.published_count + 1
                    payload_waiting = None
                try:
                    payload_waiting = self._queue.get(timeout=0.3)
                except queue.Empty:
                    pass
            except Exception as exc:               # noqa: BLE001 - any network problem: say so, wait, try again
                self.status = f"MQTT: {type(exc).__name__}, retrying"
                if client is not None:
                    try:
                        client.disconnect()
                    except Exception:               # noqa: BLE001
                        pass
                    client = None
                self._halt.wait(3.0)
        if client is not None:
            try:
                if payload_waiting is not None:
                    client.publish(self.topic, payload_waiting, qos=0, retain=self.retain)
                    time.sleep(0.2)
                client.disconnect()
            except Exception:                       # noqa: BLE001
                pass
