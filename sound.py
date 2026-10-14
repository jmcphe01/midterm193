"""Sound effects, played through Panda3D's audio (the same engine Ursina uses).

Sounds live in assets/sounds/<name>.wav (make_sounds.py generates them; replace any with your own WAV). If audio
can't start (no sound device, missing files) the game carries on silently and says so once.
"""
from pathlib import Path

from panda3d.core import Filename

SOUND_DIR = Path(__file__).parent / "assets" / "sounds"
NAMES = ("start", "tick", "hit", "opp", "bounce", "net", "gameover", "record")
POOL = 3            # copies per sound, so quick repeats (e.g. bounce then hit) can overlap


class Sounds:
    def __init__(self, enabled=True, volume=0.9):
        self.enabled, self.volume = enabled, volume
        self._pools, self._next = {}, {}
        self.loaded = []
        if enabled:
            self._load()

    def _load(self):
        loader = __import__("builtins").loader           # Panda3D injects `loader` once the engine is running
        missing = []
        for name in NAMES:
            path = SOUND_DIR / f"{name}.wav"
            if not path.exists():
                missing.append(name)
                continue
            clips = [loader.loadSfx(Filename.fromOsSpecific(str(path))) for _ in range(POOL)]
            if all(c is not None and c.length() > 0 for c in clips):
                self._pools[name], self._next[name] = clips, 0
                self.loaded.append(name)
            else:
                missing.append(name)
        if missing:
            print(f"sound: could not load {', '.join(missing)} (run make_sounds.py); those stay silent")
        if not self.loaded:
            self.enabled = False

    def play(self, name, volume=1.0, rate=1.0):
        """Play `name` (volume 0..1 relative to the master volume; rate changes the pitch)."""
        if not self.enabled or name not in self._pools:
            return
        pool = self._pools[name]
        clip = pool[self._next[name]]
        self._next[name] = (self._next[name] + 1) % len(pool)
        clip.setVolume(max(0.0, min(1.0, volume * self.volume)))
        clip.setPlayRate(rate)
        clip.play()
