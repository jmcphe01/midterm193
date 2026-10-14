"""Synthesise the game's sound effects into assets/sounds/ (no downloads, no licences).

    python make_sounds.py

Want different sounds? Drop your own WAV files into assets/sounds/ using the same names; the game plays whatever
is there. (Run this script again only if you want to regenerate the synthesised ones: it overwrites them.)

  start.wav     the whistle-up jingle at GO!
  tick.wav      countdown beep for 3, 2, 1
  hit.wav       the player's racket hitting the ball
  opp.wav       the opponent's racket (a little lower and softer)
  bounce.wav    the ball on the table
  net.wav       the ball into the net
  gameover.wav  descending tones when the game ends
  record.wav    a bright flourish for a new record
"""
import wave
from pathlib import Path

import numpy as np

SR = 44100
OUT = Path(__file__).parent / "assets" / "sounds"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def t_axis(dur):
    return np.arange(int(SR * dur)) / SR


def damped(freq, dur, tau, amp=1.0, phase=0.0):
    """A sine that dies away exponentially (time constant tau seconds): the basic 'struck object' sound."""
    t = t_axis(dur)
    return amp * np.sin(2 * np.pi * freq * t + phase) * np.exp(-t / tau)


def noise_burst(dur, tau, amp=1.0):
    t = t_axis(dur)
    return amp * rng.uniform(-1, 1, t.size) * np.exp(-t / tau)


def note(freq, dur, amp=1.0, attack=0.006, release=0.05, harmonics=((1, 1.0), (2, 0.35), (3, 0.15)), vibrato=0.0):
    t = t_axis(dur)
    phase = 2 * np.pi * freq * t + (vibrato * np.sin(2 * np.pi * 5.5 * t) if vibrato else 0)
    y = sum(a * np.sin(h * phase) for h, a in harmonics)
    env = np.minimum(1.0, t / attack) * np.minimum(1.0, (dur - t) / release) * np.exp(-t / (dur * 1.6))
    return amp * y * env / sum(a for _, a in harmonics)


def mix(*parts, length=None):
    n = length or max(p.size for p in parts)
    out = np.zeros(n)
    for p in parts:
        out[:p.size] += p[:n]
    return out


def place(events, total):
    """events: [(start_seconds, signal)] -> one signal of `total` seconds."""
    out = np.zeros(int(SR * total))
    for start, sig in events:
        i = int(SR * start)
        out[i:i + sig.size] += sig[:max(0, out.size - i)]
    return out


def save(name, signal, peak=0.85):
    signal = np.nan_to_num(signal)
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * peak
    fade = int(SR * 0.004)                                   # tiny fade-out so no sound ends with a click
    signal[-fade:] *= np.linspace(1, 0, fade)
    with wave.open(str(OUT / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((signal * 32767).astype("<i2").tobytes())


# --- ball sounds: a sharp click plus a short ringing 'pok' ---
def ball_hit(pitch, weight):
    return mix(noise_burst(0.12, 0.0025, 0.9 * weight), damped(1250 * pitch, 0.12, 0.022, 1.0),
               damped(2480 * pitch, 0.12, 0.011, 0.55), damped(620 * pitch, 0.12, 0.03, 0.4))


save("hit", ball_hit(1.0, 1.0))
save("opp", ball_hit(0.78, 0.7), peak=0.55)
save("bounce", mix(noise_burst(0.09, 0.002, 0.5), damped(760, 0.09, 0.012, 1.0), damped(1500, 0.09, 0.006, 0.4)), peak=0.5)
save("net", mix(damped(150, 0.3, 0.05, 1.0), damped(310, 0.3, 0.03, 0.5), noise_burst(0.3, 0.02, 0.35)), peak=0.8)
save("tick", note(660, 0.16, attack=0.004, release=0.06, harmonics=((1, 1.0), (2, 0.2))), peak=0.55)

# --- jingles ---
C5, E5, G5, C6, E6 = 523.25, 659.25, 783.99, 1046.5, 1318.5
save("start", place([(0.0, note(C5, 0.2)), (0.09, note(E5, 0.2)), (0.18, note(G5, 0.2)), (0.27, note(C6, 0.55, release=0.2))], 0.9), peak=0.7)
G4, F4, Eb4, C4 = 392.0, 349.23, 311.13, 261.63
save("gameover", place([(0.0, note(G4, 0.3, vibrato=0.4)), (0.28, note(F4, 0.3, vibrato=0.4)),
                        (0.56, note(Eb4, 0.3, vibrato=0.4)), (0.84, note(C4, 0.9, release=0.35, vibrato=0.6))], 1.9), peak=0.7)
save("record", place([(0.0, note(C5, 0.16)), (0.07, note(E5, 0.16)), (0.14, note(G5, 0.16)), (0.21, note(C6, 0.16)),
                      (0.28, note(E6, 0.7, release=0.3))] + [(0.28 + 0.05 * k, note(E6 * 2, 0.12, amp=0.25)) for k in range(4)], 1.1), peak=0.7)

for f in sorted(OUT.glob("*.wav")):
    with wave.open(str(f)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32768
        print(f"  {f.name:13s} {len(x) / SR:4.2f} s   peak {20 * np.log10(np.abs(x).max()):5.1f} dBFS   rms {20 * np.log10(np.sqrt((x ** 2).mean())):6.1f} dBFS")
