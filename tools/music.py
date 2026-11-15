"""Original score for the Pocket Universe trailer, synthesised from scratch and locked to the cuts."""
import numpy as np, sys
from scipy.signal import fftconvolve, butter, lfilter

SR = 44100
FPS = 24
starts_f = [0, 154, 288, 403, 528, 662, 787, 883, 1017, 1151, 1305, 1439, 1554, 1679, 1794, 1900, 2034]
T = [f / FPS for f in starts_f]
DUR = T[-1]
N = int((DUR + 4) * SR)
dry = np.zeros((N, 2)); wet = np.zeros((N, 2))
rng = np.random.default_rng(7)
hz = lambda m: 440 * 2 ** ((m - 69) / 12)

def put(buf, t, sig, pan=0.0, gain=1.0):
    i = int(t * SR)
    if i < 0 or i >= N: return
    sig = sig[: N - i]
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(sig), 0] += sig * l * gain; buf[i:i + len(sig), 1] += sig * r * gain

def piano(m, vel=.6, length=4.5):
    """Soft felt-piano tone: decaying harmonics, slight inharmonicity, gentle hammer noise."""
    t = np.arange(int(length * SR)) / SR; f = hz(m); out = np.zeros_like(t)
    bright = .55 + .5 * vel
    for k in range(1, 9):
        fk = f * k * (1 + .0004 * k * k)
        if fk > 9000: break
        amp = (1 / k ** (2.1 - bright)) * (1.0 if k > 1 else 1.15)
        out += amp * np.exp(-t * (.9 + .55 * k) * (1 + (m - 60) / 60)) * np.sin(2 * np.pi * fk * t + rng.uniform(0, .5))
    att = np.minimum(1, t / .006)
    thump = np.exp(-t * 90) * rng.normal(0, 1, len(t)) * .05
    rel = np.minimum(1, (length - t) / .3)
    return (out * att + thump) * rel * vel * .5

def pad(notes, length, level=1.0, bright=.3):
    t = np.arange(int(length * SR)) / SR; out = np.zeros((len(t), 2))
    a, r = min(1.6, length * .35), min(2.4, length * .45)
    env = np.minimum(1, t / a) * np.minimum(1, (length - t) / r)
    env = np.sin(env * np.pi / 2) ** 2
    for j, m in enumerate(notes):
        f = hz(m)
        for det, pan in ((-4, -.6), (3, .6), (0, 0)):
            ff = f * 2 ** (det / 1200); ph = rng.uniform(0, 6.28)
            vib = 1 + .0016 * np.sin(2 * np.pi * (.19 + .05 * j) * t + ph)
            s = np.sin(2 * np.pi * ff * t * vib + ph) + bright * np.sin(4 * np.pi * ff * t + ph) + bright * .45 * np.sin(6 * np.pi * ff * t)
            l, rr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            out[:, 0] += s * l; out[:, 1] += s * rr
    return out * env[:, None] * level * .028

def bass(m, length, level=1.0):
    t = np.arange(int(length * SR)) / SR; f = hz(m)
    env = np.minimum(1, t / .5) * np.minimum(1, (length - t) / 1.2)
    return (np.sin(2 * np.pi * f * t) + .25 * np.sin(4 * np.pi * f * t)) * env * level * .16

def whoosh(length=1.3):
    """Breath of filtered noise that rises into a cut."""
    n = int(length * SR); t = np.arange(n) / SR
    x = rng.normal(0, 1, (n, 2))
    b, a = butter(2, [500 / (SR / 2), 3500 / (SR / 2)], 'band'); x = lfilter(b, a, x, axis=0)
    env = (t / length) ** 2.6
    return x * env[:, None] * .16

def hit(level=1.0):
    """Soft low impact on a cut."""
    t = np.arange(int(2.2 * SR)) / SR
    f = 44 + 30 * np.exp(-t * 14)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.6) * np.minimum(1, t / .004) * level * .42

# (bass root, pitch classes), one chord per shot — D major, warm and open
D9, Bm7, G7, A9, DF, Em7, Dm = (38, {2, 6, 9, 4, 1}), (35, {11, 2, 6, 9}), (31, {7, 11, 2, 6}), (33, {9, 1, 4, 11}), (30, {2, 6, 9}), (40, {4, 7, 11, 2}), (38, {2, 6, 9, 4})
chords = [D9, Bm7, G7, A9, DF, G7, Em7, A9, Bm7, G7, DF, A9, Bm7, G7, A9, Dm]
#          neb  neb  gal  mw   sat  jup  eur  mars mars earth earth moon moon sun  comet end
energy = [.0, .25, .4, .5, .62, .68, .5, .72, .78, .92, 1.0, .6, .7, 1.0, .7, .35]
melody = [78, 74, 71, 73, 74, 78, 79, 76, 78, 83, 81, 76, 74, 78, 76, 74]
PAT = [0, 2, 1, 3, 2, 4, 3, 5]

for s, ((root, pcs), e) in enumerate(zip(chords, energy)):
    t0, t1 = T[s], T[s + 1]; d = t1 - t0
    last = s == len(chords) - 1
    hold = d + (3.5 if last else 1.6)
    # pad and bass overlap into the next shot so chords melt into each other
    pn = [m for m in range(50, 70) if m % 12 in pcs]
    if e > .85: pn += [m + 12 for m in pn[-3:]]
    padsig = pad(pn, hold + 1.2, .75 + .5 * e, .25 + .25 * e)
    i = int(max(0, t0 - .5) * SR); seg = padsig[: N - i]
    dry[i:i + len(seg)] += seg * .75; wet[i:i + len(seg)] += seg
    if s >= 2:
        put(dry, t0, bass(root + (12 if root < 33 else 0), hold, .6 + .5 * e))
    # the cut itself: a rising breath, a soft impact and a rolled chord
    if s > 0:
        put(wet, t0 - 1.3, whoosh()[:, 0], -.3, .5 + .6 * e); put(wet, t0 - 1.3, whoosh()[:, 1], .3, .5 + .6 * e)
        put(dry, t0, hit(.5 + .6 * e)); put(wet, t0, hit(.3))
    roll = [m for m in range(52, 76) if m % 12 in pcs][::2][:5]
    for j, m in enumerate(roll):
        sig = piano(m, .32 + .3 * e, 5.0)
        put(dry, t0 + j * .035, sig, (m - 64) / 22, .8); put(wet, t0 + j * .035, sig, (m - 64) / 22, .55)
    # top line: one singing note per shot, answered halfway through
    for (tt, m, v) in ((t0 + .02, melody[s], .62), (t0 + d * .52, max(x for x in range(60, melody[s]) if x % 12 in pcs), .42)):
        sig = piano(m, v + .15 * e, 6.0)
        put(dry, tt, sig, .15, .9); put(wet, tt, sig, .15, .9)
    # flowing arpeggio whose density follows the energy of the shot
    L = [m for m in range(57, 84) if m % 12 in pcs]
    step = d / max(1, round(d / (.84 if e < .3 else .42)))
    n = round(d / step)
    for k in range(n):
        if e < .05 and k % 2: continue
        if last and k > n * .6: break
        m = L[(PAT[k % 8] + k // 8 * 2 + s) % len(L)]
        v = (.2 + .34 * e) * (1.25 if k % 4 == 0 else 1.0) * rng.uniform(.82, 1.08)
        if last: v *= 1 - k / n
        tt = t0 + k * step + rng.uniform(-.008, .008)
        sig = piano(m, v, 3.2)
        pan = (-1) ** k * .35
        put(dry, tt, sig, pan, .75); put(wet, tt, sig, pan, .6)
        if e > .6 and k % 2 == 1:  # light off-beat echo an octave up for sparkle
            sig = piano(m + 12, v * .45, 2.2)
            put(dry, tt + step / 2, sig, -pan, .6); put(wet, tt + step / 2, sig, -pan, .7)
    # heartbeat under the fuller shots
    if e > .7:
        for k in range(0, n, 4):
            put(dry, t0 + k * step, hit(.22))

# reverb: decaying stereo noise as the room
irn = int(3.2 * SR); ti = np.arange(irn) / SR
ir = rng.normal(0, 1, (irn, 2)) * np.exp(-ti * 2.1)[:, None] * np.minimum(1, ti / .012)[:, None]
b, a = butter(1, 5200 / (SR / 2)); ir = lfilter(b, a, ir, axis=0); ir /= np.sqrt((ir ** 2).sum(0)).mean()
rev = np.stack([fftconvolve(wet[:, c], ir[:, c])[:N] for c in range(2)], 1)
mix = dry + rev * .34
b, a = butter(2, 28 / (SR / 2), 'high'); mix = lfilter(b, a, mix, axis=0)
# fades, gentle glue, normalise
t = np.arange(N) / SR
mix *= (np.minimum(1, t / 1.5) * np.clip((DUR + .3 - t) / 3.5, 0, 1))[:, None]
mix = mix[: int((DUR + .3) * SR)]
mix = np.tanh(mix / np.abs(mix).max() * 1.6) / np.tanh(1.6)
mix *= 10 ** (-1.0 / 20)
import wave
w = wave.open(sys.argv[1], 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((mix * 32767).astype('<i2').tobytes()); w.close()
# per-shot loudness report
for s in range(len(chords)):
    seg = mix[int(T[s] * SR): int(T[s + 1] * SR)]
    print(f'shot {s + 1:2d} {T[s]:6.2f}s rms {20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9):6.1f} dB peak {20 * np.log10(np.abs(seg).max() + 1e-9):6.1f}')
