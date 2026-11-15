"""Render the Pocket Universe trailer frame by frame (deterministic camera, no real-time capture).

Requires: Python 3, Playwright (pip install playwright), Google Chrome, ffmpeg.
Set CHROME below if Chrome lives somewhere else on your machine. On a machine with a
graphics card you can remove the swiftshader arguments for a much faster render.

  python tools/record.py /absolute/path/to/index.html out 1280 720 24 --audio
  ffmpeg -framerate 24 -i out/frames/%05d.jpg -f s16le -ar 44100 -ac 1 -i out/audio.pcm \
         -c:v libx264 -crf 17 -pix_fmt yuv420p -c:a aac -ac 2 -shortest trailer.mp4

Optional: add "I N" after the frame rate to render only every N-th frame starting at I
(run several copies in parallel), or "--bench N" to render N sample frames.
"""
import sys, math, base64, time, os
from playwright.sync_api import sync_playwright

CHROME = '/usr/bin/google-chrome'
page_path, out, Wd, Hd, FPS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rest = sys.argv[6:]
bench = int(rest[rest.index('--bench') + 1]) if '--bench' in rest else 0
want_audio = '--audio' in rest
wi, wn = (int(rest[0]), int(rest[1])) if rest and rest[0].isdigit() else (0, 1)

def norm(v):
    l = math.sqrt(sum(x * x for x in v)) or 1
    return [x / l for x in v]
def add(a, b): return [a[i] + b[i] for i in range(3)]
def sub(a, b): return [a[i] - b[i] for i in range(3)]
def mul(a, k): return [x * k for x in a]
def lerp(a, b, u): return [a[i] + (b[i] - a[i]) * u for i in range(3)]
def cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
def rot(v, ax, th):  # Rodrigues
    c, s = math.cos(th), math.sin(th); d = sum(v[i] * ax[i] for i in range(3)); cr = cross(ax, v)
    return [v[i] * c + cr[i] * s + ax[i] * d * (1 - c) for i in range(3)]
ease = lambda u: 1 - (1 - u) ** 2

TAN_V, TAN_H = math.tan(math.radians(31)), math.tan(math.radians(31)) * 16 / 9
def dot(a, b): return sum(a[i] * b[i] for i in range(3))
def in_frame(pos, look, op, orad, up=None):
    """Is an object (centre op, radius orad) inside the camera frame, with some margin?"""
    F = norm(sub(look, pos)); R = norm(cross(F, up or [0, 1, 0])); U = cross(R, F)
    v = sub(op, pos); z = dot(v, F)
    if z <= 0: return False
    pad = orad / math.sqrt(dot(v, v)) * 1.5 + .06
    return abs(dot(v, R) / z) < TAN_H + pad and abs(dot(v, U) / z) < TAN_V + pad

def build(B, GAL, S):
    comet = B['comet']['p']; away = norm(sub(comet, S))
    extras = [('nebula', [0, 0, 0], 72), ('galaxy', GAL, 115), ('tail', add(comet, mul(away, 40)), 45)]
    def clutter(fn, allowed):
        objs = [(k, v['p'], v['r']) for k, v in B.items() if k not in allowed] + [e for e in extras if e[0] not in allowed]
        n = 0
        for u in (0, .5, 1):
            c = fn(u); pos, look, up = c[0], c[1], (c[2] if len(c) > 2 else None)
            n += sum((3 if k in ('nebula', 'galaxy', 'sun') else 1) for k, p, r in objs if in_frame(pos, look, p, r, up))
        return n
    def best(id, make, allowed=(), lit=(.15, .85), face=None):
        """Pick the camera direction around a body whose background is clear of every other object."""
        allowed = set(allowed) | {id}
        ts = norm(sub(S, B[id]['p'])) if id != 'sun' else None
        cands = []
        for el in range(-65, 66, 10):
            for az in range(0, 360, 15):
                e, a = math.radians(el), math.radians(az)
                o = [math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)]
                l = dot(o, ts) if ts else .5
                if not (lit[0] <= l <= lit[1]): continue
                fn = make(o)
                sc = clutter(fn, allowed) * 10 + abs(l - .5) + abs(el) / 400
                if face:  # prefer a camera that sits over a given latitude of the body
                    le = abs(math.degrees(math.asin(max(-1, min(1, dot(o, B[id]['ax']))))) - face[1])
                    if le > (face[2] if len(face) > 2 else 30): continue
                    sc += le / 25
                cands.append((sc, o))
        cands.sort(key=lambda c: c[0])
        o = cands[0][1]
        print(f'{id:8s} clutter score {cands[0][0]:.2f}')
        ph = None
        if face:  # turn the body so that longitude face[0] is under the camera
            ax = B[id]['ax']; e1 = norm(cross(ax, [.01, 0, 1])); e2 = cross(ax, e1)
            ph = (id, math.atan2(dot(o, e2), dot(o, e1)) - math.radians(face[0]))
        return make(o), ph
    def approach(id, k0, k1, side=0.0, allowed=(), face=None, lit=(.15, .85)):
        def make(o):
            def f(u):
                k = k0 + (k1 - k0) * ease(u)
                d = rot(o, [0, 1, 0], side * (u - .5)) if side else o
                return add(B[id]['p'], mul(d, B[id]['r'] * k)), B[id]['p']
            return f
        return best(id, make, allowed, lit, face)
    def skim(id, h, th0, th1, ahead=.55, allowed=(), face=None):
        def make(o):
            ax = norm(cross(o, [0, 1, 0]))
            def f(u):
                th = th0 + (th1 - th0) * u
                up = rot(o, ax, th)
                return (add(B[id]['p'], mul(up, B[id]['r'] * h)), add(B[id]['p'], mul(rot(o, ax, th + ahead), B[id]['r'])), up)
            return f
        return best(id, make, allowed, (.3, .9), face)
    def move(p0, p1, l0, l1):
        return (lambda u: (lerp(p0, p1, ease(u)), lerp(l0, l1, ease(u)))), None
    shots = [
        (8.0, move([-18, 150, 85], [-6, 14, 26], [0, 0, 0], [8, -2, 5])),          # one dive into the nebula
        (7.0, move([-140, 55, -120], [-215, 88, -190], GAL, GAL)),                 # spiral galaxy
        (6.0, move([300, 170, 300], [560, 120, 430], [700, 60, 520], S)),          # Milky Way, Solar System ahead
        (6.5, approach('neptune', 4.4, 2.2, .4)),
        (7.0, approach('saturn', 4.6, 2.7, .5)),
        (6.5, approach('jupiter', 3.3, 1.9, .4, ('io', 'europa'), face=(49, -18), lit=(.4, .85))),
        (5.0, approach('europa', 3.2, 1.35, .5, ('jupiter',))),
        (7.0, approach('mars', 3.6, 1.25, .3)),
        (7.0, skim('mars', 1.07, -.30, .22)),
        (8.0, approach('earth', 3.6, 1.8, .5, ('moon',), face=(28, 18, 48), lit=(.35, .9))),
        (7.0, skim('earth', 1.13, -.35, .3, .7, ('moon',), face=(22, 30))),
        (6.0, approach('moon', 3.4, 1.3, .4, ('earth',))),
        (6.5, skim('moon', 1.06, -.3, .25, allowed=('earth',))),
        (6.0, approach('sun', 3.4, 1.9, .3)),
        (5.5, approach('comet', 20, 8, .6, ('tail',))),
        (7.0, move([640, 230, 820], [560, 300, 980], S, S)),
    ]
    shots = [(d, f[0], f[1]) for d, f in shots]
    pos, look = shots[0][1](0); pos2, look2 = shots[0][1](1)
    print('nebula shot sees galaxy/sun:', [in_frame(p, l, o, r) for p, l in ((pos, look), (pos2, look2)) for o, r in ((GAL, 115), (S, 60))])
    return shots

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=[
        '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--use-gl=angle', '--no-sandbox'])
    pg = b.new_page(viewport={'width': Wd, 'height': Hd})
    pg.on('pageerror', lambda e: print('PAGEERROR', e))
    pg.goto('file://' + page_path); pg.wait_for_timeout(1500)
    pg.evaluate("""()=>{const d=window.__dbg;d.manual();
      const st=document.createElement('style');st.textContent='#hud,#place,#intro,#dest{display:none!important}'+
        '#fade{position:fixed;inset:0;background:#000;opacity:0;z-index:20;pointer-events:none}'+
        '#card{position:fixed;inset:0;z-index:21;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1.2vw;opacity:0;color:#f4ead9;text-align:center;'+
        'background:radial-gradient(ellipse at center,rgba(2,3,8,.55),rgba(2,3,8,.15) 70%);font-family:system-ui,Helvetica,Arial,sans-serif}'+
        '#card b{font:400 4.6vw Georgia,serif;text-shadow:0 0 2vw rgba(241,184,102,.5)}#card span{font-size:1.7vw;opacity:.85}';
      document.head.appendChild(st);
      const f=document.createElement('div');f.id='fade';document.body.appendChild(f);
      const c=document.createElement('div');c.id='card';c.innerHTML='<b>Fly it yourself, in your browser</b><span>One small HTML file · no images · no libraries · link in the description</span>';document.body.appendChild(c);}""")
    info = pg.evaluate("()=>{const d=window.__dbg;return {B:d.BODIES.map(b=>({id:b.id,p:b.p,r:b.r,ax:b.ax,spin:b.spin})),GAL:d.GAL,S:d.S}}")
    B = {x['id']: {'p': x['p'], 'r': x['r'], 'ax': x['ax'], 'spin': x['spin']} for x in info['B']}
    shots = build(B, info['GAL'], info['S'])
    frames = []
    for si, (dur, fn, ph) in enumerate(shots):
        n = round(dur * .8 * FPS)
        for i in range(n):
            frames.append((si, i, n, fn, ph))
    total = len(frames)
    if want_audio:
        data = pg.evaluate("(d)=>window.__dbg.audio(d)", total / FPS + .5)
        open(os.path.join(out, 'audio.pcm'), 'wb').write(base64.b64decode(data))
        print('audio ok', total / FPS)
    print('total frames', total, 'seconds', total / FPS)
    os.makedirs(os.path.join(out, 'frames'), exist_ok=True)
    idx = list(range(total))
    if bench:
        idx = []; acc = 0
        for dur, fn, ph in shots:
            n = round(dur * .8 * FPS); idx += [acc + int(n * .12), acc + n // 2, acc + int(n * .9)]; acc += n
    else:
        idx = [g for g in idx if g % wn == wi]
    t0 = time.time()
    for g in idx:
        path = os.path.join(out, 'frames', f'{g:05d}.jpg')
        if os.path.exists(path) and not bench:
            continue
        si, i, n, fn, ph = frames[g]
        c = fn(i / max(n - 1, 1)); pos, look, up = c[0], c[1], (c[2] if len(c) > 2 else None)
        tsc = 40 + g / FPS
        phase = [ph[0], ph[1] - tsc * B[ph[0]]['spin']] if ph else None
        edge = min(i, n - 1 - i)
        fade = max(0.0, 1 - edge / (0.3 * FPS))
        if g < FPS * 1.5: fade = max(fade, 1 - g / (FPS * 1.5))
        if total - 1 - g < FPS * 1.2: fade = max(fade, 1 - (total - 1 - g) / (FPS * 1.2))
        card = 0.0
        if si == len(shots) - 1:
            card = max(0.0, min(1.0, (i / FPS - 1.5) / 1.2))
        pg.evaluate("([t,p,l,f,c,dt,up,ph])=>{const d=window.__dbg;if(ph)d.BODIES.find(b=>b.id===ph[0]).phase=ph[1];d.setTime(t);d.set(p,l,up);d.step(dt);document.getElementById('fade').style.opacity=f;document.getElementById('card').style.opacity=c}",
                    [tsc, pos, look, fade, card, 1 / FPS, up, phase])
        pg.screenshot(path=path, type='jpeg', quality=93)
    print('done', len(idx), 'frames in', round(time.time() - t0, 1), 's')
    b.close()
