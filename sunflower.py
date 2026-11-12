"""Pointillism sunflower: bud -> bloom -> wilt, with the sun dissolving as the flower dies.

Window keys : Esc quit | Space pause | R restart | S save current frame
Command line: python flower.py          (live window)
              python flower.py --png    (stage frames + strip)
              python flower.py --gif    (animated gif)
"""
import math
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

SEED = 88
TIME_SCALE = 1.0
TICK_MS = 10
WIN_FRAC = 0.92
LOOP = False
SOFTEN = 0.5          # gentle blur that melts the dots into a painted surface (0 = off)

# ---------------------------------------------------------------------------
# Timeline.  Columns: time, stem, size, bloom, wilt, leaf, glow
#   0-11s   stem rises, closed green bud
#   11-24s  bud opens into the flower
#   24-30s  full bloom
#   30-42s  flower dies (wilt) - the sun dissolves in sync with it
# ---------------------------------------------------------------------------
KEYS = np.array([
    [0.0,  0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [5.0,  0.40, 0.25, 0.00, 0.00, 0.30, 0.00],
    [11.0, 0.78, 0.45, 0.00, 0.00, 0.65, 0.00],
    [17.0, 1.00, 0.70, 0.22, 0.00, 0.90, 0.00],
    [24.0, 1.00, 1.00, 1.00, 0.00, 1.00, 0.80],
    [30.0, 1.00, 1.00, 1.00, 0.00, 1.00, 1.00],
    [36.0, 1.00, 1.00, 1.00, 0.45, 0.98, 0.15],
    [42.0, 1.00, 0.97, 1.00, 1.00, 0.90, 0.00],
    [48.0, 1.00, 0.97, 1.00, 1.00, 0.90, 0.00],
])
T_END = float(KEYS[-1, 0])

# ---------------------------------------------------------------------------
# Palettes (continuous gradients, not random picks -> smooth painted look)
# ---------------------------------------------------------------------------
SOIL_PAL = [(40, 30, 20), (55, 45, 30), (75, 60, 45), (30, 20, 15)]
STEM_PAL = [(38, 100, 28), (62, 142, 40), (100, 185, 58), (150, 218, 104)]
STEM_WILT = [(66, 58, 36), (104, 92, 58), (136, 120, 82)]
LEAF_PAL = [(24, 78, 26), (42, 116, 38), (70, 158, 50), (112, 198, 74)]
LEAF_WILT = [(66, 52, 28), (104, 82, 42), (132, 106, 56)]
RIB_COL = np.array([150, 225, 115], np.float32)
PETAL_PAL = [(246, 140, 0), (255, 180, 4), (255, 210, 18), (255, 230, 72)]
PETAL_PAL_IN = [(236, 126, 0), (255, 168, 0), (255, 198, 10), (255, 222, 56)]
PETAL_WILT = [(150, 102, 30), (122, 82, 26), (96, 63, 20), (70, 46, 16)]
BRACT_PAL = [(28, 82, 28), (52, 122, 36), (88, 166, 52), (132, 202, 82)]
BRACT_WILT = [(70, 66, 34), (98, 84, 44), (122, 102, 60)]
DISC_PAL = [(30, 16, 6), (52, 28, 10), (92, 54, 16), (150, 96, 26), (214, 158, 40)]
DISC_WILT = [(28, 20, 14), (46, 34, 22), (66, 50, 34)]
BACK_PAL = [(54, 90, 32), (84, 118, 46), (110, 130, 60)]
BACK_WILT = [(60, 52, 30), (84, 70, 40), (104, 88, 52)]
SUN_PAL = [(255, 252, 226), (255, 238, 166), (255, 206, 72), (255, 162, 32)]

# kind, count, length, closed/open petal angle, closed/open curl, closed/open half-width ...
LAYERS = [
    dict(kind="bract", n=14, L=0.84, ac=0.80, ao=2.35, kc=-1.20, ko=0.50, wc=0.20, wo=0.15,
         d=520, off=0.00, pal=BRACT_PAL, wpal=BRACT_WILT, fall=0.0, wk=0.40, wm=0.80),
    dict(kind="bract", n=12, L=0.66, ac=0.70, ao=2.15, kc=-1.00, ko=0.40, wc=0.20, wo=0.14,
         d=420, off=0.22, pal=BRACT_PAL, wpal=BRACT_WILT, fall=0.0, wk=0.40, wm=0.80),
    dict(kind="petal", n=34, L=1.15, ac=0.85, ao=1.40, kc=-2.00, ko=0.30, wc=0.11, wo=0.15,
         d=1400, off=0.00, pal=PETAL_PAL, wpal=PETAL_WILT, fall=0.62, wk=1.0, wm=1.0),
    dict(kind="petal", n=30, L=0.98, ac=0.95, ao=1.50, kc=-1.80, ko=0.22, wc=0.11, wo=0.14,
         d=1100, off=0.10, pal=PETAL_PAL_IN, wpal=PETAL_WILT, fall=0.62, wk=1.0, wm=1.0),
]

# stem position, side, size, start angle, curvature
LEAF_DEFS = [
    (0.14, +1, 1.10, 0.40, -0.70),
    (0.19, -1, 1.00, 0.35, -0.75),
    (0.42, +1, 0.82, 0.85, -0.60),
    (0.36, -1, 0.72, 0.80, -0.55),
    (0.60, -1, 0.55, 1.00, -0.50),
]
LS = np.array([d[0] for d in LEAF_DEFS], np.float32)
LM = np.array([d[1] for d in LEAF_DEFS], np.float32)
LF = np.array([d[2] for d in LEAF_DEFS], np.float32)
LA = np.array([d[3] for d in LEAF_DEFS], np.float32)
LC = np.array([d[4] for d in LEAF_DEFS], np.float32)

_OFF = sorted(((dx, dy) for dy in range(-3, 4) for dx in range(-3, 4) if dx * dx + dy * dy <= 9),
              key=lambda p: (p[0] ** 2 + p[1] ** 2, p))
OFF = np.array(_OFF, np.int32)
D2 = (OFF ** 2).sum(1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def sinc(x):
    return np.sinc(x / np.pi)


def profile(u):
    return np.sin(np.pi * np.clip(u, 0, 1) ** 0.8) ** 0.9


def grad(pal, g):
    """Continuous colour gradient through a palette (g in 0..1)."""
    pal = np.asarray(pal, np.float32)
    g = np.clip(np.asarray(g, np.float32), 0, 1) * (len(pal) - 1)
    i = np.clip(np.floor(g).astype(int), 0, len(pal) - 2)
    f = (g - i)[:, None]
    return pal[i] * (1 - f) + pal[i + 1] * f


def palette_pick(rng, pal, g, spread=0.55, jitter=0.07):
    pal = np.asarray(pal, np.float32)
    g = np.asarray(g, np.float32)
    x = np.clip(g, 0, 1) * (len(pal) - 1) + rng.normal(0, spread, g.shape)
    idx = np.clip(np.rint(x), 0, len(pal) - 1).astype(int)
    return (pal[idx] * (1 + rng.normal(0, jitter, (len(idx), 1)))).astype(np.float32)


def area_u(rng, n):
    out = np.empty(0)
    while out.size < n:
        u = rng.random(n * 3)
        out = np.concatenate([out, u[rng.random(n * 3) < profile(u)]])
    return out[:n]


def permute(d, rng):
    n = len(next(iter(d.values())))
    p = rng.permutation(n)
    return {k: v[p] for k, v in d.items()}


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def splat(canvas, x, y, col, rad, nmax=13):
    H, W = canvas.shape[:2]
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    off = OFF[:nmax]
    xs = xi[:, None] + off[None, :, 0]
    ys = yi[:, None] + off[None, :, 1]
    m = (D2[None, :nmax] <= (rad * rad)[:, None] + 0.01) & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    rows = np.nonzero(m)[0]
    canvas[ys[m], xs[m]] = col[rows]


def splat_add(canvas, x, y, col, rad, nmax=9):
    """Additive splat (soft light) - never leaves dark specks on top of the art."""
    H, W = canvas.shape[:2]
    xi = np.rint(x).astype(np.int32)
    yi = np.rint(y).astype(np.int32)
    off = OFF[:nmax]
    xs = xi[:, None] + off[None, :, 0]
    ys = yi[:, None] + off[None, :, 1]
    m = (D2[None, :nmax] <= (rad * rad)[:, None] + 0.01) & (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    rows = np.nonzero(m)[0]
    yy, xx = ys[m], xs[m]
    canvas[yy, xx] = np.clip(canvas[yy, xx].astype(np.float32) + col[rows], 0, 255).astype(np.uint8)


class Batch:
    """Collects points, depth-sorts them (painter's algorithm) and splats them."""

    def __init__(self):
        self.p = []

    def add(self, x, y, col, rad, z):
        if len(x):
            self.p.append((np.asarray(x), np.asarray(y), np.asarray(col),
                           np.asarray(rad), np.asarray(z)))

    def draw(self, canvas):
        if not self.p:
            return
        x, y, col, rad, z = [np.concatenate([q[i] for q in self.p]) for i in range(5)]
        o = np.argsort(z, kind="stable")
        c = np.clip(col[o], 0, 255).astype(np.uint8)
        mr = float(rad.max())
        nmax = int((D2 <= mr * mr + 0.01).sum())
        splat(canvas, x[o], y[o], c, rad[o], nmax)


def _slopes(x, y):
    """Monotone (PCHIP) tangents - smooth motion with no overshoot between keyframes."""
    h = np.diff(x)
    d = np.diff(y, axis=0) / h[:, None]
    m = np.zeros_like(y)
    for k in range(1, len(x) - 1):
        a, b = d[k - 1], d[k]
        same = a * b > 0
        w1 = 2 * h[k] + h[k - 1]
        w2 = h[k] + 2 * h[k - 1]
        den = w1 / np.where(same, a, 1.0) + w2 / np.where(same, b, 1.0)
        m[k] = np.where(same, (w1 + w2) / den, 0.0)
    return m


_TS = KEYS[:, 0]
_M = _slopes(_TS, KEYS[:, 1:])


def state(t):
    t = float(min(max(t, 0.0), T_END))
    i = min(max(int(np.searchsorted(_TS, t, side="right")) - 1, 0), len(KEYS) - 2)
    h = _TS[i + 1] - _TS[i]
    u = (t - _TS[i]) / h
    p1, p2 = KEYS[i, 1:], KEYS[i + 1, 1:]
    m1, m2 = _M[i], _M[i + 1]
    u2, u3 = u * u, u * u * u
    v = ((2 * u3 - 3 * u2 + 1) * p1 + (u3 - 2 * u2 + u) * h * m1
         + (-2 * u3 + 3 * u2) * p2 + (u3 - u2) * h * m2)
    return np.clip(v, 0, 1)


def bez(P, s):
    P0, P1, P2, P3 = P
    q = 1 - s
    x = q ** 3 * P0[0] + 3 * q * q * s * P1[0] + 3 * q * s * s * P2[0] + s ** 3 * P3[0]
    y = q ** 3 * P0[1] + 3 * q * q * s * P1[1] + 3 * q * s * s * P2[1] + s ** 3 * P3[1]
    dx = 3 * q * q * (P1[0] - P0[0]) + 6 * q * s * (P2[0] - P1[0]) + 3 * s * s * (P3[0] - P2[0])
    dy = 3 * q * q * (P1[1] - P0[1]) + 6 * q * s * (P2[1] - P1[1]) + 3 * s * s * (P3[1] - P2[1])
    return x, y, dx, dy


def mixcol(live, wilt, w):
    return live + (wilt - live) * w[:, None]


def head_basis(A):
    """Orthonormal frame for the flower head: A = face direction, U = right-ish, V = up-ish."""
    A = A / np.linalg.norm(A)
    U = np.array([A[2], 0.0, -A[0]])
    n = np.linalg.norm(U)
    U = U / n if n > 1e-3 else np.array([1.0, 0.0, 0.0])
    V = np.cross(A, U)
    return A, U, V


# ---------------------------------------------------------------------------
# the plant
# ---------------------------------------------------------------------------
class Plant:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.rng = np.random.default_rng(SEED)
        self.cx = W / 2
        self.y0 = H * 0.86
        self.Ls = H * 0.50      # full stem length (tall)
        self.Lf = H * 0.125     # flower size (kept small so the stem reads clearly)
        self.build_soil()
        self.build_mound()
        self.build_stem()
        self.build_head()
        self.build_disc()
        self.build_back()
        self.build_leaves()
        self.build_sun()
        self.build_fx()

    def surface(self, x):
        H = self.H
        return (self.y0 - 0.022 * H * np.exp(-((x - self.cx) / (0.2 * self.W)) ** 2)
                - 0.003 * H * np.sin(x * 0.045) - 0.002 * H * np.sin(x * 0.13 + 1.3))

    # -- ground -----------------------------------------------------------
    def build_soil(self):
        W, H, rng = self.W, self.H, self.rng
        n = 42000
        x = rng.uniform(0, W, n)
        d = (rng.random(n) ** 1.5) * 0.14 * H
        y = self.surface(x) + d
        depth = d / (0.14 * H)
        edge = np.clip(np.minimum(x, W - x) / (0.2 * W), 0, 1) ** 0.8
        clump = 0.5 + 0.5 * np.sin(x * 0.07 + y * 0.05) * np.sin(y * 0.11 - x * 0.03)
        bright = (1 - depth) ** 0.8 * edge * (0.55 + 0.35 * clump + 0.35 * rng.random(n))
        col = palette_pick(rng, SOIL_PAL, rng.random(n), spread=1.0) * bright[:, None]
        rad = rng.choice([1.0, 1.5, 2.0], n, p=[0.45, 0.4, 0.15])
        self.base = np.zeros((H, W, 3), np.uint8)
        splat(self.base, x, y, np.clip(col, 0, 255).astype(np.uint8), rad)

    def build_mound(self):
        """Loose soil heaped in front of the stem base so the plant looks rooted."""
        rng, n = self.rng, 3000
        x = self.cx + rng.normal(0, 0.03 * self.W, n)
        y = self.surface(x) + rng.uniform(-0.006, 0.022, n) * self.H
        col = palette_pick(rng, SOIL_PAL, rng.random(n), spread=1.0) * rng.uniform(0.85, 1.5, (n, 1))
        self.mound = dict(x=x, y=y, col=np.clip(col, 0, 255).astype(np.uint8),
                          rad=rng.choice([1.0, 1.5, 2.0], n, p=[0.4, 0.4, 0.2]))

    # -- stem -------------------------------------------------------------
    def build_stem(self):
        rng, n = self.rng, 6500
        s = rng.random(n)
        lat = rng.uniform(-1, 1, n)
        tone = np.clip(1.0 - np.abs(lat + 0.35) * 0.8 + 0.15 * (s - 0.5) + rng.normal(0, 0.06, n), 0, 1)
        live = grad(STEM_PAL, tone) * (1 + rng.normal(0, 0.03, (n, 1)))
        self.stem = permute(dict(
            s=s, lat=lat, live=live.astype(np.float32),
            wilt=grad(STEM_WILT, rng.random(n)),
            rn=rng.random(n), rad=rng.choice([1.0, 1.5], n, p=[0.6, 0.4])), rng)

    def ctrl(self, L, wilt, sway):
        cx, y0 = self.cx, self.y0
        P0 = (cx, y0 + 4.0)
        P1 = (cx - 0.02 * L * (1 - wilt) + 0.01 * L * wilt, y0 - 0.35 * L)
        P2 = (cx + 0.03 * L * (1 - wilt) + 0.12 * L * wilt + sway * 0.5, y0 - L * (0.80 + 0.25 * wilt))
        P3 = (cx + sway + 0.38 * L * wilt, y0 - L * (1 - 0.22 * wilt))
        return P0, P1, P2, P3

    # -- sun --------------------------------------------------------------
    def build_sun(self):
        rng, n = self.rng, 9000
        R = 0.06 * self.H
        r = np.sqrt(rng.random(n))
        corona = rng.random(n) < 0.18
        r[corona] = 1.0 + 0.5 * rng.random(int(corona.sum())) ** 1.5
        a = rng.uniform(0, 2 * math.pi, n)
        self.sun_dx = (r * R * np.cos(a)).astype(np.float32)
        self.sun_dy = (r * R * np.sin(a)).astype(np.float32)
        col = grad(SUN_PAL, np.minimum(r, 1.0) ** 1.6 + rng.normal(0, 0.03, n))
        col[corona] *= (np.clip(1.0 - (r[corona] - 1.0) / 0.5, 0, 1) ** 1.2 * 0.55)[:, None]
        self.sun_col = col.astype(np.float32)
        self.sun_rad = np.where(corona, 1.0, rng.choice([1.5, 2.0], n, p=[0.6, 0.4]))
        self.sun_ph = rng.uniform(0, 6.28, n).astype(np.float32)
        # dissolve: left edge erodes first, pixels blow away to the upper right
        sweep = (self.sun_dx / (1.5 * R) + 1) / 2
        self.sun_thr = (0.7 * (0.45 * rng.random(n) + 0.55 * sweep)).astype(np.float32)
        ang = rng.normal(-0.35, 0.55, n)
        self.sun_vx = np.cos(ang).astype(np.float32)
        self.sun_vy = np.sin(ang).astype(np.float32)
        self.sun_dist = (self.H * rng.uniform(0.10, 0.50, n)).astype(np.float32)

    # -- flower head ------------------------------------------------------
    def build_head(self):
        rng, H, W = self.rng, self.H, self.W
        parts, pid = [], 0
        for Ly in LAYERS:
            d, n = Ly["d"], Ly["n"]
            petal = Ly["kind"] == "petal"
            for i in range(n):
                th = 2 * math.pi * i / n + Ly["off"] + rng.normal(0, 0.04)
                aj, lj = rng.normal(0, 0.03), rng.uniform(0.93, 1.06)
                pj, wj = rng.random(), rng.uniform(-0.5, 0.5)
                tone = rng.uniform(-0.07, 0.07)
                u, v = area_u(rng, d), rng.uniform(-1, 1, d)
                g = u * 0.85 + 0.08 + tone + rng.normal(0, 0.035, d)
                live = grad(Ly["pal"], g)
                shade = (1 - 0.12 * np.abs(v) ** 3) * (1 - 0.05 * np.cos(v * 6.0))
                live = live * (shade * (1 + rng.normal(0, 0.02, d)))[:, None]
                wcol = grad(Ly["wpal"], 0.45 * u + 0.55 * rng.random(d))
                b0 = rng.random() * (0.30 if petal else 0.15)
                b1 = b0 + (0.70 if petal else 0.45 + 0.15 * rng.random())
                falls = petal and rng.random() < Ly["fall"]
                f = lambda val: np.full(d, val, np.float32)
                parts.append(dict(
                    pid=np.full(d, pid if petal else -1, np.int32),
                    cth=f(math.cos(th)), sth=f(math.sin(th)),
                    aj=f(aj), pj=f(pj), wj=f(wj),
                    u=u.astype(np.float32), v=v.astype(np.float32),
                    ac=f(Ly["ac"]), ao=f(Ly["ao"]), kc=f(Ly["kc"]), ko=f(Ly["ko"]),
                    L=f(Ly["L"] * lj), wc=f(Ly["wc"]), wo=f(Ly["wo"]),
                    b0=f(b0), b1=f(b1), wk=f(Ly["wk"]), wm=f(Ly["wm"]),
                    fthr=f(0.28 + 0.50 * rng.random() if falls else 9.0),
                    live=live.astype(np.float32), wilt=wcol.astype(np.float32),
                    rn=rng.random(d).astype(np.float32),
                    rad=rng.choice([1.0, 1.5], d, p=[0.6, 0.4]).astype(np.float32)))
                if petal:
                    pid += 1
        h = permute(cat(parts), rng)
        h["sp"] = profile(h["u"]).astype(np.float32)
        self.head = h
        # petals that fall: where they land on the ground
        self.NP = pid
        self.pm = h["pid"] >= 0
        self.pid_p = h["pid"][self.pm]
        self.pcnt = np.maximum(np.bincount(self.pid_p, minlength=pid), 1)
        tx = np.clip(self.cx + rng.normal(0.07 * H, 0.17 * H, pid), 0.06 * W, 0.94 * W)
        self.ptx = tx
        self.pty = self.surface(tx) - rng.random(pid) * 0.012 * H
        self.pph = rng.uniform(0, 6.28, pid)

    def build_disc(self):
        """Seed disc: dense dark fill + golden-angle (phyllotaxis) seed dots."""
        rng = self.rng
        nf, ns = 4200, 1500
        rf, tf = np.sqrt(rng.random(nf)), rng.uniform(0, 2 * math.pi, nf)
        i = np.arange(ns)
        rs, ts = np.sqrt((i + 0.5) / ns), i * 2.399963
        r, th = np.concatenate([rf, rs]), np.concatenate([tf, ts])
        live = grad(DISC_PAL, r ** 1.5 + rng.normal(0, 0.03, nf + ns))
        k = np.concatenate([np.ones(nf), rng.choice([0.62, 1.0, 1.4], ns, p=[0.45, 0.25, 0.30])])
        live = live * k[:, None]
        self.disc = dict(
            r=r.astype(np.float32), cth=np.cos(th).astype(np.float32), sth=np.sin(th).astype(np.float32),
            live=live.astype(np.float32), wilt=grad(DISC_WILT, r),
            zo=np.concatenate([np.zeros(nf), np.ones(ns)]).astype(np.float32),
            rad=np.concatenate([np.full(nf, 1.5), rng.choice([1.5, 2.0], ns, p=[0.6, 0.4])]).astype(np.float32))

    def build_back(self):
        """Rounded green back of the head - what you see once it hangs and turns away."""
        rng, n = self.rng, 5000
        t = rng.random(n)
        th = rng.uniform(0, 2 * math.pi, n)
        tone = np.clip(0.2 + 0.7 * t + rng.normal(0, 0.06, n), 0, 1)
        self.back = dict(
            t=t.astype(np.float32), cth=np.cos(th).astype(np.float32), sth=np.sin(th).astype(np.float32),
            live=grad(BACK_PAL, tone).astype(np.float32), wilt=grad(BACK_WILT, rng.random(n)),
            rad=rng.choice([1.5, 2.0], n, p=[0.6, 0.4]).astype(np.float32))

    # -- leaves -----------------------------------------------------------
    def build_leaves(self):
        rng, parts = self.rng, []
        d = 6500
        for j in range(len(LEAF_DEFS)):
            u, v = area_u(rng, d), rng.uniform(-1, 1, d)
            rib = rng.random(d) < 0.08
            v[rib] = rng.normal(0, 0.02, int(rib.sum()))
            tone = np.clip(0.18 + 0.55 * u + 0.22 * (1 - np.abs(v)) + rng.normal(0, 0.06, d), 0, 1)
            live = grad(LEAF_PAL, tone)
            vein = np.abs(np.sin((u * 7.0 - np.abs(v) * 2.2) * np.pi)) < 0.10
            live[vein] *= 1.22
            live[rib] = RIB_COL * (1 + rng.normal(0, 0.05, (int(rib.sum()), 1)))
            parts.append(dict(
                lid=np.full(d, j), u=u.astype(np.float32), v=v.astype(np.float32),
                live=live.astype(np.float32), wilt=grad(LEAF_WILT, rng.random(d)),
                rn=rng.random(d).astype(np.float32),
                rad=rng.choice([1.0, 1.5], d, p=[0.2, 0.8])))
        self.leaf = permute(cat(parts), rng)

    # -- sparkles ---------------------------------------------------------
    def build_fx(self):
        rng, m = self.rng, 320
        self.spark = dict(
            a=rng.uniform(0, 2 * math.pi, m), rho=rng.random(m), ph=rng.random(m),
            sp=rng.uniform(0.6, 1.4, m), rad=rng.choice([1.0, 1.5], m, p=[0.7, 0.3]),
            hue=rng.random(m))

    # ---------------------------------------------------------------------
    def render(self, t):
        H, W = self.H, self.W
        stem, size, bloom, wilt, leaf, glow = state(t)
        canvas = self.base.copy()
        B, sunB = Batch(), Batch()

        # ---- sun: fades in, crosses the sky, then dissolves with the dying flower ----
        alpha = float(np.clip(t / 8.0, 0, 1))
        track = float(np.clip((t - 8.0) / 22.0, 0, 1))
        sun_x = W * (0.25 + 0.50 * track)
        sun_y = H * (0.145 - 0.045 * math.sin(math.pi * track))
        dis = float(smoothstep(wilt, 0.0, 1.0))          # same clock as the wilting
        prog = np.clip((dis - self.sun_thr) / 0.30, 0, 1)
        fade = (1 - prog) ** 1.3
        keep = fade > 0.03
        if alpha > 0.01 and keep.any():
            trav = prog * self.sun_dist
            px = sun_x + self.sun_dx + self.sun_vx * trav + np.sin(prog * 6 + self.sun_ph) * 0.010 * H * prog
            py = (sun_y + self.sun_dy + self.sun_vy * trav - prog * prog * 0.05 * H
                  + np.cos(prog * 5 + self.sun_ph) * 0.008 * H * prog)
            shimmer = 0.93 + 0.07 * np.sin(t * 2.2 + self.sun_ph)
            col = self.sun_col * (alpha * fade * shimmer)[:, None]
            rad = np.maximum(1.0, self.sun_rad * (1 - 0.5 * prog))
            sunB.add(px[keep], py[keep], col[keep], rad[keep], np.zeros(int(keep.sum())))
        sunB.draw(canvas)

        # ---- stem ----
        L = self.Ls * stem
        sway = 0.010 * H * math.sin(t * 0.9) * stem * (1 - 0.6 * wilt)
        P = self.ctrl(L, wilt, sway)
        hx, hy = P[3]

        if stem > 0.01:
            S = self.stem
            s = S["s"]
            x, y, dx, dy = bez(P, s)
            nn = np.hypot(dx, dy) + 1e-6
            wst = (0.0105 * H * (1.18 - 0.40 * s) * (1 - 0.20 * wilt)
                   * np.clip(0.55 + stem * 1.0, 0.55, 1))
            lat = S["lat"] * wst
            wi = np.clip(wilt * (0.7 + 0.6 * s) * (0.7 + 0.5 * S["rn"]), 0, 0.9)
            B.add(x - dy / nn * lat, y + dx / nn * lat,
                  mixcol(S["live"], S["wilt"], wi), S["rad"], np.full(len(s), -500.0))

        # ---- leaves ----
        if leaf > 0.02 and stem > 0.05:
            S = self.leaf
            ax, ay, _, _ = bez(P, LS)
            lid = S["lid"]
            m, lfac = LM[lid], LF[lid]
            a0 = LA[lid] + (1 - leaf) * 0.9 - 0.6 * wilt
            c = LC[lid] - 0.5 * wilt
            u, v = S["u"], S["v"]
            LL = 0.17 * H * lfac * leaf * (1 - 0.25 * wilt)
            am = a0 + c * u * 0.5
            al = a0 + c * u
            sc = sinc(c * u * 0.5)
            lat = v * 0.30 * LL * profile(u)
            X = ax[lid] + LL * u * m * np.cos(am) * sc - lat * np.sin(al)
            Yup = LL * u * np.sin(am) * sc + lat * m * np.cos(al)
            wi = np.clip(wilt * (0.9 + 0.5 * u + 0.3 * S["rn"]) - 0.1, 0, 1)
            B.add(X, ay[lid] - Yup, mixcol(S["live"], S["wilt"], wi),
                  S["rad"], np.full(len(u), -450.0))

        # ---- soil heaped around the stem base ----
        if stem > 0.01:
            M = self.mound
            B.add(M["x"], M["y"], M["col"], M["rad"], np.full(len(M["x"]), -300.0))

        fx = fy = 0.0
        Ls_ = 0.0
        A = np.array([0.0, 0.0, 1.0])

        # ---- flower head ----
        if stem > 0.05 and size > 0.03:
            Ls_ = self.Lf * size
            openk = float(smoothstep(bloom, 0.05, 0.9))

            # orientation: bud points up -> face turns to the sun -> hangs along the bent stem
            el = 1.30 - 0.85 * openk
            sunx = float(np.clip((sun_x - hx) / (0.5 * W), -1, 1))
            az = 0.55 * sunx * openk
            A_t = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
            _, _, tdx, tdy = bez(P, np.array([1.0]))
            nrm = math.hypot(tdx[0], tdy[0]) + 1e-6
            A_w = np.array([0.88 * tdx[0] / nrm, -0.88 * tdy[0] / nrm, -0.30])
            we = float(smoothstep(wilt, 0.0, 0.85))
            A, U, V = head_basis((1 - we) * A_t + we * A_w)

            def proj(xl, zl, yl):
                px = xl * U[0] + zl * V[0] + yl * A[0]
                py = xl * U[1] + zl * V[1] + yl * A[1]
                pz = xl * U[2] + zl * V[2] + yl * A[2]
                return hx + px, hy - py, pz

            # petals + sepals
            g = self.head
            u, v = g["u"], g["v"]
            b = smoothstep(bloom, g["b0"], g["b1"])
            wp = np.clip(wilt * g["wk"] * (0.85 + 0.3 * g["pj"]), 0, 1)
            A0 = g["ac"] + (g["ao"] - g["ac"]) * b + g["aj"]
            A0 = A0 + (0.55 + 0.45 * g["pj"] - A0) * wp
            K = g["kc"] + (g["ko"] - g["kc"]) * b
            K = K + (-1.0 + 0.7 * g["wj"] - K) * wp
            Lp = Ls_ * g["L"] * (1 - 0.32 * wp)
            am = A0 + K * u * 0.5
            sc = sinc(K * u * 0.5)
            rr = Lp * u * np.sin(am) * sc
            hh = Lp * u * np.cos(am) * sc
            lat = v * (g["wc"] + (g["wo"] - g["wc"]) * b) * (1 - 0.35 * wp) * Lp * g["sp"]
            rr = rr + 0.03 * Ls_ - 0.5 * lat * lat / (Lp + 1.0)
            X, Y, zp = proj(g["cth"] * rr - g["sth"] * lat, g["sth"] * rr + g["cth"] * lat, hh)

            shade = np.clip(0.74 + 0.26 * np.clip(zp / Ls_ * 0.5 + 0.5, 0, 1) + 0.10 * (hh / Ls_), 0.5, 1.15)
            shade = shade * (0.82 + 0.18 * smoothstep(u, 0.0, 0.4)) * (1 - 0.15 * wilt)
            wi = np.clip(wilt * g["wm"] * (1.0 + 0.7 * u + 0.4 * g["rn"]) - 0.2 * g["wm"], 0, 1)

            # petals drop one by one and land on the soil
            if wilt > 0.25:
                fp = np.clip((wilt - g["fthr"]) / 0.22, 0, 1)
                mk = fp > 0
                if mk.any():
                    pm = self.pm
                    cxp = np.bincount(self.pid_p, weights=X[pm], minlength=self.NP) / self.pcnt
                    cyp = np.bincount(self.pid_p, weights=Y[pm], minlength=self.NP) / self.pcnt
                    ii = g["pid"][mk]
                    f = fp[mk]
                    ix, iy = cxp[ii], cyp[ii]
                    flut = np.sin(f * 7.0 + self.pph[ii]) * 0.03 * H * (1 - f)
                    X[mk] = ix + (X[mk] - ix) * (1 - 0.4 * f) + (self.ptx[ii] - ix) * f + flut
                    Y[mk] = iy + (Y[mk] - iy) * (1 - 0.7 * f) + (self.pty[ii] - iy) * f * f
                    zp[mk] = 700.0
                    wi = np.maximum(wi, fp)

            col = mixcol(g["live"], g["wilt"], wi) * shade[:, None]
            B.add(X, Y, col, g["rad"], zp)

            # rounded green back of the head
            Bk = self.back
            Rb = 0.52 * Ls_ * (0.40 + 0.60 * openk)
            rho = Rb * np.sqrt(1 - Bk["t"] ** 2)
            X2, Y2, z2 = proj(Bk["cth"] * rho, Bk["sth"] * rho, -0.55 * Rb * Bk["t"])
            wbk = np.full(len(Bk["t"]), min(1.0, wilt * 1.1), np.float32)
            sh2 = np.clip(0.75 + 0.25 * np.clip(z2 / Ls_ * 0.5 + 0.5, 0, 1), 0.5, 1.1)
            B.add(X2, Y2, mixcol(Bk["live"], Bk["wilt"], wbk) * sh2[:, None], Bk["rad"], z2)

            # seed disc: only drawn while the face looks toward the viewer, shrinks away as it turns
            vis = float(smoothstep(A[2], 0.02, 0.30)) * float(smoothstep(bloom, 0.04, 0.5))
            if vis > 0.02:
                D = self.disc
                R = 0.50 * Ls_ * vis
                rd = D["r"] * R
                dome = 0.04 * Ls_ + 0.08 * Ls_ * (1 - D["r"] ** 2) * vis
                X3, Y3, z3 = proj(D["cth"] * rd, D["sth"] * rd, dome)
                wdc = np.full(len(rd), min(1.0, wilt * 1.3), np.float32)
                B.add(X3, Y3, mixcol(D["live"], D["wilt"], wdc), D["rad"], z3 + 0.06 * Ls_ + D["zo"])

            fx, fy = hx + A[0] * 0.3 * Ls_, hy - A[1] * 0.3 * Ls_

        B.draw(canvas)

        # ---- soft golden sparkles around the open flower (additive, never dark) ----
        glow_e = glow * (1 - float(smoothstep(wilt, 0.0, 0.3)))
        if glow_e > 0.02 and Ls_ > 0:
            Sp = self.spark
            life = (t * 0.10 * Sp["sp"] + Sp["ph"]) % 1.0
            rho = (0.45 + 1.0 * Sp["rho"]) * Ls_ * (0.6 + 0.6 * life)
            px = fx + np.cos(Sp["a"]) * rho
            py = fy + np.sin(Sp["a"]) * rho - life * 0.35 * Ls_
            bs = (np.sin(np.pi * life) ** 0.8 * glow_e * 0.55)[:, None]
            cs = np.where(Sp["hue"][:, None] < 0.5, np.asarray([255, 240, 150], np.float32),
                          np.asarray([255, 200, 50], np.float32))
            splat_add(canvas, px, py, cs * bs, Sp["rad"])

        if SOFTEN > 0:
            canvas = np.array(Image.fromarray(canvas).filter(ImageFilter.GaussianBlur(SOFTEN)))
        return canvas


# ---------------------------------------------------------------------------
# output modes
# ---------------------------------------------------------------------------
def save_png():
    H = 1000
    plant = Plant(H, H)
    times = [4.0, 10.0, 15.0, 20.0, 26.0, 34.0, 38.0, 44.0]
    frames = []
    for i, tt in enumerate(times):
        a = plant.render(tt)
        Image.fromarray(a).save(f"sunflower_{i + 1}.png")
        frames.append(Image.fromarray(a).resize((500, 500), Image.LANCZOS))
    strip = Image.new("RGB", (500 * len(frames), 500))
    for i, f in enumerate(frames):
        strip.paste(f, (500 * i, 0))
    strip.save("sunflower_stages.png")
    print("Saved: sunflower_stages.png and stage frames")


def save_gif(fps=12, size=640):
    plant = Plant(size, size)
    frames = [Image.fromarray(plant.render(i / fps)).convert("P", palette=Image.ADAPTIVE)
              for i in range(int(T_END * fps) + 1)]
    frames[0].save("sunflower.gif", save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=False)
    print("Saved: sunflower.gif")


class App:
    def __init__(self):
        import tkinter as tk
        from PIL import ImageTk

        self.root = tk.Tk()
        self.root.title("Pointillism Sunflower Tracker")
        self.root.configure(bg="black")
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        H = int(min(sh * WIN_FRAC, sw * 0.96))
        W = H
        self.root.geometry(f"{W}x{H}+{(sw - W) // 2}+{max(0, (sh - H) // 2 - 20)}")
        self.cv = tk.Canvas(self.root, width=W, height=H, bg="black", highlightthickness=0)
        self.cv.pack(fill="both", expand=True)
        self.cv.create_text(W // 2, H // 2, text="preparing...", fill="#777777")
        self.root.update()

        self.plant = Plant(W, H)
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.plant.render(0)))
        self.cv.delete("all")
        self.cv.create_image(0, 0, anchor="nw", image=self.photo)
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<space>", lambda e: self.toggle())
        self.root.bind("<r>", lambda e: self.restart())
        self.root.bind("<s>", lambda e: Image.fromarray(self.plant.render(self.clock / TIME_SCALE)).save("sunflower_frame.png"))
        self.clock, self.last, self.paused = 0.0, time.perf_counter(), False
        self.tick()

    def toggle(self):
        self.paused = not self.paused

    def restart(self):
        self.clock = 0.0

    def tick(self):
        now = time.perf_counter()
        if not self.paused:
            self.clock += now - self.last
        self.last = now
        t = self.clock / TIME_SCALE
        if LOOP and t > T_END + 4:
            self.clock, t = 0.0, 0.0
        self.photo.paste(Image.fromarray(self.plant.render(t)))
        self.root.after(TICK_MS, self.tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if "--png" in sys.argv:
        save_png()
    elif "--gif" in sys.argv:
        save_gif()
    else:
        App().run()