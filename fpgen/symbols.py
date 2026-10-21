"""Parametric 2D drafting glyphs for fixtures.

Each glyph draws in a LOCAL mm frame: the fixture occupies
x in [-w/2, w/2], y in [-d/2, d/2]; y = -d/2 is the BACK (wall side),
y = +d/2 the front. The renderer sets up translation+rotation; glyphs are
pure strokes via the duck-typed draw helper `g` (line/rect/circle/arc/poly).

Note: bookshelf is deliberately drawn with parallel shelf lines so it
resembles a stair run — the confusable pair the training data must teach.
"""

from __future__ import annotations

import math

GLYPHS = {}


def glyph(name):
    def deco(fn):
        GLYPHS[name] = fn
        return fn
    return deco


@glyph("bed")
def _bed(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    # pillows at the back
    pw, ph = w * 0.38, d * 0.14
    g.rect(-w / 2 + w * 0.06, -d / 2 + d * 0.04, pw, ph)
    g.rect(w / 2 - w * 0.06 - pw, -d / 2 + d * 0.04, pw, ph)
    # blanket fold line
    fy = -d / 2 + d * 0.32
    g.line((-w / 2, fy), (w / 2, fy))
    g.line((-w / 2, fy + d * 0.05), (w / 2, fy + d * 0.05))


@glyph("wardrobe")
def _wardrobe(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.line((-w / 2, 0), (w / 2, 0))  # hanging rail
    n = max(2, int(w // 450))
    for i in range(1, n):
        x = -w / 2 + w * i / n
        g.line((x, -d / 2), (x, d / 2))


@glyph("sofa")
def _sofa(g, w, d, style, rng, params):
    arm = min(w * 0.12, 250.0)
    back = min(d * 0.28, 250.0)
    g.rrect(-w / 2, -d / 2, w, d, r=80)
    g.line((-w / 2 + arm, -d / 2 + back), (w / 2 - arm, -d / 2 + back))
    g.line((-w / 2 + arm, -d / 2 + back), (-w / 2 + arm, d / 2 - 40))
    g.line((w / 2 - arm, -d / 2 + back), (w / 2 - arm, d / 2 - 40))
    # seat cushions
    seats = 3 if w > 2100 else 2
    for i in range(1, seats):
        x = -w / 2 + arm + (w - 2 * arm) * i / seats
        g.line((x, -d / 2 + back), (x, d / 2 - 40))


@glyph("tv_console")
def _tv(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.line((-w / 2 + w * 0.2, 0), (w / 2 - w * 0.2, 0))  # TV mark


@glyph("coffee_table")
def _coffee(g, w, d, style, rng, params):
    g.rrect(-w / 2, -d / 2, w, d, r=min(120.0, d / 4))


@glyph("dining_table")
def _dining(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    if not (params or {}).get("chairs", True):
        return
    cw, cd = 420.0, 420.0
    n_side = max(1, int(w // 700))
    for i in range(n_side):
        cx = -w / 2 + w * (i + 0.5) / n_side
        g.rect(cx - cw / 2, -d / 2 - cd - 40, cw, cd, r=100)
        g.rect(cx - cw / 2, d / 2 + 40, cw, cd, r=100)


@glyph("bookshelf")
def _bookshelf(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    n = max(3, int(w // 280))  # parallel shelf lines — stair-like on purpose
    for i in range(1, n):
        x = -w / 2 + w * i / n
        g.line((x, -d / 2), (x, d / 2))


@glyph("desk")
def _desk(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.rect(-210, d / 2 + 60, 420, 420, r=110)  # chair


@glyph("toilet")
def _toilet(g, w, d, style, rng, params):
    tank_d = d * 0.28
    g.rect(-w / 2, -d / 2, w, tank_d)
    g.ellipse(0, -d / 2 + tank_d + (d - tank_d) / 2, w * 0.42, (d - tank_d) * 0.46)


@glyph("basin")
def _basin(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.ellipse(0, 0, w * 0.32, d * 0.3)
    g.circle(0, -d / 2 + d * 0.18, 25)  # tap


@glyph("shower")
def _shower(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.line((-w / 2, -d / 2), (w / 2, d / 2))
    g.line((w / 2, -d / 2), (-w / 2, d / 2))
    g.circle(0, 0, min(w, d) * 0.08)


@glyph("bathtub")
def _bathtub(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.rrect(-w / 2 + 70, -d / 2 + 70, w - 140, d - 140, r=min(200.0, d / 3))
    g.circle(-w / 2 + 180, 0, 35)  # drain


@glyph("counter")
def _counter(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.line((-w / 2, d / 2 - 60), (w / 2, d / 2 - 60))  # front edge double line


@glyph("stove")
def _stove(g, w, d, style, rng, params):
    n = int((params or {}).get("n_burners", 4))
    g.rect(-w / 2, -d / 2, w, d)
    r = min(w, d) * 0.17
    if n >= 4:
        for sx in (-1, 1):
            for sy in (-1, 1):
                g.circle(sx * w * 0.22, sy * d * 0.22, r)
    else:
        g.circle(-w * 0.22, 0, r)
        g.circle(w * 0.22, 0, r)


@glyph("sink")
def _sink(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.rrect(-w / 2 + 60, -d / 2 + 60, w - 120, d - 120, r=80)
    g.circle(0, -d / 2 + 45, 22)


@glyph("fridge")
def _fridge(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.line((-w / 2, -d / 2), (w / 2, d / 2))
    g.line((-w / 2 + w * 0.15, -d / 2), (-w / 2 + w * 0.15, d / 2))


@glyph("washer")
def _washer(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.circle(0, 0, min(w, d) * 0.32)
    g.circle(0, 0, min(w, d) * 0.22)


@glyph("dryer")
def _dryer(g, w, d, style, rng, params):
    _washer(g, w, d, style, rng, params)
    g.circle(0, 0, min(w, d) * 0.1)


@glyph("shelving")
def _shelving(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    n = max(2, int(w // 350))
    for i in range(1, n):
        x = -w / 2 + w * i / n
        g.line((x, -d / 2), (x, d / 2))


@glyph("ac_unit")
def _ac(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    r = min(w, d) * 0.3
    g.circle(0, 0, r)
    for k in range(3):
        a = math.radians(120 * k)
        g.line((0, 0), (r * math.cos(a), r * math.sin(a)))


@glyph("shoe_cabinet")
def _shoe(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    g.line((-w / 2, 0), (w / 2, 0))


@glyph("outdoor_chair")
def _ochair(g, w, d, style, rng, params):
    g.rrect(-w / 2, -d / 2, w, d, r=min(150.0, w / 3))
    g.circle(0, 0, min(w, d) * 0.18)


@glyph("planter_box")
def _planter(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    n = max(2, int(w // 300))
    for i in range(n):
        cx = -w / 2 + w * (i + 0.5) / n
        g.circle(cx, 0, min(w / (2 * n), d / 2) * 0.55)


@glyph("stairs")
def _stairs(g, w, d, style, rng, params):
    g.rect(-w / 2, -d / 2, w, d)
    n = max(5, int(d // 260))  # treads
    for i in range(1, n):
        y = -d / 2 + d * i / n
        g.line((-w / 2, y), (w / 2, y))
    g.line((0, d / 2 - 100), (0, -d / 2 + 200))  # direction arrow
    g.line((0, -d / 2 + 200), (-70, -d / 2 + 340))
    g.line((0, -d / 2 + 200), (70, -d / 2 + 340))
