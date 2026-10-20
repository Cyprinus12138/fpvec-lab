"""THE declarative SG room-role -> fixture rules table.

This is data, not logic: extend fpgen by adding entries here plus a glyph in
symbols.py. The solver in semantics.py interprets it.

Sizes are (width_range, depth_range) in mm — width runs along the wall the
fixture backs onto, depth extends into the room. Clearances are kept free of
OTHER fixtures (not of walls/other clearances). `hard=True` means the sample
is vetoed if the rule cannot be satisfied (e.g. a bath with no toilet is not
a bath); soft failures just drop the fixture and are logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FixtureRule:
    kind: str
    count: tuple[int, int] = (1, 1)
    prob: float = 1.0
    placement: str = "wall"  # wall | corner | center | counter_run | in_counter | near
    wall_pref: str = "any"   # any | exterior | interior | longest
    size_mm: tuple[tuple[float, float], tuple[float, float]] = ((600, 600), (600, 600))
    clearance_front_mm: float = 0.0
    clearance_sides_mm: float = 0.0
    near: str | None = None  # kind of an already-placed fixture to anchor to
    hard: bool = False
    params: dict = field(default_factory=dict)


@dataclass
class RoomRules:
    rules: list[FixtureRule] = field(default_factory=list)
    forbidden: set[str] = field(default_factory=set)


_WET = {"bed", "sofa", "wardrobe", "desk", "tv_console", "dining_table",
        "bookshelf", "coffee_table"}

RULES: dict[str, RoomRules] = {
    "living_dining": RoomRules(rules=[
        FixtureRule("sofa", placement="wall", size_mm=((1800, 2800), (850, 980)),
                    clearance_front_mm=400),
        FixtureRule("tv_console", prob=0.8, placement="near", near="sofa",
                    size_mm=((1400, 2000), (350, 450)), params={"mode": "opposite"}),
        FixtureRule("coffee_table", prob=0.7, placement="near", near="sofa",
                    size_mm=((900, 1200), (500, 650)), params={"mode": "front"}),
        FixtureRule("dining_table", placement="center",
                    size_mm=((1300, 1800), (800, 1000)), clearance_front_mm=0,
                    clearance_sides_mm=550, params={"chairs": True}),
        FixtureRule("bookshelf", prob=0.25, placement="wall",
                    size_mm=((800, 1600), (300, 400))),
    ], forbidden={"toilet", "stove", "bed", "washer"}),
    "living": RoomRules(rules=[
        FixtureRule("sofa", placement="wall", size_mm=((1800, 2600), (850, 980)),
                    clearance_front_mm=400),
        FixtureRule("tv_console", prob=0.8, placement="near", near="sofa",
                    size_mm=((1400, 2000), (350, 450)), params={"mode": "opposite"}),
        FixtureRule("coffee_table", prob=0.7, placement="near", near="sofa",
                    size_mm=((900, 1200), (500, 650)), params={"mode": "front"}),
    ], forbidden={"toilet", "stove", "bed", "washer"}),
    "dining": RoomRules(rules=[
        FixtureRule("dining_table", placement="center",
                    size_mm=((1300, 1800), (800, 1000)), clearance_sides_mm=550,
                    params={"chairs": True}),
    ], forbidden={"toilet", "stove", "bed", "washer"}),
    "bedroom": RoomRules(rules=[
        FixtureRule("bed", placement="wall", size_mm=((1400, 1900), (1900, 2100)),
                    clearance_front_mm=600, hard=True,
                    params={"master_width": (1500, 1850), "common_width": (1000, 1500)}),
        FixtureRule("wardrobe", placement="wall", wall_pref="interior",
                    size_mm=((1200, 2400), (550, 650)), clearance_front_mm=550),
        FixtureRule("desk", prob=0.3, placement="wall",
                    size_mm=((1000, 1400), (500, 650))),
        FixtureRule("bookshelf", prob=0.15, placement="wall",
                    size_mm=((800, 1200), (300, 400))),
    ], forbidden={"toilet", "stove", "sofa", "washer"}),
    "bath": RoomRules(rules=[
        FixtureRule("toilet", placement="wall", size_mm=((380, 430), (620, 700)),
                    clearance_front_mm=450, hard=True),
        FixtureRule("basin", placement="wall", size_mm=((450, 600), (420, 520)),
                    clearance_front_mm=400),
        FixtureRule("shower", prob=0.75, placement="corner",
                    size_mm=((850, 1000), (850, 1000))),
        FixtureRule("bathtub", prob=0.15, placement="wall",
                    size_mm=((1500, 1700), (700, 800))),
    ], forbidden=_WET),
    "wc": RoomRules(rules=[
        FixtureRule("toilet", placement="wall", size_mm=((380, 430), (620, 700)),
                    clearance_front_mm=450, hard=True),
        FixtureRule("basin", prob=0.8, placement="wall",
                    size_mm=((400, 500), (350, 450))),
    ], forbidden=_WET),
    "kitchen": RoomRules(rules=[
        FixtureRule("counter", placement="counter_run",
                    size_mm=((0, 0), (580, 620)), hard=False),
        FixtureRule("stove", placement="in_counter", size_mm=((580, 620), (580, 620)),
                    params={"burners": (2, 4)}),
        FixtureRule("sink", placement="in_counter", size_mm=((450, 550), (400, 500)),
                    params={"min_from": {"stove": 400}}),
        FixtureRule("fridge", placement="wall", size_mm=((600, 700), (600, 700)),
                    clearance_front_mm=550),
    ], forbidden=_WET | {"toilet"}),
    "yard": RoomRules(rules=[
        FixtureRule("washer", placement="wall", size_mm=((580, 620), (580, 620))),
        FixtureRule("dryer", prob=0.4, placement="near", near="washer",
                    size_mm=((580, 620), (580, 620)), params={"mode": "beside"}),
        FixtureRule("sink", prob=0.3, placement="wall",
                    size_mm=((450, 550), (400, 500))),
    ], forbidden=_WET),
    "utility": RoomRules(rules=[
        FixtureRule("shelving", prob=0.6, placement="wall",
                    size_mm=((800, 1400), (350, 450))),
    ]),
    "store": RoomRules(rules=[
        FixtureRule("shelving", prob=0.5, placement="wall",
                    size_mm=((700, 1200), (300, 400))),
    ]),
    "household_shelter": RoomRules(rules=[
        FixtureRule("shelving", prob=0.3, placement="wall",
                    size_mm=((700, 1100), (300, 400))),
    ], forbidden=_WET | {"toilet", "stove", "washer"}),
    "wic": RoomRules(rules=[
        FixtureRule("wardrobe", count=(1, 2), placement="wall",
                    size_mm=((1200, 2200), (550, 620)), clearance_front_mm=500),
    ]),
    "study": RoomRules(rules=[
        FixtureRule("desk", placement="wall", size_mm=((1100, 1500), (550, 700)),
                    clearance_front_mm=550),
        FixtureRule("bookshelf", prob=0.6, placement="wall",
                    size_mm=((800, 1600), (300, 400))),
    ], forbidden={"toilet", "stove", "bed"}),
    "aircon_ledge": RoomRules(rules=[
        FixtureRule("ac_unit", count=(1, 3), placement="wall",
                    size_mm=((750, 950), (300, 380))),
    ]),
    "balcony": RoomRules(rules=[
        FixtureRule("outdoor_chair", prob=0.4, count=(1, 2), placement="center",
                    size_mm=((500, 650), (500, 650))),
        FixtureRule("planter_box", prob=0.3, placement="wall",
                    size_mm=((700, 1400), (280, 380))),
    ], forbidden=_WET),
    "foyer": RoomRules(rules=[
        FixtureRule("shoe_cabinet", prob=0.35, placement="wall",
                    size_mm=((800, 1200), (300, 400))),
    ]),
    "corridor": RoomRules(rules=[]),
}


def ac_unit_count(n_bed: int) -> tuple[int, int]:
    """SG convention: one condenser per aircon'd room, capped by ledge size."""
    n = max(1, (n_bed * 7) // 10 + 1)
    return (max(1, n - 1), n)
