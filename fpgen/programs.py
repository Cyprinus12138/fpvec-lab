"""SG room programs: unit meta -> a hierarchical split program + door tree.

A program is a tree of Row groups and Leaf rooms. layout.py realizes it by
recursive proportional guillotine splits, so rooms in the same subtree end up
adjacent by construction; the residual hard adjacencies (listed per program)
are checked post-layout and failures are rejected/re-drawn.

Door topology is encoded up front via RoomSpec.access_parent (a key of another
room, or "ENTRANCE" for the unit entrance), which guarantees reachability by
construction; openings.py places the actual door geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .meta import UnitMeta

# Nominal room areas in sqm; normalized against the unit target area at layout
# time, so these encode *proportions* (SG condo conventions), not absolutes.
NOMINAL_SQM = {
    "living_dining": 21.0, "living": 14.0, "dining": 8.0,
    "kitchen": 7.5, "kitchen_small": 5.0,
    "bedroom_master": 12.0, "bedroom": 8.5, "study": 5.5,
    "bath_master": 4.2, "bath": 3.6, "wc": 2.0,
    "yard": 2.8, "utility": 2.5, "store": 1.8,
    "household_shelter": 2.8, "aircon_ledge": 1.8,
    "balcony": 4.5, "foyer": 2.5, "corridor": 3.0, "wic": 3.2,
}

MIN_DIM_MM = {
    "living": 2700, "living_dining": 2700, "dining": 2400,
    "kitchen": 1700, "bedroom": 2400, "study": 1900,
    "bath": 1400, "wc": 1100, "yard": 1100, "utility": 1100,
    "store": 900, "household_shelter": 1400, "aircon_ledge": 550,
    "balcony": 1100, "foyer": 1100, "corridor": 900, "wic": 1300,
}

# strip-like rooms are exempt from the compactness default
ASPECT_MAX = {
    "corridor": 20.0, "aircon_ledge": 20.0, "balcony": 8.0, "yard": 6.0,
    "foyer": 5.0, "wic": 4.5, "store": 4.0, "utility": 4.0,
    "wc": 3.5, "bath": 3.2, "kitchen": 3.5,
    "living": 3.4, "living_dining": 3.4,
}


@dataclass
class Carve:
    """A sub-room carved off an edge of its host room's cell."""
    spec: "RoomSpec"
    edge: str            # "exterior" (host's envelope-facing edge) | "any"
    depth_mm: tuple[float, float]
    width_frac: tuple[float, float]  # fraction of the host edge length


@dataclass
class RoomSpec:
    key: str
    role: str
    label_role: str
    nominal_sqm: float
    min_dim_mm: float
    aspect_max: float = 3.0
    label_index: int | None = None
    access_parent: str | None = None  # room key or "ENTRANCE"; None = no door
    door_kind: str = "door"           # door | sliding_door | opening
    window: bool = False
    carves: list[Carve] = field(default_factory=list)


@dataclass
class Row:
    """Split a cell among children proportionally along one axis.

    If `corridor` is set, a thin corridor strip is first carved along the
    cell edge facing the previous sibling zone, and children open off it.
    """
    children: list["Row | RoomSpec"]
    corridor: RoomSpec | None = None


@dataclass
class Program:
    root: Row
    entrance_key: str
    # pairs of room keys that must share a wall segment (checked post-layout)
    hard_adjacent: list[tuple[str, str]] = field(default_factory=list)
    meta: UnitMeta | None = None


def _spec(key, role, label_role=None, sqm=None, min_dim=None, **kw) -> RoomSpec:
    kw.setdefault("aspect_max", ASPECT_MAX.get(role, 3.0))
    return RoomSpec(
        key=key,
        role=role,
        label_role=label_role or role,
        nominal_sqm=sqm if sqm is not None else NOMINAL_SQM.get(label_role or role, 4.0),
        min_dim_mm=min_dim if min_dim is not None else MIN_DIM_MM.get(role, 1200),
        **kw,
    )


def _balcony_carve(rng: np.random.Generator) -> Carve:
    return Carve(
        spec=_spec("balcony", "balcony", access_parent="living",
                   door_kind="sliding_door"),
        edge="exterior", depth_mm=(1200, 2000), width_frac=(0.5, 1.0),
    )


def _ledge_carve(host_key: str) -> Carve:
    return Carve(
        spec=_spec("aircon_ledge", "aircon_ledge", access_parent=None),
        edge="exterior", depth_mm=(550, 950), width_frac=(0.45, 0.9),
    )


def build_program(meta: UnitMeta, rng: np.random.Generator) -> Program:
    """Meta -> program tree. Randomizes optional rooms per meta flags."""
    f = meta.flags
    n_bed, n_bath = meta.n_bed, meta.n_bath

    # area-budget gate: drop optional rooms until the nominal program fits the
    # sampled unit area, else min-dim constraints are unsatisfiable and layout
    # rejection rates explode (small units simply don't have studies/stores)
    required = (NOMINAL_SQM["living_dining"] + NOMINAL_SQM["kitchen"]
                + NOMINAL_SQM["bedroom_master"]
                + NOMINAL_SQM["bedroom"] * max(0, n_bed - 1)
                + NOMINAL_SQM["bath"] * n_bath
                + (NOMINAL_SQM["corridor"] if n_bed >= 2 else 0)
                + (NOMINAL_SQM["household_shelter"] if f.get("household_shelter") else 0)
                + (NOMINAL_SQM["aircon_ledge"] if f.get("ac_ledge") else 0))
    budget = meta.target_area_sqm - required
    for opt, cost in (("study", NOMINAL_SQM["study"]),
                      ("wic", NOMINAL_SQM["wic"]),
                      ("store", NOMINAL_SQM["store"]),
                      ("yard", NOMINAL_SQM["yard"]),
                      ("balcony", NOMINAL_SQM["balcony"])):
        if f.get(opt):
            if budget < cost:
                f[opt] = False
            else:
                budget -= cost
    # floor the unit area at the core program's needs — a 2B2b below ~60 sqm
    # scales every room under its min dims and layout can never satisfy it
    meta.target_area_sqm = max(meta.target_area_sqm, round(required * 1.06, 1))

    # --- public zone -------------------------------------------------------
    living = _spec("living", "living_dining", "living_dining",
                   access_parent=None, window=True)
    if f.get("balcony"):
        living.carves.append(_balcony_carve(rng))

    service_children: list[Row | RoomSpec] = []
    if f.get("yard"):
        yard = _spec("yard", "yard", access_parent="kitchen", door_kind="door")
        if f.get("ac_ledge"):
            yard.carves.append(_ledge_carve("yard"))
        service_children.append(yard)
        wc_parent = "yard"
    else:
        wc_parent = "kitchen"
    # a WC exists when bath count exceeds bedrooms' en-suites + common baths
    has_wc = n_bath > max(1, n_bed) and n_bed >= 2
    if has_wc:
        service_children.append(_spec("wc", "wc", access_parent=wc_parent))
    if f.get("household_shelter"):
        service_children.append(
            _spec("shelter", "household_shelter", access_parent="kitchen"))
    if f.get("store"):
        service_children.append(_spec("store", "store", access_parent="kitchen"))

    kitchen = _spec("kitchen", "kitchen",
                    sqm=NOMINAL_SQM["kitchen" if n_bed >= 2 else "kitchen_small"],
                    access_parent="living",
                    door_kind="opening" if rng.random() < 0.5 else "door",
                    window=not f.get("yard"))
    if not f.get("yard") and f.get("ac_ledge"):
        kitchen.carves.append(_ledge_carve("kitchen"))

    kitchen_block: Row | RoomSpec
    if service_children:
        kitchen_block = Row(children=[kitchen, Row(children=service_children)])
    else:
        kitchen_block = kitchen

    public_children: list[Row | RoomSpec] = []
    has_foyer = n_bed >= 2 and rng.random() < 0.6
    if has_foyer:
        # foyer = entrance alcove carved off the living cell's exterior edge
        # (a full-depth zone slice makes an unrealistic 0.7 m sliver)
        living.carves.append(Carve(
            spec=_spec("foyer", "foyer", access_parent=None),
            edge="exterior", depth_mm=(1200, 1800), width_frac=(0.25, 0.45)))
        living.access_parent = "foyer"
        living.door_kind = "opening"
        entrance_key = "foyer"
    else:
        living.access_parent = "ENTRANCE"
        entrance_key = "living"
    public_children.append(living)
    public_children.append(kitchen_block)
    public = Row(children=public_children)

    # --- private zone ------------------------------------------------------
    if n_bed == 0:  # studio: bath (+shelter cluster) off the single space
        root = Row(children=[public])
        bath = _spec("bath1", "bath", "bath", access_parent="living", label_index=1)
        # attach bath into the service cluster
        if isinstance(kitchen_block, Row):
            kitchen_block.children[-1].children.append(bath)  # type: ignore[union-attr]
        else:
            root.children.append(bath)
        prog = Program(root=root, entrance_key=entrance_key, meta=meta)
        prog.hard_adjacent = [("kitchen", "living")]
        if f.get("balcony"):
            prog.hard_adjacent.append(("balcony", "living"))
        return prog

    private_children: list[Row | RoomSpec] = []
    bed_parent = "corridor" if n_bed >= 2 else "living"

    master = _spec("master_bed", "bedroom", "bedroom_master",
                   sqm=NOMINAL_SQM["bedroom_master"],
                   min_dim=2750,  # queen bed + clearances + two doors
                   access_parent=bed_parent, window=True)
    if f.get("wic"):
        master.carves.append(Carve(
            spec=_spec("wic", "wic", access_parent="master_bed"),
            edge="any", depth_mm=(1300, 1700), width_frac=(0.4, 0.7)))
    if n_bath >= 2 or n_bed == 1:
        mbath = _spec("master_bath", "bath", "bath_master",
                      access_parent="master_bed")
        private_children.append(Row(children=[master, mbath]))
        common_baths = n_bath - 1
    else:
        private_children.append(master)
        common_baths = n_bath

    for i in range(2, n_bed + 1):
        private_children.append(
            _spec(f"bed{i}", "bedroom", "bedroom", label_index=i,
                  access_parent=bed_parent, window=True))
    if f.get("study"):
        private_children.append(
            _spec("study", "study", access_parent=bed_parent, window=True))
    for j in range(common_baths):
        # common bath opens off corridor (or living when no corridor)
        private_children.append(
            _spec(f"bath{j + 2}", "bath", "bath", label_index=j + 2,
                  access_parent=bed_parent))

    corridor = None
    if n_bed >= 2:
        corridor = _spec("corridor", "corridor",
                         access_parent="living", door_kind="opening")
    private = Row(children=private_children, corridor=corridor)

    root = Row(children=[public, private])
    prog = Program(root=root, entrance_key=entrance_key, meta=meta)
    prog.hard_adjacent = [("kitchen", "living"),
                          ("master_bed", f"{bed_parent}|living")]
    if f.get("balcony"):
        prog.hard_adjacent.append(("balcony", "living"))
    if f.get("household_shelter"):
        # shelter must be reachable off the service area: adjacency with
        # kitchen or yard (its access_parent is kitchen; layout may route
        # the door via yard when kitchen isn't adjacent)
        prog.hard_adjacent.append(("shelter", "kitchen|yard|corridor"))
    return prog


def all_leaves(node: Row | RoomSpec) -> list[RoomSpec]:
    """Flatten a program tree to its RoomSpecs (incl. corridor + carves)."""
    out: list[RoomSpec] = []

    def walk(n: Row | RoomSpec):
        if isinstance(n, RoomSpec):
            out.append(n)
            for c in n.carves:
                out.append(c.spec)
        else:
            if n.corridor is not None:
                out.append(n.corridor)
            for c in n.children:
                walk(c)

    walk(node)
    return out
