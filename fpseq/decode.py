"""Decode: flat tensor dict -> SceneEvents -> fpeval FloorPlan.

The grammar makes parsing unambiguous: three sections split by SEC, walls as
COORD/COORD/ATTR triples, rooms as WREF/OPEN cycles ended by SEP, openings as
WREF/INTERVAL pairs. Geometry comes from the continuous `coord` head (bins are
input-only). Room polygons are reconstructed by chaining walls in the emitted
canonical order and bridging OPEN edges (fpseq.geometry) so open-plan rooms get
an explicit polygon — which drives fpeval.verify's region-tiling path.
"""

from __future__ import annotations

import numpy as np

from fpeval.model import FloorPlan, Opening, Room, Wall

from . import geometry, schema
from .schema import (SceneEvents, OpeningEvent, RoomEvent, TT, WallEvent,
                     class_opening_kind, class_role, class_wall_type)


def tensors_to_events(t: dict) -> SceneEvents:
    tt = t["token_type"]
    coord = t["coord"]
    wref = t["wref"]
    sem = t["sem"]
    frame = float(t["frame"])
    n = len(tt)

    walls: list[WallEvent] = []
    rooms: list[RoomEvent] = []
    openings: list[OpeningEvent] = []
    lengths: list[float] = []

    i = 0
    # -- walls: COORD COORD ATTR triples until first SEC ----------------
    while i < n and tt[i] != TT.SEC and tt[i] != TT.EOS:
        if tt[i] == TT.COORD and i + 2 < n and tt[i + 1] == TT.COORD \
                and tt[i + 2] == TT.ATTR:
            start = (float(coord[i][0]) * frame, float(coord[i][1]) * frame)
            end = (float(coord[i + 1][0]) * frame, float(coord[i + 1][1]) * frame)
            ax, ay = float(coord[i + 2][0]), float(coord[i + 2][1])
            thickness = ax * frame / schema.THICK_SCALE
            curvature = ay * frame / schema.CURV_SCALE
            wtype = class_wall_type(int(sem[i + 2]))
            walls.append(WallEvent(start, end, thickness, curvature, wtype))
            lengths.append(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5)
            i += 3
        else:
            i += 1

    if i < n and tt[i] == TT.SEC:
        i += 1

    # -- rooms: (WREF|OPEN)+ SEP, until second SEC ----------------------
    members: list[tuple] = []
    role_cls = schema.FILLER_CLASS
    while i < n and tt[i] != TT.SEC and tt[i] != TT.EOS:
        ty = tt[i]
        if ty == TT.WREF:
            members.append(("W", int(wref[i])))
            role_cls = int(sem[i])
        elif ty == TT.OPEN:
            members.append(("OPEN",))
            role_cls = int(sem[i])
        elif ty == TT.SEP:
            if members:
                rooms.append(RoomEvent(members=members,
                                       role=class_role(role_cls)))
            members = []
            role_cls = schema.FILLER_CLASS
        i += 1
    if members:  # tolerate a missing trailing SEP
        rooms.append(RoomEvent(members=members, role=class_role(role_cls)))

    if i < n and tt[i] == TT.SEC:
        i += 1

    # -- openings: WREF INTERVAL pairs until EOS ------------------------
    while i < n and tt[i] != TT.EOS:
        if tt[i] == TT.WREF and i + 1 < n and tt[i + 1] == TT.INTERVAL:
            widx = int(wref[i])
            t_center = float(coord[i + 1][0])
            w_over_l = float(coord[i + 1][1])
            L = lengths[widx] if 0 <= widx < len(lengths) else frame
            kind = class_opening_kind(int(sem[i + 1]))
            openings.append(OpeningEvent(widx, t_center, w_over_l * L, kind))
            i += 2
        else:
            i += 1

    return SceneEvents(walls=walls, rooms=rooms, openings=openings, frame=frame)


def events_to_plan(ev: SceneEvents) -> FloorPlan:
    frame = ev.frame
    tol = frame * 5e-4

    walls = [Wall(id=f"wall_{i + 1}", start=w.start, end=w.end,
                  thickness=w.thickness, curvature=w.curvature, type=w.wtype)
             for i, w in enumerate(ev.walls)]

    for o in ev.openings:
        if not (0 <= o.wall_index < len(walls)):
            continue
        w = walls[o.wall_index]
        cx = w.start[0] + (w.end[0] - w.start[0]) * o.t_center
        cy = w.start[1] + (w.end[1] - w.start[1]) * o.t_center
        w.openings.append(Opening(kind=o.kind, center=(cx, cy),
                                  width=o.width, wall_id=w.id))

    rooms: list[Room] = []
    for r in ev.rooms:
        segs: list[tuple] = []
        open_before: list[bool] = []
        pending_open = False
        wall_ids: list[str] = []
        for m in r.members:
            if m[0] == "OPEN":
                pending_open = True
                continue
            idx = m[1]
            if not (0 <= idx < len(walls)):
                pending_open = False
                continue
            segs.append((walls[idx].start, walls[idx].end))
            open_before.append(pending_open)
            wall_ids.append(f"wall_{idx + 1}")
            pending_open = False
        # a trailing OPEN wraps to precede the first wall
        if pending_open and open_before:
            open_before[0] = True

        poly = None
        if len(segs) >= 2:
            if any(open_before):
                poly = geometry.polygon_from_cycle(segs, open_before, tol)
            else:
                poly = geometry.chain_closed(segs, tol) \
                    or geometry.polygon_from_cycle(segs, open_before, tol)
        rooms.append(Room(label="", wall_ids=wall_ids,
                          polygon=poly if poly and len(poly) >= 3 else None))

    return FloorPlan(walls=walls, rooms=rooms, frame=frame)
