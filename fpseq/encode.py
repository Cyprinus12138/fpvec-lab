"""Encode: fpeval FloorPlan -> SceneEvents -> flat tensor dict.

`plan_to_events` is authoritative for the canonical target sequence; it takes an
fpeval FloorPlan (post-H, the scoring frame) and, optionally, a per-room role
list (side-channel — role is dropped by to_vlm_json, so training-data build
supplies it from the ScenePlan; None -> filler class). `events_to_tensors`
lowers events to the flat, frame-normalized arrays a wall-first model consumes.
"""

from __future__ import annotations

import math

import numpy as np

from fpeval.model import FloorPlan

from . import canonical, geometry, schema
from .schema import (SceneEvents, OpeningEvent, RoomEvent, TT, WallEvent,
                     opening_kind_class, role_class, wall_type_class)


def _project_t(center, start, end) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return 0.0
    t = ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / L2
    return min(max(t, 0.0), 1.0)


def plan_to_events(plan: FloorPlan, roles: list[str] | None = None) -> SceneEvents:
    walls = canonical.canonical_walls(plan)
    cid = canonical.canonical_index(walls)
    frame = plan.frame
    tol = frame * 1e-3

    wall_events = [WallEvent(start=w.start, end=w.end, thickness=w.thickness,
                             curvature=w.curvature, wtype=w.type) for w in walls]

    # rooms -------------------------------------------------------------
    room_events: list[RoomEvent] = []
    for ri, r in enumerate(plan.rooms):
        wids = [wid for wid in r.wall_ids if wid in cid]
        if len(wids) < 2 or not r.polygon or len(r.polygon) < 3:
            continue
        segs = [(walls[cid[wid]].start, walls[cid[wid]].end) for wid in wids]
        order, open_before = geometry.ring_order_from_polygon(
            list(r.polygon), segs, tol)
        members: list[tuple] = []
        for pos, k in enumerate(order):
            if open_before[pos]:
                members.append(("OPEN",))
            members.append(("W", cid[wids[k]]))
        members = canonical.rotate_cycle_to_min(members)
        role = roles[ri] if roles is not None and ri < len(roles) else ""
        room_events.append(RoomEvent(members=members, role=role))

    # openings ----------------------------------------------------------
    opening_events: list[OpeningEvent] = []
    for idx, w in enumerate(walls):
        for o in w.openings:
            t = _project_t(o.center, w.start, w.end)
            opening_events.append(OpeningEvent(
                wall_index=idx, t_center=t, width=o.width, kind=o.kind))
    opening_events.sort(key=lambda o: (o.wall_index, o.t_center))

    return SceneEvents(walls=wall_events, rooms=room_events,
                       openings=opening_events, frame=frame)


def events_to_tensors(ev: SceneEvents) -> dict:
    frame = ev.frame
    tt: list[int] = []
    coord: list[list[float]] = []
    coord_mask: list[bool] = []
    wref: list[int] = []
    sem: list[int] = []
    input_ids: list[int] = []
    bins: list[list[int]] = []

    lengths = [math.dist(w.start, w.end) for w in ev.walls]

    def push(type_id, xy=None, wref_idx=-1, sem_cls=schema.FILLER_CLASS,
             in_id=None):
        tt.append(int(type_id))
        if xy is None:
            coord.append([0.0, 0.0]); coord_mask.append(False)
            bins.append([-1, -1])
        else:
            coord.append([float(xy[0]), float(xy[1])]); coord_mask.append(True)
            bx = min(max(int(xy[0] * schema.NUM_BINS), 0), schema.NUM_BINS - 1)
            by = min(max(int(xy[1] * schema.NUM_BINS), 0), schema.NUM_BINS - 1)
            bins.append([bx, by])
        wref.append(int(wref_idx))
        sem.append(int(sem_cls))
        input_ids.append(int(in_id if in_id is not None else 0))

    def coord_id(xy):
        bx = min(max(int(xy[0] * schema.NUM_BINS), 0), schema.NUM_BINS - 1)
        by = min(max(int(xy[1] * schema.NUM_BINS), 0), schema.NUM_BINS - 1)
        return by * schema.NUM_BINS + bx

    # walls
    for w in ev.walls:
        sxy = (w.start[0] / frame, w.start[1] / frame)
        exy = (w.end[0] / frame, w.end[1] / frame)
        push(TT.COORD, sxy, in_id=coord_id(sxy))
        push(TT.COORD, exy, in_id=coord_id(exy))
        attr = (w.thickness / frame * schema.THICK_SCALE,
                w.curvature / frame * schema.CURV_SCALE)
        push(TT.ATTR, attr, sem_cls=wall_type_class(w.wtype),
             in_id=schema.ATTR_ID)

    # section break: walls | rooms
    push(TT.SEC, in_id=schema.SEC_ID)

    # rooms
    for r in ev.rooms:
        rc = role_class(r.role)
        for m in r.members:
            if m[0] == "W":
                push(TT.WREF, wref_idx=m[1], sem_cls=rc,
                     in_id=schema.WREF_BASE + m[1])
            else:
                push(TT.OPEN, sem_cls=rc, in_id=schema.OPEN_ID)
        push(TT.SEP, in_id=schema.SEP_ID)

    # section break: rooms | openings
    push(TT.SEC, in_id=schema.SEC_ID)

    # openings
    for o in ev.openings:
        L = lengths[o.wall_index] if o.wall_index < len(lengths) else frame
        w_over_l = o.width / L if L > 1e-9 else 0.0
        push(TT.WREF, wref_idx=o.wall_index, sem_cls=opening_kind_class(o.kind),
             in_id=schema.WREF_BASE + o.wall_index)
        push(TT.INTERVAL, (o.t_center, w_over_l),
             sem_cls=opening_kind_class(o.kind), in_id=schema.INTERVAL_ID)

    push(TT.EOS, in_id=schema.EOS_ID)

    return {
        "token_type": np.array(tt, dtype=np.int64),
        "coord": np.array(coord, dtype=np.float32),
        "coord_mask": np.array(coord_mask, dtype=bool),
        "wref": np.array(wref, dtype=np.int64),
        "sem": np.array(sem, dtype=np.int64),
        "input_ids": np.array(input_ids, dtype=np.int64),
        "bins": np.array(bins, dtype=np.int64),
        "frame": float(frame),
    }
