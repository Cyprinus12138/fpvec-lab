"""fpseq — torch-free wall-first ScenePlan sequence codec (E3 M1).

Round-trip: fpeval FloorPlan <-> events <-> flat tensors. Shipped here because the
skv4 / ResPlan-FP converters use its capacity gate (SEQ_LEN / MAX_WALLS) when
accepting plans; the paper's wall-first model consumes these tensors.
"""

from .schema import (SceneEvents, OpeningEvent, RoomEvent, WallEvent, TT,
                     SEMANTIC_CLASSES, VOCAB_SIZE, MAX_WALLS, NUM_BINS,
                     ROOM_ROLES, WALL_TYPES, OPENING_KINDS)
from .encode import plan_to_events, events_to_tensors
from .decode import tensors_to_events, events_to_plan

__all__ = [
    "SceneEvents", "OpeningEvent", "RoomEvent", "WallEvent", "TT",
    "SEMANTIC_CLASSES", "VOCAB_SIZE", "MAX_WALLS", "NUM_BINS",
    "ROOM_ROLES", "WALL_TYPES", "OPENING_KINDS",
    "plan_to_events", "events_to_tensors",
    "tensors_to_events", "events_to_plan",
]
