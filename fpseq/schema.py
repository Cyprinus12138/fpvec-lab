"""fpseq — wall-first ScenePlan sequence codec: schema, constants, grammar.

This module is the single
source of truth for the token vocabulary, the 8 structural token *types*, the
semantic class layout, and the legal-transition grammar. It is pure-Python
(no numpy needed here, no torch, no skia) so it imports cheaply everywhere.

The sequence a wall-first model emits (upstream Raster2Seq poly2seq machinery,
forked) is a flat token stream in three dependency-ordered sections::

    <bos> ( COORD_start COORD_end ATTR )^W        # walls, canonical sort
    SEC   ( (WREF | OPEN)+ SEP )^R                 # rooms: wall-index cycles + open edges
    SEC   ( WREF INTERVAL )^O                      # openings: wall-ref + interval + kind
    EOS

Geometry is carried by a *continuous* 2-D regression head (the r2s
``coords_embed`` design — discrete bins feed only the input embedding). The
codec therefore stores continuous, frame-normalized values so encode/decode is
lossless to float precision; discrete bins are computed alongside for the model
input path but are never used to reconstruct geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

# --------------------------------------------------------------------------
# Structural token types — the 8-way `class_embed` head. Upstream ids 0–2
# (COORD, SEP, EOS) are preserved so forked weights line up.
# --------------------------------------------------------------------------

class TT(IntEnum):
    COORD = 0     # a vertex: carries a normalized (x, y) 2-vector
    SEP = 1       # end of one room's cycle
    EOS = 2       # end of sequence
    ATTR = 3      # wall attributes: carries (thickness_n, curvature_n) 2-vector
    WREF = 4      # reference to a wall by canonical index (classification)
    OPEN = 5      # an open (non-wall) room-boundary edge; carries no coordinates
    SEC = 6       # section separator (walls|rooms, rooms|openings)
    INTERVAL = 7  # opening interval: carries (t_center, width_over_len) 2-vector


NUM_TYPES = 8

# Positions whose 2-D regression head is supervised (L1 target present).
COORD_TYPES = frozenset({TT.COORD, TT.ATTR, TT.INTERVAL})

# --------------------------------------------------------------------------
# Geometry frame + head sizing
# --------------------------------------------------------------------------

NUM_BINS = 32           # discrete input-embedding grid per axis (r2s default)
MAX_WALLS = 80          # WREF classification range + run10k worst case (max 60)
SEQ_LEN = 512           # worst-case ~340 tokens fits
DEFAULT_FRAME = 1024.0  # longer-edge canonical grid (fpeval GT frame)

# Length-channel pre-scales (applied on top of frame normalization so the
# regressed 2-vectors sit in a comparable dynamic range to coords; see spec).
THICK_SCALE = 8.0
CURV_SCALE = 4.0

# --------------------------------------------------------------------------
# Semantic classes — one per-token CE head (non-gating; printed labels are
# left to OCR). Layout: [room roles ... | wall types ... | opening kinds ... | filler].
# ROOM_ROLES is frozen from fpgen's printed-label vocabulary (fpgen.vocab.LABELS
# keys) — the guaranteed superset of Room.role, since labeling is keyed by role.
# A drift guard in tests/test_fpseq_roundtrip.py asserts this stays in sync.
# --------------------------------------------------------------------------

ROOM_ROLES: tuple[str, ...] = (
    "aircon_ledge", "balcony", "bath", "bath_master", "bedroom",
    "bedroom_junior", "bedroom_master", "corridor", "db", "dining",
    "foyer", "household_shelter", "kitchen", "living", "living_dining",
    "planter", "staircase", "store", "study", "utility",
    "void", "wc", "wic", "yard",
)
WALL_TYPES: tuple[str, ...] = ("exterior", "interior", "partition", "shelter")
OPENING_KINDS: tuple[str, ...] = ("door", "sliding_door", "opening", "window")

_ROOM_BASE = 0
_WALL_BASE = _ROOM_BASE + len(ROOM_ROLES)
_OPEN_BASE = _WALL_BASE + len(WALL_TYPES)
FILLER_CLASS = _OPEN_BASE + len(OPENING_KINDS)
SEMANTIC_CLASSES = FILLER_CLASS + 1   # 24 + 4 + 4 + 1 = 33


def role_class(role: str) -> int:
    """Room role -> semantic class id (filler if unknown)."""
    try:
        return _ROOM_BASE + ROOM_ROLES.index(role)
    except ValueError:
        return FILLER_CLASS


def class_role(cls: int) -> str:
    i = cls - _ROOM_BASE
    return ROOM_ROLES[i] if 0 <= i < len(ROOM_ROLES) else ""


def wall_type_class(t: str) -> int:
    try:
        return _WALL_BASE + WALL_TYPES.index(t)
    except ValueError:
        return _WALL_BASE + WALL_TYPES.index("interior")


def class_wall_type(cls: int) -> str:
    i = cls - _WALL_BASE
    return WALL_TYPES[i] if 0 <= i < len(WALL_TYPES) else "interior"


def opening_kind_class(k: str) -> int:
    try:
        return _OPEN_BASE + OPENING_KINDS.index(k)
    except ValueError:
        return _OPEN_BASE + OPENING_KINDS.index("opening")


def class_opening_kind(cls: int) -> str:
    i = cls - _OPEN_BASE
    return OPENING_KINDS[i] if 0 <= i < len(OPENING_KINDS) else "opening"

# --------------------------------------------------------------------------
# Input vocabulary (single-stream token ids for the decoder input embedding).
# Specials + WREF ids occupy one slot each; COORD tokens map to bin ids via the
# 4-corner bilinear blend upstream. Layout:
#   [0 .. NUM_BINS^2)              coord bins (input-embedding only)
#   BOS, EOS, SEP, PAD, SEC, OPEN  specials
#   [WREF_BASE .. WREF_BASE+MAX_WALLS)   wall-reference ids
# ATTR / INTERVAL reuse a special-free slot distinguished by `type_embed`
# (an Embedding(NUM_TYPES, d) added at every position); see spec.
# --------------------------------------------------------------------------

_BIN_SLOTS = NUM_BINS * NUM_BINS
BOS_ID = _BIN_SLOTS + 0
EOS_ID = _BIN_SLOTS + 1
SEP_ID = _BIN_SLOTS + 2
PAD_ID = _BIN_SLOTS + 3
SEC_ID = _BIN_SLOTS + 4
OPEN_ID = _BIN_SLOTS + 5
ATTR_ID = _BIN_SLOTS + 6
INTERVAL_ID = _BIN_SLOTS + 7
WREF_BASE = _BIN_SLOTS + 8
VOCAB_SIZE = WREF_BASE + MAX_WALLS   # 1024 + 8 + 80 = 1112

# --------------------------------------------------------------------------
# Grammar: legal next-type transitions. Used for the day-1 grammar-masked
# decoder (every emitted sequence parses) and to validate encoded streams.
# `None` key = the start state (immediately after <bos>).
# --------------------------------------------------------------------------

LEGAL_NEXT: dict[TT | None, frozenset] = {
    None:          frozenset({TT.COORD, TT.SEC}),          # first wall, or empty→sec
    TT.COORD:      frozenset({TT.COORD, TT.ATTR}),         # start→end→(attr)
    TT.ATTR:       frozenset({TT.COORD, TT.SEC}),          # next wall, or end walls
    TT.SEC:        frozenset({TT.WREF, TT.OPEN, TT.SEC, TT.EOS}),
    TT.WREF:       frozenset({TT.WREF, TT.OPEN, TT.SEP, TT.INTERVAL}),
    TT.OPEN:       frozenset({TT.WREF, TT.SEP}),           # no two consecutive OPEN
    TT.INTERVAL:   frozenset({TT.WREF, TT.EOS}),           # opening pairs, then done
    TT.SEP:        frozenset({TT.WREF, TT.OPEN, TT.SEC}),  # next room, or end rooms
    TT.EOS:        frozenset(),
}

# --------------------------------------------------------------------------
# Event layer — the human-inspectable structured form. Points and lengths are in FRAME coordinates (not normalized); the tensor
# layer applies frame normalization. plan_to_events/events_to_plan operate here.
# --------------------------------------------------------------------------

Point = tuple[float, float]


@dataclass
class WallEvent:
    start: Point
    end: Point
    thickness: float
    curvature: float
    wtype: str                  # WALL_TYPES member


@dataclass
class RoomEvent:
    # members are ("W", wall_index) or ("OPEN",), in canonical cycle order
    members: list[tuple]
    role: str = ""              # ROOM_ROLES member (or "" -> filler); side-channel


@dataclass
class OpeningEvent:
    wall_index: int
    t_center: float             # in [0, 1] along the referenced wall chord
    width: float                # FRAME units
    kind: str                   # OPENING_KINDS member


@dataclass
class SceneEvents:
    walls: list[WallEvent] = field(default_factory=list)
    rooms: list[RoomEvent] = field(default_factory=list)
    openings: list[OpeningEvent] = field(default_factory=list)
    frame: float = DEFAULT_FRAME
