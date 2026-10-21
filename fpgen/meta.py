"""Unit-meta sampling: what kind of unit are we generating?

Draws (typology, n_bed, n_bath, area, feature flags) from distributions fitted
to the SG corpus title extractions (assets/meta_stats.json, produced by
scripts/fit_meta.py). A hardcoded fallback ships in code so fpgen runs without
the listing-title corpus (not redistributed; see scripts/fit_meta.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_STATS_PATH = Path(__file__).parent / "assets" / "meta_stats.json"

# Plausible total_area_sqm bands by bedroom count.
# Corpus-sanity bands from the SG listing corpus, not regulation.
AREA_BANDS_SQM: dict[int, tuple[float, float]] = {
    0: (25, 60),
    1: (30, 80),
    2: (50, 130),
    3: (75, 200),
    4: (110, 350),
    5: (160, 500),
}

# Fallback joint distribution P(n_bed, n_bath) for condo_apartment, roughly
# matching SG new-launch unit mixes (used only if meta_stats.json is missing).
_FALLBACK_BED_BATH: dict[tuple[int, int], float] = {
    (0, 1): 0.06,   # studio
    (1, 1): 0.18,
    (2, 1): 0.14,
    (2, 2): 0.24,
    (3, 2): 0.26,
    (3, 3): 0.04,
    (4, 3): 0.08,
}


@dataclass
class UnitMeta:
    property_type: str  # condo_apartment | landed_house (only condo is generated)
    n_bed: int          # 0 = studio
    n_bath: int
    target_area_sqm: float
    flags: dict = field(default_factory=dict)

    @property
    def typology(self) -> str:
        return "studio" if self.n_bed == 0 else f"{self.n_bed}B{self.n_bath}b"


def _load_stats() -> dict[tuple[int, int], float] | None:
    if not _STATS_PATH.exists():
        return None
    try:
        doc = json.loads(_STATS_PATH.read_text())
        pairs = {(int(k.split(",")[0]), int(k.split(",")[1])): float(v)
                 for k, v in doc["bed_bath"].items()}
        total = sum(pairs.values())
        return {k: v / total for k, v in pairs.items()} if total > 0 else None
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def sample(rng: np.random.Generator, typology: str | None = None) -> UnitMeta:
    """Draw a unit meta. `typology` (e.g. "2B2b" / "studio") pins bed/bath."""
    dist = _load_stats() or _FALLBACK_BED_BATH
    if typology is not None:
        if typology == "studio":
            n_bed, n_bath = 0, 1
        else:
            n_bed, n_bath = int(typology[0]), int(typology[2])
    else:
        keys = sorted(dist)
        probs = np.array([dist[k] for k in keys])
        n_bed, n_bath = keys[rng.choice(len(keys), p=probs / probs.sum())]

    lo, hi = AREA_BANDS_SQM.get(n_bed, (75, 200))
    # Triangular skewed toward the lower-middle of the band (marketing units
    # are efficiency-optimized, big penthouses are rare).
    area = float(rng.triangular(lo, lo + (hi - lo) * 0.35, hi))

    flags = {
        "balcony": bool(rng.random() < 0.7),
        "yard": bool(rng.random() < (0.6 if n_bed >= 2 else 0.25)),
        "household_shelter": bool(rng.random() < 0.75),
        "ac_ledge": bool(rng.random() < 0.85),
        "study": bool(n_bed >= 2 and rng.random() < 0.2),
        "wic": bool(n_bed >= 3 and rng.random() < 0.5),
        "store": bool(rng.random() < 0.3),
    }
    return UnitMeta(
        property_type="condo_apartment",
        n_bed=n_bed,
        n_bath=n_bath,
        target_area_sqm=round(area, 1),
        flags=flags,
    )
