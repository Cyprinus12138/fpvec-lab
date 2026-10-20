"""SG printed-label vocabulary and text formatting.

Label spellings vendored from the PropWise corpus vocabulary
(scripts/floorplan-extractor/dq_profile.py ROOM_WORDS) plus common marketing
variants seen in the SG corpus. Keyed by room role; each entry is a list of
(label, weight) sampled per plan so different plans use different conventions.
"""

from __future__ import annotations

import numpy as np

# role -> [(printed label, weight)]
LABELS: dict[str, list[tuple[str, float]]] = {
    "living": [("LIVING", 0.6), ("LIVING ROOM", 0.4)],
    "dining": [("DINING", 0.7), ("DINING ROOM", 0.3)],
    "living_dining": [("LIVING/DINING", 0.5), ("LIVING & DINING", 0.3), ("LIVING", 0.2)],
    "bedroom_master": [("MASTER BEDROOM", 0.8), ("MASTER BR", 0.1), ("MBR", 0.1)],
    "bedroom_junior": [("JUNIOR MASTER", 0.6), ("JR MASTER BEDROOM", 0.4)],
    "bedroom": [("BEDROOM {n}", 0.7), ("BR {n}", 0.15), ("BEDROOM {n}", 0.15)],
    "bath_master": [("MASTER BATH", 0.5), ("MASTER BATHROOM", 0.3), ("BATH 1", 0.2)],
    "bath": [("BATH {n}", 0.5), ("BATHROOM {n}", 0.3), ("BATH", 0.2)],
    "wc": [("WC", 0.8), ("POWDER ROOM", 0.2)],
    "kitchen": [("KITCHEN", 0.85), ("DRY KITCHEN", 0.1), ("WET KITCHEN", 0.05)],
    "yard": [("YARD", 0.5), ("SERVICE YARD", 0.5)],
    "utility": [("UTILITY", 0.9), ("UTILITY ROOM", 0.1)],
    "store": [("STORE", 0.8), ("STORAGE", 0.2)],
    "household_shelter": [("HOUSEHOLD SHELTER", 0.6), ("HS", 0.3), ("SHELTER", 0.1)],
    "aircon_ledge": [("A/C LEDGE", 0.7), ("AC LEDGE", 0.2), ("A/C", 0.1)],
    "balcony": [("BALCONY", 0.9), ("PES", 0.1)],
    "foyer": [("FOYER", 0.7), ("ENTRANCE", 0.15), ("PRIVATE LIFT LOBBY", 0.15)],
    "corridor": [("", 0.8), ("HALLWAY", 0.2)],  # corridors usually unlabeled
    "study": [("STUDY", 0.9), ("STUDY ROOM", 0.1)],
    "wic": [("WIC", 0.5), ("WALK-IN WARDROBE", 0.5)],
    "planter": [("PLANTER", 1.0)],
    "void": [("VOID", 1.0)],
    "staircase": [("", 1.0)],
    "db": [("DB", 1.0)],
}


def pick_label(rng: np.random.Generator, role: str, index: int | None = None) -> str:
    """Sample a printed label for a room role. `index` fills BEDROOM {n} slots."""
    options = LABELS.get(role)
    if not options:
        return role.upper().replace("_", " ")
    weights = np.array([w for _, w in options])
    label = options[rng.choice(len(options), p=weights / weights.sum())][0]
    if "{n}" in label:
        label = label.format(n=index if index is not None else 2)
    return label


def area_text(rng: np.random.Generator, area_sqm: float) -> str:
    style = rng.choice(["sqm", "sqm_short", "sqft"], p=[0.5, 0.3, 0.2])
    if style == "sqm":
        return f"{area_sqm:.1f} m²"
    if style == "sqm_short":
        return f"{area_sqm:.0f} sqm"
    return f"{area_sqm * 10.7639:.0f} sq ft"


def dim_text(length_mm: float) -> str:
    return f"{length_mm / 1000:.2f}".rstrip("0").rstrip(".") + " m"
