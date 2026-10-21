"""Style-family parameter sampling.

A StyleParams is drawn once per image; families are priors over the same
parameter space: clean_cad, marketing, brochure, hand_drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StyleParams:
    family: str
    bg_color: tuple[int, int, int] = (255, 255, 255)
    wall_fill: tuple[int, int, int] = (20, 20, 20)
    wall_render: str = "solid"  # solid | hollow
    wall_outline_mm: float = 0.0  # >0: stroke wall band outline
    stroke_color: tuple[int, int, int] = (30, 30, 30)
    furniture_stroke_mm: float = 22.0
    opening_stroke_mm: float = 18.0
    floor_fills: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    font_name: str = "DejaVu LGC Sans"
    label_font_mm: float = 260.0
    area_font_mm: float = 190.0
    show_area: bool = True
    window_lines: int = 2
    detail: str = "outline"  # outline | rich (marketing)
    show_dims: bool = False       # dimension chains along two envelope sides
    show_scalebar: bool = False
    dim_font_mm: float = 170.0
    stroke_jitter_mm: float = 0.0  # hand_drawn: per-vertex stroke wobble
    title_text: str | None = None  # brochure: project/unit title line
    draw_soft_furniture: bool = True  # line_cad bare skeleton: drop movable furniture


# soft floor fills for the marketing family; clean_cad uses none
_MARKETING_FLOORS = {
    "living_dining": (245, 238, 226), "living": (245, 238, 226),
    "dining": (245, 238, 226), "bedroom": (240, 231, 219),
    "kitchen": (228, 232, 236), "bath": (219, 231, 235), "wc": (219, 231, 235),
    "balcony": (226, 238, 226), "yard": (232, 236, 232),
    "household_shelter": (212, 207, 198), "aircon_ledge": (220, 220, 220),
    "store": (230, 226, 220), "wic": (240, 231, 219), "study": (240, 235, 224),
    "corridor": (246, 243, 238), "foyer": (246, 243, 238),
    "utility": (232, 236, 232),
}


def sample_style(rng: np.random.Generator, family: str = "clean_cad") -> StyleParams:
    if family == "clean_cad":
        gray = int(rng.integers(0, 40))
        return StyleParams(
            family=family,
            wall_fill=(gray, gray, gray),
            wall_render="solid" if rng.random() < 0.8 else "hollow",
            furniture_stroke_mm=float(rng.uniform(16, 28)),
            opening_stroke_mm=float(rng.uniform(14, 24)),
            label_font_mm=float(rng.uniform(220, 320)),
            area_font_mm=float(rng.uniform(160, 220)),
            show_area=bool(rng.random() < 0.7),
            window_lines=int(rng.choice([1, 2, 3], p=[0.2, 0.55, 0.25])),
            show_dims=bool(rng.random() < 0.65),
            show_scalebar=bool(rng.random() < 0.4),
            dim_font_mm=float(rng.uniform(140, 200)),
        )
    if family == "line_cad":
        # Austere residential-CAD skeleton (the cc5k-like sub-genre, but the point
        # is invariance training, not cc5k mimicry): bold black walls, no floor
        # fills, no dims/scalebar/title, built-in fixtures kept, soft furniture
        # dropped. No spurious context cue -> corners must be read from geometry.
        gray = int(rng.integers(0, 25))
        return StyleParams(
            family=family,
            wall_fill=(gray, gray, gray),
            wall_render="solid" if rng.random() < 0.85 else "hollow",
            opening_stroke_mm=float(rng.uniform(14, 22)),
            label_font_mm=float(rng.uniform(220, 320)),
            show_area=bool(rng.random() < 0.25),
            window_lines=int(rng.choice([1, 2], p=[0.35, 0.65])),
            show_dims=False,
            show_scalebar=False,
            draw_soft_furniture=False,
        )
    if family == "marketing":
        wall_gray = int(rng.integers(10, 70))
        return StyleParams(
            family=family,
            bg_color=(255, 255, 255) if rng.random() < 0.7 else (250, 247, 240),
            wall_fill=(wall_gray, wall_gray, wall_gray),
            wall_render="solid",
            floor_fills=dict(_MARKETING_FLOORS),
            furniture_stroke_mm=float(rng.uniform(18, 30)),
            opening_stroke_mm=float(rng.uniform(16, 26)),
            label_font_mm=float(rng.uniform(240, 340)),
            area_font_mm=float(rng.uniform(170, 230)),
            show_area=True,
            window_lines=int(rng.choice([2, 3])),
            detail="rich",
            show_dims=bool(rng.random() < 0.35),
            show_scalebar=bool(rng.random() < 0.25),
            dim_font_mm=float(rng.uniform(150, 210)),
        )
    if family == "brochure":
        style = sample_style(rng, "marketing")
        style.family = "brochure"
        tint = int(rng.integers(238, 252))
        style.bg_color = (tint, tint - int(rng.integers(0, 8)),
                          tint - int(rng.integers(2, 14)))
        project = str(rng.choice(["THE ORIE", "LENTOR MODERN", "GRAND DUNMAN",
                                  "PINETREE HILL", "THE CONTINUUM",
                                  "TEMBUSU GRAND", "J'DEN RESIDENCES"]))
        # "{n_bed}" placeholder is filled by the pipeline from the actual unit
        # meta so the title never contradicts the plan
        style.title_text = f"{project}  ·  {{n_bed}} BEDROOM  ·  TYPE {chr(65 + int(rng.integers(0, 5)))}{int(rng.integers(1, 9))}"
        return style
    if family == "hand_drawn":
        style = sample_style(rng, "clean_cad")
        style.family = "hand_drawn"
        style.wall_render = "solid" if rng.random() < 0.5 else "hollow"
        style.stroke_jitter_mm = float(rng.uniform(12, 32))
        style.furniture_stroke_mm = float(rng.uniform(20, 34))
        style.show_dims = bool(rng.random() < 0.3)
        style.show_area = bool(rng.random() < 0.4)
        return style
    raise ValueError(f"unknown style family: {family}")
