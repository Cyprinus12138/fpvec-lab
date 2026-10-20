"""fpgen — synthetic floorplan training-data engine (research-plan §4 E2).

Public API: GenConfig, generate. CLI: python -m fpgen --help
"""

from .pipeline import GenConfig, generate

__all__ = ["GenConfig", "generate"]
