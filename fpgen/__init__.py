"""fpgen — synthetic floorplan training-data engine.

Public API: GenConfig, generate. CLI: python -m fpgen --help
"""

from .pipeline import GenConfig, generate

__all__ = ["GenConfig", "generate"]
