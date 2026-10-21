"""fpeval — geometric verifier + eval harness for floorplan vectorization.

Triple duty:
  1. RL reward function (verifiable geometric rewards, FloorplanVLM-style gating)
  2. automatic QA filter for training data
  3. evaluation harness (validity, Ext-IoU, Room-IoU, Room-F1@0.5, Opening-F1, Wall-F1)
"""

from .model import FloorPlan, Wall, Opening, Room, from_vlm_json
from .verify import verify, VerifyResult
from .metrics import evaluate, alpha_gate, EvalReport

__all__ = [
    "FloorPlan", "Wall", "Opening", "Room",
    "from_vlm_json",
    "verify", "VerifyResult",
    "evaluate", "alpha_gate", "EvalReport",
]
