"""ResPlan-FP: a clean vector-render floorplan benchmark derived from ResPlan.

* ``convert``  — ResPlan.pkl -> wall-first floorplan JSON (one row per plan)
* ``render``   — the ``clean_band_v1`` render protocol (256 px, solid wall band,
                 door arcs, window gaps)
* ``build``    — convert -> freeze splits (seed 42, fingerprinted) -> render
* ``splits.json`` — the frozen split fingerprints + the full test-id list
"""
