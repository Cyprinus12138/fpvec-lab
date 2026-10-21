# fpeval test fixtures

Hand-authored plan JSONs of one real marketing floor plan (24 walls).
`manual_ballota_v5.dsl.json` is the final manual vectorization;
`manual_ballota_v4.dsl.json` is an earlier pass that still contains phantom
walls (thin furniture outlines taken for structure), kept so the tests can
exercise additive-error scoring. `manual_v5_overlay.jpg` overlays v5 on the
source image.
