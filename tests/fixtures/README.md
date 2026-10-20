# Manual vectorization study (Ballota Type C)

Hand-authored ground-truth-quality DSLs from the recognition ceiling study.
`manual_ballota_v5.dsl.json` is the best manual result (24 walls) after three
rounds of self-review; `v4` kept for the diff. `manual_v5_overlay.jpg` shows it.

Core lesson (cost three phantom-wall bugs to learn): **a line is only a wall if
it has a wall band (thickness). Thin outlines are furniture** — classify them
(bookshelf / wardrobe / counter) and emit Item nodes, never structure. When two
parallel lines flank a space nobody can stand in, it's a built-in, not a
corridor. Errors from priors are consistently *additive* (inventing structure);
the recognition prompt must classify ambiguous symbols BEFORE deriving walls.
