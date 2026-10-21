# skv4 — corrected CubiCasa5K annotations and their converter

Non-commercial: the vendored CubiCasa5K parser (`svg_utils.py`) is CC BY-NC 4.0
and the CubiCasa5K data is CC BY-NC-SA 4.0. The released annotations
(`Tristan12138/cubicasa5k-skv4` on the Hub) inherit CC BY-NC-SA 4.0.

Rebuild from the source release:

```bash
pip install -e ".[skv4]"
# CubiCasa5K: https://zenodo.org/record/2613548 -> unzip to <cc5k>/ (contains train.txt val.txt test.txt + sample dirs)
python -m skv4.convert --svg-dir <cc5k> --out out/skv4_raw            # add --shard i --nshards n to parallelise
python -m skv4.split --converted out/skv4_raw --cc5k-root <cc5k> --out out/skv4 --copy-images
```

`convert` parses each `model.svg` (wall centerlines from the vendored parser;
rooms and openings from our own lean SVG walk), welds wall endpoints into a
planar graph, takes rooms as the faces of that graph, projects openings onto
their host walls, rescales to the 1024 longer-edge frame, then applies the two
skv4 gates: the **ink gate** (drop a GT wall whose centerline shows under 10%
dark pixels outside its declared openings) and the **spur prune** (drop
free-ended stubs shorter than twice the wall thickness). Each plan must pass
`fpeval.verify` and the sequence-capacity gate. `split` repartitions by the
official CubiCasa5K lists: 3,328 / 318 / 296 of the 3,942 plans that convert.
