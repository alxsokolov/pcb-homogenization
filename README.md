# pcb_homogenization

PCB copper-layout homogenization and solder-joint thermal-cycling simulation, developed in the MikroVAL project.

The tool takes a 2-layer PCB (Gerber), rasterizes its copper, and coarse-grains it with a quadtree that applies per-leaf Voigt-Reuss homogenization. It then builds a board + CR0805 + SAC305 solder assembly in ANSYS MAPDL (via PyMAPDL), runs a thermal-cycling profile with creep, and post-processes the solder creep strain per cycle for fatigue-life evaluation.

## Pipeline

```
Gerber ──► import_real_board.py
       ──► discretize.py            copper rasters      → <sample>/mesh/
       ──► quadtree.py              Voigt-Reuss leaves  → <sample>/mesh/quadtree.json
       ──► simulate_pyansys_assembly.py   MAPDL solve   → <sample>/results/pyansys_assembly.rst
                                          (+ thermal_profile.yaml/.png: the applied load)
       ──► zone_creep_strain.py ──► plot_cycle_creep_increment.py   per-cycle Δε_creep
```

Supporting scripts:
- `simulate_pyansys_benchmark.py`: full-resolution board-only benchmark.
- `solder_reliability.py`: analytical CTE-mismatch estimate. It also defines the tracked pad coordinates.
- `plot_board_model.py` (top view of the board as simulated, read from the `.rst`), `plot_creep_strain_history.py`, `plot_component_materials.py`: plots.
- `plot_comparison_creep.py`: overlays zone-averaged creep history and per-cycle Δε across several `--board-model` variants of the same board (`results/` JSON only, no `.rst` needed).

The library modules are `homogenization.py`, `component_import.py` and `thermal_profile.py`.

`docs/workflow_overview.pptx` gives an overview of the workflow.

## n-layer pipeline

A separate set of scripts handles boards with more than two copper layers. The 2-layer scripts above are untouched and keep working exactly as before; nothing in the n-layer track imports them except to be checked against them.

```
ODB++  ──► import_odb_board_nlayer.py    stackup + all copper layers
Gerber ──► import_gerber_board_nlayer.py (+ stackup YAML)
       ──► discretize.py                 copper rasters, one per layer (shared, unchanged)
       ──► coarsegrain_nlayer.py         regions × materials → <sample>/mesh/coarsegrain_nlayer.json
       ──► simulate_pyansys_nlayer.py    MAPDL board solve  → <sample>/results/pyansys_nlayer_<tag>.json
```

The pipeline separates two choices that the 2-layer path mixes together:

- **How the board plane is discretized** (`coarsegrain_nlayer.py --mode`): `grid` with the FE mesh's own `--nx/--ny` (one region per element), `grid` with fewer (regions uniformly coarser than elements), or `quadtree` (adaptive, refined where the layout is heterogeneous).
- **How each region's copper becomes material** (`--region-model`): `homogenized_zlayers` (each copper layer its own Voigt-Reuss material from its own area fraction), `homogenized` (one smeared material through the full thickness), or `projection` (black/white per region and layer).

`simulate_pyansys_nlayer.py --board-model raster` skips the coarse-graining entirely and paints each element pure copper or dielectric from the layout, as the full-resolution reference to compare the others against.

The FE mesh bands z by **physical layer**: every copper and dielectric slab gets its own element rows, so no element straddles two layers and each layer's copper volume is exact. This is why the n-layer path needs no copper-"dilution" correction.

Supporting scripts:
- `plot_coarsegrain_nlayer.py`: per-layer region maps of copper fraction, and the stackup through thickness — inspect the model before solving.
- `compare_nlayer_models.py`: tabulates every solved board model against the `raster` reference.

Library modules: `odb_stackup.py` (ODB++ → physical stackup), `stackup_nlayer.py` (sample → z slabs + rasters), `homogenization_nlayer.py` (n-slab Voigt-Reuss).

`python homogenization_nlayer.py` runs a self-test that checks the n-layer formulas reduce exactly to `homogenization.py`'s on a 3-slab stackup. `stackup_nlayer.py` also reads the legacy 2-layer metadata schema, so the n-layer tools can be run on the existing 2-layer samples as a cross-check.

### Where the ODB++ stackup comes from

`odb_stackup.py` reads `matrix/matrix` for layer order and kind, each layer's `attrlist` for thicknesses, and `misc/attrlist` `.board_thickness` as a cross-check. Three decoding rules it implements, each verified against the JODY-W6 (Altium, 8 layers) and Hella (Allegro) jobs:

- `.layer_dielectric` on a **conductive** layer is the thickness of the dielectric *below* it, not that layer's copper. Copper thickness comes from `.copper_weight`.
- `.copper_weight` units vary by exporter: Altium writes µm (35.001), Allegro writes oz (1). Values above 10 are read as µm.
- The resolved total is checked against `.board_thickness`. For JODY-W6: 8 copper (0.266008 mm) + 7 dielectric (0.462 mm) = 0.728008 mm against 0.72801226 — agreement to 4 nm.

`matrix/stackup.xml` (ODB++ 8.1u3+, mainly for rigid-flex) is detected and reported but not parsed. Boards whose ODB++ carries no usable stackup — Hella has no `DIELECTRIC` rows at all — need `--stackup <yaml>` to supply the missing thicknesses.

### n-layer quickstart (8-layer ublox board)

```
python import_odb_board_nlayer.py --odb-dir path/to/extracted/job --sample-id ublox_nlayer
python discretize.py              data/samples/ublox_nlayer --resolution 20
python coarsegrain_nlayer.py      data/samples/ublox_nlayer --mode grid --nx 40 --ny 40
python plot_coarsegrain_nlayer.py data/samples/ublox_nlayer
python simulate_pyansys_nlayer.py data/samples/ublox_nlayer --nx 40 --ny 40 `
    --mapdl-dir C:\temp\mapdl_nlayer

python simulate_pyansys_nlayer.py data/samples/ublox_nlayer --nx 40 --ny 40 `
    --board-model raster --tag raster --mapdl-dir C:\temp\mapdl_nlayer
python compare_nlayer_models.py   data/samples/ublox_nlayer
```

`--dry-run` reports the mesh, material count and analytical laminate reference without launching MAPDL.

## Install

```
conda create -n pyansys python=3.14
conda activate pyansys
pip install -r requirements.txt
```

ANSYS MAPDL 2025 R2 (v251) with a valid license is needed for the `simulate_pyansys_*` scripts. The postprocessing scripts read the `.rst` directly and need no MAPDL session.

On Windows, set `PYTHONUTF8=1` (`$env:PYTHONUTF8=1` in PowerShell). The scripts print Unicode characters and otherwise crash when their output is piped or redirected.

## Data (not in git)

`data/` and `mapdl_run/` are gitignored. Partner-provided input files are distributed through the MikroVAL project share (`\\berlin\simulation\ere\MikroVAL\03_Simulationsdaten\pcb_homogenization\data`), not through this repository.

```
data/
  components/cr0805/DemonstratorB_CR0805_Export.cdb   component archive (+ MANIFEST.md)
  materials/sac305/SAC305_MikroVal.mac                solder material macro (+ MANIFEST.md)
  samples/<sample_id>/                                 one board per directory
    metadata.json, gerber/, mesh/, results/
configs/
  materials.yaml         copper / FR4 properties
  thermal_profile.yaml   reflow + -40/125 °C cycling profile
```

## Quickstart (example board `rv3032_example`)

```
python discretize.py data/samples/rv3032_example
python quadtree.py data/samples/rv3032_example --max-level 6

python thermal_profile.py configs/thermal_profile_example.yaml      # check the load before solving

python simulate_pyansys_assembly.py data/samples/rv3032_example --nx 20 --ny 20 --creep `
    --thermal-profile configs/thermal_profile_example.yaml --mapdl-dir C:\temp\mapdl_example

python zone_creep_strain.py        data/samples/rv3032_example --target-ratio 0.25
python plot_cycle_creep_increment.py data/samples/rv3032_example
python plot_board_model.py         data/samples/rv3032_example
```

`configs/thermal_profile_example.yaml` is a shortened profile for demonstration runs. It starts at the stress-free reference temperature (180 °C), cools down to 22 °C in 600 s, holds there for 300 s, then runs one −40/125 °C cycle with 300 s holds. With the command above it solves in about 8.5 min on 8 cores. For a production fatigue evaluation, use the full `configs/thermal_profile.yaml` instead. It is the default when `--thermal-profile` is omitted.

Put `--mapdl-dir` on a local disk. MAPDL writes multi-GB scratch and result files there, and a network share slows the solve down considerably.

Pass `--help` to any script to see all of its options.

To import your own board:

```
python import_real_board.py --f-cu F_Cu.gbr --b-cu B_Cu.gbr --bbox XMIN YMIN XMAX YMAX --sample-id my_board
```

Then set the tracked component pads (`PAD_1_XY_MM` / `PAD_2_XY_MM`) at the top of `solder_reliability.py`.

## Known limitations

- The 2-layer pipeline above supports only 2-layer boards (outer copper pair + dielectric core). Multilayer boards go through the separate n-layer track described above, which is board-only — no component, no solder, no creep yet.
- Via copper is not represented anywhere: both tracks model copper only in the layer planes. On a thin HDI board this matters — the JODY-W6 reference job is 0.728 mm over 8 layers with blind and stacked micro vias between every adjacent layer pair, and that via copper carries real through-thickness stiffness.
- One component (CR0805) is placed manually per run.
- TREF is 180 °C, not the 205 °C reflow peak, because the SAC305 material data only extends to 180 °C (see `configs/thermal_profile.yaml`).
- Gerber arcs (G02/G03) are approximated as straight strokes.

## Credits

`zone_creep_strain.py` ports R. Schwerz's (Fraunhofer IKTS) APDL post-processing macro (POST_V3a_fix5). It is based on Darveaux's volume-weighted averaging method.
