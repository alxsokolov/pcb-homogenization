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

- Only 2-layer boards are supported (outer copper pair + dielectric core).
- One component (CR0805) is placed manually per run.
- TREF is 180 °C, not the 205 °C reflow peak, because the SAC305 material data only extends to 180 °C (see `configs/thermal_profile.yaml`).
- Gerber arcs (G02/G03) are approximated as straight strokes.

## Credits

`zone_creep_strain.py` ports R. Schwerz's (Fraunhofer IKTS) APDL post-processing macro (POST_V3a_fix5). It is based on Darveaux's volume-weighted averaging method.
