# Changelog

## Unreleased

### n-layer PCB support

A separate track of scripts for boards with more than two copper layers. The 2-layer scripts are untouched; nothing in the n-layer track imports them except to be verified against them.

- `import_odb_board_nlayer.py` / `odb_stackup.py`: ODB++ import keeping **all** conductive layers (SIGNAL, POWER_GROUND and MIXED — inner planes are usually the most copper-rich layers on a multilayer board), with the physical stackup resolved from `matrix/matrix` + per-layer `attrlist`, cross-checked against `misc/attrlist` `.board_thickness`. `--stackup <yaml>` supplies thicknesses for jobs that carry none.
- `import_gerber_board_nlayer.py`: n copper Gerbers plus a stackup YAML (Gerber carries no stackup at all, so it is an input, not something to recover).
- `homogenization_nlayer.py`: Voigt-Reuss homogenization of an arbitrary n-slab laminate. `python homogenization_nlayer.py` self-tests that it reduces exactly to `homogenization.py` on a 3-slab stackup over 400 random cases, plus slab-count invariance.
- `stackup_nlayer.py`: sample → ordered z slabs + per-layer rasters. Also reads the legacy 2-layer schema, so the n-layer tools can run on existing 2-layer samples as a cross-check.
- `coarsegrain_nlayer.py`: separates the two choices the 2-layer path mixes — the in-plane discretization (`--mode grid` at element resolution or coarser, or `--mode quadtree`) and the material model per region (`--region-model homogenized_zlayers | homogenized | projection`). Writes a self-contained, inspectable `mesh/coarsegrain_nlayer.json` with region geometry, per-layer copper fractions and a deduplicated material table, before MAPDL is involved.
- `simulate_pyansys_nlayer.py`: board-only MAPDL model. The mesh bands z by **physical layer**, so every slab gets its own element rows, no element straddles two layers, and each layer's copper volume is exact — which is why the n-layer path needs no copper-"dilution" correction. Reports effective in-plane CTE, warpage and peak von Mises, and checks the solved CTE against the analytical laminate value.
- `plot_coarsegrain_nlayer.py`, `compare_nlayer_models.py`: inspection plots and a model comparison against the full-resolution `raster` reference.
- `discretize.py` is shared and unchanged — it already rasterizes every copper layer listed in `metadata.json`.

**The "structural" Voigt/Reuss bounds carry no information the global ones don't.** `homogenization.py` describes its structural bounds (`E_xy_voigt`, `E_z_reuss`) as tighter than the geometry-blind global pair, exploiting the known layer geometry. They are in fact algebraically identical to it, for any number of slabs, any thicknesses and any per-layer copper fractions — both slab quantities are linear in that slab's φ, so thickness-weighting commutes with evaluating at the mean φ. Verified to machine precision (bit-identical in plane, 1e-16 relative out of plane) on an 8-layer stackup with per-layer φ from 0.02 to 0.99. The consequence: a single smeared through-thickness material depends on the stackup *only* through φ_3D, so smearing discards the entire layer structure. That costs little on a 2-layer board (copper ~4% of thickness) and a lot on the 8-layer reference board (copper 37% of thickness, per-layer φ from 0.67 to 0.86), which is why `homogenized_zlayers` is the n-layer default. The per-slab (z-resolved) materials are the only place layer geometry genuinely enters the homogenization.

**Known issue in the 2-layer ODB++ path (not fixed, to leave 2-layer results unchanged):** `odb_parser.parse_odb_attrlist()` splits on `=` without stripping whitespace, so a job writing `.layer_dielectric = 0.065` (Altium, Allegro) lands the key with a trailing space and every lookup misses it. Jobs writing `.layer_dielectric=1.51` with no spaces parse fine, which is why `odb_kitchen_sink` worked and `ublox`/`hella` silently fell back to defaults. `odb_stackup.py` carries its own whitespace-tolerant parser; fixing the shared one would change `hella`'s resolved copper thickness from 0.035 to 0.036614 mm.

## v0.1.2 (2026-10-05)

- **Fixed: the component and solder never followed the thermal profile.** The CR0805 archive contains a `BFBLOCK` that fixes all of its nodes at 20 °C, and nodal temperatures override `BFUNIF`. In every earlier run, only the board cycled while the component and solder stayed at 20 °C. With TREF = 180 °C, they also started with a 160 K thermal mismatch: above 200 MPa in the solder during a constant 180 °C hold. `component_import.py` now deletes these nodal temperatures after `CDREAD`; the same hold now gives 0 MPa. **Results from runs before this fix are not valid.**
- Fixed: the first segment of the thermal profile was applied as a step instead of a ramp. In load step 1, MAPDL ramps from TUNIF, and `BFUNIF,TEMP` itself overwrites TUNIF. The profile is now applied as a time table (`BFUNIF,TEMP,%TPROFILE%`).
- The example profile starts at the stress-free reference temperature (180 °C), cools down to 22 °C in 600 s, holds there for 300 s, and then runs one thermal cycle with 300 s holds.
- The applied load is plotted before the solve. `simulate_pyansys_assembly.py` saves `results/thermal_profile.yaml` and `results/thermal_profile.png`, and `python thermal_profile.py <profile.yaml>` plots a profile without simulating.
- Removed ODB++ support (`odb_parser.py`, `import_odb_board.py`) and the shared `raster_geometry.py`. Board input is Gerber only, and `discretize.py` is back to its self-contained Gerber version, which produces identical rasters.
- Removed the synthetic board generator `generate_gerber.py` and its `configs/default.yaml` (only needed for the removed ML dataset). Also removed `homogenization.orthotropic_engineering_constants()`, which only the removed APDL/STEP exports used, and the remaining docstring references to removed scripts.
- **Fixed: the component was placed on the wrong pads.** `PAD_1_XY_MM`/`PAD_2_XY_MM` in `solder_reliability.py` pointed at two pin-header pads (18, 12.2)/(18, 9.55), not at R1. They now hold the R1 pads from the Gerber, (15.0, 13.2)/(15.0, 11.55). `simulate_pyansys_assembly.py` now aligns the component with the pad axis by default (here 90°); before, it defaulted to 0°, crosswise to the pads. R1 is an 0603 footprint while the partner archive is a CR0805, so the component overhangs the pads.
- Replaced `plot_quadtree.py` with `plot_board_model.py`. It reads the board back from the solved `.rst` and shows the top copper layer as it was simulated: copper/FR4 per element when the layout was only projected (as now), or a per-element modulus colormap when the board is homogenized. The old plot always showed the quadtree, although the MAPDL model doesn't use it. `discretize.py` no longer writes `material_map.png`.
- Removed `temperature_history.py`. `plot_cycle_creep_increment.py` now takes the cycle boundaries (starts of the cold holds) from the saved profile.
- `--creep-ratio-limit` no longer defaults to 10. MAPDL ignored that value anyway, because `CUTCONTROL,CRPLIMIT` needs option 1 for implicit creep. By default no limit is issued now, and MAPDL's automatic time stepping applies. A value passed explicitly is now enforced with option 1.
- Added `docs/overview/`: a LaTeX module overview with example figures, and `make_figures.py` to regenerate the figures from a solved sample.
- Added `plot_comparison_creep.py`, which overlays the zone-averaged creep history and per-cycle Δε of several board-model variants (projection, homogenized, homogenized quadtree, and the new `homogenized_zlayers`, which resolves the two copper layers into separate per-leaf materials instead of smearing them together) read from their saved `results/` JSON. `docs/overview/main.tex` now includes the comparison figure for the `rv3032_example` board.

## v0.1.1 (2026-10-01)

- Added `configs/thermal_profile_example.yaml`, a shortened 2-cycle profile for quick demonstration runs.
- The README quickstart now uses the example profile and a local MAPDL run directory.

## v0.1.0 (2026-10-01)

First internal release for MikroVAL partners.

- Board input: Gerber (`import_real_board.py`), ODB++ (`import_odb_board.py`) and synthetic layouts (`generate_gerber.py`).
- Rasterization (`discretize.py`) and quadtree Voigt-Reuss homogenization (`quadtree.py`).
- PyMAPDL board benchmark and the board + CR0805 + SAC305 assembly with MPC bonded contact and a thermal-cycling profile.
- Creep postprocessing: temperature history, zone-averaged creep strain and per-cycle increments.

These experimental tracks are not part of this release. They are kept on branch `wip/2026-10`:
- FEniCSx dataset generation and the tensile/joint-QoI solves;
- the CNN homogenization surrogate;
- the quadtree lifetime sweep;
- the APDL/STEP Workbench exports.
