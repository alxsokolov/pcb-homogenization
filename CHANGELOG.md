# Changelog

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
