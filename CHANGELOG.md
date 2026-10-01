# Changelog

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
