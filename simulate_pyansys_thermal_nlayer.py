#!/usr/bin/env python3
"""
Whole-board effective thermal conductivity, n-layer PCB, via PyMAPDL.

Reuses the already-generated discretization from the n-layer pipeline --
mesh/coarsegrain_nlayer.json (or the raw copper rasters, with
--board-model raster) -- the same mesh and material-assignment code as
simulate_pyansys_nlayer.py, just a different element type and a thermal
(not structural) load/BC set:

    import_odb_board_nlayer.py   ODB++ -> sample + resolved stackup
    discretize.py                copper rasters (unchanged, already n-layer)
    coarsegrain_nlayer.py        regions x homogenize-or-project -> materials
    simulate_pyansys_thermal_nlayer.py   mesh + thermal BCs + solve   <- here

For each of x, y and z in turn: fix one face's temperature, apply a total
heat flow of --heat-w (default 1 W) into the opposite face, solve steady-
state conduction, and read the average temperature rise across the board.
That gives an effective conductivity

    k_eff = Q * L / (A * delta_T)

directly comparable to the analytical Voigt/Reuss laminate conductivity
(parallel in-plane, series through-thickness -- homogenization_nlayer's
k_xy_voigt / k_z_reuss), the same check simulate_pyansys_nlayer.py runs for
CTE against the laminate alpha_xy.

Heat flow BC: the total --heat-w is split over the hot face's elements as a
uniform flux and lumped to nodes with the standard bilinear weights for a
structured grid (corner 1/4, edge 1/2, interior 1 share of its cell, per
cell), so it sums to exactly --heat-w regardless of mesh density -- not an
approximation that degrades as the face mesh is refined.

Only the n-layer path's materials carry thermal conductivity (KXX/KYY/KZZ
added to homogenization_nlayer.py's MP dicts); the 2-layer path
(homogenization.py) does not, so this script has no 2-layer counterpart.

Requires `ansys-mapdl-core` and a reachable MAPDL install/license.

Usage:
    python simulate_pyansys_thermal_nlayer.py data/samples/ublox_nlayer --nx 40 --ny 40
    python simulate_pyansys_thermal_nlayer.py data/samples/ublox_nlayer --board-model raster
    python simulate_pyansys_thermal_nlayer.py data/samples/ublox_nlayer --directions z --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from homogenization import load_materials
from homogenization_nlayer import PhaseProps, SlabSpec, laminate_bounds
from simulate_pyansys_nlayer import (
    _ELEMENT_WARN_LIMIT,
    _force_kill_mapdl_processes,
    build_mesh,
    build_z_grid,
    load_coarsegrain,
    register_materials,
)
from stackup_nlayer import BoardStackup, load_board_stackup

_THERMAL_ELEM_TYPE = "SOLID70"

# direction -> (hot face key, cold face key, board length attr, cross-section
# area from the other two board dimensions). Heat flows from the hot face
# (loaded) to the cold face (fixed temperature).
_DIRECTIONS = {
    "x": ("xL_nodes", "x0_nodes", "width_mm", ("height_mm", "total_thickness_mm")),
    "y": ("yL_nodes", "y0_nodes", "height_mm", ("width_mm", "total_thickness_mm")),
    # z: hot face on top (as for a heat source mounted on the board),
    # fixed temperature at the bottom (as for a cold plate).
    "z": ("zL_nodes", "z0_nodes", "total_thickness_mm", ("width_mm", "height_mm")),
}


def analytical_k_reference(board: BoardStackup, p: PhaseProps) -> dict:
    """Whole-board analytical laminate conductivity, same construction as
    simulate_pyansys_nlayer.analytical_reference() for CTE."""
    slabs = [SlabSpec(s.thickness_mm,
                      float(s.raster.mean()) if (s.is_copper and s.raster is not None) else 0.0,
                      s.kind, s.name)
             for s in board.slabs]
    b = laminate_bounds(slabs, p)
    return {"phi_3D": b["phi_3D"],
            "k_xy_W_mK": b["k_xy_voigt_W_mmK"] * 1e3,
            "k_z_W_mK": b["k_z_reuss_W_mmK"] * 1e3}


def apply_fixed_temp(mapdl, node_grid: list[list[int]], temp_c: float) -> None:
    with mapdl.non_interactive:
        for row in node_grid:
            for node in row:
                mapdl.d(node, "TEMP", temp_c)


def apply_face_heat_flow(mapdl, node_grid: list[list[int]], total_watts: float) -> None:
    """
    Lump a uniform total heat flow onto a structured 2D face grid of node
    IDs: each cell contributes total_watts / n_cells, split evenly across
    its 4 corner nodes. On a uniform grid this is exactly the nodal
    equivalent of a constant surface heat flux (consistent bilinear
    weighting), so it sums to total_watts exactly -- interior nodes end up
    shared by 4 cells (full weight), edge nodes by 2 (half weight), corners
    by 1 (quarter weight).
    """
    n1, n2 = len(node_grid), len(node_grid[0])
    n_cells = (n1 - 1) * (n2 - 1)
    if n_cells < 1:
        raise ValueError("face grid needs at least a 2x2 node grid")
    q_cell = total_watts / n_cells
    loads: dict[int, float] = {}
    for a in range(n1 - 1):
        for b in range(n2 - 1):
            for node in (node_grid[a][b], node_grid[a + 1][b],
                        node_grid[a][b + 1], node_grid[a + 1][b + 1]):
                loads[node] = loads.get(node, 0.0) + q_cell / 4.0
    with mapdl.non_interactive:
        for node, watts in loads.items():
            mapdl.f(node, "HEAT", watts)


def clear_thermal_bcs(mapdl) -> None:
    """
    Drop into /SOLU and clear every D/F from the previous direction's solve.
    Also the reason this has to run before apply_fixed_temp()/
    apply_face_heat_flow(): after the first direction's mapdl.post1(), MAPDL
    is sitting in POST1, where D and F are not valid commands at all -- this
    FINISH + /SOLU is what makes it legal to issue them again.
    """
    mapdl.finish()
    mapdl.slashsolu()
    mapdl.allsel()
    mapdl.ddele("ALL", "ALL")
    mapdl.fdele("ALL", "ALL")


def solve_steady_state(mapdl) -> None:
    mapdl.allsel()
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()


def face_avg_temp(mapdl, node_grid: list[list[int]]) -> float:
    """
    Mean nodal temperature over a face. Node IDs are 1..n_nodes with no
    gaps (build_mesh()'s nid() numbering), and mapdl.post_processing's
    nodal arrays are indexed in that same ascending order -- the same
    assumption simulate_pyansys_nlayer.evaluate() relies on for its
    coordinate/field arrays -- so array[id - 1] is node id's value.
    """
    temps = np.asarray(mapdl.post_processing.nodal_temperature())
    ids = np.array([n for row in node_grid for n in row], dtype=int)
    return float(temps[ids - 1].mean())


def run_direction(mapdl, node_ids: dict, board: BoardStackup, axis: str,
                  heat_w: float, cold_temp_c: float) -> dict:
    hot_key, cold_key, length_attr, area_attrs = _DIRECTIONS[axis]
    hot_grid, cold_grid = node_ids[hot_key], node_ids[cold_key]

    clear_thermal_bcs(mapdl)
    apply_fixed_temp(mapdl, cold_grid, cold_temp_c)
    apply_face_heat_flow(mapdl, hot_grid, heat_w)
    solve_steady_state(mapdl)

    mapdl.post1()
    mapdl.set("LAST")
    t_hot = face_avg_temp(mapdl, hot_grid)
    t_cold = face_avg_temp(mapdl, cold_grid)
    delta_t = t_hot - t_cold

    length_mm = getattr(board, length_attr)
    area_mm2 = getattr(board, area_attrs[0]) * getattr(board, area_attrs[1])
    k_eff_W_mmK = heat_w * length_mm / (area_mm2 * delta_t) if delta_t else float("nan")
    return {
        "axis": axis,
        "heat_w": heat_w,
        "length_mm": length_mm,
        "area_mm2": area_mm2,
        "t_hot_avg_c": t_hot,
        "t_cold_avg_c": t_cold,
        "delta_t_K": delta_t,
        "k_eff_W_mK": k_eff_W_mmK * 1e3,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Whole-board effective thermal conductivity, n-layer PCB, via PyMAPDL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--nz-per-layer", type=int, default=1,
                    help="Element rows per physical layer, as in simulate_pyansys_nlayer.py.")
    ap.add_argument("--board-model", choices=["coarsegrain", "raster"], default="coarsegrain",
                    help="coarsegrain: materials from mesh/coarsegrain_nlayer.json. "
                         "raster: pure copper/dielectric per element straight from the "
                         "copper rasters, full resolution.")
    ap.add_argument("--directions", default="xyz",
                    help="Which axes to run, any subset of x/y/z (e.g. 'z' for just "
                         "through-thickness).")
    ap.add_argument("--heat-w", type=float, default=1.0, help="Total heat flow applied [W]")
    ap.add_argument("--cold-temp-c", type=float, default=20.0,
                    help="Fixed temperature of the cold face [C]")
    ap.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--mapdl-dir", type=Path, default=Path("mapdl_run_thermal_nlayer"))
    ap.add_argument("--nproc", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the mesh and analytical reference, then stop without "
                         "launching MAPDL")
    args = ap.parse_args()

    sample_dir = args.sample_dir
    if not sample_dir.exists():
        sys.exit(f"Not found: {sample_dir}")
    if args.nx < 1 or args.ny < 1:
        sys.exit("--nx/--ny must be >= 1")
    axes = list(dict.fromkeys(args.directions.lower()))
    if not axes or any(a not in _DIRECTIONS for a in axes):
        sys.exit(f"--directions must be a subset of x/y/z, got {args.directions!r}")

    try:
        board = load_board_stackup(sample_dir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    p = PhaseProps.from_materials(load_materials(args.materials))

    cg = None
    if args.board_model == "coarsegrain":
        try:
            cg = load_coarsegrain(sample_dir)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(str(e))

    zgrid = build_z_grid(board, args.nz_per_layer)
    n_elems = args.nx * args.ny * zgrid.nz
    n_nodes = (args.nx + 1) * (args.ny + 1) * len(zgrid.z_pts)

    print(board.describe())
    print(f"\nMesh: {args.nx} x {args.ny} x {zgrid.nz} = {n_elems} elements, {n_nodes} nodes")

    ref = analytical_k_reference(board, p)
    print(f"\nAnalytical laminate conductivity reference (whole board, true per-layer phi):")
    print(f"  phi_3D = {ref['phi_3D']:.6f}")
    print(f"  k_xy   = {ref['k_xy_W_mK']:.4f} W/(m*K)  (in-plane, parallel paths)")
    print(f"  k_z    = {ref['k_z_W_mK']:.4f} W/(m*K)  (through-thickness, series)")

    if n_elems > _ELEMENT_WARN_LIMIT:
        print(f"\n[warn] {n_elems} elements is a lot -- consider lowering --nx/--ny or "
              f"--nz-per-layer.", file=sys.stderr)

    if args.dry_run:
        print("\n--dry-run: stopping before MAPDL.")
        return

    tag = args.tag or (cg.region_model if cg is not None else "raster")
    args.mapdl_dir.mkdir(parents=True, exist_ok=True)
    run_location = str(args.mapdl_dir.resolve())

    from ansys.mapdl.core import launch_mapdl
    kwargs = {"run_location": run_location, "override": True}
    if args.nproc:
        kwargs["nproc"] = args.nproc
    mapdl = launch_mapdl(**kwargs)
    print(f"\nMAPDL working directory: {mapdl.directory}")
    results: list[dict] = []
    try:
        mat_ids = register_materials(mapdl, cg, p)
        print(f"Registered {len(mat_ids)} materials in MAPDL")
        node_ids = build_mesh(mapdl, board, zgrid, args.nx, args.ny, cg, mat_ids,
                              args.board_model, elem_type=_THERMAL_ELEM_TYPE)
        for axis in axes:
            print(f"\nSolving direction {axis}: {args.heat_w} W into the "
                  f"{_DIRECTIONS[axis][0]!r} face, {args.cold_temp_c} C fixed on "
                  f"the {_DIRECTIONS[axis][1]!r} face …")
            result = run_direction(mapdl, node_ids, board, axis, args.heat_w, args.cold_temp_c)
            results.append(result)
    finally:
        try:
            mapdl.exit()
        except Exception as e:                       # noqa: BLE001 - best effort
            print(f"[warn] mapdl.exit() failed: {e}", file=sys.stderr)
        _force_kill_mapdl_processes(run_location)

    print(f"\nEffective thermal conductivity (analytical reference in parens):")
    for result in results:
        axis = result["axis"]
        a_ref = ref["k_z_W_mK"] if axis == "z" else ref["k_xy_W_mK"]
        rel = (result["k_eff_W_mK"] - a_ref) / a_ref if a_ref else float("nan")
        result["k_ref_W_mK"] = a_ref
        result["k_eff_vs_analytical_rel"] = rel
        print(f"  {axis}: delta_T = {result['delta_t_K']:7.4f} K, "
              f"k_eff = {result['k_eff_W_mK']:9.4f} W/(m*K)  "
              f"({a_ref:.4f} W/(m*K), {rel * 100:+.2f} %)")

    out = {
        "sample_id": board.sample_id,
        "board_model": args.board_model,
        "region_model": cg.region_model if cg is not None else None,
        "heat_w": args.heat_w,
        "cold_temp_c": args.cold_temp_c,
        "mesh": {"nx": args.nx, "ny": args.ny, "nz": zgrid.nz,
                 "nz_per_layer": args.nz_per_layer},
        "analytical_reference": ref,
        "directions": results,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    results_dir = sample_dir / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"pyansys_thermal_nlayer_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
