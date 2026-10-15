#!/usr/bin/env python3
"""
Board-only MAPDL model, via PyMAPDL: mesh + material assignment (from
mesh/homogenization.json, written by homogenize.py) + thermal load + solve.
No component, no solder, no creep -- that's simulate_assembly.py.

Mesh: a structured hex grid, nx x ny in plane, z banded by PHYSICAL LAYER
(see pcbh.mesh) -- every slab boundary is an element boundary, so no element
ever straddles two layers and each layer's copper volume is exact.

--analysis cte (default): effective in-plane CTE, warpage, and peak von
    Mises stress, from a single static load step to --ref-temp-c +
    --delta-t. Linear-elastic materials have no path dependence, so a ramp
    and a single end state give the same answer. The solved alpha_eff is
    compared against the analytical laminate value (pcbh.homogenization).

--analysis k: effective thermal conductivity along x, y, z in turn (SOLID70,
    steady-state): fix one face's temperature, apply a total heat flow of
    --heat-w into the opposite face, read k_eff = Q*L/(A*delta_T), compared
    against the analytical laminate conductivity.

To solve the full-resolution reference a coarse-grained model should be
compared against, run homogenize.py --model projection --gx <nx> --gy <ny>
first (an element is copper if its region's copper area fraction >= the
projection threshold) and solve that.

Usage:
    python simulate_board.py data/samples/my_board --nx 40 --ny 40
    python simulate_board.py data/samples/my_board --nx 40 --ny 40 --analysis k
    python simulate_board.py data/samples/my_board --nx 40 --ny 40 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pcbh import grid as gridmod
from pcbh import mesh as meshmod
from pcbh.homogenization import SlabSpec, laminate_bounds
from pcbh.materials import PhaseProps, load_materials
from pcbh.stackup import BoardStackup, load_board

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


def analytical_reference(board: BoardStackup, p: PhaseProps) -> dict:
    """Whole-board analytical laminate properties at the true per-layer
    copper fractions -- the independent yardstick for the solved QoIs."""
    slabs = [SlabSpec(s.thickness_mm,
                      float(s.raster.mean()) if (s.is_copper and s.raster is not None) else 0.0,
                      s.kind, s.name)
             for s in board.slabs]
    b = laminate_bounds(slabs, p)
    return {"phi_3D": b["phi_3D"],
            "alpha_xy_per_K": b["alpha_xy_global_per_K"],
            "alpha_z_per_K": b["alpha_z_reuss_per_K"],
            "E_xy_MPa": b["E_xy_voigt_MPa"], "E_z_MPa": b["E_z_reuss_MPa"],
            "k_xy_W_mK": b["k_xy_voigt_W_mmK"] * 1e3, "k_z_W_mK": b["k_z_reuss_W_mmK"] * 1e3}


# ---------------------------------------------------------------------------
# --analysis cte
# ---------------------------------------------------------------------------


def _fit_plane_residual(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """w minus its best-fit plane -- warpage with rigid tilt removed."""
    A = np.column_stack([x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    return w - A @ coef


def solve_thermal_step(mapdl, ref_temp_c: float, delta_t: float) -> None:
    mapdl.finish()
    mapdl.slashsolu()
    mapdl.allsel()
    mapdl.tref(ref_temp_c)
    mapdl.bfunif("TEMP", ref_temp_c + delta_t)
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()


def evaluate_cte(mapdl, board: BoardStackup, mesh: meshmod.MeshInfo,
                 delta_t: float) -> tuple[dict, np.ndarray, np.ndarray]:
    mapdl.post1()
    mapdl.set(1, 1)

    def disp(node_id: int) -> np.ndarray:
        return np.array([mapdl.get_value("NODE", node_id, "U", c) for c in ("X", "Y", "Z")])

    u_x = disp(mesh.corner_Lx00)
    u_y = disp(mesh.corner_0Ly0)
    alpha_x = float(u_x[0] / (delta_t * board.width_mm)) if delta_t else float("nan")
    alpha_y = float(u_y[1] / (delta_t * board.height_mm)) if delta_t else float("nan")

    mapdl.allsel()
    coords = np.asarray(mapdl.mesh.nodes)
    uz_all = np.asarray(mapdl.post_processing.nodal_displacement("Z"))
    eqv = np.asarray(mapdl.post_processing.nodal_eqv_stress())

    top_mask = np.isclose(coords[:, 2], board.total_thickness_mm)
    xt, yt, wt = coords[top_mask, 0], coords[top_mask, 1], uz_all[top_mask]
    resid = _fit_plane_residual(xt, yt, wt)
    result = {
        "delta_t_K": delta_t, "alpha_eff_x_per_K": alpha_x, "alpha_eff_y_per_K": alpha_y,
        "warpage_pv_um": float((resid.max() - resid.min()) * 1e3),
        "warpage_rms_um": float(np.sqrt(np.mean(resid ** 2)) * 1e3),
        "von_mises_max_MPa": float(eqv.max()),
        "von_mises_top_max_MPa": float(eqv[top_mask].max()),
    }
    return result, coords, eqv


def plot_von_mises_top(coords: np.ndarray, eqv: np.ndarray, board: BoardStackup,
                       out_path: Path, title_suffix: str = "") -> None:
    top = np.isclose(coords[:, 2], board.total_thickness_mm)
    x, y, s = coords[top, 0], coords[top, 1], eqv[top]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    tpc = ax.tricontourf(x, y, s, levels=20, cmap="inferno")
    ax.set_xlim(0, board.width_mm)
    ax.set_ylim(0, board.height_mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Top-surface von Mises{title_suffix}")
    fig.colorbar(tpc, ax=ax, label="von Mises [MPa]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_cte(mapdl, board, mesh, args, ref: dict, tag: str, results_dir: Path) -> dict:
    print(f"Solving: TREF={args.ref_temp_c} C, uniform "
          f"{args.ref_temp_c + args.delta_t} C (delta_t={args.delta_t} K) …")
    solve_thermal_step(mapdl, args.ref_temp_c, args.delta_t)
    result, coords, eqv = evaluate_cte(mapdl, board, mesh, args.delta_t)

    a_ref = ref["alpha_xy_per_K"]
    for axis in ("x", "y"):
        a = result[f"alpha_eff_{axis}_per_K"]
        result[f"alpha_eff_{axis}_vs_analytical_rel"] = (a - a_ref) / abs(a_ref) if a_ref else float("nan")
    print(f"\nEffective in-plane CTE, bottom face (analytical laminate {a_ref * 1e6:.4f} ppm/K):")
    for axis in ("x", "y"):
        a = result[f"alpha_eff_{axis}_per_K"]
        print(f"  alpha_eff_{axis} = {a * 1e6:9.4f} ppm/K  "
              f"({result[f'alpha_eff_{axis}_vs_analytical_rel'] * 100:+.2f} %)")
    print(f"\nWarpage (top surface, best-fit plane removed):")
    print(f"  peak-to-valley = {result['warpage_pv_um']:.3f} um")
    print(f"  RMS            = {result['warpage_rms_um']:.3f} um")
    print(f"von Mises max = {result['von_mises_max_MPa']:.3f} MPa "
          f"(top surface {result['von_mises_top_max_MPa']:.3f} MPa)")

    plot_path = results_dir / f"von_mises_top_{tag}.png"
    plot_von_mises_top(coords, eqv, board, plot_path, f" — {board.n_copper} layers, {tag}")
    print(f"Wrote {plot_path}")
    return result


# ---------------------------------------------------------------------------
# --analysis k
# ---------------------------------------------------------------------------


def apply_fixed_temp(mapdl, node_grid: list[list[int]], temp_c: float) -> None:
    with mapdl.non_interactive:
        for row in node_grid:
            for node in row:
                mapdl.d(node, "TEMP", temp_c)


def apply_face_heat_flow(mapdl, node_grid: list[list[int]], total_watts: float) -> None:
    """Lump a uniform total heat flow onto a structured 2D face grid of node
    IDs: each cell contributes total_watts / n_cells, split evenly across
    its 4 corner nodes -- the consistent bilinear nodal equivalent of a
    constant surface heat flux, so it sums to total_watts exactly regardless
    of mesh density."""
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
    mapdl.finish()
    mapdl.slashsolu()
    mapdl.allsel()
    mapdl.ddele("ALL", "ALL")
    mapdl.fdele("ALL", "ALL")


def face_avg_temp(mapdl, node_grid: list[list[int]]) -> float:
    temps = np.asarray(mapdl.post_processing.nodal_temperature())
    ids = np.array([n for row in node_grid for n in row], dtype=int)
    return float(temps[ids - 1].mean())


def run_direction(mapdl, mesh: meshmod.MeshInfo, board: BoardStackup, axis: str,
                  heat_w: float, cold_temp_c: float) -> dict:
    hot_key, cold_key, length_attr, area_attrs = _DIRECTIONS[axis]
    hot_grid, cold_grid = getattr(mesh, hot_key), getattr(mesh, cold_key)

    clear_thermal_bcs(mapdl)
    apply_fixed_temp(mapdl, cold_grid, cold_temp_c)
    apply_face_heat_flow(mapdl, hot_grid, heat_w)
    mapdl.allsel()
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()

    mapdl.post1()
    mapdl.set("LAST")
    t_hot, t_cold = face_avg_temp(mapdl, hot_grid), face_avg_temp(mapdl, cold_grid)
    delta_t = t_hot - t_cold

    length_mm = getattr(board, length_attr)
    area_mm2 = getattr(board, area_attrs[0]) * getattr(board, area_attrs[1])
    k_eff = heat_w * length_mm / (area_mm2 * delta_t) if delta_t else float("nan")
    return {"axis": axis, "heat_w": heat_w, "length_mm": length_mm, "area_mm2": area_mm2,
            "t_hot_avg_c": t_hot, "t_cold_avg_c": t_cold, "delta_t_K": delta_t,
            "k_eff_W_mK": k_eff * 1e3}


def run_k(mapdl, board, mesh, args, ref: dict) -> list[dict]:
    axes = list(dict.fromkeys(args.directions.lower()))
    results = []
    for axis in axes:
        hot_key, cold_key, _, _ = _DIRECTIONS[axis]
        print(f"\nSolving direction {axis}: {args.heat_w} W into the {hot_key!r} face, "
              f"{args.cold_temp_c} C fixed on the {cold_key!r} face …")
        result = run_direction(mapdl, mesh, board, axis, args.heat_w, args.cold_temp_c)
        a_ref = ref["k_z_W_mK"] if axis == "z" else ref["k_xy_W_mK"]
        result["k_ref_W_mK"] = a_ref
        result["k_eff_vs_analytical_rel"] = (result["k_eff_W_mK"] - a_ref) / a_ref if a_ref else float("nan")
        print(f"  delta_T = {result['delta_t_K']:7.4f} K, k_eff = {result['k_eff_W_mK']:9.4f} "
              f"W/(m*K)  ({a_ref:.4f} W/(m*K), {result['k_eff_vs_analytical_rel'] * 100:+.2f} %)")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Board-only MAPDL model (mesh + homogenized materials + thermal load), "
                    "via PyMAPDL.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--nz-per-layer", type=int, default=1,
                    help="Element rows per physical layer. z is banded by slab, so 1 already "
                         "resolves every slab exactly.")
    ap.add_argument("--analysis", choices=["cte", "k"], default="cte",
                    help="cte: effective in-plane CTE, warpage, von Mises (SOLID186, one static "
                         "step). k: effective thermal conductivity x/y/z (SOLID70, steady-state).")
    ap.add_argument("--delta-t", type=float, default=100.0, help="--analysis cte: thermal step [K]")
    ap.add_argument("--ref-temp-c", type=float, default=20.0)
    ap.add_argument("--heat-w", type=float, default=1.0, help="--analysis k: total heat flow [W]")
    ap.add_argument("--cold-temp-c", type=float, default=20.0,
                    help="--analysis k: fixed temperature of the cold face [C]")
    ap.add_argument("--directions", default="xyz",
                    help="--analysis k: which axes to run, any subset of x/y/z")
    ap.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    ap.add_argument("--tag", default=None,
                    help="Suffix for the results file names (default: the region model)")
    ap.add_argument("--mapdl-dir", type=Path, default=Path("mapdl_run_board"),
                    help="MAPDL working directory. Put it on a local disk -- MAPDL writes "
                         "multi-GB scratch there and a network share slows the solve badly.")
    ap.add_argument("--nproc", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the mesh, materials and analytical reference, then stop "
                         "without launching MAPDL")
    args = ap.parse_args()

    sample_dir = args.sample_dir
    if not sample_dir.exists():
        sys.exit(f"Not found: {sample_dir}")
    if args.nx < 1 or args.ny < 1:
        sys.exit("--nx/--ny must be >= 1")
    axes = list(dict.fromkeys(args.directions.lower()))
    if args.analysis == "k" and (not axes or any(a not in _DIRECTIONS for a in axes)):
        sys.exit(f"--directions must be a subset of x/y/z, got {args.directions!r}")

    try:
        board = load_board(sample_dir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    p = PhaseProps.from_materials(load_materials(args.materials))

    homog_path = sample_dir / "mesh" / gridmod.OUTPUT_NAME
    if not homog_path.exists():
        sys.exit(f"Not found: {homog_path} -- run homogenize.py first")
    model = gridmod.load(homog_path)

    zgrid = meshmod.build_z_grid(board, args.nz_per_layer)
    n_elems = args.nx * args.ny * zgrid.nz
    n_nodes = (args.nx + 1) * (args.ny + 1) * len(zgrid.z_pts)

    print(board.describe())
    print(f"\nMesh: {args.nx} x {args.ny} x {zgrid.nz} = {n_elems} elements, {n_nodes} nodes")
    print(f"  in-plane element {board.width_mm / args.nx:.4f} x "
          f"{board.height_mm / args.ny:.4f} mm")
    thin = min(board.slabs, key=lambda s: s.thickness_mm)
    dz_thin = thin.thickness_mm / args.nz_per_layer
    print(f"  thinnest slab {thin.name!r}: {thin.thickness_mm * 1e3:.1f} um -> "
          f"{args.nz_per_layer} row(s) of {dz_thin * 1e3:.1f} um "
          f"(in-plane/through-thickness aspect ratio "
          f"{(board.width_mm / args.nx) / dz_thin:.0f}:1)")
    print(f"\nHomogenization: {model.gx} x {model.gy} grid, region model {model.region_model}, "
          f"{len(model.materials)} materials")
    if (model.gx, model.gy) == (args.nx, args.ny):
        print("  regions coincide with elements (one region per element)")

    ref = analytical_reference(board, p)
    print(f"\nAnalytical laminate reference (whole board, true per-layer phi):")
    print(f"  phi_3D     = {ref['phi_3D']:.6f}")
    print(f"  alpha_xy   = {ref['alpha_xy_per_K'] * 1e6:.4f} ppm/K")
    print(f"  alpha_z    = {ref['alpha_z_per_K'] * 1e6:.4f} ppm/K")
    print(f"  E_xy/E_z   = {ref['E_xy_MPa']:.0f} / {ref['E_z_MPa']:.0f} MPa")
    print(f"  k_xy/k_z   = {ref['k_xy_W_mK']:.4f} / {ref['k_z_W_mK']:.4f} W/(m*K)")

    if n_elems > meshmod.ELEMENT_WARN_LIMIT:
        print(f"\n[warn] {n_elems} elements is a lot -- nodes and elements are generated "
              f"through APDL commands, so this will take a long time before the solver "
              f"even starts. Consider lowering --nx/--ny or --nz-per-layer.", file=sys.stderr)

    if args.dry_run:
        print("\n--dry-run: stopping before MAPDL.")
        return

    tag = args.tag or model.region_model
    elem_type = "SOLID186" if args.analysis == "cte" else _THERMAL_ELEM_TYPE

    mapdl = meshmod.launch(args.mapdl_dir, args.nproc)
    print(f"\nMAPDL working directory: {mapdl.directory}")
    try:
        mat_ids = meshmod.register_materials(mapdl, model, p)
        print(f"Registered {len(mat_ids)} materials in MAPDL (MAT {min(mat_ids.values())}-"
              f"{max(mat_ids.values())})")
        mat_of_cell = gridmod.mat_resolver(model, mat_ids, args.nx, args.ny)
        mesh = meshmod.build_mesh(mapdl, board, zgrid, args.nx, args.ny, mat_of_cell, mat_ids,
                                  elem_type=elem_type)
        meshmod.apply_321_fixture(mapdl, mesh)

        results_dir = sample_dir / "results"
        results_dir.mkdir(exist_ok=True)

        if args.analysis == "cte":
            result = run_cte(mapdl, board, mesh, args, ref, tag, results_dir)
        else:
            result = {"directions": run_k(mapdl, board, mesh, args, ref)}
    finally:
        meshmod.shutdown(mapdl, args.mapdl_dir)

    out = {
        "sample_id": board.sample_id, "analysis": args.analysis, "region_model": model.region_model,
        "grid": {"gx": model.gx, "gy": model.gy},
        "mesh": {"nx": args.nx, "ny": args.ny, "nz": zgrid.nz, "nz_per_layer": args.nz_per_layer,
                 "n_elements": mesh.n_elements, "n_nodes": mesh.n_nodes,
                 "n_materials_used": mesh.n_materials_used},
        "stackup": {"n_copper_layers": board.n_copper, "total_thickness_mm": board.total_thickness_mm},
        "analytical_reference": ref,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        **result,
    }
    out_path = results_dir / f"board_{args.analysis}_{tag}.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
