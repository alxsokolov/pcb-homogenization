#!/usr/bin/env python3
"""
PyMAPDL port of simulate_joint_qoi.py's full-resolution benchmark: a
structured hex mesh painted per-element with copper/FR4 straight from the
Gerber-derived rasters (mesh/discretization.json + F_Cu_map.npy/
B_Cu_map.npy, from discretize.py) -- no quadtree homogenization, no CAD
trace geometry. Drives MAPDL directly via PyMAPDL, bypassing Workbench's
GUI entirely (no Geometry cell, no Engineering Data import, none of the
STEP/MatML friction from export_ansys_step.py -- that whole path exists
only because that other export specifically needed to show up in
Workbench's GUI tree; this script doesn't).

Nodes and elements are generated directly (mapdl.n()/mapdl.e(), batched
via mapdl.non_interactive) rather than through ANSYS's own BLOCK+VMESH
volume mesher, so the mesh is deterministic and exactly mirrors
simulate_tensile.py's create_layered_mesh(): same per-physical-layer z
banding (z_segs), same nx/ny/nz_per_layer resolution knobs, same
raster-lookup convention as assign_materials()/assign_alpha() (cell
centroid -> nearest raster pixel, threshold 0.5). This also sidesteps
ANSYS's own line-selection/mapped-meshing mechanics entirely, and lets
each element's MAT be set at creation time (mapdl.mat() pointer) instead
of a post-mesh EMODIF pass.

Board-only, elastic materials, single monotonic temperature ramp to an
end-state (one static load step) -- matches the project's existing
finding that a linear-elastic solve has no path dependence, so a "ramp"
and a single end-state step give the same answer; that distinction only
starts to matter once a path-dependent (creep/plasticity) solder material
is added, which is explicitly out of scope here (see project notes).

Requires `ansys-mapdl-core` (pip install ansys-mapdl-core) AND a reachable
MAPDL install/license -- launch_mapdl() auto-detects a local install.
This has only been syntax-checked, not run against a real MAPDL session.

Usage:
    python simulate_pyansys_benchmark.py data/samples/rv3032_real
    python simulate_pyansys_benchmark.py data/samples/rv3032_real --nx 50 --ny 50 --delta-t 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from homogenization import load_materials
from quadtree import _load_layer_maps
from solder_reliability import (
    COMPONENT_NAME, PAD_1_XY_MM, PAD_2_XY_MM,
    coffin_manson_cycles,
    _DEFAULT_COMPONENT_CTE_PER_K, _DEFAULT_STANDOFF_MM, _DEFAULT_CM_C, _DEFAULT_CM_M,
)

_CU_MAT_ID = 1
_FR4_MAT_ID = 2
_ELEM_TYPE_ID = 1


# ---------------------------------------------------------------------------
# Mesh construction -- mirrors simulate_tensile.py's create_layered_mesh()
# node-grid scheme exactly, just emitted as APDL N/E commands instead of a
# dolfinx mesh, and using ANSYS's own I-J-K-L-M-N-O-P brick node order
# (a proper CCW quad loop on the bottom face, then the matching top face --
# NOT the same node order as basix's hexahedron convention, which is a
# tensor-product/lexicographic order that would produce a twisted element
# if handed to ANSYS's E command as-is).
# ---------------------------------------------------------------------------

def _grid_points(Lx_mm: float, Ly_mm: float, z_layers_mm: list[float],
                 nx: int, ny: int, nz_per_layer: int):
    z_segs = [np.linspace(z_layers_mm[i], z_layers_mm[i + 1], nz_per_layer + 1)
              for i in range(len(z_layers_mm) - 1)]
    z_pts = np.concatenate([s[:-1] for s in z_segs] + [z_segs[-1][-1:]])
    x_pts = np.linspace(0.0, Lx_mm, nx + 1)
    y_pts = np.linspace(0.0, Ly_mm, ny + 1)
    return x_pts, y_pts, z_pts


def _raster_lookup(xc: float, yc: float, Lx_mm: float, Ly_mm: float, raster: np.ndarray) -> bool:
    """True if (xc, yc) [mm] falls on a copper pixel. Same convention as
    simulate_tensile.py's assign_materials(): row -> y, col -> x, threshold 0.5."""
    ny_map, nx_map = raster.shape
    ix = min(int(xc / Lx_mm * nx_map), nx_map - 1)
    iy = min(int(yc / Ly_mm * ny_map), ny_map - 1)
    return bool(raster[iy, ix] > 0.5)


def build_mesh_and_materials(mapdl, layer_maps: dict, Lx_mm: float, Ly_mm: float,
                             z_cu_bot_mm: float, z_cu_top_mm: float, Lz_mm: float,
                             nx: int, ny: int, nz_per_layer: int) -> dict:
    """
    Builds the full structured hex mesh directly (N/E commands, batched via
    mapdl.non_interactive) with MAT painted per element from the rasters at
    creation time. Returns a dict of node IDs needed later (BC corners,
    pad-nearest top-surface nodes) so the caller never has to guess/search
    for them geometrically.
    """
    x_pts, y_pts, z_pts = _grid_points(Lx_mm, Ly_mm, [0.0, z_cu_bot_mm, z_cu_top_mm, Lz_mm],
                                       nx, ny, nz_per_layer)
    nxn, nyn, nzn = len(x_pts), len(y_pts), len(z_pts)
    nz = nzn - 1

    def nid(i, j, k) -> int:
        return (i * nyn + j) * nzn + k + 1   # 1-based APDL node numbering

    mapdl.prep7()
    mapdl.et(_ELEM_TYPE_ID, "SOLID186")
    mapdl.type(_ELEM_TYPE_ID)

    print(f"Generating {nxn * nyn * nzn} nodes / {nx * ny * nz} elements …")
    with mapdl.non_interactive:
        for i in range(nxn):
            for j in range(nyn):
                for k in range(nzn):
                    mapdl.n(nid(i, j, k), x_pts[i], y_pts[j], z_pts[k])

    # Vectorized copper/FR4 classification per cell, exactly mirroring
    # assign_materials(): z-band selects which raster (or none, for the
    # always-FR4 core) to look up at the cell's (x,y) centroid.
    cu_cells, fr4_cells = [], []
    for k in range(nz):
        zc = (z_pts[k] + z_pts[k + 1]) / 2.0
        if zc < z_cu_bot_mm:
            raster = layer_maps.get("bottom")
        elif zc > z_cu_top_mm:
            raster = layer_maps.get("top")
        else:
            raster = None
        for i in range(nx):
            xc = (x_pts[i] + x_pts[i + 1]) / 2.0
            for j in range(ny):
                yc = (y_pts[j] + y_pts[j + 1]) / 2.0
                nodes8 = [
                    nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ]
                is_cu = raster is not None and _raster_lookup(xc, yc, Lx_mm, Ly_mm, raster)
                (cu_cells if is_cu else fr4_cells).append(nodes8)

    print(f"  {len(cu_cells)} copper elements, {len(fr4_cells)} FR4 elements")
    with mapdl.non_interactive:
        mapdl.mat(_CU_MAT_ID)
        for nodes8 in cu_cells:
            mapdl.e(*nodes8)
        mapdl.mat(_FR4_MAT_ID)
        for nodes8 in fr4_cells:
            mapdl.e(*nodes8)

    def nearest_top_node(x_mm: float, y_mm: float) -> int:
        i = int(np.clip(round(x_mm / Lx_mm * nx), 0, nx))
        j = int(np.clip(round(y_mm / Ly_mm * ny), 0, ny))
        return nid(i, j, nzn - 1)

    return {
        "corner_000": nid(0, 0, 0),
        "corner_Lx00": nid(nxn - 1, 0, 0),
        "corner_0Ly0": nid(0, nyn - 1, 0),
        "pad1_top": nearest_top_node(*PAD_1_XY_MM),
        "pad2_top": nearest_top_node(*PAD_2_XY_MM),
    }


# ---------------------------------------------------------------------------
# Materials, BCs, load step
# ---------------------------------------------------------------------------

def assign_material_properties(mapdl, materials: dict) -> None:
    cu, fr4 = materials["copper"], materials["FR4"]
    mapdl.mp("EX", _CU_MAT_ID, cu["E_GPa"] * 1e3)     # GPa -> MPa (mm/N/MPa unit system)
    mapdl.mp("EY", _CU_MAT_ID, cu["E_GPa"] * 1e3)
    mapdl.mp("EZ", _CU_MAT_ID, cu["E_GPa"] * 1e3)
    mapdl.mp("PRXY", _CU_MAT_ID, cu["nu"])
    mapdl.mp("PRYZ", _CU_MAT_ID, cu["nu"])
    mapdl.mp("PRXZ", _CU_MAT_ID, cu["nu"])
    alpha_cu = cu.get("CTE_per_K", 17e-6)
    mapdl.mp("ALPX", _CU_MAT_ID, alpha_cu)
    mapdl.mp("ALPY", _CU_MAT_ID, alpha_cu)
    mapdl.mp("ALPZ", _CU_MAT_ID, alpha_cu)

    mapdl.mp("EX", _FR4_MAT_ID, fr4["E_GPa"] * 1e3)
    mapdl.mp("EY", _FR4_MAT_ID, fr4["E_GPa"] * 1e3)
    mapdl.mp("EZ", _FR4_MAT_ID, fr4["E_GPa"] * 1e3)
    mapdl.mp("PRXY", _FR4_MAT_ID, fr4["nu"])
    mapdl.mp("PRYZ", _FR4_MAT_ID, fr4["nu"])
    mapdl.mp("PRXZ", _FR4_MAT_ID, fr4["nu"])
    alpha_fr4_ip = fr4.get("CTE_inplane_per_K", fr4.get("CTE_per_K", 14e-6))
    alpha_fr4_op = fr4.get("CTE_outofplane_per_K", fr4.get("CTE_per_K", 60e-6))
    mapdl.mp("ALPX", _FR4_MAT_ID, alpha_fr4_ip)
    mapdl.mp("ALPY", _FR4_MAT_ID, alpha_fr4_ip)
    mapdl.mp("ALPZ", _FR4_MAT_ID, alpha_fr4_op)


def apply_321_fixture(mapdl, node_ids: dict) -> None:
    """Same 3-2-1 fixture as simulate_tensile.py's build_cte_bcs(): removes
    all 6 rigid-body modes without restraining free thermal expansion."""
    n0, n1, n2 = node_ids["corner_000"], node_ids["corner_Lx00"], node_ids["corner_0Ly0"]
    mapdl.d(n0, "UX", 0)
    mapdl.d(n0, "UY", 0)
    mapdl.d(n0, "UZ", 0)
    mapdl.d(n1, "UY", 0)
    mapdl.d(n1, "UZ", 0)
    mapdl.d(n2, "UZ", 0)


def solve_thermal_step(mapdl, ref_temp_c: float, delta_t: float) -> None:
    mapdl.allsel()
    mapdl.tref(ref_temp_c)
    mapdl.bfunif("TEMP", ref_temp_c + delta_t)
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()


# ---------------------------------------------------------------------------
# Post-processing / QoI -- mirrors simulate_joint_qoi.py's evaluate_from_fe()
# ---------------------------------------------------------------------------

def evaluate_from_mapdl(mapdl, node_ids: dict,
                        alpha_component_per_K: float, delta_T_K: float, standoff_mm: float,
                        cm_C: float, cm_m: float) -> dict:
    mapdl.post1()
    mapdl.set(1, 1)

    def disp(node_id: int) -> np.ndarray:
        return np.array([mapdl.get_value("NODE", node_id, "U", c) for c in ("X", "Y", "Z")])

    u1 = disp(node_ids["pad1_top"])
    u2 = disp(node_ids["pad2_top"])

    du_board_mm = float(np.linalg.norm(u1[:2] - u2[:2]))
    dnp_total_mm = ((PAD_1_XY_MM[0] - PAD_2_XY_MM[0]) ** 2
                    + (PAD_1_XY_MM[1] - PAD_2_XY_MM[1]) ** 2) ** 0.5
    du_component_mm = alpha_component_per_K * delta_T_K * dnp_total_mm
    gamma = abs(du_board_mm - du_component_mm) / standoff_mm
    n_f = coffin_manson_cycles(gamma, cm_C, cm_m)

    result = {
        "u_pad1_mm": u1.tolist(), "u_pad2_mm": u2.tolist(),
        "du_board_mm": du_board_mm, "du_component_mm": du_component_mm,
        "gamma": gamma, "N_f_cycles": n_f,
    }
    try:
        vm1 = mapdl.get_value("NODE", node_ids["pad1_top"], "S", "EQV")
        vm2 = mapdl.get_value("NODE", node_ids["pad2_top"], "S", "EQV")
        result["von_mises_pad1_MPa"] = float(vm1)
        result["von_mises_pad2_MPa"] = float(vm2)
    except Exception as e:
        print(f"[warn] von Mises query failed, skipping (main QoI unaffected): {e}", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-resolution raster-mapped board benchmark, via PyMAPDL "
                    "(no quadtree, no CAD trace geometry).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--nz-per-layer", type=int, default=2)
    parser.add_argument("--delta-t", type=float, default=100.0, help="Thermal ramp range [K]")
    parser.add_argument("--ref-temp-c", type=float, default=20.0)
    parser.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    parser.add_argument("--component-cte", type=float, default=None,
                        help="Component CTE [1/K]; defaults to materials.yaml's component_ceramic_chip.CTE_per_K")
    parser.add_argument("--standoff-mm", type=float, default=_DEFAULT_STANDOFF_MM)
    parser.add_argument("--cm-c", type=float, default=_DEFAULT_CM_C)
    parser.add_argument("--cm-m", type=float, default=_DEFAULT_CM_M)
    args = parser.parse_args()

    sample_dir = args.sample_dir
    if not sample_dir.exists():
        sys.exit(f"Not found: {sample_dir}")
    with open(sample_dir / "metadata.json") as f:
        meta = json.load(f)

    layer_maps = _load_layer_maps(sample_dir)
    materials = load_materials(args.materials)

    Lx_mm = meta["board"]["width_mm"]
    Ly_mm = meta["board"]["height_mm"]
    z_cu_bot_mm = meta["stackup"]["copper_thickness_mm"]
    z_cu_top_mm = z_cu_bot_mm + meta["stackup"]["dielectric_thickness_mm"]
    Lz_mm = meta["stackup"]["total_thickness_mm"]

    alpha_component = args.component_cte
    if alpha_component is None:
        alpha_component = materials.get("component_ceramic_chip", {}).get(
            "CTE_per_K", _DEFAULT_COMPONENT_CTE_PER_K)

    print(f"Board {Lx_mm:.1f} x {Ly_mm:.1f} mm   Component {COMPONENT_NAME}   "
         f"pads {PAD_1_XY_MM}, {PAD_2_XY_MM} mm")
    print(f"delta_T = {args.delta_t:.1f} K   alpha_component = {alpha_component*1e6:.2f} ppm/K   "
         f"standoff = {args.standoff_mm:.4f} mm\n")

    from ansys.mapdl.core import launch_mapdl
    mapdl = launch_mapdl()
    try:
        node_ids = build_mesh_and_materials(
            mapdl, layer_maps, Lx_mm, Ly_mm, z_cu_bot_mm, z_cu_top_mm, Lz_mm,
            args.nx, args.ny, args.nz_per_layer)
        assign_material_properties(mapdl, materials)
        apply_321_fixture(mapdl, node_ids)
        solve_thermal_step(mapdl, args.ref_temp_c, args.delta_t)
        result = evaluate_from_mapdl(mapdl, node_ids, alpha_component, args.delta_t,
                                     args.standoff_mm, args.cm_c, args.cm_m)
    finally:
        mapdl.exit()

    print(f"du_board   = {result['du_board_mm']*1e3:.4f} um")
    print(f"du_component = {result['du_component_mm']*1e3:.4f} um")
    print(f"gamma = {result['gamma']:.6f}   N_f = {result['N_f_cycles']:.1f} cycles")
    if "von_mises_pad1_MPa" in result:
        print(f"von Mises @ pad1/pad2 = {result['von_mises_pad1_MPa']:.3f} / "
             f"{result['von_mises_pad2_MPa']:.3f} MPa")

    results_dir = sample_dir / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "pyansys_benchmark.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
