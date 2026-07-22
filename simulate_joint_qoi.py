#!/usr/bin/env python3
"""
FE-based joint QoI: homogenized-sheet study vs full-resolution benchmark.

Replaces solder_reliability.py's closed-form "uniform CTE mismatch times a
straight-line DNP" assumption with an actual FEniCSx thermal-expansion
solve, evaluating real displacement at the two pad locations (see
solder_reliability.py's COMPONENT_NAME/PAD_1_XY_MM/PAD_2_XY_MM).

Two models share the SAME mesh (this is a homogenization-accuracy study,
not a mesh-cost study):

  Benchmark  -- the true 3D layered stack (bottom Cu / FR4 core / top Cu),
                exact per-z-band copper/FR4 assignment from the raster.
                This is exactly simulate_tensile.py's assign_materials()/
                assign_alpha(), reused unchanged.

  Study (a)  -- simplest possible homogenization: collapse the 3-layer
                stack into ONE isotropic effective sheet per quadtree leaf
                (mesh.py's quadtree.json), using that leaf's already-
                computed E_xy_voigt_GPa/alpha_xy_voigt_per_K uniformly
                through the whole board thickness. Deliberately throws
                away the top/bottom copper-asymmetry effect (which can
                otherwise cause bimetallic-strip-like bending) -- the
                point of this pass is to see how much that costs, by
                comparing against the benchmark.

Also prints solder_reliability.py's closed-form analytical number
alongside, for a 3-way comparison.

Requires mesh/quadtree.json (from quadtree.py) and mesh/discretization.json
+ per-layer raster maps (from discretize.py).

Usage:
    python simulate_joint_qoi.py data/samples/rv3032_real
    python simulate_joint_qoi.py data/samples/rv3032_real --delta-t 100 --nx 50 --ny 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import ufl
import basix
from mpi4py import MPI

from dolfinx import fem, geometry
from dolfinx.io import XDMFFile

from homogenization import load_materials
from simulate_tensile import (
    create_layered_mesh, assign_materials, assign_alpha, build_cte_bcs, run_cte, sig_fn,
)
from solder_reliability import (
    COMPONENT_NAME, PAD_1_XY_MM, PAD_2_XY_MM,
    coffin_manson_cycles, evaluate_component,
    _DEFAULT_COMPONENT_CTE_PER_K, _DEFAULT_STANDOFF_MM, _DEFAULT_CM_C, _DEFAULT_CM_M,
)


def _load_layer_maps(sample_dir: Path) -> dict[str, np.ndarray]:
    """Same loading pattern as simulate_tensile.py's main() / quadtree.py's _load_layer_maps."""
    mesh_dir = sample_dir / "mesh"
    disc_path = mesh_dir / "discretization.json"
    if not disc_path.exists():
        sys.exit(f"Run discretize.py first — not found: {disc_path}")
    with open(disc_path) as f:
        disc = json.load(f)

    layer_maps: dict[str, np.ndarray] = {}
    for _lname, linfo in disc.get("layer_maps", {}).items():
        pos = linfo.get("position")
        fpath = mesh_dir / linfo["file"]
        if pos in ("top", "bottom") and fpath.exists():
            layer_maps[pos] = np.load(fpath)
    if not layer_maps:
        combined = np.load(mesh_dir / "material_map.npy")
        layer_maps = {"top": combined, "bottom": combined}
    return layer_maps


# ---------------------------------------------------------------------------
# Study (a): single isotropic effective sheet per quadtree leaf
# ---------------------------------------------------------------------------

def assign_uniform_from_quadtree(msh, V0, leaves: list[dict],
                                 Lx_mm: float, Ly_mm: float, mat: dict):
    """
    Collapse the 3-layer stack into one isotropic effective sheet per leaf:
    every fine mesh cell (regardless of z-band) gets its (x,y) centroid's
    quadtree leaf's E_xy_voigt_GPa/alpha_xy_voigt_per_K, applied uniformly
    through the whole thickness -- the "simplest possible" homogenization.

    nu_eff = phi_3D * nu_cu + (1 - phi_3D) * nu_fr4 (volume-fraction
    weighted -- Voigt-Reuss bounds don't give an effective nu directly,
    this is a flagged simplification, same spirit as solder_reliability.py's
    illustrative constants). alpha is isotropic (alpha_ip = alpha_op),
    collapsing FR4's real through-thickness CTE anisotropy too.
    """
    from dolfinx.mesh import compute_midpoints

    tdim = msh.topology.dim
    n_cells = msh.topology.index_map(tdim).size_local
    msh.topology.create_connectivity(tdim, 0)
    mid = compute_midpoints(msh, tdim, np.arange(n_cells, dtype=np.int32))

    nu_cu = mat["copper"]["nu"]
    nu_fr4 = mat["FR4"]["nu"]

    xc_mm = np.clip(mid[:, 0] * 1e3, 0.0, Lx_mm - 1e-9)
    yc_mm = np.clip(mid[:, 1] * 1e3, 0.0, Ly_mm - 1e-9)

    E_arr = np.empty(n_cells, dtype=np.float64)
    nu_arr = np.empty(n_cells, dtype=np.float64)
    alpha_arr = np.empty(n_cells, dtype=np.float64)

    # Loop over the (few) leaves rather than the (many) cells: vectorized
    # numpy masking per leaf, O(n_leaves) numpy ops instead of an
    # O(n_cells * n_leaves) Python-level per-cell linear scan -- matters
    # once this gets called dozens of times in a sweep.
    assigned = np.zeros(n_cells, dtype=bool)
    for leaf in leaves:
        x0, y0, x1, y1 = leaf["bbox_mm"]
        mask = (xc_mm >= x0) & (xc_mm < x1) & (yc_mm >= y0) & (yc_mm < y1)
        E_arr[mask] = leaf["E_xy_voigt_GPa"] * 1e9
        nu_arr[mask] = leaf["phi_3D"] * nu_cu + (1.0 - leaf["phi_3D"]) * nu_fr4
        alpha_arr[mask] = leaf["alpha_xy_voigt_per_K"]
        assigned |= mask
    if not assigned.all():
        n_missing = int((~assigned).sum())
        raise ValueError(f"{n_missing} mesh cell(s) matched no quadtree leaf -- "
                        f"leaves don't cover the full board?")

    lam = fem.Function(V0)
    mu = fem.Function(V0)
    lam.x.array[:] = (E_arr * nu_arr) / ((1 + nu_arr) * (1 - 2 * nu_arr))
    mu.x.array[:] = E_arr / (2 * (1 + nu_arr))

    alpha_ip = fem.Function(V0)
    alpha_op = fem.Function(V0)
    alpha_ip.x.array[:] = alpha_arr
    alpha_op.x.array[:] = alpha_arr

    return lam, mu, alpha_ip, alpha_op


# ---------------------------------------------------------------------------
# Point evaluation + joint QoI from an FE displacement field
# ---------------------------------------------------------------------------

def eval_at_point(msh, uh: fem.Function, point_xyz) -> np.ndarray:
    points = np.array([point_xyz], dtype=np.float64)
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    cells, pts_found = [], []
    for i, point in enumerate(points):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_found.append(point)
            cells.append(links[0])
    if not pts_found:
        raise ValueError(f"Point {point_xyz} not found in any mesh cell (outside domain?)")
    values = uh.eval(np.array(pts_found, dtype=np.float64), cells)
    # eval() squeezes to a flat (value_size,) array for a single point rather
    # than (1, value_size) -- reshape explicitly so both cases are handled.
    return values.reshape(len(pts_found), -1)[0]


def evaluate_from_fe(
    msh, uh: fem.Function, Lz: float,
    alpha_component_per_K: float, delta_T_K: float, standoff_mm: float,
    cm_C: float, cm_m: float,
) -> dict:
    p1 = (PAD_1_XY_MM[0] * 1e-3, PAD_1_XY_MM[1] * 1e-3, Lz)
    p2 = (PAD_2_XY_MM[0] * 1e-3, PAD_2_XY_MM[1] * 1e-3, Lz)
    u1 = eval_at_point(msh, uh, p1)
    u2 = eval_at_point(msh, uh, p2)

    du_board_m = float(np.linalg.norm(u1[:2] - u2[:2]))
    dnp_total_m = ((PAD_1_XY_MM[0] - PAD_2_XY_MM[0]) ** 2
                  + (PAD_1_XY_MM[1] - PAD_2_XY_MM[1]) ** 2) ** 0.5 * 1e-3
    du_component_m = alpha_component_per_K * delta_T_K * dnp_total_m
    standoff_m = standoff_mm * 1e-3
    gamma = abs(du_board_m - du_component_m) / standoff_m
    n_f = coffin_manson_cycles(gamma, cm_C, cm_m)

    return {
        "u_pad1_m": u1.tolist(), "u_pad2_m": u2.tolist(),
        "du_board_m": du_board_m, "du_component_m": du_component_m,
        "gamma": gamma, "N_f_cycles": n_f,
    }


# ---------------------------------------------------------------------------
# XDMF/H5 output for ParaView
# ---------------------------------------------------------------------------

def _effective_E(lam: fem.Function, mu: fem.Function) -> np.ndarray:
    """Recover isotropic E from (lambda, mu): E = mu*(3*lambda + 2*mu)/(lambda + mu)."""
    l, m = lam.x.array, mu.x.array
    return m * (3 * l + 2 * m) / (l + m)


def _write_vector_field(msh, uh: fem.Function, name: str, path: Path) -> None:
    uh.name = name
    with XDMFFile(msh.comm, str(path), "w") as f:
        f.write_mesh(msh)
        f.write_function(uh)


def _write_scalar_field(msh, V0, arr: np.ndarray, name: str, path: Path) -> None:
    fn = fem.Function(V0, name=name)
    fn.x.array[:] = arr
    with XDMFFile(msh.comm, str(path), "w") as f:
        f.write_mesh(msh)
        f.write_function(fn)


def _von_mises_field(V0, uh: fem.Function, lam: fem.Function, mu: fem.Function,
                     alpha_ip: fem.Function, alpha_op: fem.Function, delta_T: float,
                     name: str) -> fem.Function:
    """
    True physical stress for a thermal-expansion solve is the mechanical
    stress MINUS the thermal eigenstress -- run_cte solves
    a(u,v) = L(v) i.e. sig_mech(u) = sig_thermal in the weak sense, so
    sig_true = sig_mech(u) - sig_thermal is what's actually driving strain
    energy/failure (a free-standing unconstrained expansion should read
    near-zero stress, not the raw sig_mech(u), which conflates the free
    thermal strain with an elastic one). Mirrors run_cte's own sig_th
    exactly (simulate_tensile.py).
    """
    alpha_t = ufl.as_tensor([[alpha_ip, 0, 0],
                             [0, alpha_ip, 0],
                             [0, 0,        alpha_op]])
    tr_alpha = 2 * alpha_ip + alpha_op
    sig_th = lam * tr_alpha * delta_T * ufl.Identity(3) + 2 * mu * alpha_t * delta_T
    sig_true = sig_fn(uh, lam, mu) - sig_th

    sdev = sig_true - (ufl.tr(sig_true) / 3) * ufl.Identity(3)
    svm = ufl.sqrt((3 / 2) * ufl.inner(sdev, sdev))
    svm_expr = fem.Expression(svm, V0.element.interpolation_points())
    svm_h = fem.Function(V0, name=name)
    svm_h.interpolate(svm_expr)
    return svm_h


def write_model_results(msh, V0, uh: fem.Function, lam: fem.Function, mu: fem.Function,
                        alpha_ip: fem.Function, alpha_op: fem.Function, delta_T: float,
                        results_dir: Path, tag: str) -> None:
    """Write displacement + effective E + in-plane CTE + von Mises stress to <tag>_*.xdmf/.h5."""
    _write_vector_field(msh, uh, f"{tag}_displacement_m", results_dir / f"{tag}_disp.xdmf")
    _write_scalar_field(msh, V0, _effective_E(lam, mu), f"{tag}_E_Pa", results_dir / f"{tag}_E.xdmf")
    _write_scalar_field(msh, V0, alpha_ip.x.array, f"{tag}_alpha_ip_per_K",
                        results_dir / f"{tag}_alpha_ip.xdmf")
    svm_h = _von_mises_field(V0, uh, lam, mu, alpha_ip, alpha_op, delta_T, f"{tag}_von_mises_Pa")
    with XDMFFile(msh.comm, str(results_dir / f"{tag}_von_mises.xdmf"), "w") as f:
        f.write_mesh(msh)
        f.write_function(svm_h)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FE-based (benchmark vs single-leaf-homogenized) joint QoI "
                    "against the closed-form analytical estimate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--nz-per-layer", type=int, default=2)
    parser.add_argument("--delta-t", type=float, default=100.0, help="Thermal cycle range [K]")
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
    qt_path = sample_dir / "mesh" / "quadtree.json"
    if not qt_path.exists():
        sys.exit(f"Run quadtree.py first — not found: {qt_path}")
    qt = json.loads(qt_path.read_text())
    leaves = qt["leaves"]

    layer_maps = _load_layer_maps(sample_dir)
    mat = load_materials(args.materials)

    Lx_mm = meta["board"]["width_mm"]
    Ly_mm = meta["board"]["height_mm"]
    Lx, Ly = Lx_mm * 1e-3, Ly_mm * 1e-3
    cu_t = meta["stackup"]["copper_thickness_mm"] * 1e-3
    fr4_t = meta["stackup"]["dielectric_thickness_mm"] * 1e-3
    Lz = meta["stackup"]["total_thickness_mm"] * 1e-3
    z_cu_bot, z_cu_top = cu_t, cu_t + fr4_t

    alpha_component = args.component_cte
    if alpha_component is None:
        alpha_component = mat.get("component_ceramic_chip", {}).get(
            "CTE_per_K", _DEFAULT_COMPONENT_CTE_PER_K)

    print(f"Board {Lx_mm:.1f} x {Ly_mm:.1f} mm   Component {COMPONENT_NAME}   "
         f"pads {PAD_1_XY_MM}, {PAD_2_XY_MM} mm")
    print(f"delta_T = {args.delta_t:.1f} K   alpha_component = {alpha_component*1e6:.2f} ppm/K   "
         f"standoff = {args.standoff_mm:.4f} mm\n")

    comm = MPI.COMM_WORLD
    msh, _ = create_layered_mesh(comm, Lx, Ly, [0.0, z_cu_bot, z_cu_top, Lz],
                                 args.nx, args.ny, args.nz_per_layer)
    cname = msh.ufl_cell().cellname()
    V = fem.functionspace(msh, basix.ufl.element("Lagrange", cname, 1, shape=(3,)))
    V0 = fem.functionspace(msh, basix.ufl.element("DG", cname, 0))
    bcs = build_cte_bcs(msh, V, Lx, Ly)

    results_dir = sample_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("Solving benchmark (full-resolution raster, true 3-layer stack) …")
    lam_b, mu_b, _ = assign_materials(msh, V0, layer_maps, Lx, Ly, z_cu_bot, z_cu_top, mat)
    alpha_ip_b, alpha_op_b = assign_alpha(msh, V0, layer_maps, Lx, Ly, z_cu_bot, z_cu_top, mat)
    cte_b = run_cte(msh, V, lam_b, mu_b, alpha_ip_b, alpha_op_b, bcs, args.delta_t, Lx, Ly, Lz)
    benchmark = evaluate_from_fe(msh, cte_b["uh"], Lz, alpha_component, args.delta_t,
                                 args.standoff_mm, args.cm_c, args.cm_m)
    write_model_results(msh, V0, cte_b["uh"], lam_b, mu_b, alpha_ip_b, alpha_op_b,
                        args.delta_t, results_dir, "benchmark")

    print("Solving study (a) (single-leaf-homogenized isotropic sheet) …")
    lam_s, mu_s, alpha_ip_s, alpha_op_s = assign_uniform_from_quadtree(
        msh, V0, leaves, Lx_mm, Ly_mm, mat)
    cte_s = run_cte(msh, V, lam_s, mu_s, alpha_ip_s, alpha_op_s, bcs, args.delta_t, Lx, Ly, Lz)
    study_a = evaluate_from_fe(msh, cte_s["uh"], Lz, alpha_component, args.delta_t,
                               args.standoff_mm, args.cm_c, args.cm_m)
    write_model_results(msh, V0, cte_s["uh"], lam_s, mu_s, alpha_ip_s, alpha_op_s,
                        args.delta_t, results_dir, "study_a")
    print(f"Wrote XDMF/H5 results to {results_dir}/  (benchmark_*, study_a_*)")

    analytical = evaluate_component(
        sample_dir, alpha_component_per_K=alpha_component, delta_T_K=args.delta_t,
        standoff_mm=args.standoff_mm, cm_C=args.cm_c, cm_m=args.cm_m,
    )

    print(f"\n{'model':32s} {'du_board_um':>12s} {'du_comp_um':>11s} {'gamma':>10s} {'N_f':>12s}")
    print(f"{'closed-form analytical':32s} {'--':>12s} {'--':>11s} "
         f"{analytical['qoi_gamma']:10.6f} {analytical['qoi_N_f_cycles']:12.1f}")
    print(f"{'FE benchmark (full raster)':32s} {benchmark['du_board_m']*1e6:12.4f} "
         f"{benchmark['du_component_m']*1e6:11.4f} {benchmark['gamma']:10.6f} {benchmark['N_f_cycles']:12.1f}")
    print(f"{'FE study (a) (1-leaf sheet)':32s} {study_a['du_board_m']*1e6:12.4f} "
         f"{study_a['du_component_m']*1e6:11.4f} {study_a['gamma']:10.6f} {study_a['N_f_cycles']:12.1f}")

    gap = abs(study_a["gamma"] - benchmark["gamma"]) / benchmark["gamma"] * 100
    print(f"\nStudy(a) vs benchmark gamma gap: {gap:.2f}%  "
         f"[illustrative Coffin-Manson constants, not calibrated]")


if __name__ == "__main__":
    main()
