#!/usr/bin/env python3
"""
Sweep the quadtree's two control knobs -- depth cap (max_level) and leaf
budget (max_leaves), both via build_quadtree_by_budget() (quadtree.py) --
and see how the FE-evaluated joint lifetime (N_f) responds, against the one
benchmark value (full-resolution raster, true layered stack -- doesn't
depend on the quadtree at all, so it's computed once and shown as a
constant reference).

For each (max_level, max_leaves) combination: build a budget-driven
quadtree, homogenize it (quadtree.py's homogenize_leaves), collapse to a
single isotropic effective sheet (simulate_joint_qoi.py's
assign_uniform_from_quadtree -- same "study (a)" model as the single-run
script), solve the same fixed FE mesh, and evaluate the joint QoI at
COMPONENT_NAME's two pads (solder_reliability.py).

Compute only -- no plotting here. See plot_lifetime_sweep.py to render the
results, so plot styling can be iterated on without re-running the (much
more expensive) FE sweep.

Requires mesh/discretization.json + per-layer raster maps (from
discretize.py) -- does NOT require a pre-built mesh/quadtree.json, since
this script builds its own quadtree per sweep point.

Output: <sample_dir>/results/lifetime_sweep.json (all combinations' records
+ the benchmark value).

Usage:
    python sweep_quadtree_lifetime.py data/samples/rv3032_real
    python sweep_quadtree_lifetime.py data/samples/rv3032_real \\
        --leaves 4 8 16 32 64 --levels 3 4 5 6 8
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import basix
from mpi4py import MPI

from dolfinx import fem

from homogenization import load_materials
from simulate_tensile import create_layered_mesh, assign_materials, assign_alpha, build_cte_bcs, run_cte
from simulate_joint_qoi import assign_uniform_from_quadtree, evaluate_from_fe
from quadtree import build_quadtree_by_budget, homogenize_leaves, to_records, _load_layer_maps
from solder_reliability import (
    COMPONENT_NAME, PAD_1_XY_MM, PAD_2_XY_MM,
    _DEFAULT_COMPONENT_CTE_PER_K, _DEFAULT_STANDOFF_MM, _DEFAULT_CM_C, _DEFAULT_CM_M,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep quadtree max_level x max_leaves and evaluate FE-based joint "
                    "lifetime for each, against the full-resolution benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--leaves", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 512, 1024])
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 4, 5, 8])
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

    layer_maps = _load_layer_maps(sample_dir)
    mat = load_materials(args.materials)

    Lx_mm = meta["board"]["width_mm"]
    Ly_mm = meta["board"]["height_mm"]
    Lx, Ly = Lx_mm * 1e-3, Ly_mm * 1e-3
    cu_t_mm = meta["stackup"]["copper_thickness_mm"]
    fr4_t_mm = meta["stackup"]["dielectric_thickness_mm"]
    cu_t, fr4_t = cu_t_mm * 1e-3, fr4_t_mm * 1e-3
    Lz = meta["stackup"]["total_thickness_mm"] * 1e-3
    z_cu_bot, z_cu_top = cu_t, cu_t + fr4_t

    alpha_component = args.component_cte
    if alpha_component is None:
        alpha_component = mat.get("component_ceramic_chip", {}).get(
            "CTE_per_K", _DEFAULT_COMPONENT_CTE_PER_K)

    print(f"Board {Lx_mm:.1f} x {Ly_mm:.1f} mm   Component {COMPONENT_NAME}   "
         f"pads {PAD_1_XY_MM}, {PAD_2_XY_MM} mm")
    print(f"Sweep: leaves={args.leaves}  levels={args.levels}  "
         f"({len(args.leaves) * len(args.levels)} combinations)\n")

    comm = MPI.COMM_WORLD
    msh, _ = create_layered_mesh(comm, Lx, Ly, [0.0, z_cu_bot, z_cu_top, Lz],
                                 args.nx, args.ny, args.nz_per_layer)
    cname = msh.ufl_cell().cellname()
    V = fem.functionspace(msh, basix.ufl.element("Lagrange", cname, 1, shape=(3,)))
    V0 = fem.functionspace(msh, basix.ufl.element("DG", cname, 0))
    bcs = build_cte_bcs(msh, V, Lx, Ly)

    print("Solving benchmark (full-resolution raster, true 3-layer stack) …")
    lam_b, mu_b, _ = assign_materials(msh, V0, layer_maps, Lx, Ly, z_cu_bot, z_cu_top, mat)
    alpha_ip_b, alpha_op_b = assign_alpha(msh, V0, layer_maps, Lx, Ly, z_cu_bot, z_cu_top, mat)
    cte_b = run_cte(msh, V, lam_b, mu_b, alpha_ip_b, alpha_op_b, bcs, args.delta_t, Lx, Ly, Lz)
    benchmark = evaluate_from_fe(msh, cte_b["uh"], Lz, alpha_component, args.delta_t,
                                 args.standoff_mm, args.cm_c, args.cm_m)
    print(f"  benchmark: gamma={benchmark['gamma']:.6f}  N_f={benchmark['N_f_cycles']:.1f}\n")

    records = []
    combos = list(product(args.levels, args.leaves))
    for i, (max_level, max_leaves) in enumerate(combos):
        root = build_quadtree_by_budget(layer_maps["top"], layer_maps["bottom"],
                                        Lx_mm, Ly_mm, max_leaves, max_level)
        homogenize_leaves(root, mat, cu_t_mm, fr4_t_mm)
        leaves = to_records(root)
        n_leaves_actual = len(leaves)

        lam_s, mu_s, alpha_ip_s, alpha_op_s = assign_uniform_from_quadtree(
            msh, V0, leaves, Lx_mm, Ly_mm, mat)
        cte_s = run_cte(msh, V, lam_s, mu_s, alpha_ip_s, alpha_op_s, bcs, args.delta_t, Lx, Ly, Lz)
        result = evaluate_from_fe(msh, cte_s["uh"], Lz, alpha_component, args.delta_t,
                                  args.standoff_mm, args.cm_c, args.cm_m)

        pct_diff = (result["N_f_cycles"] - benchmark["N_f_cycles"]) / benchmark["N_f_cycles"] * 100

        print(f"  [{i+1:2d}/{len(combos)}] max_level={max_level} max_leaves={max_leaves:3d} "
             f"-> n_leaves={n_leaves_actual:3d}  gamma={result['gamma']:.6f}  "
             f"N_f={result['N_f_cycles']:.1f}  ({pct_diff:+.2f}% vs benchmark)")
        records.append({
            "max_level": max_level, "max_leaves_requested": max_leaves,
            "n_leaves_actual": n_leaves_actual,
            "gamma": result["gamma"], "N_f_cycles": result["N_f_cycles"],
            "pct_diff_from_benchmark": pct_diff,
        })

    results_dir = sample_dir / "results"
    results_dir.mkdir(exist_ok=True)
    out = {
        "component": COMPONENT_NAME,
        "benchmark": {"gamma": benchmark["gamma"], "N_f_cycles": benchmark["N_f_cycles"]},
        "sweep": records,
    }
    json_path = results_dir / "lifetime_sweep.json"
    json_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {json_path}")
    print("Run plot_lifetime_sweep.py to render it.")


if __name__ == "__main__":
    main()
