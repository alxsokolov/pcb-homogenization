#!/usr/bin/env python3
"""
Analytical CTE-mismatch shear-strain QoI at a solder joint.

Simplified (Sherlock-style) estimate: a 2-terminal component's own CTE
mismatches the *local* effective board CTE around each of its pads, and
that mismatch drives a shear strain at the joint during a thermal cycle.
The local board CTE is read from whichever quadtree leaf (produced by
quadtree.py) contains the pad -- this is the goal-oriented hook: as the
quadtree's discretization changes, the leaf covering the joint changes
size/composition, and so does this QoI.

    DNP   = distance from the component's own centroid (mean of its pad
            coordinates) to this pad                              [mm]
    gamma = |alpha_board_local - alpha_component| * delta_T * DNP / standoff
                                                    [dimensionless shear strain]
    N_f   = C * gamma ** (-m)          (Coffin-Manson -- illustrative only)

alpha_component, delta_T, standoff, and the Coffin-Manson C/m constants are
physically-uncertain and default to illustrative literature values -- NOT
validated for any specific joint or package. gamma (the strain) is the
quantity that actually matters for a convergence study; N_f is a bonus
downstream mapping.

Deliberately single-component for now: edit COMPONENT_NAME/PAD_1_XY_MM/
PAD_2_XY_MM below by hand for whatever joint you're looking at. Coordinates
are in the sample's local mm frame (same as quadtree.json/material_map.npy).
The R1 defaults below are the flash coordinates tagged %TO.P,R1,1*%/
%TO.P,R1,2*% in data/samples/rv3032_example/gerber/F_Cu.gbr (an 0603
footprint: 0.55 x 0.40 mm pads, 1.65 mm apart along y).
simulate_pyansys_assembly.py places the component at their midpoint and
aligns it with them.

Requires mesh/quadtree.json (from quadtree.py).

Usage:
    python solder_reliability.py data/samples/rv3032_example
    python solder_reliability.py data/samples/rv3032_example \\
        --component-cte 6.5e-6 --delta-t 100 --standoff-mm 0.075 \\
        --cm-c 0.5 --cm-m 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from homogenization import load_materials

# ---------------------------------------------------------------------------
# One component, hand-set. Edit these for whatever joint you're evaluating.
# ---------------------------------------------------------------------------
COMPONENT_NAME = "R1"
PAD_1_XY_MM = (15.0, 13.2)
PAD_2_XY_MM = (15.0, 11.55)

_DEFAULT_COMPONENT_CTE_PER_K = 6.5e-6
_DEFAULT_DELTA_T_K = 100.0
_DEFAULT_STANDOFF_MM = 0.075
_DEFAULT_CM_C = 0.5
_DEFAULT_CM_M = 2.0


def find_leaf_at(leaf_records: list[dict], x: float, y: float) -> dict:
    """Linear scan of quadtree.json's "leaves" list for the record covering (x, y)."""
    for leaf in leaf_records:
        x0, y0, x1, y1 = leaf["bbox_mm"]
        if x0 <= x < x1 and y0 <= y < y1:
            return leaf
    raise ValueError(f"No quadtree leaf contains point ({x}, {y}) -- outside board bounds?")


def dnp_shear_strain(alpha_board_per_K: float, alpha_component_per_K: float,
                     delta_T_K: float, dnp_mm: float, standoff_mm: float) -> float:
    return abs(alpha_board_per_K - alpha_component_per_K) * delta_T_K * dnp_mm / standoff_mm


def coffin_manson_cycles(gamma: float, C: float = _DEFAULT_CM_C, m: float = _DEFAULT_CM_M) -> float:
    """N_f = C * gamma^-m. C, m are illustrative SAC-solder-like placeholders, uncalibrated."""
    return C * gamma ** (-m)


def evaluate_component(
    sample_dir: Path,
    alpha_component_per_K: float = _DEFAULT_COMPONENT_CTE_PER_K,
    delta_T_K: float = _DEFAULT_DELTA_T_K,
    standoff_mm: float = _DEFAULT_STANDOFF_MM,
    cm_C: float = _DEFAULT_CM_C,
    cm_m: float = _DEFAULT_CM_M,
) -> dict:
    qt_path = sample_dir / "mesh" / "quadtree.json"
    if not qt_path.exists():
        sys.exit(f"Run quadtree.py first — not found: {qt_path}")
    qt = json.loads(qt_path.read_text())

    pads = [{"name": "pad1", "x_mm": PAD_1_XY_MM[0], "y_mm": PAD_1_XY_MM[1]},
           {"name": "pad2", "x_mm": PAD_2_XY_MM[0], "y_mm": PAD_2_XY_MM[1]}]
    cx = sum(p["x_mm"] for p in pads) / len(pads)
    cy = sum(p["y_mm"] for p in pads) / len(pads)

    pad_results = []
    for pad in pads:
        x, y = pad["x_mm"], pad["y_mm"]
        leaf = find_leaf_at(qt["leaves"], x, y)
        alpha_board = leaf["alpha_xy_voigt_per_K"]
        dnp = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        gamma = dnp_shear_strain(alpha_board, alpha_component_per_K, delta_T_K, dnp, standoff_mm)
        n_f = coffin_manson_cycles(gamma, cm_C, cm_m)
        pad_results.append({
            "name": pad["name"], "x_mm": x, "y_mm": y,
            "leaf_level": leaf["level"], "leaf_bbox_mm": leaf["bbox_mm"],
            "alpha_board_per_K": alpha_board,
            "dnp_mm": dnp, "gamma": gamma, "N_f_cycles": n_f,
        })

    worst = max(pad_results, key=lambda r: r["gamma"])
    return {
        "component": COMPONENT_NAME,
        "centroid_mm": [cx, cy],
        "params": {
            "alpha_component_per_K": alpha_component_per_K,
            "delta_T_K": delta_T_K, "standoff_mm": standoff_mm,
            "cm_C": cm_C, "cm_m": cm_m,
        },
        "pads": pad_results,
        "qoi_worst_pad": worst["name"],
        "qoi_gamma": worst["gamma"],
        "qoi_N_f_cycles": worst["N_f_cycles"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analytical CTE-mismatch shear-strain QoI at a solder joint, "
                    "using the local homogenized board CTE from a quadtree leaf. "
                    "Component/pad coordinates are hardcoded at the top of this file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    parser.add_argument("--component-cte", type=float, default=None,
                        help="Component CTE [1/K]; defaults to materials.yaml's "
                             "component_ceramic_chip.CTE_per_K")
    parser.add_argument("--delta-t", type=float, default=_DEFAULT_DELTA_T_K, help="Thermal cycle range [K]")
    parser.add_argument("--standoff-mm", type=float, default=_DEFAULT_STANDOFF_MM,
                        help="Solder joint standoff/fillet height [mm]")
    parser.add_argument("--cm-c", type=float, default=_DEFAULT_CM_C, help="Coffin-Manson C (illustrative)")
    parser.add_argument("--cm-m", type=float, default=_DEFAULT_CM_M, help="Coffin-Manson m (illustrative)")
    args = parser.parse_args()

    if not args.sample_dir.exists():
        sys.exit(f"Not found: {args.sample_dir}")

    materials = load_materials(args.materials)
    alpha_component = args.component_cte
    if alpha_component is None:
        alpha_component = materials.get("component_ceramic_chip", {}).get(
            "CTE_per_K", _DEFAULT_COMPONENT_CTE_PER_K)

    result = evaluate_component(
        args.sample_dir,
        alpha_component_per_K=alpha_component,
        delta_T_K=args.delta_t, standoff_mm=args.standoff_mm,
        cm_C=args.cm_c, cm_m=args.cm_m,
    )

    print(f"Component {result['component']}  (centroid {result['centroid_mm'][0]:.4f}, "
         f"{result['centroid_mm'][1]:.4f} mm)")
    print(f"alpha_component = {alpha_component * 1e6:.2f} ppm/K   "
         f"delta_T = {args.delta_t:.1f} K   standoff = {args.standoff_mm:.4f} mm")
    print(f"{'pad':8s} {'x_mm':>8s} {'y_mm':>8s} {'level':>6s} "
         f"{'alpha_board':>13s} {'DNP_mm':>8s} {'gamma':>10s} {'N_f':>12s}")
    for p in result["pads"]:
        print(f"{p['name']:8s} {p['x_mm']:8.3f} {p['y_mm']:8.3f} {p['leaf_level']:6d} "
             f"{p['alpha_board_per_K']*1e6:10.3f}ppm {p['dnp_mm']:8.4f} "
             f"{p['gamma']:10.6f} {p['N_f_cycles']:12.1f}")
    print(f"\nQoI (worst pad = {result['qoi_worst_pad']}): "
         f"gamma = {result['qoi_gamma']:.6f}   N_f ~ {result['qoi_N_f_cycles']:.1f} cycles "
         f"[illustrative Coffin-Manson constants, not calibrated]")


if __name__ == "__main__":
    main()
