#!/usr/bin/env python3
"""
A quantity of interest vs. resolution, read from each sample's saved
results/ JSON -- no MAPDL session or .rst access needed. Two independent
resolutions can be swept, matching the project's own two discretizations:

  --grid N DIR    homogenization-grid resolution (homogenize.py --gx/--gy N),
                  FE mesh held fixed across the sweep -- "how many regions
                  does the homogenization need".
  --mesh N DIR    FE mesh resolution (simulate_board.py/simulate_assembly.py
                  --nx/--ny N), homogenization model held fixed -- "does the
                  FE mesh itself need to be finer".

Both flags are repeatable and independent; a sample can appear under either
or both. --reference DIR adds a dashed 0% line and a deviation column, from
a sample not necessarily on either sweep (e.g. the full-resolution
--model projection reference).

The quantity of interest is read per sample, preferring (in order):
  1. results/cycle_creep_increment.json's cycle-1 delta_cracc_zone -- the
     per-cycle solder creep-strain increment (needs simulate_assembly.py +
     postprocess_creep.py already run).
  2. results/board_cte_*.json's alpha_eff_x_per_K -- the board-only
     effective in-plane CTE (needs simulate_board.py --analysis cte).
A sample with neither is reported and skipped rather than failing the whole
sweep.

Usage:
    python plot_convergence.py \\
        --grid 5  data/samples/my_board_grid5  --grid 10 data/samples/my_board_grid10 \\
        --grid 25 data/samples/my_board_grid25 --grid 50 data/samples/my_board_grid50 \\
        --mesh 20 data/samples/my_board_mesh20 --mesh 50 data/samples/my_board_mesh50 \\
        --reference data/samples/my_board_reference \\
        --out docs/overview/figures/convergence.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.8,
    "grid.color": "#d5d5d0",
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

_GRID_COLOR = "#e8732c"
_MESH_COLOR = "#2ca02c"


def read_qoi(sample_dir: Path) -> tuple[float, str]:
    """(value, label) for a sample, preferring the creep increment over the
    board-only CTE -- see module docstring."""
    creep_path = sample_dir / "results" / "cycle_creep_increment.json"
    if creep_path.exists():
        cycles = json.loads(creep_path.read_text())["cycles"]
        if cycles:
            return float(cycles[0]["delta_cracc_zone"]), r"per-cycle $\Delta\varepsilon_{creep}$ [-]"
    board_paths = sorted((sample_dir / "results").glob("board_cte_*.json"))
    if board_paths:
        doc = json.loads(board_paths[0].read_text())
        return float(doc["alpha_eff_x_per_K"]), r"$\alpha_{eff,x}$ [per K]"
    raise FileNotFoundError(
        f"Neither results/cycle_creep_increment.json nor results/board_cte_*.json found "
        f"under {sample_dir} -- run postprocess_creep.py or simulate_board.py first")


def _read_points(pairs: list[list[str]]) -> list[tuple[int, float, str, str]]:
    pts = []
    for n_str, d_str in pairs:
        n, d = int(n_str), Path(d_str)
        if not d.is_dir():
            raise SystemExit(f"Not a directory: {d}")
        value, label = read_qoi(d)
        pts.append((n, value, d.name, label))
    pts.sort(key=lambda p: p[0])
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(
        description="QoI vs. homogenization-grid resolution and/or FE mesh resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--grid", nargs=2, action="append", default=[], metavar=("N", "SAMPLE_DIR"),
                    help="Homogenization-grid resolution N and its sample dir (repeatable)")
    ap.add_argument("--mesh", nargs=2, action="append", default=[], metavar=("N", "SAMPLE_DIR"),
                    help="FE mesh resolution N and its sample dir (repeatable)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="Reference sample for a dashed 0%% deviation line (e.g. the "
                         "full-resolution --model projection run)")
    ap.add_argument("--out", type=Path, default=Path("convergence.png"))
    args = ap.parse_args()

    if not args.grid and not args.mesh:
        ap.error("Provide at least one --grid N DIR or --mesh N DIR")

    grid_points = _read_points(args.grid)
    mesh_points = _read_points(args.mesh)
    label = (grid_points or mesh_points)[0][3]

    ref_value = None
    if args.reference is not None:
        ref_value, _ = read_qoi(args.reference)
        print(f"reference  {args.reference.name:28s} = {ref_value:.6e}")
    for n, v, name, _ in grid_points:
        dev = f"  ({100 * (v - ref_value) / ref_value:+.2f}% vs reference)" if ref_value else ""
        print(f"grid N={n:4d}  {name:28s} = {v:.6e}{dev}")
    for n, v, name, _ in mesh_points:
        dev = f"  ({100 * (v - ref_value) / ref_value:+.2f}% vs reference)" if ref_value else ""
        print(f"mesh N={n:4d}  {name:28s} = {v:.6e}{dev}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if grid_points:
        xs = [p[0] for p in grid_points]
        ys = [p[1] for p in grid_points]
        ax.plot(xs, ys, "o-", color=_GRID_COLOR, linewidth=2, markersize=8, zorder=3,
                label="homogenization grid resolution")
        for n, v, _, _ in grid_points:
            ax.annotate(f"{v:.3e}", (n, v), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=9)
    if mesh_points:
        xs = [p[0] for p in mesh_points]
        ys = [p[1] for p in mesh_points]
        ax.plot(xs, ys, "D-", color=_MESH_COLOR, linewidth=2, markersize=9, zorder=4,
                label="FE mesh resolution")
        for n, v, _, _ in mesh_points:
            ax.annotate(f"{v:.3e}", (n, v), textcoords="offset points",
                       xytext=(0, -16), ha="center", fontsize=9, color=_MESH_COLOR,
                       fontweight="bold")
    if ref_value is not None:
        ax.axhline(ref_value, color="#6b6b6b", linewidth=1.4, linestyle="--", zorder=2,
                  label="reference")

    ax.set_xlabel("Resolution N (regions/side for grid, elements/side for mesh)")
    ax.set_ylabel(label)
    ax.set_title("Convergence: homogenization grid vs. FE mesh resolution")
    ax.grid(True, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
