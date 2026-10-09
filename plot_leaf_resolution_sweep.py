#!/usr/bin/env python3
"""
Per-cycle creep-strain increment vs. in-plane leaf resolution, FE mesh held
fixed (all samples meshed at the same --nx --ny --nz-per-layer), reading
each sample's saved results/cycle_creep_increment.json -- no MAPDL session
or .rst access needed.

--leaves N SAMPLE_DIR points at a sample homogenized on an N x N leaf grid
(quadtree.py --per-element-grid N N) re-applied to one fixed, fine FE mesh
-- an in-plane *material*-resolution sweep (orange). --projection N
SAMPLE_DIR points at a binary copper/FR4 projection meshed at N x N
in-plane (same nz-per-layer throughout) -- an in-plane *mesh*-resolution
sweep (green), answering a different question: not "how many leaves does
the homogenization need" but "does the binary projection itself need a
finer mesh". Both series share the plot only for visual comparison; their
x-axis N means a leaf count for one and a mesh count for the other.

Usage:
    python plot_leaf_resolution_sweep.py \
        --leaves 5  data/samples/rv3032_homog_leaf5 \
        --leaves 10 data/samples/rv3032_homog_leaf10 \
        --leaves 25 data/samples/rv3032_homog_leaf25 \
        --leaves 50 data/samples/rv3032_homog_leaf50 \
        --projection 20 data/samples/rv3032_reference \
        --projection 30 data/samples/rv3032_projection_30 \
        --projection 50 data/samples/rv3032_projection_fine \
        --out docs/overview/figures/leaf_resolution_sweep.png
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

_HOMOG_COLOR = "#e8732c"
_REF_COLOR = "#2ca02c"
_MISALIGNED_COLOR = "#9467bd"


def _delta_cracc(sample_dir: Path) -> float:
    path = sample_dir / "results" / "cycle_creep_increment.json"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    cycles = json.loads(path.read_text())["cycles"]
    if not cycles:
        raise ValueError(f"No cycles recorded in {path}")
    return float(cycles[0]["delta_cracc_zone"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-cycle creep-strain increment vs. in-plane leaf "
                    "resolution at a fixed FE mesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--leaves", nargs=2, action="append", default=[],
                        metavar=("N", "SAMPLE_DIR"),
                        help="Leaf-grid resolution N and its sample dir (repeatable)")
    parser.add_argument("--leaves-misaligned", nargs=2, action="append", default=[],
                        metavar=("N", "SAMPLE_DIR"),
                        help="Same as --leaves, but N does not evenly divide the FE mesh's "
                             "--nx/--ny (leaf boundaries fall mid-element) -- plotted separately, "
                             "off the main convergence line, not implying a trend.")
    parser.add_argument("--projection", nargs=2, action="append", default=[],
                        metavar=("N", "SAMPLE_DIR"),
                        help="Binary projection meshed at N x N in-plane (repeatable) -- an "
                             "in-plane mesh-resolution sweep, plotted as its own series")
    parser.add_argument("--out", type=Path, default=Path("leaf_resolution_sweep.png"))
    args = parser.parse_args()

    if not args.leaves:
        parser.error("Provide at least one --leaves N SAMPLE_DIR")

    def _read_points(pairs):
        pts = []
        for n_str, d_str in pairs:
            n, d = int(n_str), Path(d_str)
            if not d.is_dir():
                parser.error(f"Not a directory: {d}")
            pts.append((n, _delta_cracc(d), d.name))
        pts.sort(key=lambda p: p[0])
        return pts

    points = _read_points(args.leaves)
    misaligned_points = _read_points(args.leaves_misaligned)
    projection_points = _read_points(args.projection)

    for n, delta, name in points:
        print(f"{n:3d} leaves/side  delta_CRACC_zone = {delta:.6e}  [{name}, homogenized]")
    for n, delta, name in misaligned_points:
        print(f"{n:3d} leaves/side  delta_CRACC_zone = {delta:.6e}  [{name}, homogenized, MISALIGNED]")
    for n, delta, name in projection_points:
        print(f"{n:3d} mesh/side     delta_CRACC_zone = {delta:.6e}  [{name}, binary projection]")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, "o-", color=_HOMOG_COLOR, linewidth=2, markersize=8, zorder=3,
            label="homogenized (per-leaf, z-resolved, clean)")
    for n, delta, _ in points:
        ax.annotate(f"{delta:.3e}", (n, delta), textcoords="offset points",
                   xytext=(0, 10), ha="center", fontsize=9)

    if misaligned_points:
        mxs = [p[0] for p in misaligned_points]
        mys = [p[1] for p in misaligned_points]
        ax.plot(mxs, mys, "^", color=_MISALIGNED_COLOR, markersize=10, zorder=3, linestyle="none",
                label="homogenized (leaf grid misaligned with FE mesh)")
        for n, delta, _ in misaligned_points:
            ax.annotate(f"{delta:.3e}", (n, delta), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=9, color=_MISALIGNED_COLOR)

    if projection_points:
        pxs = [p[0] for p in projection_points]
        pys = [p[1] for p in projection_points]
        ax.plot(pxs, pys, "D-", color=_REF_COLOR, linewidth=2, markersize=10, zorder=4,
                label="binary projection (mesh resolution)")
        for n, delta, _ in projection_points:
            ax.annotate(f"{delta:.3e}", (n, delta), textcoords="offset points",
                       xytext=(0, -16), ha="center", fontsize=9, color=_REF_COLOR,
                       fontweight="bold")

    ax.set_xlabel("Resolution N (leaves/side for homogenized, mesh/side for projection)")
    ax.set_ylabel(r"Per-cycle $\Delta\varepsilon$ [-]")
    ax.set_title("Homogenized leaf resolution vs. projection mesh resolution")
    ax.grid(True, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10)

    fig.tight_layout()
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
