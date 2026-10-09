#!/usr/bin/env python3
"""
Per-cycle creep-strain increment vs. artificial copper-penetration depth,
in-plane mesh and leaf grid held fixed (all samples at --nx 50 --ny 50
--nz-per-layer 10), reading each sample's saved
results/cycle_creep_increment.json -- no MAPDL session or .rst needed.

--penetration N SAMPLE_DIR points at a sample run with
--cu-penetration-elements N (simulate_pyansys_assembly.py): the copper/FR4
classification (binary) or per-leaf bottom/top homogenized material
(--board-model homogenized_zlayers) extends N extra FR4-core z-elements
deep from each surface, beyond the true copper thickness. Pass --kind to
label the series ("binary projection" or "homogenized").

Usage:
    python plot_cu_penetration_sweep.py \
        --binary      0 data/samples/rv3032_projection_fine \
        --binary      1 data/samples/rv3032_cupen1_proj \
        --binary      3 data/samples/rv3032_cupen3_proj \
        --binary      5 data/samples/rv3032_cupen5_proj \
        --homogenized 0 data/samples/rv3032_homog_leaf50 \
        --homogenized 1 data/samples/rv3032_cupen1_homog \
        --homogenized 3 data/samples/rv3032_cupen3_homog \
        --homogenized 5 data/samples/rv3032_cupen5_homog \
        --out docs/overview/figures/cu_penetration_sweep.png
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

_BINARY_COLOR = "#2ca02c"
_HOMOG_COLOR = "#e8732c"


def _delta_cracc(sample_dir: Path) -> float:
    path = sample_dir / "results" / "cycle_creep_increment.json"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    cycles = json.loads(path.read_text())["cycles"]
    if not cycles:
        raise ValueError(f"No cycles recorded in {path}")
    return float(cycles[0]["delta_cracc_zone"])


def _read_points(parser, pairs):
    pts = []
    for n_str, d_str in pairs:
        n, d = int(n_str), Path(d_str)
        if not d.is_dir():
            parser.error(f"Not a directory: {d}")
        pts.append((n, _delta_cracc(d), d.name))
    pts.sort(key=lambda p: p[0])
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-cycle creep-strain increment vs. artificial copper "
                    "penetration depth into the FR4 core, binary vs. homogenized.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--binary", nargs=2, action="append", default=[],
                        metavar=("N", "SAMPLE_DIR"),
                        help="Penetration depth N (FR4-core elements) and its binary-"
                             "projection sample dir (repeatable)")
    parser.add_argument("--homogenized", nargs=2, action="append", default=[],
                        metavar=("N", "SAMPLE_DIR"),
                        help="Same as --binary, for homogenized_zlayers samples")
    parser.add_argument("--out", type=Path, default=Path("cu_penetration_sweep.png"))
    args = parser.parse_args()

    if not args.binary and not args.homogenized:
        parser.error("Provide at least one --binary or --homogenized N SAMPLE_DIR")

    binary_points = _read_points(parser, args.binary)
    homog_points = _read_points(parser, args.homogenized)

    for n, delta, name in binary_points:
        print(f"penetration={n}  delta_CRACC_zone = {delta:.6e}  [{name}, binary projection]")
    for n, delta, name in homog_points:
        print(f"penetration={n}  delta_CRACC_zone = {delta:.6e}  [{name}, homogenized]")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if binary_points:
        xs = [p[0] for p in binary_points]
        ys = [p[1] for p in binary_points]
        ax.plot(xs, ys, "D-", color=_BINARY_COLOR, linewidth=2, markersize=9, zorder=3,
                label="binary projection")
        for n, delta, _ in binary_points:
            ax.annotate(f"{delta:.3e}", (n, delta), textcoords="offset points",
                       xytext=(0, 10), ha="center", fontsize=9, color=_BINARY_COLOR,
                       fontweight="bold")

    if homog_points:
        xs = [p[0] for p in homog_points]
        ys = [p[1] for p in homog_points]
        ax.plot(xs, ys, "o-", color=_HOMOG_COLOR, linewidth=2, markersize=9, zorder=3,
                label="homogenized (per-leaf, z-resolved)")
        for n, delta, _ in homog_points:
            ax.annotate(f"{delta:.3e}", (n, delta), textcoords="offset points",
                       xytext=(0, -16), ha="center", fontsize=9, color=_HOMOG_COLOR,
                       fontweight="bold")

    ax.set_xlabel("Cu penetration depth (FR4-core elements beyond the true copper thickness)")
    ax.set_ylabel(r"Per-cycle $\Delta\varepsilon$ [-]")
    ax.set_title("Sensitivity to artificial copper-layer z-thickness")
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
