#!/usr/bin/env python3
"""
Show how in-plane (quadtree leaf count) and out-of-plane (smeared vs.
per-copper-layer) discretization each shift the per-cycle creep-strain
increment away from a full-fidelity projection reference, reading the
pre-computed cycle_creep_increment.json from each sample's results/
directory -- no MAPDL session or .rst access needed.

Grouped bar chart: two groups on the x-axis (coarse / fine in-plane quadtree),
two bars per group (smeared through thickness / resolved per copper layer),
plotted as % deviation of delta_CRACC_zone (cycle 1) from the projection
reference, which sits at the dashed 0% line.

Usage:
    python plot_discretization_sensitivity.py \
        --reference       data/samples/rv3032_example \
        --coarse-smeared  data/samples/rv3032_homog_quadtree \
        --coarse-zlayers  data/samples/rv3032_homog_quadtree_z \
        --fine-smeared    data/samples/rv3032_homog \
        --fine-zlayers    data/samples/rv3032_homog_z \
        --out docs/overview/figures/discretization_sensitivity.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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

_SMEARED_COLOR = "#2a78d6"   # blue -- homogenized, smeared through thickness
_ZLAYERS_COLOR = "#e8732c"  # orange -- homogenized, per-copper-layer


def _delta_cracc(sample_dir: Path) -> float:
    """Cycle-1 per-cycle creep-strain increment from a sample's results/."""
    path = sample_dir / "results" / "cycle_creep_increment.json"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    cycles = json.loads(path.read_text())["cycles"]
    if not cycles:
        raise ValueError(f"No cycles recorded in {path}")
    return float(cycles[0]["delta_cracc_zone"])


def plot_sensitivity(reference: float, coarse_smeared: float, coarse_zlayers: float,
                     fine_smeared: float, fine_zlayers: float,
                     n_leaves_coarse: int, n_leaves_fine: int, out_path: Path) -> None:
    pct = lambda v: 100.0 * (v - reference) / reference

    groups = [f"coarse quadtree\n({n_leaves_coarse} leaves)",
              f"fine quadtree\n({n_leaves_fine} leaves)"]
    smeared_vals = [pct(coarse_smeared), pct(fine_smeared)]
    zlayers_vals = [pct(coarse_zlayers), pct(fine_zlayers)]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(groups))
    width = 0.32

    bars_s = ax.bar(x - width / 2, smeared_vals, width, color=_SMEARED_COLOR,
                    edgecolor="white", linewidth=0.8, zorder=3,
                    label="smeared through thickness")
    bars_z = ax.bar(x + width / 2, zlayers_vals, width, color=_ZLAYERS_COLOR,
                    edgecolor="white", linewidth=0.8, zorder=3,
                    label="resolved per copper layer")

    ax.axhline(0.0, color="#2ca02c", linewidth=1.8, linestyle="--", zorder=2,
              label="projection reference")

    for bars in (bars_s, bars_z):
        for bar in bars:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = 0.4 if h >= 0 else -0.4
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:+.1f}%",
                   ha="center", va=va, fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(r"Per-cycle $\Delta\varepsilon$ deviation from projection [%]")
    ax.set_title("Sensitivity to in-plane vs. out-of-plane discretization")
    ax.grid(True, axis="y", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="center", bbox_to_anchor=(0.5, 0.28))

    y_vals = smeared_vals + zlayers_vals + [0.0]
    pad = 0.15 * (max(y_vals) - min(y_vals) if max(y_vals) != min(y_vals) else 1.0)
    ax.set_ylim(min(y_vals) - pad, max(y_vals) + pad)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped bar chart of in-plane vs. out-of-plane discretization "
                    "sensitivity, relative to a full-fidelity projection reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", type=Path, required=True,
                        help="Full-fidelity projection sample (the 0%% baseline)")
    parser.add_argument("--coarse-smeared", type=Path, required=True)
    parser.add_argument("--coarse-zlayers", type=Path, required=True)
    parser.add_argument("--fine-smeared", type=Path, required=True)
    parser.add_argument("--fine-zlayers", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("discretization_sensitivity.png"))
    args = parser.parse_args()

    dirs = {
        "reference": args.reference,
        "coarse_smeared": args.coarse_smeared,
        "coarse_zlayers": args.coarse_zlayers,
        "fine_smeared": args.fine_smeared,
        "fine_zlayers": args.fine_zlayers,
    }
    for label, d in dirs.items():
        if not d.is_dir():
            parser.error(f"Not a directory: {d}")

    vals = {label: _delta_cracc(d) for label, d in dirs.items()}
    for label, v in vals.items():
        print(f"{label:16s} {dirs[label].name:28s} delta_CRACC_zone = {v:.6e}")

    def n_leaves(sample_dir: Path) -> int:
        qt = json.loads((sample_dir / "mesh" / "quadtree.json").read_text())
        return int(qt["n_leaves"])

    n_coarse = n_leaves(args.coarse_smeared)
    n_fine = n_leaves(args.fine_smeared)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_sensitivity(vals["reference"], vals["coarse_smeared"], vals["coarse_zlayers"],
                     vals["fine_smeared"], vals["fine_zlayers"], n_coarse, n_fine, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
