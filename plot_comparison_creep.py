#!/usr/bin/env python3
"""
Compare accumulated creep strain and per-cycle increment across board-model
variants (projection, homogenized, homogenized z-layers), reading the
pre-computed zone_creep_strain.json and cycle_creep_increment.json from each
sample's results/ directory -- no MAPDL session or .rst access needed.

Two-panel figure:
  Left  -- zone-averaged accumulated creep strain (CRACC_zone) vs. time, one
           curve per sample.
  Right -- per-cycle creep-strain increment (delta_CRACC_zone) as a bar chart,
           one bar per sample.

Usage:
    python plot_comparison_creep.py \
        data/samples/rv3032_example \
        data/samples/rv3032_homog \
        data/samples/rv3032_homog_quadtree \
        data/samples/rv3032_homog_quadtree_z

    python plot_comparison_creep.py \
        data/samples/rv3032_example \
        data/samples/rv3032_homog_quadtree_z \
        --out docs/overview/figures/comparison_creep.png
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

# One colour per sample, stable across argument order changes.
_COLORS = [
    "#2a78d6",  # blue
    "#e8732c",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#d62728",  # red
]


def _default_label(sample_dir: Path) -> str:
    """Derive a readable label from the directory name."""
    name = sample_dir.name
    if name == "rv3032_example":
        return "projection"
    if name == "rv3032_reference":
        return "projection (fine z-mesh reference)"
    if name == "rv3032_homog":
        return "homogenized (241 leaves)"
    if name == "rv3032_homog_z":
        return "homogenized z-layers (241 leaves)"
    if name == "rv3032_homog_quadtree":
        return "homogenized (19 leaves)"
    if name == "rv3032_homog_quadtree_z":
        return "homogenized z-layers (19 leaves)"
    if name == "rv3032_homog_perelement":
        return "homogenized z-layers (1 leaf/element, 20x20 mesh)"
    if name == "rv3032_projection_fine":
        return "projection (50x50 mesh)"
    if name == "rv3032_homog_perelement_finemesh":
        return "homogenized z-layers (20x20 leaves, 50x50 mesh)"
    return name


def _load_sample(sample_dir: Path) -> dict:
    """Load zone_creep_strain.json and cycle_creep_increment.json from a sample."""
    results = sample_dir / "results"
    zone_path = results / "zone_creep_strain.json"
    cycle_path = results / "cycle_creep_increment.json"
    if not zone_path.exists():
        raise FileNotFoundError(f"Not found: {zone_path}")
    if not cycle_path.exists():
        raise FileNotFoundError(f"Not found: {cycle_path}")
    zone = json.loads(zone_path.read_text())
    cycle = json.loads(cycle_path.read_text())
    return {
        "label": _default_label(sample_dir),
        "time_s": np.asarray(zone["time_s"], dtype=float),
        "cracc_zone": np.asarray(zone["cracc_zone"], dtype=float),
        "cycles": cycle["cycles"],
    }


def plot_comparison(samples: list[dict], out_path: Path) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left panel: accumulated creep strain vs time ---
    for i, s in enumerate(samples):
        color = _COLORS[i % len(_COLORS)]
        axL.plot(s["time_s"], s["cracc_zone"], "-", color=color, linewidth=2,
                 label=s["label"])
    axL.set_xlabel("Time [s]")
    axL.set_ylabel("Zone-averaged accumulated creep strain [-]")
    axL.set_title("Accumulated creep strain (CRACC_zone)")
    axL.legend(fontsize=10)
    axL.grid(True, alpha=0.6, zorder=0)
    axL.set_axisbelow(True)
    axL.spines[["top", "right"]].set_visible(False)

    # --- Right panel: per-cycle increment bar chart ---
    labels = [s["label"] for s in samples]
    deltas = [s["cycles"][-1]["delta_cracc_zone"] if s["cycles"] else np.nan
              for s in samples]
    x = np.arange(len(samples))
    bars = axR.bar(x, deltas, color=[_COLORS[i % len(_COLORS)] for i in range(len(samples))],
                   width=0.6, edgecolor="white", linewidth=0.8, zorder=3)
    axR.set_xticks(x)
    axR.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    axR.set_ylabel("Per-cycle creep strain increment [-]")
    axR.set_title("Per-cycle $\\Delta\\varepsilon$ (cycle 1)")
    axR.grid(True, axis="y", alpha=0.6, zorder=0)
    axR.set_axisbelow(True)
    axR.spines[["top", "right"]].set_visible(False)
    # Value labels on top of each bar
    y_max = max(deltas) if deltas else 0
    for bar, val in zip(bars, deltas):
        axR.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * y_max,
                 f"{val:.4e}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axR.set_ylim(0, 1.15 * y_max if y_max > 0 else 1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare accumulated creep strain and per-cycle increment "
                    "across board-model variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dirs", type=Path, nargs="+",
                        help="Two or more sample directories (each must have "
                             "results/zone_creep_strain.json and "
                             "results/cycle_creep_increment.json)")
    parser.add_argument("--out", type=Path, default=Path("comparison_creep.png"),
                        help="Output PNG path")
    args = parser.parse_args()

    if len(args.sample_dirs) < 2:
        parser.error("Provide at least two sample directories to compare.")

    samples = []
    for d in args.sample_dirs:
        if not d.is_dir():
            parser.error(f"Not a directory: {d}")
        s = _load_sample(d)
        print(f"Loaded {d.name}: {s['label']}, "
              f"{s['time_s'].size} time points, "
              f"{len(s['cycles'])} cycle(s)")
        samples.append(s)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(samples, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()