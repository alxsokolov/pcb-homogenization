#!/usr/bin/env python3
"""
One grouped bar chart of per-cycle creep-strain increment across several
board-model/mesh-resolution scenarios, reading each sample's saved
results/cycle_creep_increment.json -- no MAPDL session or .rst access
needed. Consolidates what used to be several separate
plot_comparison_creep.py time-series+bar figures into one summary.

--bar LABEL CATEGORY SAMPLE_DIR adds one bar (repeatable). CATEGORY is one
of "projection", "smeared", "zresolved" and only selects the bar's color
and the legend -- it does not change what's plotted.

Usage:
    python plot_delta_summary.py \
        --bar "20x20, nz=2\nprojection"              projection data/samples/rv3032_example \
        --bar "20x20, nz=2\nsmeared (241 leaves)"     smeared    data/samples/rv3032_homog \
        --out docs/overview/figures/delta_summary.png
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
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 11,
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.8,
    "grid.color": "#d5d5d0",
    "grid.linewidth": 0.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

_CATEGORY_COLOR = {
    "projection": "#2ca02c",
    "smeared": "#2a78d6",
    "zresolved": "#e8732c",
}
_CATEGORY_LABEL = {
    "projection": "binary projection",
    "smeared": "homogenized, smeared through thickness",
    "zresolved": "homogenized, z-resolved",
}


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
        description="One grouped bar chart of per-cycle Delta-epsilon across several "
                    "board-model/mesh scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bar", nargs=3, action="append", default=[],
                        metavar=("LABEL", "CATEGORY", "SAMPLE_DIR"),
                        help="x-axis label (\\n for a line break), category "
                             "(projection/smeared/zresolved), sample dir (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("delta_summary.png"))
    args = parser.parse_args()

    if not args.bar:
        parser.error("Provide at least one --bar LABEL CATEGORY SAMPLE_DIR")

    bars = []
    for label, category, d_str in args.bar:
        if category not in _CATEGORY_COLOR:
            parser.error(f"Unknown category {category!r} -- must be one of {list(_CATEGORY_COLOR)}")
        d = Path(d_str)
        if not d.is_dir():
            parser.error(f"Not a directory: {d}")
        bars.append((label.replace("\\n", "\n"), category, _delta_cracc(d), d.name))

    for label, category, delta, name in bars:
        print(f"{delta:.6e}  [{category:10s}] {name}  ({label!r})")

    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    x = range(len(bars))
    colors = [_CATEGORY_COLOR[c] for _, c, _, _ in bars]
    heights = [d for _, _, d, _ in bars]
    bar_objs = ax.bar(x, heights, color=colors, width=0.62, edgecolor="white",
                      linewidth=0.8, zorder=3)
    for bar, h in zip(bar_objs, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * max(heights), f"{h:.3e}",
               ha="center", va="bottom", fontsize=8.5, fontweight="bold", rotation=0)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for label, _, _, _ in bars], fontsize=8.2)
    ax.set_ylabel(r"Per-cycle $\Delta\varepsilon$ [-]")
    ax.set_title("Summary: board model and mesh resolution across the study")
    ax.grid(True, axis="y", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.3 * max(heights))

    seen = set()
    handles, labels = [], []
    for _, category, _, _ in bars:
        if category in seen:
            continue
        seen.add(category)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=_CATEGORY_COLOR[category]))
        labels.append(_CATEGORY_LABEL[category])
    ax.legend(handles, labels, fontsize=9.5, loc="upper center", ncol=3,
             bbox_to_anchor=(0.5, 1.13), frameon=False)

    fig.tight_layout()
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
