#!/usr/bin/env python3
"""
Compare n-layer board models solved on the same board.

Reads every <sample_dir>/results/pyansys_nlayer_*.json written by
simulate_pyansys_nlayer.py and reports how far each coarse-grained model
lands from the full-resolution raster reference -- which is the question the
whole homogenization pipeline exists to answer.

The raster model (--board-model raster: every element pure copper or pure
dielectric, straight from the layout) is treated as the reference whenever it
is present. It is not ground truth in an absolute sense, since it still
discretizes the layout onto the FE mesh; it is the same mesh with no
homogenization applied, which is exactly the right control for isolating what
the homogenization itself costs.

Metrics compared, all board-level:
  alpha_eff_x/y   effective in-plane CTE
  warpage_pv      top-surface peak-to-valley, best-fit plane removed
  warpage_rms     top-surface RMS
  von_mises_max   peak equivalent stress

Usage:
    python compare_nlayer_models.py data/samples/ublox_nlayer
    python compare_nlayer_models.py data/samples/ublox_nlayer --reference smeared
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots in fixed order, assigned per model and never recycled.
# Validated as a 6-slot set against the light surface #fcfcfb: lightness band
# PASS, chroma floor PASS, worst adjacent CVD dE 9.1 (protan) PASS,
# normal-vision floor dE 19.6 PASS. Three slots carry a contrast WARN against
# the surface, which is why every bar also gets a visible value label and the
# table above the plot repeats every number -- identity and magnitude are
# never left to colour alone.
_SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
_INK = "#1a1a1a"
_INK_MUTED = "#6b6b6b"

_METRICS = [
    ("alpha_eff_x_per_K", "α_eff,x", 1e6, "ppm/K"),
    ("alpha_eff_y_per_K", "α_eff,y", 1e6, "ppm/K"),
    ("warpage_pv_um", "warpage P–V", 1.0, "µm"),
    ("warpage_rms_um", "warpage RMS", 1.0, "µm"),
    ("von_mises_max_MPa", "von Mises max", 1.0, "MPa"),
]


def load_runs(sample_dir: Path) -> dict[str, dict]:
    results_dir = Path(sample_dir) / "results"
    runs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("pyansys_nlayer_*.json")):
        tag = path.stem[len("pyansys_nlayer_"):]
        try:
            runs[tag] = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"[warn] skipping {path.name}: {e}", file=sys.stderr)
    return runs


def print_table(runs: dict[str, dict], ref_tag: str | None) -> None:
    tags = sorted(runs, key=lambda t: (t != ref_tag, t))
    w = max(len(t) for t in tags)

    print(f"{'model':<{w}} {'regions':>8} {'materials':>9} "
          + " ".join(f"{label:>14}" for _, label, _, _ in _METRICS))
    print("-" * (w + 20 + 15 * len(_METRICS)))
    for tag in tags:
        r = runs[tag]
        disc = r.get("discretization") or {}
        n_reg = (disc.get("nx", 0) * disc.get("ny", 0) if disc.get("mode") == "grid"
                 else disc.get("max_leaves") or "-")
        n_mat = (r.get("mesh") or {}).get("n_materials_used", "-")
        cells = " ".join(f"{r.get(key, float('nan')) * scale:14.4f}"
                         for key, _, scale, _ in _METRICS)
        mark = "  <- reference" if tag == ref_tag else ""
        print(f"{tag:<{w}} {str(n_reg):>8} {str(n_mat):>9} {cells}{mark}")

    print("\nunits: " + ", ".join(f"{label} [{unit}]" for _, label, _, unit in _METRICS))

    if ref_tag is None or ref_tag not in runs:
        return
    ref = runs[ref_tag]
    print(f"\nDeviation from '{ref_tag}' [%]:")
    print(f"{'model':<{w}} " + " ".join(f"{label:>14}" for _, label, _, _ in _METRICS))
    print("-" * (w + 1 + 15 * len(_METRICS)))
    for tag in tags:
        if tag == ref_tag:
            continue
        cells = []
        for key, _, _, _ in _METRICS:
            a, b = runs[tag].get(key), ref.get(key)
            cells.append(f"{(a - b) / abs(b) * 100:+14.2f}"
                         if a is not None and b else f"{'-':>14}")
        print(f"{tag:<{w}} " + " ".join(cells))


def plot_deviation(runs: dict[str, dict], ref_tag: str, out_path: Path,
                   sample_id: str) -> None:
    """
    Signed deviation from the reference, one bar group per metric. One x axis
    in one unit (%), so the metrics are directly comparable -- which is the
    reason for plotting deviations rather than the raw values, whose units
    differ and would otherwise invite a second axis.
    """
    tags = [t for t in sorted(runs) if t != ref_tag]
    if not tags:
        return
    if len(tags) > len(_SERIES_COLORS):
        # Never recycle a categorical hue -- two models sharing a colour is
        # worse than not plotting one of them. The table above still lists
        # every run.
        print(f"[warn] {len(tags)} models to plot but only {len(_SERIES_COLORS)} "
              f"distinct colours; plotting the first {len(_SERIES_COLORS)}. "
              f"Use --reference to narrow the comparison.", file=sys.stderr)
        tags = tags[:len(_SERIES_COLORS)]
    ref = runs[ref_tag]
    labels = [label for _, label, _, _ in _METRICS]
    n_m, n_t = len(_METRICS), len(tags)

    fig, ax = plt.subplots(figsize=(8.4, 1.2 + 0.52 * n_m * n_t))
    height = 0.8 / n_t
    ypos = np.arange(n_m)

    for ti, tag in enumerate(tags):
        color = _SERIES_COLORS[ti % len(_SERIES_COLORS)]
        vals = []
        for key, _, _, _ in _METRICS:
            a, b = runs[tag].get(key), ref.get(key)
            vals.append((a - b) / abs(b) * 100 if a is not None and b else np.nan)
        offs = ypos - 0.4 + height * (ti + 0.5)
        ax.barh(offs, vals, height=height * 0.88, color=color,
                edgecolor="white", linewidth=0.8, label=tag, zorder=3)
        # Visible value labels on every bar: the aqua slot's contrast WARN
        # means magnitude must not rely on the fill alone.
        for y, v in zip(offs, vals):
            if np.isnan(v):
                continue
            ax.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:+.2f}%",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=8, color=_INK, zorder=4)

    ax.axvline(0, color=_INK_MUTED, linewidth=1.0, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.5, color=_INK)
    ax.invert_yaxis()
    ax.set_xlabel(f"deviation from '{ref_tag}' [%]", fontsize=10, color=_INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_INK_MUTED)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=_INK_MUTED, labelsize=8.5, width=0.8, length=3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=_INK_MUTED, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    # Headroom so the outermost value labels are not clipped.
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))

    n_cu = (ref.get("stackup") or {}).get("n_copper_layers", "?")
    # Legend above the axes: inside the plot it sits on top of the bars in
    # whichever metric group happens to be smallest. The title has to clear
    # however many rows the legend wraps onto.
    ncol = min(n_t, 4)
    legend_rows = int(np.ceil(n_t / ncol))
    ax.set_title(f"{sample_id} — board models vs '{ref_tag}' reference "
                 f"({n_cu} copper layers)",
                 fontsize=11.5, color=_INK, loc="left", pad=14 + 17 * legend_rows)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK, ncol=ncol,
              loc="lower left", bbox_to_anchor=(0.0, 1.005), handlelength=1.4,
              columnspacing=1.6, borderpad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare n-layer board models solved on the same board.")
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--reference", default=None,
                    help="Tag to compare against (default: 'raster' if present)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.sample_dir)
    if not runs:
        sys.exit(f"No results/pyansys_nlayer_*.json under {args.sample_dir} -- "
                 "run simulate_pyansys_nlayer.py first")

    ref_tag = args.reference or ("raster" if "raster" in runs else None)
    if args.reference and args.reference not in runs:
        sys.exit(f"--reference {args.reference!r} not among {sorted(runs)}")

    sample_id = next(iter(runs.values())).get("sample_id", args.sample_dir.name)
    print(f"{sample_id}: {len(runs)} board model(s)\n")
    print_table(runs, ref_tag)

    if ref_tag is None:
        print("\n[note] no 'raster' run present, so no reference to compare against. "
              "Solve one with --board-model raster for the deviations.", file=sys.stderr)
        return
    if args.no_plot or len(runs) < 2:
        return
    out = args.sample_dir / "results" / "nlayer_model_comparison.png"
    plot_deviation(runs, ref_tag, out, sample_id)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
