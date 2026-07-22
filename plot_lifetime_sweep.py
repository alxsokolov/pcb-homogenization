#!/usr/bin/env python3
"""
Plot <sample_dir>/results/lifetime_sweep.json (produced by
sweep_quadtree_lifetime.py) -- separated from the sweep itself so plot
styling can be iterated on without re-running the (much more expensive) FE
sweep.

Two figures, each independently stylable via --style / --pct-style:
  - lifetime_sweep_3d_<style>.png       N_f vs (n_leaves, max_level), with
                                        the benchmark N_f as a reference
                                        plane (scatter/surface only --
                                        skipped for bar3d, see below).
  - lifetime_sweep_pct_diff_3d_<style>.png
                                        percent difference from benchmark
                                        vs (n_leaves, max_level).

--style choices:
  scatter  one point per (leaves, level) combination.
  surface  plot_surface over the regular (leaves x level) grid -- reads as
           a continuous trend, but only as smooth as the sweep is dense.
  bar3d    one 3D bar per combination, height = value. No benchmark plane
           for the first plot in this mode (a translucent plane mixed with
           opaque bars reads poorly) -- the benchmark value is still
           printed in the title.

Output filenames are tagged with the style used, so trying several styles
doesn't overwrite previous ones -- easy to compare side by side.

Usage:
    python plot_lifetime_sweep.py data/samples/rv3032_real
    python plot_lifetime_sweep.py data/samples/rv3032_real --style bar3d
    python plot_lifetime_sweep.py data/samples/rv3032_real --style surface --pct-style bar3d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3D projection

_STYLES = ("scatter", "surface", "bar3d")


def _build_grid(records: list[dict], value_key: str):
    """Regular (max_level x max_leaves_requested) grid -> X=n_leaves_actual,
    Y=max_level, Z=value_key, indexed by dict lookup so grid assembly is
    correct regardless of the records' order."""
    levels_sorted = sorted({r["max_level"] for r in records})
    leaves_sorted = sorted({r["max_leaves_requested"] for r in records})
    by_key = {(r["max_level"], r["max_leaves_requested"]): r for r in records}

    X = np.zeros((len(levels_sorted), len(leaves_sorted)))
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for i, lvl in enumerate(levels_sorted):
        for j, lf in enumerate(leaves_sorted):
            r = by_key[(lvl, lf)]
            X[i, j] = r["n_leaves_actual"]
            Y[i, j] = lvl
            Z[i, j] = r[value_key]
    return X, Y, Z


def _draw(fig, ax, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
         style: str, cmap: str, cbar_label: str) -> None:
    norm = plt.Normalize(np.nanmin(Z), np.nanmax(Z))

    if style == "scatter":
        sc = ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c=Z.ravel(),
                        cmap=cmap, norm=norm, s=60, depthshade=True)
        fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.6, pad=0.1)

    elif style == "surface":
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, norm=norm,
                               edgecolor="k", linewidth=0.3, antialiased=True)
        fig.colorbar(surf, ax=ax, label=cbar_label, shrink=0.6, pad=0.1)

    elif style == "bar3d":
        xs, ys, zs = X.ravel(), Y.ravel(), Z.ravel()
        dx = (X.max() - X.min()) / max(X.shape[1] - 1, 1) * 0.6 if X.shape[1] > 1 else 1.0
        dy = (Y.max() - Y.min()) / max(Y.shape[0] - 1, 1) * 0.6 if Y.shape[0] > 1 else 1.0
        colors = plt.get_cmap(cmap)(norm(zs))
        # Base bars near the data's own minimum (with a small margin so the
        # shortest bar stays visible), not at 0 -- these sweeps vary by only
        # ~1% of their absolute value, so a base at 0 makes every bar look
        # the same height (color would still vary, height wouldn't). Basing
        # near Z.min() makes bar height actually carry the variation, matching
        # the scatter/surface styles' autoscaled z-axis.
        zrange = Z.max() - Z.min()
        zbase = Z.min() - 0.05 * zrange if zrange > 0 else Z.min() - 1.0
        ax.bar3d(xs - dx / 2, ys - dy / 2, np.full_like(zs, zbase), dx, dy, zs - zbase,
                 color=colors, shade=True)
        ax.set_zlim(zbase, Z.max() + 0.05 * zrange if zrange > 0 else Z.max() + 1.0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=cbar_label, shrink=0.6, pad=0.1)

    else:
        raise ValueError(f"Unknown style {style!r} -- choose from {_STYLES}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--style", choices=_STYLES, default="scatter",
                        help="Style for the raw N_f plot")
    parser.add_argument("--pct-style", choices=_STYLES, default=None,
                        help="Style for the percent-diff plot; defaults to --style")
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()

    json_path = args.sample_dir / "results" / "lifetime_sweep.json"
    if not json_path.exists():
        sys.exit(f"Run sweep_quadtree_lifetime.py first — not found: {json_path}")
    data = json.loads(json_path.read_text())
    records = data["sweep"]
    benchmark_N_f = data["benchmark"]["N_f_cycles"]
    component = data.get("component", "component")
    pct_style = args.pct_style or args.style

    results_dir = args.sample_dir / "results"

    # --- Plot 1: raw N_f ----------------------------------------------------
    X, Y, Z = _build_grid(records, "N_f_cycles")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw(fig, ax, X, Y, Z, args.style, args.cmap, "N_f [cycles]")
    if args.style != "bar3d":
        xx, yy = np.meshgrid(np.linspace(X.min(), X.max(), 2), np.linspace(Y.min(), Y.max(), 2))
        zz = np.full_like(xx, benchmark_N_f)
        ax.plot_surface(xx, yy, zz, alpha=0.25, color="red")
        ax.text(X.max(), Y.max(), benchmark_N_f, f"  benchmark N_f={benchmark_N_f:.0f}", color="red")
    ax.set_xlabel("n_leaves (actual)")
    ax.set_ylabel("max_level")
    ax.set_zlabel("N_f [cycles]")
    title = f"Joint lifetime vs quadtree coarse-graining ({component}) [{args.style}]"
    if args.style == "bar3d":
        title += f"  (benchmark N_f={benchmark_N_f:.0f})"
    ax.set_title(title)
    fig.tight_layout()
    out1 = results_dir / f"lifetime_sweep_3d_{args.style}.png"
    fig.savefig(out1, dpi=150)
    print(f"Wrote {out1}")

    # --- Plot 2: percent difference from benchmark --------------------------
    X2, Y2, Z2 = _build_grid(records, "pct_diff_from_benchmark")
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection="3d")
    _draw(fig2, ax2, X2, Y2, Z2, pct_style, args.cmap, "N_f % difference from benchmark")
    ax2.set_xlabel("n_leaves (actual)")
    ax2.set_ylabel("max_level")
    ax2.set_zlabel("N_f % difference from benchmark")
    ax2.set_title(f"Lifetime bias vs quadtree coarse-graining ({component}) [{pct_style}]")
    fig2.tight_layout()
    out2 = results_dir / f"lifetime_sweep_pct_diff_3d_{pct_style}.png"
    fig2.savefig(out2, dpi=150)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
