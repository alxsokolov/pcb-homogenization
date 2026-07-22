#!/usr/bin/env python3
"""
Visualize a sample's quadtree coarse-graining (produced by quadtree.py).

Panel A: copper raster with quadtree leaf boundaries overlaid, edge color by
         refinement level -- shows refinement concentrating at copper/FR4
         boundaries and staying coarse over uniform copper or open FR4.
Panel B: leaves filled by their per-leaf effective in-plane E (Voigt) --
         the piecewise-constant coarse-grained stiffness field.
Panel C: zoomed-in view around the one component tracked in
         solder_reliability.py (COMPONENT_NAME/PAD_1_XY_MM/PAD_2_XY_MM) --
         at whole-board scale a small SMD pad is barely distinguishable from
         a similarly-thick trace, so the zoom is what actually lets you
         confirm the marker sits on the real pad, not just near it.

Output: <sample_dir>/mesh/quadtree_preview.png

Usage:
    python plot_quadtree.py data/samples/dummy_test
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from solder_reliability import COMPONENT_NAME, PAD_1_XY_MM, PAD_2_XY_MM


def _load_background(mesh_dir: Path) -> np.ndarray:
    """Combined (top OR bottom) copper raster for the panels' backdrop."""
    disc_path = mesh_dir / "discretization.json"
    top = bot = None
    if disc_path.exists():
        disc = json.loads(disc_path.read_text())
        for _lname, linfo in disc.get("layer_maps", {}).items():
            pos = linfo.get("position")
            fpath = mesh_dir / linfo["file"]
            if pos == "top" and fpath.exists():
                top = np.load(fpath)
            elif pos == "bottom" and fpath.exists():
                bot = np.load(fpath)
    if top is None and bot is None:
        return np.load(mesh_dir / "material_map.npy")
    if top is None:
        top = np.zeros_like(bot)
    if bot is None:
        bot = np.zeros_like(top)
    return np.maximum(top, bot)


def _plot_component_marker(ax, label_pads: bool = False) -> None:
    xs = [PAD_1_XY_MM[0], PAD_2_XY_MM[0]]
    ys = [PAD_1_XY_MM[1], PAD_2_XY_MM[1]]
    ax.scatter(xs, ys, marker="+", c="red", s=120, linewidths=1.8, zorder=5)
    if label_pads:
        for (x, y), name in zip([PAD_1_XY_MM, PAD_2_XY_MM], ["pad1", "pad2"]):
            ax.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points",
                       color="red", fontsize=9, zorder=5)
    else:
        cx, cy = sum(xs) / 2, sum(ys) / 2
        ax.annotate(COMPONENT_NAME, (cx, cy), xytext=(4, 4), textcoords="offset points",
                   color="red", fontsize=9, fontweight="bold", zorder=5)


def _plot_component_zoom(ax, background: np.ndarray, leaves: list[dict],
                         Lx: float, Ly: float, max_level: int) -> None:
    xs = [PAD_1_XY_MM[0], PAD_2_XY_MM[0]]
    ys = [PAD_1_XY_MM[1], PAD_2_XY_MM[1]]
    cx, cy = sum(xs) / 2, sum(ys) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.5)
    margin = max(1.5, span * 1.5)
    x0, x1 = max(0.0, cx - margin), min(Lx, cx + margin)
    y0, y1 = max(0.0, cy - margin), min(Ly, cy + margin)

    ax.imshow(background, origin="lower", extent=(0, Lx, 0, Ly),
              cmap="gray", vmin=0, vmax=1)
    for leaf in leaves:
        lx0, ly0, lx1, ly1 = leaf["bbox_mm"]
        if lx1 < x0 or lx0 > x1 or ly1 < y0 or ly0 > y1:
            continue
        color = plt.cm.viridis(leaf["level"] / max_level)
        ax.add_patch(Rectangle((lx0, ly0), lx1 - lx0, ly1 - ly0,
                               fill=False, edgecolor=color, linewidth=1.0))
    _plot_component_marker(ax, label_pads=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{COMPONENT_NAME} zoom (pad spacing {span:.3f} mm)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a sample's quadtree coarse-graining and per-leaf homogenized stiffness.",
    )
    parser.add_argument("sample_dir", type=Path)
    args = parser.parse_args()

    mesh_dir = args.sample_dir / "mesh"
    qt_path = mesh_dir / "quadtree.json"
    if not qt_path.exists():
        sys.exit(f"Run quadtree.py first — not found: {qt_path}")

    qt = json.loads(qt_path.read_text())
    Lx, Ly = qt["board"]["width_mm"], qt["board"]["height_mm"]
    leaves = qt["leaves"]
    max_level = max(1, qt["max_level"])
    levels = [leaf["level"] for leaf in leaves]

    background = _load_background(mesh_dir)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(19, 6))

    axA.imshow(background, origin="lower", extent=(0, Lx, 0, Ly),
              cmap="gray", vmin=0, vmax=1)
    for leaf in leaves:
        x0, y0, x1, y1 = leaf["bbox_mm"]
        color = plt.cm.viridis(leaf["level"] / max_level)
        axA.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                fill=False, edgecolor=color, linewidth=0.6))
    axA.set_title(f"Quadtree ({len(leaves)} leaves, levels {min(levels)}–{max(levels)})")
    axA.set_xlabel("x [mm]")
    axA.set_ylabel("y [mm]")
    axA.set_xlim(0, Lx)
    axA.set_ylim(0, Ly)
    axA.set_aspect("equal")
    _plot_component_marker(axA)

    vals = [leaf["E_xy_voigt_GPa"] for leaf in leaves]
    norm = Normalize(vmin=min(vals), vmax=max(vals))
    cmap = plt.cm.plasma
    for leaf in leaves:
        x0, y0, x1, y1 = leaf["bbox_mm"]
        axB.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                facecolor=cmap(norm(leaf["E_xy_voigt_GPa"])),
                                edgecolor="none"))
    axB.set_title("Per-leaf effective in-plane E (Voigt)")
    axB.set_xlabel("x [mm]")
    axB.set_ylabel("y [mm]")
    axB.set_xlim(0, Lx)
    axB.set_ylim(0, Ly)
    axB.set_aspect("equal")
    _plot_component_marker(axB)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axB, label="E_xy_voigt [GPa]")

    _plot_component_zoom(axC, background, leaves, Lx, Ly, max_level)

    fig.tight_layout()
    out_path = mesh_dir / "quadtree_preview.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
