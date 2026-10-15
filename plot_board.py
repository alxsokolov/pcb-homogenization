#!/usr/bin/env python3
"""
Preview a board model before any MAPDL run: the copper layout, the
homogenization grid's per-layer copper fraction, and the stackup through
thickness. Needs only discretize.py's output; mesh/homogenization.json
(from homogenize.py) is optional -- without it, only panels A and C are
drawn.

Panel A: top copper layout (mesh/raster_<top layer>.npy from discretize.py).
Panel B: one small-multiple per copper layer, each homogenization-grid
         region shaded by its copper area fraction, on ONE shared 0-1
         colorbar so panels are directly comparable -- which is the whole
         point, and why this is not one colored series per layer on a
         single plot.
Panel C: the stackup through thickness: each copper slab's board-wide
         copper fraction against its true z position and thickness, with
         the dielectric slabs drawn to scale between them.

Output: <sample_dir>/results/board_preview.png

Usage:
    python plot_board.py data/samples/my_board
    python plot_board.py data/samples/my_board --tag grid40x40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from pcbh import grid as gridmod
from pcbh.stackup import load_board

LAYOUT_CU, LAYOUT_FR4 = "#c8702a", "#cfd8c4"
_SEQ_CMAP = "Blues"
_PHI_VMIN, _PHI_VMAX = 0.0, 1.0
_INK = "#1a1a1a"
_INK_MUTED = "#6b6b6b"
_DIELECTRIC_FILL = "#e4e0d8"
_DIELECTRIC_EDGE = "#c9c3b8"


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_INK_MUTED, labelsize=8, width=0.8)


def plot_layout(ax, raster: np.ndarray, W: float, H: float) -> None:
    ax.imshow(raster, origin="lower", extent=(0, W, 0, H), interpolation="nearest",
              cmap=ListedColormap([LAYOUT_FR4, LAYOUT_CU]), vmin=0, vmax=1)
    ax.set_title(f"Top copper layout ({raster.mean():.1%} copper)", fontsize=10)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")


def plot_layer_panels(fig, model, board, out_path: Path) -> None:
    """One small-multiple per copper layer, top of board first."""
    copper_slabs_top_first = sorted(board.copper_slabs, key=lambda s: -s.z0_mm)
    n = len(copper_slabs_top_first)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    W, H = board.width_mm, board.height_mm
    panel_w = 2.9
    panel_h = panel_w * (H / W)
    fig2, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * panel_w + 1.4, nrows * panel_h + 1.5),
                              squeeze=False)

    mappable = None
    for pos, s in enumerate(copper_slabs_top_first):
        ax = axes[pos // ncols][pos % ncols]
        phi_grid = model.layer_phi[s.index]
        im = ax.imshow(phi_grid, origin="lower", extent=(0, W, 0, H),
                       cmap=_SEQ_CMAP, vmin=_PHI_VMIN, vmax=_PHI_VMAX, interpolation="nearest")
        mappable = im
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        _style_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        name = s.layer_name or s.name
        ax.set_title(name, fontsize=10, color=_INK, pad=14)
        ax.text(0.0, 1.015,
                f"z {s.z0_mm:.3f}–{s.z1_mm:.3f} mm · {s.thickness_mm * 1e3:.0f} µm "
                f"· φ̄ {float(phi_grid.mean()):.3f}",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5, color=_INK_MUTED)

    for pos in range(n, nrows * ncols):
        axes[pos // ncols][pos % ncols].axis("off")

    fig2.tight_layout(rect=(0, 0.02, 0.93, 0.80))
    fig2.suptitle(f"{board.sample_id} — copper area fraction per region, {n} copper layers "
                  f"(top left = top of board)", fontsize=12, color=_INK, y=0.97)
    fig2.text(0.5, 0.90,
              f"{model.gx} x {model.gy} grid · region model {model.region_model} · "
              f"{len(model.materials)} distinct materials",
              ha="center", fontsize=9, color=_INK_MUTED)
    cax = fig2.add_axes([0.945, 0.12, 0.015, 0.72])
    cb = fig2.colorbar(mappable, cax=cax)
    cb.set_label("copper area fraction φ", fontsize=9, color=_INK)
    cb.ax.tick_params(colors=_INK_MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig2.savefig(out_path, dpi=150)
    plt.close(fig2)


def plot_thickness_profile(ax, board, model=None) -> None:
    """Copper fraction against true z. Copper slabs are bars at their real
    thickness and position; dielectrics are drawn to scale between them."""
    cmap = matplotlib.colormaps[_SEQ_CMAP]
    for s in board.slabs:
        if not s.is_copper:
            ax.add_patch(Rectangle((0, s.z0_mm), 1.0, s.thickness_mm,
                                   facecolor=_DIELECTRIC_FILL,
                                   edgecolor=_DIELECTRIC_EDGE, linewidth=0.6, zorder=1))
            continue
        phi = (float(model.layer_phi[s.index].mean()) if model is not None
              else float(s.raster.mean()))
        ax.add_patch(Rectangle((0, s.z0_mm), phi, s.thickness_mm,
                               facecolor=cmap(0.25 + 0.7 * phi),
                               edgecolor="white", linewidth=0.8, zorder=3))
        name = s.layer_name or s.name
        ax.text(min(phi + 0.015, 0.99), s.z_mid_mm,
                f"{name}  φ {phi:.3f}  ({s.thickness_mm * 1e3:.0f} µm)",
                va="center", ha="left", fontsize=8.5, color=_INK, zorder=4)

    total_t = board.total_thickness_mm
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, total_t)
    _style_axes(ax)
    ax.set_xlabel("copper area fraction φ", fontsize=10, color=_INK)
    ax.set_ylabel("z above bottom face [mm]", fontsize=10, color=_INK)
    ax.grid(axis="x", color=_INK_MUTED, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    cu_t = sum(s.thickness_mm for s in board.copper_slabs)
    ax.set_title(f"Stackup through thickness\n{board.n_copper} copper layers, "
                f"{total_t * 1e3:.0f} µm total, copper {cu_t / total_t * 100:.0f}% of thickness",
                fontsize=10.5, color=_INK, loc="left")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preview a board model (layout + homogenization grid + stackup) before "
                    "any MAPDL run.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--tag", default=None,
                    help="Suffix for the output file names (default: the grid resolution)")
    args = ap.parse_args()

    try:
        board = load_board(args.sample_dir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))

    homog_path = args.sample_dir / "mesh" / gridmod.OUTPUT_NAME
    model = gridmod.load(homog_path) if homog_path.exists() else None

    results_dir = args.sample_dir / "results"
    results_dir.mkdir(exist_ok=True)
    tag = args.tag or (f"{model.gx}x{model.gy}" if model is not None else "raw")

    # Panel A + C: layout + stackup (always available after discretize.py)
    top_slab = max(board.copper_slabs, key=lambda s: s.z0_mm)
    fig, (axA, axC) = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained",
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    plot_layout(axA, top_slab.raster, board.width_mm, board.height_mm)
    plot_thickness_profile(axC, board, model)
    out1 = results_dir / f"board_preview_{tag}.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Wrote {out1}")

    if model is None:
        print(f"[note] {homog_path} not found -- skipping the per-layer grid panel "
              "(run homogenize.py first for that view)")
        return

    out2 = results_dir / f"board_layers_{tag}.png"
    plot_layer_panels(None, model, board, out2)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
