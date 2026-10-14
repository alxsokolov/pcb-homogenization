#!/usr/bin/env python3
"""
Inspect a coarse-grained n-layer board model before solving it.

Reads <sample_dir>/mesh/coarsegrain_nlayer.json (from coarsegrain_nlayer.py)
and renders what the FE model will actually be given:

  coarsegrain_layers_<tag>.png   one panel per copper layer, each region
                                 shaded by its copper area fraction. Small
                                 multiples on ONE shared 0-1 scale with one
                                 shared colorbar, so panels are directly
                                 comparable -- which is the whole point, and
                                 is also why this is not eight coloured
                                 series on one plot.
  coarsegrain_profile_<tag>.png  the stackup through thickness: each copper
                                 slab's board-wide copper fraction against
                                 its true z position and thickness, with the
                                 dielectric slabs drawn to scale between
                                 them.

The layer panels are drawn as the regions themselves (one rectangle per
region), not as a resampled image, so an adaptive quadtree's varying region
sizes are visible rather than smoothed away -- seeing where the
coarse-graining spent its detail is most of the reason to look at this.

Usage:
    python plot_coarsegrain_nlayer.py data/samples/ublox_nlayer
    python plot_coarsegrain_nlayer.py data/samples/ublox_nlayer --tag grid40
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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from coarsegrain_nlayer import OUTPUT_NAME as COARSEGRAIN_NAME

# Copper fraction is a magnitude, so: one hue, light to dark, fixed 0-1
# domain. Not a rainbow and not a per-panel autoscale -- both would make the
# panels incomparable, which is the one thing this figure has to get right.
_SEQ_CMAP = "Blues"
_PHI_VMIN, _PHI_VMAX = 0.0, 1.0

# Recessive ink, so the data carries the colour and the chrome does not.
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


def plot_layer_panels(doc: dict, out_path: Path) -> None:
    regions = doc["regions"]
    stackup = {s["index"]: s for s in doc["stackup"]}
    cu_idx = list(doc["copper_slab_indices"])
    # Top of board first, so the grid reads like a stackup drawing.
    cu_idx_top_first = sorted(cu_idx, key=lambda i: -stackup[i]["z0_mm"])

    W = doc["board"]["width_mm"]
    H = doc["board"]["height_mm"]
    bboxes = [r["bbox_mm"] for r in regions]

    n = len(cu_idx_top_first)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    # Panels sized to the board's aspect ratio so the layout is not distorted.
    panel_w = 2.9
    panel_h = panel_w * (H / W)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * panel_w + 1.4, nrows * panel_h + 1.5),
                             squeeze=False)

    mappable = None
    for pos, slab_idx in enumerate(cu_idx_top_first):
        ax = axes[pos // ncols][pos % ncols]
        slab = stackup[slab_idx]
        phis = np.array([r["phi"][str(slab_idx)] for r in regions], dtype=float)

        patches = [Rectangle((x0, y0), x1 - x0, y1 - y0) for x0, y0, x1, y1 in bboxes]
        pc = PatchCollection(patches, cmap=_SEQ_CMAP, edgecolor="none")
        pc.set_array(phis)
        pc.set_clim(_PHI_VMIN, _PHI_VMAX)
        ax.add_collection(pc)
        mappable = pc

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        _style_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        # Direct labels: name the layer and its numbers on the panel, rather
        # than making the reader cross-reference a legend. Kept ABOVE the
        # axes -- inside, muted text sits on near-black fill at high phi and
        # becomes unreadable on exactly the panels that matter most.
        name = slab.get("layer_name") or slab["name"]
        ax.set_title(f"{name}", fontsize=10, color=_INK, pad=14)
        ax.text(0.0, 1.015,
                f"z {slab['z0_mm']:.3f}–{slab['z1_mm']:.3f} mm · "
                f"{slab['thickness_mm'] * 1e3:.0f} µm · φ̄ {phis.mean():.3f}",
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=7.5, color=_INK_MUTED)

    for pos in range(n, nrows * ncols):
        axes[pos // ncols][pos % ncols].axis("off")

    disc = doc.get("discretization", {})
    mode = disc.get("mode", "?")
    detail = (f"{disc.get('nx')}×{disc.get('ny')}" if mode == "grid"
              else f"max_level {disc.get('max_level')}"
                   + (f", max_leaves {disc.get('max_leaves')}"
                      if disc.get("max_leaves") else ""))
    fig.suptitle(
        f"{doc['sample_id']} — copper area fraction per region, {n} copper layers "
        f"(top left = top of board)",
        fontsize=12, color=_INK, y=0.995)
    fig.text(0.5, 0.955,
             f"{mode} discretization ({detail}), {doc['n_regions']} regions · "
             f"region model {doc['region_model']} · {doc['n_materials']} distinct materials",
             ha="center", fontsize=9, color=_INK_MUTED)

    fig.tight_layout(rect=(0, 0.02, 0.93, 0.945))
    cax = fig.add_axes([0.945, 0.12, 0.015, 0.72])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label("copper area fraction φ", fontsize=9, color=_INK)
    cb.ax.tick_params(colors=_INK_MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_thickness_profile(doc: dict, out_path: Path) -> None:
    """
    Copper fraction against true z. Copper slabs are bars at their real
    thickness and position; dielectrics are drawn to scale between them, so
    the figure shows the actual stackup proportions rather than a schematic.
    """
    regions = doc["regions"]
    areas = np.array([(r["bbox_mm"][2] - r["bbox_mm"][0])
                      * (r["bbox_mm"][3] - r["bbox_mm"][1]) for r in regions])
    total_area = areas.sum()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    cmap = matplotlib.colormaps[_SEQ_CMAP]

    for slab in doc["stackup"]:
        z0, z1 = slab["z0_mm"], slab["z1_mm"]
        if slab["kind"] != "copper":
            ax.add_patch(Rectangle((0, z0), 1.0, z1 - z0,
                                   facecolor=_DIELECTRIC_FILL,
                                   edgecolor=_DIELECTRIC_EDGE, linewidth=0.6, zorder=1))
            continue
        phis = np.array([r["phi"][str(slab["index"])] for r in regions], dtype=float)
        phi = float((phis * areas).sum() / total_area)      # area-weighted
        ax.add_patch(Rectangle((0, z0), phi, z1 - z0,
                               facecolor=cmap(0.25 + 0.7 * phi),
                               edgecolor="white", linewidth=0.8, zorder=3))
        name = slab.get("layer_name") or slab["name"]
        ax.text(min(phi + 0.015, 0.99), (z0 + z1) / 2,
                f"{name}  φ {phi:.3f}  ({slab['thickness_mm'] * 1e3:.0f} µm)",
                va="center", ha="left", fontsize=8.5, color=_INK, zorder=4)

    total_t = doc["board"]["total_thickness_mm"]
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, total_t)
    _style_axes(ax)
    ax.set_xlabel("copper area fraction φ", fontsize=10, color=_INK)
    ax.set_ylabel("z above bottom face [mm]", fontsize=10, color=_INK)
    ax.grid(axis="x", color=_INK_MUTED, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)

    n_cu = len(doc["copper_slab_indices"])
    cu_t = sum(s["thickness_mm"] for s in doc["stackup"] if s["kind"] == "copper")
    ax.set_title(f"{doc['sample_id']} — stackup through thickness\n"
                 f"{n_cu} copper layers, {total_t * 1e3:.0f} µm total, "
                 f"copper {cu_t / total_t * 100:.0f}% of thickness",
                 fontsize=11.5, color=_INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot a coarse-grained n-layer board model for inspection.")
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--tag", default=None,
                    help="Suffix for the output file names (default: the discretization mode)")
    args = ap.parse_args()

    path = args.sample_dir / "mesh" / COARSEGRAIN_NAME
    if not path.exists():
        sys.exit(f"Not found: {path} -- run coarsegrain_nlayer.py first")
    doc = json.loads(path.read_text())
    if not doc.get("regions"):
        sys.exit(f"{path} contains no regions")
    if not doc.get("copper_slab_indices"):
        sys.exit(f"{path} lists no copper layers")

    disc = doc.get("discretization", {})
    tag = args.tag or (f"grid{disc.get('nx')}x{disc.get('ny')}"
                       if disc.get("mode") == "grid" else "quadtree")

    results_dir = args.sample_dir / "results"
    results_dir.mkdir(exist_ok=True)

    p1 = results_dir / f"coarsegrain_layers_{tag}.png"
    plot_layer_panels(doc, p1)
    print(f"Wrote {p1}")

    p2 = results_dir / f"coarsegrain_profile_{tag}.png"
    plot_thickness_profile(doc, p2)
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()
