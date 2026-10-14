#!/usr/bin/env python3
"""
Preview the board model before running any MAPDL simulation.

Unlike plot_board_model.py (which reads back the solved .rst to show exactly
what MAPDL saw, including the placed component), this only needs
discretize.py's and optionally quadtree.py's output -- so it works right
after importing a board, before any component placement or solve.

Panel A: top copper layout (mesh/<top layer>_map.npy from discretize.py).
Panel B: board exactly as --board-model says simulate_pyansys_assembly.py will
         mesh it (same flag, same default "projection", same choices) --
         "projection": the raw per-pixel copper/FR4 grid from discretize.py
           (copper red / FR4 blue), i.e. no homogenization.
         "homogenized": mesh/quadtree.json leaves colored by the full-thickness
           homogenized in-plane modulus (E_xy_voigt_GPa).
         "homogenized_zlayers": mesh/quadtree.json leaves colored by the TOP
           copper layer's own per-leaf modulus (E_topCu_xy_GPa) -- the bottom
           layer is homogenized separately with its own phi_bot and isn't
           shown here, since panel A/B are both top-view.
         Both homogenized modes need mesh/quadtree.json from quadtree.py, same
         as the simulate script -- this script errors the same way if it's
         missing, rather than silently falling back to projection.
Panel C: side view of the stackup (schematic z, true thickness labels) --
         no component, since nothing has been placed yet.

Output: <sample_dir>/results/board_preview.png

Usage:
    python plot_board_preview.py data/samples/hella
    python plot_board_preview.py data/samples/hella --board-model homogenized
    python plot_board_preview.py data/samples/hella --board-model homogenized_zlayers
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
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, Normalize

from quadtree import _load_layer_maps

LAYOUT_CU, LAYOUT_FR4 = "#c8702a", "#cfd8c4"   # panel A layout
CU_COLOR, FR4_COLOR = "#d62728", "#0c00ad"     # projection fallback (panel B)
HOMOG_CMAP = "plasma"                          # quadtree leaves (panel B)


def _plot_projection(ax, top_map: np.ndarray, bot_map: np.ndarray, Lx: float, Ly: float) -> str:
    combined = top_map | bot_map
    ax.imshow(combined, origin="lower", extent=(0, Lx, 0, Ly), interpolation="nearest",
               cmap=ListedColormap([FR4_COLOR, CU_COLOR]), vmin=0, vmax=1)
    return f"projection, no homogenization: copper {combined.mean():.0%} (top|bottom OR)"


def _plot_quadtree(ax, leaves: list[dict], ex_key: str, label: str) -> tuple:
    boxes = np.array([leaf["bbox_mm"] for leaf in leaves])
    ex = np.array([leaf.get(ex_key, np.nan) for leaf in leaves])
    polys = np.stack([
        boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]],
    ], axis=1)
    norm = Normalize(np.nanmin(ex), np.nanmax(ex))
    lw = 0.2 if len(leaves) > 1000 else 0.5
    coll = PolyCollection(polys, array=ex, cmap=HOMOG_CMAP, norm=norm,
                          edgecolors="k", linewidths=lw)
    ax.add_collection(coll)
    return coll, f"{label}: {len(leaves)} leaves"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview the board model (layout + stackup) before any MAPDL run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--board-model", choices=["projection", "homogenized", "homogenized_zlayers"],
                        default="projection",
                        help="Same flag/default/choices as simulate_pyansys_assembly.py -- panel B "
                             "shows exactly what that --board-model would mesh.")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Defaults to <sample_dir>/results/board_preview.png")
    args = parser.parse_args()

    if args.board_model in ("homogenized", "homogenized_zlayers"):
        quadtree_path = args.sample_dir / "mesh" / "quadtree.json"
        if not quadtree_path.exists():
            sys.exit(f"--board-model {args.board_model} needs {quadtree_path} -- run quadtree.py first")

    meta_path = args.sample_dir / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"Not found: {meta_path} -- run import_odb_board.py / import_real_board.py first")
    meta = json.loads(meta_path.read_text())
    Lx, Ly = meta["board"]["width_mm"], meta["board"]["height_mm"]
    st = meta["stackup"]
    board_top_z = st["total_thickness_mm"]
    z_cu_bot = st["copper_thickness_mm"]
    z_cu_top = z_cu_bot + st["dielectric_thickness_mm"]

    maps = _load_layer_maps(args.sample_dir)
    top_map, bot_map = maps["top"] > 0.5, maps["bottom"] > 0.5

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained",
                                        gridspec_kw={"width_ratios": [1, 1, 1.1]})

    # --- A: top copper layout ---
    axA.imshow(top_map, origin="lower", extent=(0, Lx, 0, Ly), interpolation="nearest",
               cmap=ListedColormap([LAYOUT_FR4, LAYOUT_CU]), vmin=0, vmax=1)
    axA.set_title(f"Top copper layout ({top_map.mean():.1%} copper)", fontsize=10)

    # --- B: board as it will be meshed, per --board-model ---
    coll_b = None
    if args.board_model == "projection":
        kind = _plot_projection(axB, top_map, bot_map, Lx, Ly)
    else:
        qt = json.loads((args.sample_dir / "mesh" / "quadtree.json").read_text())
        if args.board_model == "homogenized":
            coll_b, kind = _plot_quadtree(axB, qt["leaves"], "E_xy_voigt_GPa",
                                          "homogenized (full thickness)")
        else:
            coll_b, kind = _plot_quadtree(axB, qt["leaves"], "E_topCu_xy_GPa",
                                          "homogenized z-layers (top Cu layer shown)")
    axB.set_title(f"Board to be meshed, top view\n{kind}", fontsize=10)

    for ax in (axA, axB):
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_aspect("equal")
    if coll_b is not None:
        fig.colorbar(coll_b, ax=axB, label="E_xy (homogenized, per leaf) [GPa]", shrink=0.9)

    # --- C: stackup side view (schematic z, true thickness labels) ---
    head = 0.25 * board_top_z
    for z in (0.0, z_cu_bot, z_cu_top, board_top_z):
        axC.axhline(z, color="k", linewidth=0.9, linestyle="--", zorder=1)
    axC.axhspan(0.0, z_cu_bot, color=CU_COLOR, alpha=0.5)
    axC.axhspan(z_cu_bot, z_cu_top, color="#cfd8c4", alpha=0.7)
    axC.axhspan(z_cu_top, board_top_z, color=CU_COLOR, alpha=0.5)
    for z_lab, va, text in ((z_cu_top + 0.08 * board_top_z, "bottom",
                             f"top Cu: z {z_cu_top:.3f}-{board_top_z:.3f} mm"),
                            ((z_cu_bot + z_cu_top) / 2, "center",
                             f"{st.get('dielectric_material', 'FR4')} core: "
                             f"z {z_cu_bot:.3f}-{z_cu_top:.3f} mm"),
                            (z_cu_bot - 0.08 * board_top_z, "top",
                             f"bottom Cu: z 0.000-{z_cu_bot:.3f} mm")):
        axC.text(0.03, z_lab, text, fontsize=8, va=va, ha="left", transform=axC.get_yaxis_transform())
    axC.set_xlim(0, 1)
    axC.set_xticks([])
    axC.set_ylim(-0.05 * board_top_z, board_top_z + head)
    axC.set_ylabel("z [mm]")
    axC.set_title(f"Stackup (total {board_top_z:.3f} mm, schematic -- no component placed yet)",
                  fontsize=10)

    out = args.out_png if args.out_png is not None else args.sample_dir / "results" / "board_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
