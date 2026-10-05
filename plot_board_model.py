#!/usr/bin/env python3
"""
The board as it was actually simulated, read back from the solved .rst -- so
the picture shows what MAPDL got, not what a separate preprocessing step
produced.

Panel A: top copper layout (mesh/<top layer>_map.npy from discretize.py).
Panel B: top view of the board's top element row with its mesh. A projected
         board (two materials) is shown copper red / FR4 blue; a homogenized
         board as a colormap of each element's in-plane Young's modulus EX.
         Dashed line: the plane of panel C.
Panel C: side view of the whole board along x -- a vertical x-z section at
         the component's y, every element row with its mesh and material, same
         colors as panel B. Rows are drawn at equal height and labelled with
         their layer and true thickness (at true scale the 35 um copper layers
         would vanish next to the 1.5 mm core), so z is schematic, x to scale.
         The component is marked schematically above the board.
Panels A, B: outlines of the imported component as simulated and the pad
positions configured in solder_reliability.py, to check the placement.

Board elements are the solid elements below the board's top surface; the
component and solder sit above it.

Output: <sample_dir>/results/board_model.png

Usage:
    python plot_board_model.py data/samples/rv3032_example
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
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

from quadtree import _load_layer_maps
from solder_reliability import COMPONENT_NAME, PAD_1_XY_MM, PAD_2_XY_MM

LAYOUT_CU, LAYOUT_FR4 = "#c8702a", "#cfd8c4"                  # panel A layout
CU_COLOR, FR4_COLOR = "#d62728", "#0c00ad"                      # projected board elements
HOMOG_CMAP = "plasma"                                           # homogenized board elements
COMPONENT_COLORS = {2: "#9aa3ad", 6: "#eda100", 9: "#1f6fd1"}   # body, terminations, solder
COMPONENT_NAMES = {2: "body", 6: "termination", 9: "solder"}


def _plot_component_marker(ax) -> None:
    """Pad positions configured in solder_reliability.py (used for placement)."""
    xs = [PAD_1_XY_MM[0], PAD_2_XY_MM[0]]
    ys = [PAD_1_XY_MM[1], PAD_2_XY_MM[1]]
    ax.scatter(xs, ys, marker="+", c="red", s=120, linewidths=1.8, zorder=7,
               path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
               label=f"{COMPONENT_NAME} pads (solder_reliability.py)")


def _scalar_ex(prop) -> float:
    """EX may be stored as a constant or as a (temperature, value) table."""
    arr = np.asarray(prop, dtype=float)
    return float(arr) if arr.ndim == 0 else float(np.mean(arr[-1]))


def read_model(rst_path: Path, board_top_z: float) -> dict:
    """Board elements (boxes x0 x1 y0 y1 z0 z1, MAT), {MAT: EX [MPa]} for the
    board materials, and the component's node coordinates per MAT above the board."""
    from ansys.mapdl.reader import read_binary
    res = read_binary(str(rst_path))
    grid = res.grid
    mat = np.asarray(grid.cell_data["ansys_material_type"])
    zc = grid.cell_centers().points[:, 2]
    tol = 1e-6 * max(1.0, board_top_z)

    board = np.where(zc < board_top_z - tol)[0]
    if board.size == 0:
        sys.exit("No board elements found below the board's top surface.")
    boxes = np.empty((board.size, 6))
    for i, cid in enumerate(board):
        # first 8 = hex corners; unused mid-side node slots of a 20-node
        # element type point at node 0 (the board origin) and must be ignored
        p = grid.points[grid.get_cell(cid).point_ids[:8]]
        boxes[i] = (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max(),
                    p[:, 2].min(), p[:, 2].max())
    ex = {}
    for m in np.unique(mat[board]):
        props = res.materials.get(int(m), {})
        if "EX" in props:
            ex[int(m)] = _scalar_ex(props["EX"])

    above = zc > board_top_z + tol
    comp = {}
    for m in np.unique(mat[above]):
        pts = grid.extract_cells(np.where(above & (mat == m))[0]).points
        comp[int(m)] = pts[pts[:, 2] > board_top_z - 1e-3]   # drop node-0 placeholders
    return {"boxes": boxes, "mats": mat[board], "ex": ex, "component": comp}


def _split_parts(xy: np.ndarray, gap: float = 0.1) -> list[np.ndarray]:
    """Split a point cloud in two along its longer axis if there is a clear gap
    (both terminations / both solder fillets of a 2-terminal part)."""
    ax = int(np.ptp(xy[:, 1]) > np.ptp(xy[:, 0]))
    s = np.sort(xy[:, ax])
    gaps = np.diff(s)
    if gaps.size and gaps.max() > gap:
        cut = s[gaps.argmax()] + gaps.max() / 2
        return [xy[xy[:, ax] < cut], xy[xy[:, ax] >= cut]]
    return [xy]


def component_outlines(component: dict) -> dict[int, list[np.ndarray]]:
    """{MAT: [convex outline (k, 2), ...]} of the component in the board plane."""
    from scipy.spatial import ConvexHull
    return {m: [p[ConvexHull(p).vertices] for p in _split_parts(pts[:, :2]) if len(p) >= 3]
            for m, pts in component.items()}


def _plot_component_outlines(ax, outlines: dict[int, list[np.ndarray]]) -> None:
    for m, polys in sorted(outlines.items()):
        color = COMPONENT_COLORS.get(m, "#555555")
        for j, poly in enumerate(polys):
            closed = np.vstack([poly, poly[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.4, zorder=6,
                    label=f"{COMPONENT_NAMES.get(m, 'component')} (MAT {m})" if j == 0 else None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The simulated board: layout, top view and a side view along x, read from "
                    "the .rst.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--rst", type=Path, default=None,
                        help="Solved .rst; defaults to <sample_dir>/results/pyansys_assembly.rst")
    parser.add_argument("--section-y-mm", type=float, default=None,
                        help="y of the x-z side-view section; defaults to the PAD_1/PAD_2 midpoint")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Defaults to <sample_dir>/results/board_model.png")
    args = parser.parse_args()

    rst = args.rst if args.rst is not None else args.sample_dir / "results" / "pyansys_assembly.rst"
    if not rst.exists():
        sys.exit(f"Not found: {rst} -- run simulate_pyansys_assembly.py first")
    meta = json.loads((args.sample_dir / "metadata.json").read_text())
    Lx, Ly = meta["board"]["width_mm"], meta["board"]["height_mm"]
    st = meta["stackup"]
    board_top_z = st["total_thickness_mm"]
    z_cu_bot = st["copper_thickness_mm"]
    z_cu_top = z_cu_bot + st["dielectric_thickness_mm"]

    top_map = _load_layer_maps(args.sample_dir)["top"] > 0.5
    model = read_model(rst, board_top_z)
    boxes, mats, ex = model["boxes"], model["mats"], model["ex"]
    outlines = component_outlines(model["component"])
    board_mats = sorted(set(mats.tolist()))
    homogenized = len(board_mats) > 2
    egpa = np.array([ex.get(int(m), np.nan) for m in mats]) / 1e3          # MPa -> GPa

    if homogenized:
        norm = Normalize(np.nanmin(egpa), np.nanmax(egpa))
    else:
        cu_mat = max(board_mats, key=lambda m: ex.get(m, 0.0))
        fr4_mat = [m for m in board_mats if m != cu_mat]
        colors = np.where(mats == cu_mat, CU_COLOR, FR4_COLOR)

    def collection(polys, sel, lw):
        if homogenized:
            return PolyCollection(polys, array=egpa[sel], cmap=HOMOG_CMAP, norm=norm,
                                  edgecolors="k", linewidths=lw)
        return PolyCollection(polys, facecolors=colors[sel], edgecolors="k", linewidths=lw)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(21, 5.8), layout="constrained",
                                        gridspec_kw={"width_ratios": [1, 1, 1.25]})

    # --- A: layout ---
    axA.imshow(top_map, origin="lower", extent=(0, Lx, 0, Ly), interpolation="nearest",
               cmap=ListedColormap([LAYOUT_FR4, LAYOUT_CU]), vmin=0, vmax=1)
    axA.set_title("Top copper layout", fontsize=10)

    # --- B: top element row, top view ---
    zmid = (boxes[:, 4] + boxes[:, 5]) / 2
    top = np.abs(zmid - zmid.max()) < 1e-3 * board_top_z
    tb = boxes[top]
    foot = np.stack([tb[:, [0, 2]], tb[:, [1, 2]], tb[:, [1, 3]], tb[:, [0, 3]]], axis=1)
    n_top = int(top.sum())
    lw = 0.2 if n_top > 1000 else 0.4
    coll_b = axB.add_collection(collection(foot, top, lw))
    if homogenized:
        kind = f"homogenized: {len(board_mats)} materials"
    else:
        area = (tb[:, 1] - tb[:, 0]) * (tb[:, 3] - tb[:, 2])
        kind = (f"projection, no homogenization: copper "
                f"{area[mats[top] == cu_mat].sum() / area.sum():.0%}")
    axB.set_title(f"Simulated board, top layer ({n_top} elements)\n{kind}", fontsize=10)

    ys = args.section_y_mm if args.section_y_mm is not None else (PAD_1_XY_MM[1] + PAD_2_XY_MM[1]) / 2
    for ax in (axA, axB):
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_aspect("equal")
        _plot_component_outlines(ax, outlines)
        _plot_component_marker(ax)
    axB.axhline(ys, color="white", linewidth=2.2, zorder=5)
    axB.axhline(ys, color="k", linewidth=1.0, linestyle="--", zorder=5,
                label=f"side-view plane y = {ys:g} mm")
    handles, _ = axB.get_legend_handles_labels()
    if not homogenized:
        handles = [Patch(color=CU_COLOR, label=f"copper (MAT {cu_mat}, {ex[cu_mat] / 1e3:.0f} GPa)"),
                   Patch(color=FR4_COLOR, label=f"FR4 (MAT {fr4_mat[0]}, "
                                                f"{ex[fr4_mat[0]] / 1e3:.0f} GPa)")] + handles
    axB.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9)

    # --- C: side view along x = x-z section at y = ys, true z coordinates ---
    # an element whose face lies exactly on the plane is taken from the +y side
    yq = ys + 1e-6
    cut = (boxes[:, 2] <= yq) & (boxes[:, 3] > yq)
    cb = boxes[cut]
    sec = np.stack([cb[:, [0, 4]], cb[:, [1, 4]], cb[:, [1, 5]], cb[:, [0, 5]]], axis=1)
    axC.add_collection(collection(sec, cut, 0.4))
    if homogenized:
        fig.colorbar(coll_b, ax=[axB, axC], label="EX of the element material [GPa]", shrink=0.9)

    # stackup from the layout (metadata.json): layer boundaries as dashed lines
    head = 0.45 * board_top_z                       # space above the board for the marker
    for z in (0.0, z_cu_bot, z_cu_top, board_top_z):
        axC.axhline(z, color="k", linewidth=0.9, linestyle="--", zorder=4)
    # one label per layer, placed so the two thin copper layers don't collide
    for z_lab, va, text in ((z_cu_top + 0.03 * board_top_z, "bottom",
                             f"top Cu: z {z_cu_top:.3f}-{board_top_z:.3f} mm"),
                            ((z_cu_bot + z_cu_top) / 2, "center",
                             f"FR4 core: z {z_cu_bot:.3f}-{z_cu_top:.3f} mm"),
                            (z_cu_bot - 0.03 * board_top_z, "top",
                             f"bottom Cu: z 0.000-{z_cu_bot:.3f} mm")):
        axC.text(Lx * 1.01, z_lab, text, fontsize=8, va=va, ha="left", clip_on=False)

    # component: schematic marker only -- its x extent on the board surface and an arrow
    comp_xy = np.concatenate([pts[:, :2] for pts in model["component"].values()])
    cx0, cx1 = comp_xy[:, 0].min(), comp_xy[:, 0].max()
    axC.plot([cx0, cx1], [board_top_z + 0.04 * head] * 2, color="k", linewidth=4,
             solid_capstyle="butt", zorder=6)
    axC.annotate(f"{COMPONENT_NAME} (x {cx0:.1f}-{cx1:.1f} mm)",
                 xy=((cx0 + cx1) / 2, board_top_z + 0.08 * head),
                 xytext=((cx0 + cx1) / 2, board_top_z + 0.7 * head), ha="center", fontsize=9,
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
    axC.set_xlim(0, Lx)
    axC.set_ylim(-0.05 * board_top_z, board_top_z + head)
    axC.set_xlabel("x [mm]")
    axC.set_ylabel("z [mm]")
    axC.set_title(f"Side view along x: section at y = {ys:g} mm "
                  f"(true z, vertically stretched)", fontsize=10)

    out = args.out_png if args.out_png is not None else args.sample_dir / "results" / "board_model.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
