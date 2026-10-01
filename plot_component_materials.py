#!/usr/bin/env python3
"""
3-D render of a component-on-PCB assembly, colored by material, from a
solved (or merely meshed) .rst -- one color per ansys_material_type, with a
legend naming each MAT id.

Material names come from data/components/cr0805/MANIFEST.md (the partner
archive's own material identification write-up) and component_import.py's
module docstring (which confirms MAT 2/6/9 are the only CR0805 archive
materials actually referenced by imported elements, and that
remap_board_materials() moves the *board's* copper/FR4 up to MAT 9001/9002
before import so they don't collide with the archive's own MAT 1-10):

    MAT 2    Ceramic (alumina) resistor body
    MAT 6    Nickel metallization/barrier plating
    MAT 9    SAC305 solder -- CONFIRMED (matches SAC305_MikroVal.mac exactly)
    MAT 9001 Copper (board)
    MAT 9002 FR4 board laminate

Any other MAT id found in the mesh is labeled generically ("MAT <id>") and
flagged, rather than silently guessed at.

Elements sharing a MAT id but not actually solid (CONTA174/TARGE170 contact
elements tagged with the underlying part's material -- see
zone_creep_strain.py's solder_solid_mask() docstring for the confirmed case
of MAT 9) are excluded the same way: requiring a genuine 3-D solid VTK cell
type.

Auto-crop: the component (ceramic/nickel/solder) occupies a tiny fraction of
the board's footprint, so a straight isometric view of the whole board would
render it as an invisible speck. Materials whose XY footprint covers most of
the overall mesh's footprint (the board's own copper/FR4) are treated as
"board" (context, shown in full); the rest are treated as "component", and
the view is cropped to their combined bounding box plus a margin -- pass
--no-crop to see the whole board instead.

Usage:
    python plot_component_materials.py data/samples/rv3032_real
    python plot_component_materials.py data/samples/rv3032_real --no-crop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_MAT_NAMES = {
    2: "Ceramic",
    6: "Nickel",
    9: "SAC305",
    9001: "Copper",
    9002: "FR4",
}

# Fixed categorical hue order (see dataviz skill's reference palette) --
# assigned by material identity below, not cycled or re-ordered per-sample.
_MAT_COLORS = {
    9: "#2a78d6",     # solder -- blue, matches this project's other creep-strain plots
    9001: "#eb6834",  # copper -- orange
    2: "#1baf7a",     # ceramic -- aqua
    6: "#eda100",     # nickel -- yellow
    9002: "#e87ba4",  # FR4 -- magenta
}
_FALLBACK_COLORS = ["#008300", "#4a3aa7", "#e34948"]  # remaining palette slots, in order

# Real 3-D solid VTK cell types -- see zone_creep_strain.py's _SOLID_CELLTYPES
_SOLID_CELLTYPES = (10, 12, 13, 14, 24, 25, 26, 27)


def solid_mask(grid) -> np.ndarray:
    return np.isin(grid.celltypes, _SOLID_CELLTYPES)


def material_name(mat_id: int) -> str:
    return _MAT_NAMES.get(mat_id, "unidentified")


def material_color(mat_id: int, fallback_idx: list[int]) -> str:
    if mat_id in _MAT_COLORS:
        return _MAT_COLORS[mat_id]
    color = _FALLBACK_COLORS[fallback_idx[0] % len(_FALLBACK_COLORS)]
    fallback_idx[0] += 1
    return color


def crop_bounds(grid, mat: np.ndarray, mask: np.ndarray, board_footprint_ratio: float = 0.5):
    """Bounding box (with margin) of the "component" materials -- those
    whose own XY footprint covers less than board_footprint_ratio of the
    overall mesh's XY footprint. Returns None if every material is
    board-sized (nothing to crop to)."""
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    board_area = (xmax - xmin) * (ymax - ymin)

    component_bounds = []
    for mat_id in np.unique(mat[mask]):
        idx = np.where(mask & (mat == mat_id))[0]
        sub = grid.extract_cells(idx)
        x0, x1, y0, y1, z0, z1 = sub.bounds
        area = (x1 - x0) * (y1 - y0)
        if area < board_footprint_ratio * board_area:
            component_bounds.append((x0, x1, y0, y1, z0, z1))

    if not component_bounds:
        return None

    arr = np.array(component_bounds)
    x0, y0, z0 = arr[:, 0].min(), arr[:, 2].min(), arr[:, 4].min()
    x1, y1, z1 = arr[:, 1].max(), arr[:, 3].max(), arr[:, 5].max()
    margin = max(2.0, 0.5 * max(x1 - x0, y1 - y0))
    return (
        max(x0 - margin, xmin), min(x1 + margin, xmax),
        max(y0 - margin, ymin), min(y1 + margin, ymax),
        zmin, zmax,
    )


def plot_component_materials(grid, out_path: Path, crop: bool = True) -> None:
    import pyvista as pv

    mat = grid.cell_data["ansys_material_type"]
    mask = solid_mask(grid)
    n_excluded = int((~mask).sum())
    if n_excluded:
        print(f"[info] excluded {n_excluded} non-solid element(s) (contact/target elements)")

    bounds = crop_bounds(grid, mat, mask) if crop else None
    if bounds is not None:
        print(f"[info] cropped view to x=[{bounds[0]:.2f},{bounds[1]:.2f}] "
             f"y=[{bounds[2]:.2f},{bounds[3]:.2f}] mm")
        clip_idx = np.where(mask)[0]
        clipped = grid.extract_cells(clip_idx).clip_box(bounds, invert=False)
        clipped_mat = clipped.cell_data["ansys_material_type"]
        mat_ids = np.unique(clipped_mat)
        get_cells = lambda mat_id: clipped.extract_cells(np.where(clipped_mat == mat_id)[0])
    else:
        mat_ids = np.unique(mat[mask])
        get_cells = lambda mat_id: grid.extract_cells(np.where(mask & (mat == mat_id))[0])

    fallback_idx = [0]
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    for mat_id in mat_ids:
        sub = get_cells(int(mat_id))
        if sub.n_cells == 0:
            continue
        color = material_color(int(mat_id), fallback_idx)
        pl.add_mesh(sub, color=color, show_edges=True, edge_color="black", line_width=0.5,
                    label=f"MAT {int(mat_id)}: {material_name(int(mat_id))}")

    unidentified = sorted(int(m) for m in mat_ids if int(m) not in _MAT_NAMES)
    if unidentified:
        print(f"[warn] unidentified material id(s) in mesh: {unidentified} -- "
             "add them to _MAT_NAMES once confirmed, see this script's docstring")

    pl.add_legend(bcolor="white", border=True, face="rectangle", size=(0.26, 0.014 * len(mat_ids)))
    pl.add_axes()
    pl.view_isometric()
    pl.reset_camera()
    pl.screenshot(str(out_path), scale=2)
    pl.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-D render of a component-on-PCB assembly, colored by material.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--rst", type=Path, default=None,
                        help="Path to the .rst (or .rth/.cdb-derived mesh file); defaults to "
                             "<sample_dir>/results/pyansys_assembly.rst")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Output PNG path; defaults to "
                             "<sample_dir>/results/component_materials.png")
    parser.add_argument("--no-crop", action="store_true",
                        help="Show the whole board instead of auto-cropping to the component")
    args = parser.parse_args()

    rst_path = args.rst if args.rst is not None else args.sample_dir / "results" / "pyansys_assembly.rst"
    if not rst_path.exists():
        sys.exit(f"Not found: {rst_path}")

    from ansys.mapdl.reader import read_binary
    res = read_binary(str(rst_path))

    out_png = args.out_png if args.out_png is not None else args.sample_dir / "results" / "component_materials.png"
    out_png.parent.mkdir(exist_ok=True)
    plot_component_materials(res.grid, out_png, crop=not args.no_crop)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
