#!/usr/bin/env python3
"""
Zone-averaged accumulated creep strain, ported from Robert Schwerz's
(Fraunhofer IKTS) APDL /POST1 macro (POST_V3a_fix5) to run entirely against a
solved .rst via ansys.mapdl.reader -- no live MAPDL session, matching this
project's existing plot_creep_strain_history.py approach (avoids the
launch/crash/hang problems of driving MAPDL live).

The critical-volume selection below refines the volume-weighted-averaging
technique of R. Darveaux ("Fatigue Life Predictions for Thermally Loaded
Solder Joints Using a Volume-Weighted Averaging Technique", ASME J. Electron.
Packag. 119(4), 1997; "Effect of Simulation Methodology on Solder Joint Crack
Growth Correlation and Fatigue Life Prediction", Proc. 50th ECTC, 2000) with
the accumulated-creep-strain fatigue law of A. Syed ("Accumulated Creep
Strain and Energy Density Based Thermal Fatigue Life Prediction Models for
SnAgCu Solder Joints", Proc. 54th ECTC, 2004) -- see Schwerz's own group's
Berger, Schwerz, Roellig, Heuer, "Influence analysis of joint attributes on
the fatigue progress of SnAgCu solder joints under thermomechanical
loading", Microelectronics Reliability 140 (2023) 114870,
https://doi.org/10.1016/j.microrel.2022.114870 for this line of work on
two-pole SMT solder joints.

Method (unchanged from the source macro): rather than tracking a single
element's peak NL,CREQ (mesh-sensitive), pick a *critical volume* -- the
solder elements that together account for the largest share of the total
volume-weighted creep-strain increment over one reference interval
[--time-start, --time-end] -- up to a target fraction of the total solder
volume (--target-ratio). That fixed set of elements ("EvalElements") is then
volume-averaged at *every* stored result set to produce a CRACC_zone(t)
history for a fatigue-life law (e.g. Nf = C * (delta_eps_creep)^-n).

Ranking: elements are ranked by volume * delta_creep (equivalent to the
macro's eCRINC, which divides by a constant total volume -- an order-
preserving transform, so it doesn't change which elements get selected). One
correctness fix vs. the source macro: if even the single worst element's
volume already exceeds --target-ratio, this always keeps that one element,
where the original APDL script's `icut` fallback would silently keep *every*
element (icut initialized to N, and the *do loop's *exit on the first failed
iteration leaves it there) -- an edge case not expected to matter for the
target ratios in use (0.10-0.75) but worth not inheriting.

NL,CREQ is only ever available at corner nodes (SOLID186's midside nodes
carry no nonlinear history, see plot_creep_strain_history.py); the
element-averaged value here is the mean of whichever of an element's nodes
have data, same convention as ETABLE,,NL,CREQ and as the NaN-filtering
already used in plot_creep_strain_history.py.

Elements sharing the solder MAT id but not actually solid (CONTA174/TARGE170
contact elements tagged with the underlying part's material) are excluded by
requiring a genuine 3-D solid VTK cell type -- confirmed against
data/samples/rv3032_real: MAT=9 covers 650 cells, only 520 of which are real
SOLID186 hexahedra; the other 130 are contact quads with zero volume that
would otherwise dilute the volume-weighted ranking with junk entries.

Two 3-D renders (via pyvista, already a hard dependency of
ansys.mapdl.reader -- res.grid *is* a pyvista UnstructuredGrid, see
plot_creep_strain_history.py's own grid.extract_cells() use) are produced
alongside the CRACC_zone(t) plot, to sanity-check *which* elements the
volume-weighted ranking actually picked and what the field looks like where
they sit: a semi-transparent solder volume with the selected critical-zone
elements highlighted opaque (zone_elements_highlight.png), and the solder
volume shaded by accumulated equivalent creep strain at the last stored
result set (creep_strain_field.png). Both are best-effort -- a headless
environment without a working OpenGL context will fail the screenshot, which
is caught and warned about rather than losing the JSON/history-plot outputs
that already succeeded.

Usage:
    python zone_creep_strain.py data/samples/rv3032_real
    python zone_creep_strain.py data/samples/rv3032_real \\
        --target-ratio 0.25 --time-start 0 --time-end 3600
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
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

_DEFAULT_SOLDER_MAT = 9  # see component_import.py module docstring / MANIFEST.md
_ENL_CREQ_COLUMN = 4  # see plot_creep_strain_history.py

# Real 3-D solid VTK cell types -- excludes CONTA174/TARGE170 surface
# elements (VTK_QUAD / VTK_TRIANGLE) that can share the solder MAT id.
_SOLID_CELLTYPES = (10, 12, 13, 14, 24, 25, 26, 27)  # tet, hex, wedge, pyramid + quadratic variants


def solder_solid_mask(grid, solder_mat_id: int) -> np.ndarray:
    """Boolean mask over grid.cell_data selecting genuine solid solder elements."""
    mat = grid.cell_data["ansys_material_type"]
    is_solder = mat == solder_mat_id
    is_solid = np.isin(grid.celltypes, _SOLID_CELLTYPES)
    n_excluded = int((is_solder & ~is_solid).sum())
    if n_excluded:
        print(f"[info] excluded {n_excluded} non-solid element(s) sharing MAT={solder_mat_id} "
              "(contact/target elements, not solder volume)")
    return is_solder & is_solid


def element_node_ids(grid, cell_indices: np.ndarray) -> list[np.ndarray]:
    """ANSYS node numbers for each cell in cell_indices, in grid.get_cell() order."""
    node_num = grid.point_data["ansys_node_num"]
    return [node_num[grid.get_cell(int(i)).point_ids] for i in cell_indices]


def element_mean_creq(res, rnum: int, elem_node_ids: list[np.ndarray]) -> np.ndarray:
    """Per-element mean NL,CREQ at result set rnum -- same convention as
    ETABLE,,NL,CREQ: average of whichever nodes on the element actually carry
    nonlinear history data (corner nodes only; midside nodes are NaN)."""
    needed = np.unique(np.concatenate(elem_node_ids))
    nnum, enl = res._nodal_result(rnum, "ENL", nnum_of_interest=needed)
    creq = enl[:, _ENL_CREQ_COLUMN]  # nnum is sorted ascending -- see plot_creep_strain_history.py
    out = np.empty(len(elem_node_ids))
    for i, nids in enumerate(elem_node_ids):
        vals = creq[np.searchsorted(nnum, nids)]
        vals = vals[~np.isnan(vals)]
        out[i] = vals.mean() if vals.size else np.nan
    return out


def select_zone_elements(vol: np.ndarray, delta_creq: np.ndarray, target_ratio: float) -> np.ndarray:
    """Indices (into vol/delta_creq) of the elements making up the critical
    volume: ranked by vol * delta_creq descending, kept while cumulative
    volume stays within target_ratio of the total (at least one element)."""
    weight = vol * delta_creq
    order = np.argsort(-weight)
    cumvol = np.cumsum(vol[order])
    vtot = vol.sum()
    icut = max(int(np.searchsorted(cumvol / vtot, target_ratio, side="right")), 1)
    return order[:icut]


def zone_creep_history(
    rst_path: Path,
    solder_mat_id: int,
    target_ratio: float,
    time_start: float | None,
    time_end: float | None,
) -> dict:
    from ansys.mapdl.reader import read_binary
    res = read_binary(str(rst_path))
    grid = res.grid

    mask = solder_solid_mask(grid, solder_mat_id)
    cell_idx = np.where(mask)[0]
    if cell_idx.size == 0:
        raise ValueError(f"No solid elements with MAT={solder_mat_id} found in this .rst")

    vol = grid.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"][mask]
    elem_num = grid.cell_data["ansys_elem_num"][mask]
    node_ids = element_node_ids(grid, cell_idx)

    times = res.time_values
    t_start = times[0] if time_start is None else time_start
    t_end = times[-1] if time_end is None else time_end
    rnum_start = int(np.argmin(np.abs(times - t_start)))
    rnum_end = int(np.argmin(np.abs(times - t_end)))
    print(f"{cell_idx.size} solder elements, {vol.sum():.6g} mm^3 total volume; "
         f"reference interval t={times[rnum_start]:.4g}s (set {rnum_start + 1}) -> "
         f"t={times[rnum_end]:.4g}s (set {rnum_end + 1})")

    creq_start = element_mean_creq(res, rnum_start, node_ids)
    creq_end = element_mean_creq(res, rnum_end, node_ids)
    delta_creq = creq_end - creq_start

    zone_idx = select_zone_elements(vol, delta_creq, target_ratio)
    zone_vol = vol[zone_idx]
    achieved_ratio = zone_vol.sum() / vol.sum()
    zone_node_ids = [node_ids[i] for i in zone_idx]
    print(f"critical volume: {zone_idx.size} elements, {achieved_ratio * 100:.2f}% of solder volume "
         f"(target {target_ratio * 100:.2f}%)")

    zone_times, zone_cracc = [], []
    for rnum in range(res.nsets):
        creq = element_mean_creq(res, rnum, zone_node_ids)
        valid = ~np.isnan(creq)
        cracc = float((creq[valid] * zone_vol[valid]).sum() / zone_vol[valid].sum())
        zone_times.append(float(times[rnum]))
        zone_cracc.append(cracc)

    history = {
        "rst_path": str(rst_path),
        "solder_mat_id": solder_mat_id,
        "target_ratio": target_ratio,
        "achieved_volume_ratio": achieved_ratio,
        "reference_time_start_s": float(times[rnum_start]),
        "reference_time_end_s": float(times[rnum_end]),
        "n_solder_elements": int(cell_idx.size),
        "n_zone_elements": int(zone_idx.size),
        "zone_element_numbers": elem_num[zone_idx].tolist(),
        "time_s": zone_times,
        "cracc_zone": zone_cracc,
    }

    rnum_last = res.nsets - 1
    creq_last = element_mean_creq(res, rnum_last, node_ids)
    geometry = {
        "grid": grid,
        "cell_idx": cell_idx,
        "zone_idx": zone_idx,
        "achieved_volume_ratio": achieved_ratio,
        "creq_last": creq_last,
        "time_last_s": float(times[rnum_last]),
    }

    return history, geometry


def _nice_bounds(vmax: float, n_intervals: int = 5) -> tuple[float, str]:
    """Round vmax up to a "nice" number (1/2/2.5/5/10 x 10^k) and pick a
    fixed-point format with just enough decimals to tell the n_intervals+1
    evenly spaced tick labels apart -- avoids a colorbar/axis whose ticks
    are the raw data range's arbitrary digits (e.g. 0.00113 / 0.0561 / 0.111)."""
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0, "%.2f"
    exp = math.floor(math.log10(vmax))
    frac = vmax / 10 ** exp
    nice = next(step for step in (1, 2, 2.5, 5, 10) if frac <= step)
    nice_max = nice * 10 ** exp
    decimals = max(0, -math.floor(math.log10(nice_max / n_intervals)))
    return nice_max, f"%.{decimals}f"


def plot_history(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(history["time_s"], history["cracc_zone"], "o-",
            color="#2a78d6", linewidth=2, markersize=6,
            markerfacecolor="#2a78d6", markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Zone-averaged accumulated creep strain, CRACC$_{zone}$ [-]")
    title = (f"Critical-volume CRACC_zone (MAT={history['solder_mat_id']}, "
            f"{history['n_zone_elements']} elements, "
            f"{history['achieved_volume_ratio'] * 100:.1f}% of solder volume)")
    ax.set_title("\n".join(textwrap.wrap(title, width=48)), fontsize=12, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


_FONT_FAMILY = "arial"  # pyvista/VTK built-in font -- no external font file needed


def _autocrop_whitespace(path: Path, pad: int = 40) -> None:
    """Trim the uniform white margin VTK leaves around the rendered scene
    (pyvista has no equivalent of matplotlib's savefig(bbox_inches="tight")),
    with a small padding kept around the content. This assembly's two
    solder joints sit far apart, which otherwise leaves most of the frame
    blank -- see plot_zone_highlight()/plot_creep_strain_field()."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return
    img = Image.open(path)
    rgb = img.convert("RGB")
    bbox = ImageChops.difference(rgb, Image.new("RGB", rgb.size, (255, 255, 255))).getbbox()
    if bbox is None:
        return
    left, top, right, bottom = bbox
    left = max(left - pad, 0)
    top = max(top - pad, 0)
    right = min(right + pad, img.width)
    bottom = min(bottom + pad, img.height)
    img.crop((left, top, right, bottom)).save(path)


def plot_zone_highlight(geometry: dict, out_path: Path) -> None:
    """Semi-transparent render of the full solder volume with the selected
    critical-zone elements picked out opaque, so the volume-weighted
    selection in select_zone_elements() can be sanity-checked visually."""
    import pyvista as pv

    grid = geometry["grid"]
    cell_idx = geometry["cell_idx"]
    zone_idx = geometry["zone_idx"]
    solder = grid.extract_cells(cell_idx)
    zone = grid.extract_cells(cell_idx[zone_idx])

    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    pl.add_mesh(solder, color="#2a78d6", opacity=0.15,
                show_edges=True, edge_color="#184f95", line_width=0.5,
                label="Solder volume")
    pl.add_mesh(zone, color="#d03b3b", opacity=1.0,
                show_edges=True, edge_color="#6e1414", line_width=1.2,
                label="Critical zone")
    pl.add_legend(size=(0.22, 0.025), bcolor="white", border=True,
                  face="rectangle", font_family=_FONT_FAMILY)
    pl.add_text(
        f"Critical zone: {zone.n_cells} / {solder.n_cells} solder elements "
        f"({geometry['achieved_volume_ratio'] * 100:.1f}% of volume)",
        font_size=16, color="black", font=_FONT_FAMILY,
    )
    pl.add_axes()
    pl.view_isometric()
    pl.reset_camera()
    pl.screenshot(str(out_path), scale=2)
    pl.close()
    _autocrop_whitespace(out_path)


def plot_creep_strain_field(geometry: dict, out_path: Path) -> None:
    """Render of the solder volume shaded by accumulated equivalent creep
    strain (NL,CREQ) at the last stored result set. The colorbar's upper
    bound is rounded up to a "nice" number (0 is the physically meaningful
    floor -- NL,CREQ only ever accumulates) so its ticks land on round,
    evenly-spaced values instead of the raw data range's arbitrary digits."""
    import pyvista as pv

    grid = geometry["grid"]
    cell_idx = geometry["cell_idx"]
    field = np.full(grid.n_cells, np.nan)
    field[cell_idx] = geometry["creq_last"]
    grid.cell_data["CREQ"] = field
    solder = grid.extract_cells(cell_idx)

    vmax_nice, fmt = _nice_bounds(float(np.nanmax(geometry["creq_last"])))

    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")
    pl.enable_anti_aliasing("ssaa")
    pl.add_mesh(
        solder, scalars="CREQ", cmap="viridis", clim=(0.0, vmax_nice), show_edges=False,
        scalar_bar_args={
            "title": "Accumulated equivalent creep strain, NL,CREQ [-]",
            "fmt": fmt,
            "n_labels": 6,
            "title_font_size": 50,
            "label_font_size": 40,
            "font_family": _FONT_FAMILY,
            "color": "black",
            "position_x": 0.18, "position_y": 0.04, "width": 0.64, "height": 0.06,
            "bold": True,
        },
    )
    pl.add_text(f"t = {geometry['time_last_s']:.1f} s (last stored step)",
                font_size=16, color="black", font=_FONT_FAMILY)
    pl.add_axes()
    pl.view_isometric()
    pl.reset_camera()
    pl.screenshot(str(out_path), scale=2)
    pl.close()
    _autocrop_whitespace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Volume-averaged accumulated creep strain over a critical sub-volume of solder "
                    "(zone method), ported from R. Schwerz's APDL macro to run against a solved .rst.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--rst", type=Path, default=None,
                        help="Path to the .rst file; defaults to "
                             "<sample_dir>/results/pyansys_assembly.rst")
    parser.add_argument("--solder-mat", type=int, default=_DEFAULT_SOLDER_MAT)
    parser.add_argument("--target-ratio", type=float, default=0.25,
                        help="Volume fraction of solder to include in the critical zone")
    parser.add_argument("--time-start", type=float, default=None,
                        help="Reference interval start [s]; defaults to the first stored set")
    parser.add_argument("--time-end", type=float, default=None,
                        help="Reference interval end [s]; defaults to the last stored set")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path; defaults to "
                             "<sample_dir>/results/zone_creep_strain.json")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Output PNG path; defaults to "
                             "<sample_dir>/results/zone_creep_strain.png")
    parser.add_argument("--out-highlight-png", type=Path, default=None,
                        help="Output PNG path for the zone-highlight render; defaults to "
                             "<sample_dir>/results/zone_elements_highlight.png")
    parser.add_argument("--out-field-png", type=Path, default=None,
                        help="Output PNG path for the last-step creep-strain field render; "
                             "defaults to <sample_dir>/results/creep_strain_field.png")
    parser.add_argument("--no-3d-plots", action="store_true",
                        help="Skip the pyvista renders (JSON + CRACC_zone(t) plot still written)")
    args = parser.parse_args()

    rst_path = args.rst if args.rst is not None else args.sample_dir / "results" / "pyansys_assembly.rst"
    if not rst_path.exists():
        sys.exit(f"Not found: {rst_path}")

    history, geometry = zone_creep_history(
        rst_path, args.solder_mat, args.target_ratio, args.time_start, args.time_end,
    )

    for t, c in zip(history["time_s"], history["cracc_zone"]):
        print(f"  t={t:9.2f} s   CRACC_zone={c:.6e}")

    results_dir = args.sample_dir / "results"
    results_dir.mkdir(exist_ok=True)
    out_json = args.out if args.out is not None else results_dir / "zone_creep_strain.json"
    out_json.write_text(json.dumps(history, indent=2) + "\n")
    print(f"\nWrote {out_json}")

    out_png = args.out_png if args.out_png is not None else results_dir / "zone_creep_strain.png"
    plot_history(history, out_png)
    print(f"Wrote {out_png}")

    if not args.no_3d_plots:
        out_highlight_png = (args.out_highlight_png if args.out_highlight_png is not None
                             else results_dir / "zone_elements_highlight.png")
        try:
            plot_zone_highlight(geometry, out_highlight_png)
            print(f"Wrote {out_highlight_png}")
        except Exception as exc:
            print(f"[warn] Skipped {out_highlight_png} (3D render failed): {exc}")

        out_field_png = (args.out_field_png if args.out_field_png is not None
                         else results_dir / "creep_strain_field.png")
        try:
            plot_creep_strain_field(geometry, out_field_png)
            print(f"Wrote {out_field_png}")
        except Exception as exc:
            print(f"[warn] Skipped {out_field_png} (3D render failed): {exc}")


if __name__ == "__main__":
    main()
