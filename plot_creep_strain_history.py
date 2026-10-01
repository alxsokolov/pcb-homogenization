#!/usr/bin/env python3
"""
Extracts the solder's accumulated equivalent creep strain (MAPDL's
NL,CREQ -- a monotonically non-decreasing history variable, integrated
over the whole solve, not a snapshot of the current state) from every
stored result set in a solved .rst, and plots it against TIME.

Reads the .rst directly via ansys.mapdl.reader.read_binary() -- no live
MAPDL session, no license checkout, none of this project's repeated
launch/crash/hang problems with actually running MAPDL. NL,CREQ lives in
the .rst's "ENL" (nodal nonlinear) record, specifically column index 4 --
confirmed empirically against a live MAPDL session's own
*GET,Par,NODE,n,NL,CREQ for several nodes (matched to 6 significant
figures). This is NOT the same as the "ECR" record's EQV column (index
6, what an earlier version of this script used by mistake): ECR,EQV is a
von-Mises-style combination of the CURRENT creep strain TENSOR's 6
components, which can legitimately decrease as stress direction reverses
between hot/cold dwells even though creep itself only ever accumulates --
that produced a visibly non-monotonic plot, which is what caught the bug
(accumulated creep strain cannot decrease; NL,CREQ is the quantity that
is actually guaranteed not to). Most solder nodes have no NL data at all
(NaN) -- SOLID186's midside nodes don't store nonlinear history data,
only corner nodes do -- filtered out before taking max/mean, same as the
elastic/plastic/thermal strain records' own documented NaN convention.

Solder elements are identified via the .rst's own per-element MAT id
(grid.cell_data["ansys_material_type"] == --solder-mat, default 9 -- the
CR0805 archive's own solder MAT id, see component_import.py's module
docstring), not re-derived from the .cdb -- the .rst is self-contained.

This example only has one load step (see simulate_pyansys_assembly.py),
so "time" here means substep TIME within that one step, not multiple load
steps -- each stored result set is one substep of the thermal ramp/dwell.

Usage:
    python plot_creep_strain_history.py data/samples/rv3032_real
    python plot_creep_strain_history.py data/samples/rv3032_real --rst mapdl_run/file.rst
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

_DEFAULT_SOLDER_MAT = 9  # see component_import.py module docstring / MANIFEST.md


def solder_node_ids(grid, solder_mat_id: int) -> np.ndarray:
    """ANSYS node numbers belonging to any element whose MAT id is
    solder_mat_id, read straight from the .rst's own per-element material
    record (grid.cell_data["ansys_material_type"]) -- no dependency on the
    .cdb or a live session's selection state."""
    mat = grid.cell_data["ansys_material_type"]
    mask = mat == solder_mat_id
    if not mask.any():
        raise ValueError(f"No elements with MAT={solder_mat_id} found in this .rst -- "
                         "check --solder-mat against the actual result file.")
    solder_grid = grid.extract_cells(np.where(mask)[0])
    return np.unique(solder_grid.point_data["ansys_node_num"])


_ENL_CREQ_COLUMN = 4  # confirmed against a live MAPDL session's *GET,,NODE,n,NL,CREQ


def creep_strain_history(rst_path: Path, solder_mat_id: int) -> dict:
    """(time, max/mean accumulated equivalent creep strain among solder
    nodes) for every result set stored in the .rst, plus the peak's
    location."""
    from ansys.mapdl.reader import read_binary
    res = read_binary(str(rst_path))

    node_ids = solder_node_ids(res.grid, solder_mat_id)
    print(f"{res.nsets} result set(s) in {rst_path.name}, "
         f"{node_ids.size} solder nodes (MAT={solder_mat_id})")
    if res.nsets == 1:
        print("[warn] Only 1 result set stored -- this .rst can't show a time history, "
             "just the final state. The solve needs `mapdl.outres(\"ALL\", \"ALL\")` "
             "before SOLVE so every substep (not just the last) gets written.")

    times, max_creep, mean_creep = [], [], []
    for rnum in range(res.nsets):
        # nnum_of_interest restricts the read to just the solder nodes --
        # confirmed empirically this isn't just post-hoc filtering, it cuts
        # actual per-set read time ~5-10x (an unrestricted read pulls all
        # 37603 nodes off a 4.5GB file on a network drive every time, most
        # of which get thrown away immediately since only ~3000 of them are
        # solder). This is what makes 196 result sets tractable at all.
        nnum, enl = res._nodal_result(rnum, "ENL", nnum_of_interest=node_ids)
        creq = enl[:, _ENL_CREQ_COLUMN]
        creq = creq[~np.isnan(creq)]
        times.append(float(res.time_values[rnum]))
        max_creep.append(float(creq.max()) if creq.size else float("nan"))
        mean_creep.append(float(creq.mean()) if creq.size else float("nan"))
        print(f"  [{rnum + 1}/{res.nsets}] t={res.time_values[rnum]:.2f}s processed")

    return {
        "rst_path": str(rst_path),
        "solder_mat_id": solder_mat_id,
        "n_solder_nodes": int(node_ids.size),
        "time_s": times,
        "max_accum_creep_strain": max_creep,
        "mean_accum_creep_strain": mean_creep,
    }


def plot_history(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["time_s"], history["max_accum_creep_strain"], "o-", label="max (solder)")
    ax.plot(history["time_s"], history["mean_accum_creep_strain"], "s--", label="mean (solder)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Accumulated equivalent creep strain, NL,CREQ [-]")
    ax.set_title(f"Solder accumulated creep strain vs. time (MAT={history['solder_mat_id']})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the solder's max/mean equivalent creep strain vs. time from a solved .rst.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path,
                        help="Sample directory (used for the default --rst/--out locations)")
    parser.add_argument("--rst", type=Path, default=None,
                        help="Path to the .rst file; defaults to "
                             "<sample_dir>/results/pyansys_assembly.rst")
    parser.add_argument("--solder-mat", type=int, default=_DEFAULT_SOLDER_MAT)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path; defaults to "
                             "<sample_dir>/results/creep_strain_history.png")
    args = parser.parse_args()

    rst_path = args.rst if args.rst is not None else args.sample_dir / "results" / "pyansys_assembly.rst"
    if not rst_path.exists():
        sys.exit(f"Not found: {rst_path}")

    history = creep_strain_history(rst_path, args.solder_mat)

    for t, mx, mn in zip(history["time_s"], history["max_accum_creep_strain"], history["mean_accum_creep_strain"]):
        print(f"  t={t:9.2f} s   max={mx:.6e}   mean={mn:.6e}")

    results_dir = args.sample_dir / "results"
    results_dir.mkdir(exist_ok=True)
    out_json = results_dir / "creep_strain_history.json"
    out_json.write_text(json.dumps(history, indent=2) + "\n")
    print(f"\nWrote {out_json}")

    out_png = args.out if args.out is not None else results_dir / "creep_strain_history.png"
    plot_history(history, out_png)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
