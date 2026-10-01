#!/usr/bin/env python3
"""
Applied temperature vs. time for every stored result set in a solved .rst --
used to find thermal-cycle boundaries (see plot_cycle_creep_increment.py).

Reads the actual solved temperature at only the LAST substep of each load
step (a handful of slow reads -- see below), then linearly interpolates the
rest: this project's structural analysis imposes a single uniform BF,TEMP
field (see thermal_profile.apply_thermal_profile()), and MAPDL's default
KBC,0 ramped load-step behavior makes that field exactly piecewise-linear in
time, with the pieces breaking only at load-step boundaries. Two solved
anchor values per load step (its start -- shared with the previous step's
end -- and its end) fully determine every substep in between, so there is no
need to either read every substep or trust an external profile file for
what was actually solved.

ansys.mapdl.reader's nodal_temperature() falls back to the full
nodal_solution() for a structural analysis (no fast path like ENL's
nnum_of_interest, see plot_creep_strain_history.py) -- confirmed empirically
at 5+ minutes for all 196 substeps on this project's network-drive .rst.
Reading only the ~16 load-step-end substeps (plus the very first stored
substep, needed to anchor the first load step) cuts that to ~16 reads.

Load-step boundaries come from the .rst's own ls_table (each stored
substep's [load step, substep] pair) via the reader's private
_resultheader -- there's no public accessor, but this project already reads
private reader internals where the public API has no fast path (see
zone_creep_strain.py's res._nodal_result()).

Usage:
    python temperature_history.py data/samples/rv3032_real
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


def load_step_boundary_rnums(res) -> list[int]:
    """Cumulative (0-based) result-set indices worth an actual solved read:
    the very first stored substep, plus the last substep of every load
    step -- the corners of the piecewise-linear applied-temperature curve."""
    load_step = res._resultheader["ls_table"][:, 0]
    last_of_step = (int(np.where(load_step == s)[0].max()) for s in np.unique(load_step))
    return sorted({0, *last_of_step})


def temperature_history(rst_path: Path) -> dict:
    from ansys.mapdl.reader import read_binary
    res = read_binary(str(rst_path))

    node_id = int(res.grid.point_data["ansys_node_num"][0])
    anchor_rnums = load_step_boundary_rnums(res)
    anchor_temps = []
    for rnum in anchor_rnums:
        _, temp = res.nodal_temperature(rnum, nodes=[node_id])
        anchor_temps.append(float(temp[0]))
        print(f"  load-step boundary [{rnum + 1}/{res.nsets}] "
             f"t={res.time_values[rnum]:.2f}s T={anchor_temps[-1]:.3f}")

    time_s = res.time_values
    temp_C = np.interp(time_s, time_s[anchor_rnums], anchor_temps)

    return {
        "rst_path": str(rst_path),
        "watch_node": node_id,
        "time_s": time_s.tolist(),
        "temp_C": temp_C.tolist(),
    }


def plot_temperature_history(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(history["time_s"], history["temp_C"], "o-",
            color="#eb6834", linewidth=2, markersize=4,
            markerfacecolor="#eb6834", markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Applied temperature [\N{DEGREE SIGN}C]")
    ax.set_title("Applied temperature profile")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Applied temperature history at every stored result set, reconstructed by "
                    "reading only the load-step-boundary substeps and interpolating in between.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--rst", type=Path, default=None,
                        help="Path to the .rst file; defaults to "
                             "<sample_dir>/results/pyansys_assembly.rst")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path; defaults to "
                             "<sample_dir>/results/temperature_history.json")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Output PNG path; defaults to "
                             "<sample_dir>/results/temperature_history.png")
    args = parser.parse_args()

    rst_path = args.rst if args.rst is not None else args.sample_dir / "results" / "pyansys_assembly.rst"
    if not rst_path.exists():
        sys.exit(f"Not found: {rst_path}")

    history = temperature_history(rst_path)

    results_dir = args.sample_dir / "results"
    results_dir.mkdir(exist_ok=True)
    out_json = args.out if args.out is not None else results_dir / "temperature_history.json"
    out_json.write_text(json.dumps(history, indent=2) + "\n")
    print(f"\nWrote {out_json}")

    out_png = args.out_png if args.out_png is not None else results_dir / "temperature_history.png"
    plot_temperature_history(history, out_png)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
