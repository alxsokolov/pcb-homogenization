#!/usr/bin/env python3
"""
Per-cycle zone-averaged creep-strain increment, split out of the full
CRACC_zone(t) history (zone_creep_strain.py) using thermal-cycle boundaries
found in the applied temperature history (temperature_history.py).

Why per-cycle and not the run's total: Coffin-Manson-type fatigue laws
(Nf = C * (delta_eps_creep)^-n) take a *per-cycle* strain range as input.
The first cycle of a simulation is usually not representative -- it carries
the initial ramp-in transient before the material's cyclic response
stabilizes (shakedown) -- so plotting all per-cycle increments (not just
taking the last one on faith) is the way to check whether/when the response
has actually stabilized before picking a value for the fatigue law.

Cycle-boundary detection: a thermal cycle here is cold-dwell-start to
cold-dwell-start (JEDEC-style profile: dwell / ramp-up / dwell / ramp-down,
repeated). Boundaries are the first sample of each contiguous run at the
profile's minimum temperature -- robust to the irregular autotime-stepped
sampling (many result sets landing on the same flat dwell) without assuming
a fixed dwell temperature or duration. N boundaries give N-1 full cycles.

Usage:
    python plot_cycle_creep_increment.py data/samples/rv3032_real
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


def cold_dwell_start_indices(temp_C: list[float]) -> list[int]:
    """Index of the first sample in each contiguous run at the profile's
    minimum temperature -- one per cold dwell, in time order."""
    temp = np.asarray(temp_C)
    at_min = temp <= temp.min() + 1e-6
    starts = np.where(at_min & ~np.roll(at_min, 1))[0]
    if at_min[0]:
        starts = starts[starts != 0]
        starts = np.insert(starts, 0, 0)
    return starts.tolist()


def per_cycle_increments(zone_history: dict, temp_history: dict) -> dict:
    zt = np.asarray(zone_history["time_s"])
    zc = np.asarray(zone_history["cracc_zone"])
    tt = np.asarray(temp_history["time_s"])

    if not np.allclose(zt, tt):
        raise ValueError("zone_creep_strain.json and temperature_history.json were computed "
                         "on different result-set time grids -- regenerate both from the same .rst")

    boundary_idx = cold_dwell_start_indices(temp_history["temp_C"])
    if len(boundary_idx) < 2:
        raise ValueError(f"Only {len(boundary_idx)} cold-dwell boundary found -- need at least 2 "
                         "(i.e. one full cycle) to compute a per-cycle increment")

    cycles = []
    for i in range(len(boundary_idx) - 1):
        i0, i1 = boundary_idx[i], boundary_idx[i + 1]
        cycles.append({
            "cycle": i + 1,
            "time_start_s": float(zt[i0]),
            "time_end_s": float(zt[i1]),
            "cracc_zone_start": float(zc[i0]),
            "cracc_zone_end": float(zc[i1]),
            "delta_cracc_zone": float(zc[i1] - zc[i0]),
        })
    return {"boundary_time_s": zt[boundary_idx].tolist(), "cycles": cycles}


def plot_cycles(result: dict, out_path: Path) -> None:
    cycles = result["cycles"]
    x = [c["cycle"] for c in cycles]
    y = [c["delta_cracc_zone"] for c in cycles]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, "-o", color="#2a78d6", linewidth=2, markersize=9,
            markerfacecolor="#2a78d6", markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xlabel("Cycle number [-]")
    ax.set_ylabel("Avg. creep strain per cycle [m/m]")
    ax.set_title("Per-cycle CRACC_zone increment")
    ax.set_xticks(x)
    ax.set_ylim(0.8*np.max(y), 1.05 * np.max(y))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split the CRACC_zone(t) history into per-cycle increments and plot them, "
                    "to check for shakedown/stabilization before picking a value for a fatigue law.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--zone-json", type=Path, default=None,
                        help="Output of zone_creep_strain.py; defaults to "
                             "<sample_dir>/results/zone_creep_strain.json")
    parser.add_argument("--temp-json", type=Path, default=None,
                        help="Output of temperature_history.py; defaults to "
                             "<sample_dir>/results/temperature_history.json")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path; defaults to "
                             "<sample_dir>/results/cycle_creep_increment.json")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Output PNG path; defaults to "
                             "<sample_dir>/results/cycle_creep_increment.png")
    args = parser.parse_args()

    results_dir = args.sample_dir / "results"
    zone_json = args.zone_json if args.zone_json is not None else results_dir / "zone_creep_strain.json"
    temp_json = args.temp_json if args.temp_json is not None else results_dir / "temperature_history.json"
    for p in (zone_json, temp_json):
        if not p.exists():
            sys.exit(f"Not found: {p} -- run zone_creep_strain.py / temperature_history.py first")

    zone_history = json.loads(zone_json.read_text())
    temp_history = json.loads(temp_json.read_text())
    result = per_cycle_increments(zone_history, temp_history)

    print(f"{len(result['cycles'])} full cycle(s) found "
         f"(boundaries at t={[f'{t:.1f}' for t in result['boundary_time_s']]} s)")
    for c in result["cycles"]:
        print(f"  cycle {c['cycle']}: t={c['time_start_s']:.1f}-{c['time_end_s']:.1f}s   "
             f"delta_CRACC_zone={c['delta_cracc_zone']:.6e}")

    out_json = args.out if args.out is not None else results_dir / "cycle_creep_increment.json"
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {out_json}")

    out_png = args.out_png if args.out_png is not None else results_dir / "cycle_creep_increment.png"
    plot_cycles(result, out_png)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
