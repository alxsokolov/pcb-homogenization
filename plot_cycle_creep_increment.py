#!/usr/bin/env python3
"""
Per-cycle zone-averaged creep-strain increment, split out of the full
CRACC_zone(t) history (zone_creep_strain.py) using the thermal-cycle
boundaries of the applied profile (thermal_profile.cold_dwell_start_times()
on the profile copy simulate_pyansys_assembly.py saves next to the results).

Why per-cycle and not the run's total: Coffin-Manson-type fatigue laws
(Nf = C * (delta_eps_creep)^-n) take a *per-cycle* strain range as input.
The first cycle of a simulation is usually not representative -- it carries
the initial ramp-in transient before the material's cyclic response
stabilizes (shakedown) -- so plotting all per-cycle increments (not just
taking the last one on faith) is the way to check whether/when the response
has actually stabilized before picking a value for the fatigue law.

Cycle-boundary detection: a thermal cycle here is cold-dwell-start to
cold-dwell-start (JEDEC-style profile: dwell / ramp-up / dwell / ramp-down,
repeated). Boundaries come straight from the profile points; every profile
point is a load-step end and therefore a stored result set, so each boundary
is looked up exactly in the creep history's time grid. N boundaries give
N-1 full cycles.

Usage:
    python plot_cycle_creep_increment.py data/samples/rv3032_example
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

from thermal_profile import ThermalProfile, cold_dwell_start_times, load_thermal_profile

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


def per_cycle_increments(zone_history: dict, profile: ThermalProfile) -> dict:
    zt = np.asarray(zone_history["time_s"])
    zc = np.asarray(zone_history["cracc_zone"])

    boundary_t = cold_dwell_start_times(profile)
    if len(boundary_t) < 2:
        raise ValueError(f"Only {len(boundary_t)} cold-dwell boundary in the profile -- need at "
                         "least 2 (i.e. one full cycle) to compute a per-cycle increment")
    boundary_idx = []
    for t in boundary_t:
        hit = np.where(np.isclose(zt, t, rtol=0, atol=1e-6 * max(1.0, abs(t))))[0]
        if hit.size == 0:
            raise ValueError(f"Cycle boundary t={t:g}s is not a stored result set -- was the .rst "
                             "solved with this thermal profile?")
        boundary_idx.append(int(hit[0]))

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
    parser.add_argument("--thermal-profile", type=Path, default=None,
                        help="Profile the .rst was solved with; defaults to the copy "
                             "simulate_pyansys_assembly.py saves, <sample_dir>/results/thermal_profile.yaml")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path; defaults to "
                             "<sample_dir>/results/cycle_creep_increment.json")
    parser.add_argument("--out-png", type=Path, default=None,
                        help="Output PNG path; defaults to "
                             "<sample_dir>/results/cycle_creep_increment.png")
    args = parser.parse_args()

    results_dir = args.sample_dir / "results"
    zone_json = args.zone_json if args.zone_json is not None else results_dir / "zone_creep_strain.json"
    profile_yaml = (args.thermal_profile if args.thermal_profile is not None
                    else results_dir / "thermal_profile.yaml")
    for p in (zone_json, profile_yaml):
        if not p.exists():
            sys.exit(f"Not found: {p} -- run simulate_pyansys_assembly.py / zone_creep_strain.py first")

    zone_history = json.loads(zone_json.read_text())
    result = per_cycle_increments(zone_history, load_thermal_profile(profile_yaml))

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
