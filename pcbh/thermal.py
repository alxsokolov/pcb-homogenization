#!/usr/bin/env python3
"""
Loads a time/temperature thermal-cycling profile from YAML (e.g.
configs/thermal_profile.yaml) and drives an MAPDL solve through it as one
ramped load step per consecutive pair of points.

The profile's FIRST point is normally the physically stress-free
temperature (e.g. the reflow peak solder solidifies at) and so the
natural choice for TREF -- but the YAML can set an explicit
`reference_temp_c` instead, when the true stress-free temperature falls
outside the solder's own characterized material data range (confirmed
empirically for this project's CR0805/SAC305 data: TREF at the actual
205C reflow peak raised "Tref is outside of the temperature range
defined by the MPTEMP data used for CTEX" for the solder material, whose
CTE table only extends to 180C -- see configs/thermal_profile.yaml's own
comment on this). Falls back to the profile's first point if
`reference_temp_c` isn't given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class ThermalProfile:
    time_s: np.ndarray
    temp_c: np.ndarray
    reference_temp_c: float


def load_thermal_profile(path: Path) -> ThermalProfile:
    with open(path) as f:
        data = yaml.safe_load(f)
    rows = data["profile"]
    time_s = np.array([r[0] for r in rows], dtype=float)
    temp_c = np.array([r[1] for r in rows], dtype=float)
    if not np.all(np.diff(time_s) > 0):
        raise ValueError(f"{path}: profile times must be strictly increasing")
    reference_temp_c = float(data.get("reference_temp_c", temp_c[0]))
    return ThermalProfile(time_s=time_s, temp_c=temp_c, reference_temp_c=reference_temp_c)


def cold_dwell_start_times(profile: ThermalProfile) -> list[float]:
    """Thermal-cycle boundaries: the time each hold at the profile's minimum
    temperature begins (JEDEC-style dwell / ramp-up / dwell / ramp-down
    cycles, counted cold-dwell-start to cold-dwell-start). Every profile
    point is a load-step end, so these times are always stored result sets.
    N boundaries give N-1 full cycles."""
    at_min = profile.temp_c <= profile.temp_c.min() + 1e-6
    return [float(profile.time_s[i]) for i in range(len(at_min))
            if at_min[i] and (i == 0 or not at_min[i - 1])]


def plot_thermal_profile(profile: ThermalProfile, out_path: Path, title: str | None = None) -> None:
    """The applied load as given: one marker per profile point (= load-step
    end), straight ramps in between -- exactly what apply_thermal_profile()
    hands to MAPDL, so it can be checked before a solve is started."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(profile.time_s, profile.temp_c, "-o", color="#eb6834", linewidth=2, markersize=6,
            markeredgecolor="white", markeredgewidth=0.8, label="load-step end")
    ax.axhline(profile.reference_temp_c, color="#5b6470", linewidth=1, linestyle="--",
               label=f"TREF = {profile.reference_temp_c:g} °C")
    for i, t in enumerate(cold_dwell_start_times(profile)):
        ax.axvline(t, color="#2a78d6", linewidth=1, linestyle=":",
                   label="cycle boundary" if i == 0 else None)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Applied temperature [°C]")
    ax.set_title(title or "Applied temperature profile", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def apply_thermal_profile(mapdl, profile: ThermalProfile) -> None:
    """Drives the current /SOLU session through the profile.

    The uniform temperature is a TIME table (BFUNIF,TEMP,%TPROFILE%) holding
    the whole profile, so MAPDL evaluates the exact profile temperature at
    every substep. This deliberately does NOT rely on KBC,0 ramping between
    per-step BFUNIF values: in the first load step that ramp starts from
    TUNIF -- which BFUNIF,TEMP itself overwrites -- so the first segment was
    silently applied as a step change instead of a ramp (seen in solved
    results: 22C already at t=1s of a 205->22C/1800s cool-down).

    One load step per segment is still solved (TIME = segment end), so
    load-step boundaries stay at profile points -- postprocess_creep.py's
    cycle-boundary lookup relies on that.

    TREF is set to profile.reference_temp_c (the profile's first point,
    unless the YAML overrode it -- see this module's docstring). If the two
    differ, the model carries a residual thermal strain at t=0 rather than
    starting stress-free.

    Call this once, after entering /SOLU and setting ANTYPE/RATE/OUTRES,
    before any SOLVE.
    """
    mapdl.tref(profile.reference_temp_c)
    table = np.column_stack([profile.time_s, profile.temp_c])
    mapdl.load_table("TPROFILE", table, "TIME")
    mapdl.bfunif("TEMP", "%TPROFILE%")
    for t in profile.time_s[1:]:
        mapdl.time(float(t))
        mapdl.solve()
