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


def apply_thermal_profile(mapdl, profile: ThermalProfile) -> None:
    """Drives the current /SOLU session through the profile.

    APDL's default KBC,0 (ramped) load-step behavior linearly interpolates
    BF,TEMP between each load step's start and end value over that step's
    own TIME span -- so a piecewise-linear input profile needs nothing
    more than one TIME + BFUNIF + SOLVE per segment; no custom
    sub-stepping/interpolation of our own is needed. A "dwell" segment
    (consecutive points at the same temperature, e.g. this project's
    1800s/3600s room-temp hold) works the same way -- the ramp's start and
    end value are simply equal, so nothing changes over that step.

    TREF is set to profile.reference_temp_c (the profile's first point,
    unless the YAML overrode it -- see this module's docstring). Note the
    BF,TEMP ramp still starts from profile.temp_c[0] (the profile's actual
    first temperature) regardless of reference_temp_c, so if the two
    differ, the model already carries a small residual thermal strain at
    t=0 rather than starting perfectly stress-free -- a deliberate,
    documented approximation in that case, not a bug.

    Call this once, after entering /SOLU and setting ANTYPE/RATE/OUTRES,
    before any SOLVE.
    """
    mapdl.tref(profile.reference_temp_c)
    mapdl.bfunif("TEMP", profile.temp_c[0])
    for t, temp in zip(profile.time_s[1:], profile.temp_c[1:]):
        mapdl.time(float(t))
        mapdl.bfunif("TEMP", float(temp))
        mapdl.solve()
