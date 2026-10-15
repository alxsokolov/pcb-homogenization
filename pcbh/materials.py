#!/usr/bin/env python3
"""
Material loading and the two-phase (copper / dielectric) property bundle
every homogenization formula in pcbh.homogenization is built on.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_MAT = {
    "copper": {"E_GPa": 120.0, "nu": 0.34},
    "FR4":    {"E_GPa":  20.0, "nu": 0.20},
}


def load_materials(path: Path | None) -> dict:
    if path is None or not path.exists():
        if path is not None:
            print(f"[warn] {path} not found — using built-in defaults.", file=sys.stderr)
        return _DEFAULT_MAT
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        print("[warn] PyYAML not installed — using built-in defaults.", file=sys.stderr)
        return _DEFAULT_MAT


@dataclass
class PhaseProps:
    """
    The two constituent phases, in MAPDL mm-N-MPa units.

    E_d / nu_d / alpha_d_* describe the dielectric. A board with several
    distinct dielectrics (e.g. a CORE at one Dk mixed with PREPREGs at
    another) is still homogenized against one mechanical dielectric:
    configs/materials.yaml carries no per-resin elastic data, and inventing a
    stiffness difference from a dielectric constant would be worse than
    treating them alike. Per-dielectric properties can be added later by
    giving each slab its own PhaseProps.
    """
    E_cu_MPa: float
    nu_cu: float
    alpha_cu: float
    E_d_MPa: float
    nu_d: float
    alpha_d_ip: float
    alpha_d_op: float
    k_cu: float = 385e-3    # thermal conductivity, W/(mm*K) -- MAPDL mm-N-MPa units
    k_d: float = 0.3e-3

    @classmethod
    def from_materials(cls, materials: dict) -> "PhaseProps":
        cu, d = materials["copper"], materials["FR4"]
        return cls(
            E_cu_MPa=float(cu["E_GPa"]) * 1e3,
            nu_cu=float(cu["nu"]),
            alpha_cu=float(cu.get("CTE_per_K", 17e-6)),
            E_d_MPa=float(d["E_GPa"]) * 1e3,
            nu_d=float(d["nu"]),
            alpha_d_ip=float(d.get("CTE_inplane_per_K", d.get("CTE_per_K", 14e-6))),
            alpha_d_op=float(d.get("CTE_outofplane_per_K", d.get("CTE_per_K", 60e-6))),
            # W/(m*K) in materials.yaml -> W/(mm*K) for the mm-N-MPa unit system
            # (1 W/(m*K) = 1e-3 W/(mm*K)).
            k_cu=float(cu.get("thermal_conductivity_W_mK", 385.0)) * 1e-3,
            k_d=float(d.get("thermal_conductivity_W_mK", 0.3)) * 1e-3,
        )
