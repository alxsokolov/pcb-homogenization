#!/usr/bin/env python3
"""
Material loading and analytical Voigt/Reuss homogenization.

Pure-numpy helpers shared by simulate_tensile.py (full 3D FEniCSx solve) and
quadtree.py (per-leaf analytical homogenization) — kept dependency-free
(no dolfinx/petsc4py) so quadtree prototyping doesn't need an FEniCSx
environment.
"""

from __future__ import annotations

import sys
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


# ---------------------------------------------------------------------------
# Voigt / Reuss analytical bounds
# ---------------------------------------------------------------------------

def voigt_reuss_bounds(
    phi_top: float,
    phi_bot: float,
    E_cu: float,
    E_fr4: float,
    t_cu: float,
    t_fr4: float,
    alpha_cu: float = 17e-6,
    alpha_fr4_ip: float = 14e-6,
    alpha_fr4_op: float = 60e-6,
) -> dict:
    """
    Compute Voigt and Reuss bounds for the PCB two-phase (Cu / FR4) stackup.

    Two levels of bounds are returned:

    Global bounds — use only the total 3D copper volume fraction phi_3D:
      E_voigt_global   upper bound (all directions)
      E_reuss_global   lower bound (all directions)

    Structural bounds — exploit the known layer geometry (tighter):
      E_xy_voigt   upper bound for in-plane E_x, E_y
                   (layers parallel → Voigt layer-by-layer)
      E_z_reuss    lower bound for out-of-plane E_z
                   (layers in series → Reuss within each Cu layer, then series)

    The structural bounds serve as the ML baseline: the network learns
    the residual (E_actual − bound) due to spatial copper non-uniformity.

    All E values in GPa (inputs and outputs).
    """
    Lz    = 2.0 * t_cu + t_fr4
    phi_3D = t_cu * (phi_top + phi_bot) / Lz

    # ── Global bounds ────────────────────────────────────────────────────────
    E_voigt_global = phi_3D * E_cu + (1.0 - phi_3D) * E_fr4
    E_reuss_global = 1.0 / (phi_3D / E_cu + (1.0 - phi_3D) / E_fr4)

    # ── Structural in-plane Voigt ────────────────────────────────────────────
    # Each z-slab contributes its Voigt in-plane modulus; slabs are parallel.
    E_botCu_xy = phi_bot * E_cu + (1.0 - phi_bot) * E_fr4
    E_topCu_xy = phi_top * E_cu + (1.0 - phi_top) * E_fr4
    E_xy_voigt = (t_cu * E_botCu_xy + t_fr4 * E_fr4 + t_cu * E_topCu_xy) / Lz

    # ── Structural out-of-plane Reuss ────────────────────────────────────────
    # Within each Cu slab, z-loading is Reuss; then all slabs are in series.
    E_botCu_z = 1.0 / (phi_bot / E_cu + (1.0 - phi_bot) / E_fr4)
    E_topCu_z = 1.0 / (phi_top / E_cu + (1.0 - phi_top) / E_fr4)
    E_z_reuss  = Lz / (t_cu / E_botCu_z + t_fr4 / E_fr4 + t_cu / E_topCu_z)

    # ── CTE bounds ───────────────────────────────────────────────────────────
    # In-plane: Voigt (E-weighted) — exact for ideal layered composite
    # α_xy_voigt = (Σ E_i φ_i α_i) / E_voigt
    alpha_xy_voigt = (phi_3D * E_cu * alpha_cu
                      + (1.0 - phi_3D) * E_fr4 * alpha_fr4_ip) / E_voigt_global

    # Out-of-plane: Reuss (volume-weighted) — exact for ideal layered composite
    # α_z_reuss = Σ φ_i α_z_i
    alpha_z_reuss = phi_3D * alpha_cu + (1.0 - phi_3D) * alpha_fr4_op

    return {
        "phi_top":            round(phi_top,               6),
        "phi_bot":            round(phi_bot,               6),
        "phi_3D":             round(phi_3D,                6),
        "E_voigt_global_GPa": round(E_voigt_global / 1e9,  6),
        "E_reuss_global_GPa": round(E_reuss_global / 1e9,  6),
        "E_xy_voigt_GPa":     round(E_xy_voigt     / 1e9,  6),
        "E_z_reuss_GPa":      round(E_z_reuss      / 1e9,  6),
        "alpha_xy_voigt_per_K": round(alpha_xy_voigt,      9),
        "alpha_z_reuss_per_K":  round(alpha_z_reuss,       9),
    }


# ---------------------------------------------------------------------------
# Orthotropic engineering constants for FE export (export_ansys*.py)
# ---------------------------------------------------------------------------

def orthotropic_engineering_constants(leaf: dict, materials: dict) -> dict:
    """
    Derive a full orthotropic engineering-constant set (E, nu, G, CTE per
    axis) from a quadtree leaf's homogenized record (phi_top/phi_bot/phi_3D,
    E_xy_voigt_GPa, E_z_reuss_GPa, alpha_xy_voigt_per_K, alpha_z_reuss_per_K
    -- present in leaf.props from either voigt_reuss_bounds() or
    ml_homogenization.predict_leaf_properties(), same schema).

    Voigt-Reuss gives no effective Poisson's ratio or shear modulus
    directly:
      nu_eff = phi_3D*nu_cu + (1-phi_3D)*nu_fr4 (volume-fraction weighted;
        same simplification already used by simulate_joint_qoi.py's
        assign_uniform_from_quadtree()), applied isotropically
        (PRXY=PRYZ=PRXZ).
      G derived per-plane via the isotropic relationship G = E/(2*(1+nu))
        using that plane's own E -- the same default ANSYS's MP command
        applies when orthotropic shear moduli aren't given explicitly.

    Units: GPa/per-K in (leaf's own units), MPa/per-K out (ANSYS
    MP/Engineering-Data convention, mm-based unit system).
    """
    nu_cu = materials["copper"]["nu"]
    nu_fr4 = materials["FR4"]["nu"]
    nu_eff = leaf["phi_3D"] * nu_cu + (1.0 - leaf["phi_3D"]) * nu_fr4

    EX = EY = leaf["E_xy_voigt_GPa"] * 1e3
    EZ = leaf["E_z_reuss_GPa"] * 1e3
    GXY = EX / (2.0 * (1.0 + nu_eff))
    GYZ = GXZ = EZ / (2.0 * (1.0 + nu_eff))

    return {
        "EX": EX, "EY": EY, "EZ": EZ,
        "PRXY": nu_eff, "PRYZ": nu_eff, "PRXZ": nu_eff,
        "GXY": GXY, "GYZ": GYZ, "GXZ": GXZ,
        "ALPX": leaf["alpha_xy_voigt_per_K"], "ALPY": leaf["alpha_xy_voigt_per_K"],
        "ALPZ": leaf["alpha_z_reuss_per_K"],
    }
