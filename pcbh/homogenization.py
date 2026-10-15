#!/usr/bin/env python3
"""
Analytical Voigt/Reuss homogenization of an n-layer PCB laminate.

Takes an arbitrary ordered list of slabs, each with its own thickness and its
own copper area fraction phi. Dielectric slabs are simply slabs with phi = 0,
so a 2-copper-layer board and an 8-copper-layer board go through identical
code -- there is no separate 2-layer formula set.

IMPORTANT -- the "structural" bounds carry no information the "global" ones
don't. A naive comparison might expect the structural pair (which exploits
the known layer geometry) to be tighter than the geometry-blind global pair.
They are in fact algebraically identical, for any number of slabs, any
thicknesses and any per-layer copper fractions. Both slab-level quantities
are linear in that slab's phi, so thickness-weighting them commutes with
evaluating them at the mean phi:

  in plane, stiffnesses add:
    sum_i t_i E_i_xy = sum_i t_i (phi_i E_cu + (1-phi_i) E_d)
                     = E_cu sum_i t_i phi_i  +  E_d sum_i t_i (1-phi_i)
                     = Lz [phi_3D E_cu + (1-phi_3D) E_d]
    => E_xy_voigt == E_voigt_global, exactly

  out of plane, compliances add:
    sum_i t_i / E_i_z = sum_i t_i (phi_i/E_cu + (1-phi_i)/E_d)
                      = Lz [phi_3D/E_cu + (1-phi_3D)/E_d]
    => E_z_reuss == E_reuss_global, exactly

  and the same linearity makes the thickness-and-stiffness-weighted in-plane
  CTE equal to the plain volume-fraction one.

See tests/test_homogenization.py for the numerical verification of this (bit-
identical in plane, 1e-16 relative out of plane) on an 8-copper-layer
stackup with unequal thicknesses and per-layer phi from 0.02 to 0.99. Both
key names are still returned, since callers may want either name, but they
are the same number -- not two bounds to choose between.

The consequence matters for how an n-layer board should be modelled: a single
smeared through-thickness material depends on the stackup ONLY through
phi_3D. Per-layer copper fractions cannot influence it, so smearing discards
the entire layer structure. That loss is far larger on a multilayer board
than on a 2-layer one -- an 8-layer reference job can be 36% copper by
thickness against about 4% for a typical 2-layer board -- which is why the
z-resolved model, not the smeared one, is this project's default.

  Per-slab values (E_xy, E_z, alpha_xy, alpha_z, nu for one slab alone) are
  what the z-resolved model uses: every copper layer gets its own material
  from its own phi, and the dielectric slabs stay plain dielectric rather
  than being smeared into one through-thickness material. This is the only
  place the layer geometry genuinely enters the homogenization.

Units: everything in and out of this module is MPa and 1/K -- the MAPDL
mm-N-MPa system, so the returned material dicts go straight into mapdl.mp()
with no conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

from pcbh.materials import PhaseProps


@dataclass
class SlabSpec:
    """One slab of the laminate for homogenization purposes."""
    thickness_mm: float
    phi: float = 0.0            # copper area fraction, 0 for a dielectric slab
    kind: str = "copper"        # "copper" | "dielectric"
    name: str = ""

    @property
    def is_copper(self) -> bool:
        return self.kind == "copper"


# ---------------------------------------------------------------------------
# Per-slab (single layer alone)
# ---------------------------------------------------------------------------

def slab_properties(phi: float, p: PhaseProps) -> dict:
    """
    Voigt/Reuss properties of ONE slab at copper area fraction phi, taken
    alone rather than smeared with the rest of the laminate.

    In-plane the copper and dielectric within the slab carry load side by
    side (Voigt); through-thickness they stack in series (Reuss). phi = 0
    returns the plain dielectric, phi = 1 plain copper, both exactly.
    """
    phi = min(max(float(phi), 0.0), 1.0)
    E_xy = phi * p.E_cu_MPa + (1.0 - phi) * p.E_d_MPa
    E_z = 1.0 / (phi / p.E_cu_MPa + (1.0 - phi) / p.E_d_MPa)
    alpha_xy = (phi * p.E_cu_MPa * p.alpha_cu
                + (1.0 - phi) * p.E_d_MPa * p.alpha_d_ip) / E_xy
    alpha_z = phi * p.alpha_cu + (1.0 - phi) * p.alpha_d_op
    nu = phi * p.nu_cu + (1.0 - phi) * p.nu_d
    # Thermal conductivity follows the same Voigt/Reuss split as stiffness:
    # in-plane the copper and dielectric conduct side by side (parallel
    # paths, conductances add), through-thickness they conduct in series
    # (resistances add).
    k_xy = phi * p.k_cu + (1.0 - phi) * p.k_d
    k_z = 1.0 / (phi / p.k_cu + (1.0 - phi) / p.k_d)
    return {"phi": phi, "E_xy_MPa": E_xy, "E_z_MPa": E_z,
            "alpha_xy_per_K": alpha_xy, "alpha_z_per_K": alpha_z, "nu": nu,
            "k_xy_W_mmK": k_xy, "k_z_W_mmK": k_z}


# ---------------------------------------------------------------------------
# Whole-laminate bounds
# ---------------------------------------------------------------------------

def laminate_bounds(slabs: list[SlabSpec], p: PhaseProps) -> dict:
    """
    Voigt/Reuss bounds for an arbitrary n-slab Cu/dielectric laminate.

    slabs must be in z order with positive thicknesses; dielectric slabs
    carry phi = 0. Returns global bounds, structural bounds, both in-plane
    CTE definitions, and the per-slab property list (same order as `slabs`).
    """
    if not slabs:
        raise ValueError("laminate_bounds() needs at least one slab")
    Lz = sum(s.thickness_mm for s in slabs)
    if Lz <= 0:
        raise ValueError(f"laminate total thickness must be > 0, got {Lz}")

    per_slab = [slab_properties(s.phi if s.is_copper else 0.0, p) for s in slabs]

    # Total copper volume fraction: thickness-weighted mean of the slab phis.
    phi_3D = sum(s.thickness_mm * ps["phi"] for s, ps in zip(slabs, per_slab)) / Lz

    # -- Global bounds (geometry-blind) -------------------------------------
    E_voigt_global = phi_3D * p.E_cu_MPa + (1.0 - phi_3D) * p.E_d_MPa
    E_reuss_global = 1.0 / (phi_3D / p.E_cu_MPa + (1.0 - phi_3D) / p.E_d_MPa)

    # -- Structural bounds (layering-aware) --------------------------------
    # In plane the slabs are parallel: stiffness adds, thickness-weighted.
    E_xy_voigt = sum(s.thickness_mm * ps["E_xy_MPa"]
                     for s, ps in zip(slabs, per_slab)) / Lz
    # Through thickness the slabs are in series: compliance adds.
    E_z_reuss = Lz / sum(s.thickness_mm / ps["E_z_MPa"]
                         for s, ps in zip(slabs, per_slab))

    # -- CTE ----------------------------------------------------------------
    # Volume-fraction in-plane form.
    # alpha_xy_voigt = (sum E_i phi_i alpha_i) / E_voigt
    alpha_xy_global = (phi_3D * p.E_cu_MPa * p.alpha_cu
                       + (1.0 - phi_3D) * p.E_d_MPa * p.alpha_d_ip) / E_voigt_global
    # Thickness-and-stiffness-weighted in-plane form: each slab resists the
    # others' expansion in proportion to its own in-plane stiffness x
    # thickness. Computed independently rather than aliased, so
    # tests/test_homogenization.py can verify it really does equal
    # alpha_xy_global (the module docstring's proof: E_i_xy * alpha_i_xy is
    # linear in phi_i, so the weighting commutes).
    num = sum(s.thickness_mm * ps["E_xy_MPa"] * ps["alpha_xy_per_K"]
              for s, ps in zip(slabs, per_slab))
    den = sum(s.thickness_mm * ps["E_xy_MPa"] for s, ps in zip(slabs, per_slab))
    alpha_xy_struct = num / den
    # Out of plane the slabs expand freely in series: thickness-weighted mean
    # (identical to the volume-fraction form phi_3D*alpha_cu + (1-phi_3D)*alpha_d_op).
    alpha_z_reuss = sum(s.thickness_mm * ps["alpha_z_per_K"]
                        for s, ps in zip(slabs, per_slab)) / Lz

    nu_eff = phi_3D * p.nu_cu + (1.0 - phi_3D) * p.nu_d

    # -- Thermal conductivity -----------------------------------------------
    # Same global/structural pair as E, same algebraic identity between them
    # (both are linear in each slab's phi).
    k_voigt_global = phi_3D * p.k_cu + (1.0 - phi_3D) * p.k_d
    k_reuss_global = 1.0 / (phi_3D / p.k_cu + (1.0 - phi_3D) / p.k_d)
    k_xy_voigt = sum(s.thickness_mm * ps["k_xy_W_mmK"]
                     for s, ps in zip(slabs, per_slab)) / Lz
    k_z_reuss = Lz / sum(s.thickness_mm / ps["k_z_W_mmK"]
                        for s, ps in zip(slabs, per_slab))

    return {
        "n_slabs": len(slabs),
        "n_copper_slabs": sum(1 for s in slabs if s.is_copper),
        "total_thickness_mm": Lz,
        "phi_3D": phi_3D,
        "phi_per_slab": [ps["phi"] for ps in per_slab],
        "E_voigt_global_MPa": E_voigt_global,
        "E_reuss_global_MPa": E_reuss_global,
        "E_xy_voigt_MPa": E_xy_voigt,
        "E_z_reuss_MPa": E_z_reuss,
        "alpha_xy_global_per_K": alpha_xy_global,
        "alpha_xy_struct_per_K": alpha_xy_struct,
        "alpha_z_reuss_per_K": alpha_z_reuss,
        "nu_eff": nu_eff,
        "k_voigt_global_W_mmK": k_voigt_global,
        "k_reuss_global_W_mmK": k_reuss_global,
        "k_xy_voigt_W_mmK": k_xy_voigt,
        "k_z_reuss_W_mmK": k_z_reuss,
        "slabs": per_slab,
    }


# ---------------------------------------------------------------------------
# MAPDL orthotropic material sets
# ---------------------------------------------------------------------------

def _orthotropic(EX: float, EZ: float, nu: float,
                 alpha_xy: float, alpha_z: float,
                 k_xy: float, k_z: float) -> dict:
    """
    Assemble a MAPDL MP label set from an in-plane / out-of-plane pair.

    Voigt-Reuss yields neither an effective Poisson's ratio nor shear moduli,
    so nu is applied isotropically (PRXY = PRYZ = PRXZ), and each plane's
    shear modulus comes from the isotropic relation G = E/(2(1+nu)) using
    that plane's own E -- the same default MAPDL's MP applies when
    orthotropic shear moduli are not supplied.

    KXX/KYY/KZZ (thermal conductivity) ride along on every material this
    builds, mechanical or thermal use alike -- a structural element ignores
    them, so there is no reason to keep a separate thermal-only material set.
    """
    return {
        "EX": EX, "EY": EX, "EZ": EZ,
        "PRXY": nu, "PRYZ": nu, "PRXZ": nu,
        "GXY": EX / (2.0 * (1.0 + nu)),
        "GYZ": EZ / (2.0 * (1.0 + nu)),
        "GXZ": EZ / (2.0 * (1.0 + nu)),
        "ALPX": alpha_xy, "ALPY": alpha_xy, "ALPZ": alpha_z,
        "KXX": k_xy, "KYY": k_xy, "KZZ": k_z,
    }


def orthotropic_smeared(bounds: dict) -> dict:
    """
    One orthotropic material for the WHOLE laminate thickness, from
    laminate_bounds() output.

    Every element in a region's column gets this material, so the model has
    no z resolution of the copper distribution. Because the smeared bounds
    depend on the stackup only through phi_3D (see module docstring), this
    material is the same whether the board has 2 copper layers or 8, and
    whether its copper sits in one plane or is spread evenly -- it is the
    reference the z-resolved model should be compared against, not a
    recommended model for a multilayer board.

    There is no alpha_model choice: the volume-fraction and
    stiffness-weighted in-plane CTE definitions are provably equal.
    """
    return _orthotropic(bounds["E_xy_voigt_MPa"], bounds["E_z_reuss_MPa"],
                        bounds["nu_eff"], bounds["alpha_xy_global_per_K"],
                        bounds["alpha_z_reuss_per_K"],
                        bounds["k_xy_voigt_W_mmK"], bounds["k_z_reuss_W_mmK"])


def orthotropic_slab(phi: float, p: PhaseProps) -> dict:
    """
    One orthotropic material for ONE copper slab alone, at area fraction
    phi. The z-resolved model's per-(region, copper layer) material.

    The mesh bands z by physical slab (see pcbh.mesh), so every copper layer
    is always represented by element rows at its true thickness -- there is
    no "thicker than the real copper" case to dilute a fraction for.
    """
    ps = slab_properties(float(phi), p)
    return _orthotropic(ps["E_xy_MPa"], ps["E_z_MPa"], ps["nu"],
                        ps["alpha_xy_per_K"], ps["alpha_z_per_K"],
                        ps["k_xy_W_mmK"], ps["k_z_W_mmK"])


def pure_phase_material(is_copper: bool, p: PhaseProps) -> dict:
    """
    The isotropic constants of a single phase, for the black/white
    (projection) model where a region's slab is resolved to all-copper or
    all-dielectric rather than homogenized. phi = 0 / 1 through
    slab_properties(), so projection and homogenization come out of the same
    formulas and differ only in the phi fed to them.
    """
    return orthotropic_slab(1.0 if is_copper else 0.0, p)
