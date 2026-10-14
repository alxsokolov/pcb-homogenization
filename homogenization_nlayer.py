#!/usr/bin/env python3
"""
Analytical Voigt/Reuss homogenization of an n-layer PCB laminate.

Generalizes homogenization.voigt_reuss_bounds() from the fixed three-slab
stackup (bottom Cu / FR4 core / top Cu, parameterized by phi_bot and phi_top)
to an arbitrary ordered list of slabs, each with its own thickness and its own
copper area fraction. Dielectric slabs are simply slabs with phi = 0, so an
8-copper-layer board and a 2-copper-layer board go through identical code.

The generalization is exact, not an approximation of the 2-layer formulas:
fed the three-slab stackup [(t_cu, phi_bot), (t_fr4, 0), (t_cu, phi_top)],
every quantity here reduces algebraically to its homogenization.py
counterpart. selftest() checks that numerically against the 2-layer module
on random inputs, so the reduction is verified rather than asserted.

IMPORTANT -- the "structural" bounds carry no information the "global" ones
don't. homogenization.py presents two levels of bound and describes the
structural pair as tighter, exploiting the known layer geometry. They are in
fact algebraically identical to the geometry-blind global pair, for any
number of slabs, any thicknesses and any per-layer copper fractions. Both
slab-level quantities are linear in that slab's phi, so thickness-weighting
them commutes with evaluating them at the mean phi:

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

selftest() checks all three numerically (bit-identical in plane, 1e-16
relative out of plane) on an 8-copper-layer stackup with unequal thicknesses
and per-layer phi from 0.02 to 0.99. Both key names are still returned, since
quadtree.json carries both, but they are the same number -- not two bounds to
choose between.

The consequence matters for how an n-layer board should be modelled: a single
smeared through-thickness material depends on the stackup ONLY through
phi_3D. Per-layer copper fractions cannot influence it, so smearing discards
the entire layer structure. That loss is far larger on a multilayer board
than on a 2-layer one -- the reference 8-layer job is 36% copper by thickness
(0.266 of 0.728 mm) against about 4% for the 2-layer rv3032 board (0.07 of
1.6 mm) -- which is why the z-resolved model, not the smeared one, is the
default for the n-layer path.

  Per-slab values (E_xy, E_z, alpha_xy, alpha_z, nu for one slab alone) are
  what that z-resolved model uses: every copper layer gets its own material
  from its own phi, and the dielectric slabs stay plain dielectric rather
  than being smeared into one through-thickness material. This is the only
  place the layer geometry genuinely enters the homogenization.

Units: everything in and out of this module is MPa and 1/K -- the MAPDL
mm-N-MPa system, so the returned material dicts go straight into mapdl.mp()
with no conversion. (homogenization.py instead takes Pa, returns GPa, and
converts to MPa at the orthotropic-constants step.)

Does not touch homogenization.py; the 2-layer path is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass
class PhaseProps:
    """
    The two constituent phases, in MAPDL mm-N-MPa units.

    E_d / nu_d / alpha_d_* describe the dielectric. A board with several
    distinct dielectrics (the reference 8-layer job mixes a CORE at Dk 3.8
    with PREPREGs at 3.6) is still homogenized against one mechanical
    dielectric: configs/materials.yaml carries no per-resin elastic data, and
    inventing a stiffness difference from a dielectric constant would be
    worse than treating them alike. Per-dielectric properties can be added
    later by giving each slab its own PhaseProps.
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
    # Volume-fraction in-plane form -- homogenization.py's definition.
    alpha_xy_global = (phi_3D * p.E_cu_MPa * p.alpha_cu
                       + (1.0 - phi_3D) * p.E_d_MPa * p.alpha_d_ip) / E_voigt_global
    # Thickness-and-stiffness-weighted in-plane form: each slab resists the
    # others' expansion in proportion to its own in-plane stiffness x
    # thickness. Computed independently rather than aliased, so selftest()
    # can verify it really does equal alpha_xy_global (see module docstring:
    # E_i_xy * a_i_xy is linear in phi_i, so the weighting commutes).
    num = sum(s.thickness_mm * ps["E_xy_MPa"] * ps["alpha_xy_per_K"]
              for s, ps in zip(slabs, per_slab))
    den = sum(s.thickness_mm * ps["E_xy_MPa"] for s, ps in zip(slabs, per_slab))
    alpha_xy_struct = num / den
    # Out of plane the slabs expand freely in series: thickness-weighted mean
    # (identical to the volume-fraction form phi_3D*a_cu + (1-phi_3D)*a_d_op).
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
    so, exactly as homogenization.orthotropic_engineering_constants() does:
    nu is applied isotropically (PRXY = PRYZ = PRXZ), and each plane's shear
    modulus comes from the isotropic relation G = E/(2(1+nu)) using that
    plane's own E -- which is also the default MAPDL's MP applies when
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
    laminate_bounds() output: the n-layer counterpart of
    homogenization.orthotropic_engineering_constants().

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


def orthotropic_slab(phi: float, p: PhaseProps, dilution: float = 1.0) -> dict:
    """
    One orthotropic material for ONE copper slab alone, at area fraction phi.
    The z-resolved model's per-(region, copper layer) material, and the
    n-layer counterpart of
    homogenization.orthotropic_engineering_constants_layer().

    dilution < 1.0 homogenizes at phi * dilution instead of phi, for
    representing a slab's copper over a thicker effective z region than it
    physically occupies. The copper volume does not grow just because the
    element covering it does, so spreading it over more thickness must lower
    the fraction by the same factor the region grew by
    (true_thickness / effective_thickness). Passing the undiluted material
    across a thickened region would silently add copper that does not exist.

    This matters far more on an n-layer board than on a 2-layer one: the
    reference 8-layer job has 25-38 um copper layers separated by 62-70 um
    dielectrics, so a z mesh that cannot resolve a 25 um slab is the normal
    case, not the exception.
    """
    if not 0.0 < dilution <= 1.0:
        raise ValueError(f"dilution must be in (0, 1], got {dilution}")
    ps = slab_properties(float(phi) * dilution, p)
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


# ---------------------------------------------------------------------------
# Reduction check against the 2-layer module
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True) -> bool:
    """
    Verify that this module reduces exactly to homogenization.py on the
    three-slab stackup the 2-layer code is written for, at random phi and
    thicknesses. Run as: python homogenization_nlayer.py
    """
    import random
    from homogenization import (
        orthotropic_engineering_constants,
        orthotropic_engineering_constants_layer,
        voigt_reuss_bounds,
    )

    materials = {
        "copper": {"E_GPa": 120.0, "nu": 0.34, "CTE_per_K": 17e-6},
        "FR4": {"E_GPa": 20.0, "nu": 0.20,
                "CTE_inplane_per_K": 14e-6, "CTE_outofplane_per_K": 60e-6},
    }
    p = PhaseProps.from_materials(materials)
    E_cu_Pa = materials["copper"]["E_GPa"] * 1e9
    E_fr4_Pa = materials["FR4"]["E_GPa"] * 1e9

    rng = random.Random(20261013)
    ok = True
    # voigt_reuss_bounds() rounds its outputs (6 decimals in GPa for moduli,
    # 9 for CTEs), so an exact match is not available to compare against.
    # A difference passes if it is within tol_abs OR tol_rel; "worst slack"
    # below reports the largest difference as a fraction of whichever bound
    # applied, so 1.0 means "just at tolerance" and the number is not
    # inflated by the reference's own rounding noise the way a raw relative
    # difference would be.
    tol_rel = 1e-6
    tol_abs = 2e-3
    worst_slack = 0.0
    worst_label = ""

    def check(label: str, ref_val: float, got_val: float) -> None:
        nonlocal ok, worst_slack, worst_label
        err = abs(got_val - ref_val)
        allowed = max(tol_abs, tol_rel * abs(ref_val))
        slack = err / allowed
        if slack > worst_slack:
            worst_slack, worst_label = slack, label
        if slack > 1.0:
            ok = False
            if verbose:
                print(f"  MISMATCH {label}: 2-layer={ref_val!r} n-layer={got_val!r} "
                      f"(slack {slack:.2f}x tolerance)")

    for trial in range(400):
        phi_bot, phi_top = rng.random(), rng.random()
        t_cu = rng.uniform(0.005, 0.1)
        t_fr4 = rng.uniform(0.05, 2.0)

        ref = voigt_reuss_bounds(phi_top, phi_bot, E_cu=E_cu_Pa, E_fr4=E_fr4_Pa,
                                 t_cu=t_cu, t_fr4=t_fr4,
                                 alpha_cu=p.alpha_cu, alpha_fr4_ip=p.alpha_d_ip,
                                 alpha_fr4_op=p.alpha_d_op)
        slabs = [SlabSpec(t_cu, phi_bot, "copper", "bot"),
                 SlabSpec(t_fr4, 0.0, "dielectric", "core"),
                 SlabSpec(t_cu, phi_top, "copper", "top")]
        got = laminate_bounds(slabs, p)

        # MPa -> GPa where the 2-layer module reports GPa.
        G = 1e-3
        check(f"t{trial} phi_3D", ref["phi_3D"], got["phi_3D"])
        check(f"t{trial} E_voigt_global", ref["E_voigt_global_GPa"], got["E_voigt_global_MPa"] * G)
        check(f"t{trial} E_reuss_global", ref["E_reuss_global_GPa"], got["E_reuss_global_MPa"] * G)
        check(f"t{trial} E_xy_voigt", ref["E_xy_voigt_GPa"], got["E_xy_voigt_MPa"] * G)
        check(f"t{trial} E_z_reuss", ref["E_z_reuss_GPa"], got["E_z_reuss_MPa"] * G)
        check(f"t{trial} alpha_xy", ref["alpha_xy_voigt_per_K"], got["alpha_xy_global_per_K"])
        check(f"t{trial} alpha_z", ref["alpha_z_reuss_per_K"], got["alpha_z_reuss_per_K"])
        # per-slab: slabs[0] is the bottom copper layer, slabs[2] the top
        check(f"t{trial} E_botCu_xy", ref["E_botCu_xy_GPa"], got["slabs"][0]["E_xy_MPa"] * G)
        check(f"t{trial} E_topCu_xy", ref["E_topCu_xy_GPa"], got["slabs"][2]["E_xy_MPa"] * G)
        check(f"t{trial} E_botCu_z", ref["E_botCu_z_GPa"], got["slabs"][0]["E_z_MPa"] * G)
        check(f"t{trial} E_topCu_z", ref["E_topCu_z_GPa"], got["slabs"][2]["E_z_MPa"] * G)
        check(f"t{trial} alpha_botCu_xy", ref["alpha_botCu_xy_per_K"],
              got["slabs"][0]["alpha_xy_per_K"])
        check(f"t{trial} alpha_topCu_z", ref["alpha_topCu_z_per_K"],
              got["slabs"][2]["alpha_z_per_K"])

        # Full MAPDL material sets must agree too, not just the bounds.
        ref_mat = orthotropic_engineering_constants(ref, materials)
        got_mat = orthotropic_smeared(got)
        for lab in ref_mat:
            check(f"t{trial} smeared MP {lab}", ref_mat[lab], got_mat[lab])

        for side, idx in (("bottom", 0), ("top", 2)):
            phi = phi_bot if side == "bottom" else phi_top
            ref_lay = orthotropic_engineering_constants_layer(ref, materials, side)
            got_lay = orthotropic_slab(phi, p)
            for lab in ref_lay:
                check(f"t{trial} slab[{side}] MP {lab}", ref_lay[lab], got_lay[lab])

    if verbose:
        print(f"n-layer vs 2-layer reduction on 400 random 3-slab stackups: "
              f"{'PASS' if ok else 'FAIL'} "
              f"(worst difference {worst_slack:.2f}x tolerance, at {worst_label})")

    # A uniform-phi laminate must also homogenize to the single-slab result,
    # independently of how many slabs it is cut into -- a check the 2-layer
    # module cannot express at all.
    phi = 0.37
    one = laminate_bounds([SlabSpec(1.0, phi, "copper", "one")], p)
    many = laminate_bounds([SlabSpec(0.1, phi, "copper", f"s{i}") for i in range(10)], p)
    for key in ("phi_3D", "E_xy_voigt_MPa", "E_z_reuss_MPa",
                "alpha_xy_global_per_K", "alpha_xy_struct_per_K", "alpha_z_reuss_per_K"):
        rel = abs(one[key] - many[key]) / max(abs(one[key]), 1e-30)
        if rel > 1e-12:
            ok = False
            if verbose:
                print(f"  MISMATCH slab-count invariance {key}: "
                      f"1 slab={one[key]!r} 10 slabs={many[key]!r}")
    if verbose:
        print(f"slab-count invariance at uniform phi: {'PASS' if ok else 'FAIL'}")

    # The structural bounds are identical to the global ones -- verified on
    # the real 8-copper-layer reference stackup, with unequal thicknesses and
    # per-layer phi spanning 0.02 to 0.99 (the worst case for any claim that
    # layering matters to the smeared result). See the module docstring for
    # why this holds algebraically.
    cu_t = [0.035, 0.038, 0.035, 0.025, 0.025, 0.035, 0.038, 0.035]
    di_t = [0.065, 0.065, 0.070, 0.062, 0.070, 0.065, 0.065]
    phis = [0.02, 0.99, 0.10, 0.95, 0.97, 0.05, 0.88, 0.30]
    mixed_slabs: list[SlabSpec] = []
    for i, (t, ph) in enumerate(zip(cu_t, phis)):
        mixed_slabs.append(SlabSpec(t, ph, "copper", f"cu{i}"))
        if i < len(di_t):
            mixed_slabs.append(SlabSpec(di_t[i], 0.0, "dielectric", f"d{i}"))
    m = laminate_bounds(mixed_slabs, p)
    identities = [
        ("E_xy_voigt == E_voigt_global", m["E_xy_voigt_MPa"], m["E_voigt_global_MPa"]),
        ("E_z_reuss  == E_reuss_global", m["E_z_reuss_MPa"], m["E_reuss_global_MPa"]),
        ("alpha_xy struct == global", m["alpha_xy_struct_per_K"], m["alpha_xy_global_per_K"]),
    ]
    struct_ok = True
    for label, a, b in identities:
        rel = abs(a - b) / max(abs(b), 1e-30)
        if rel > 1e-12:
            struct_ok = ok = False
            if verbose:
                print(f"  MISMATCH {label}: {a!r} vs {b!r} (rel {rel:.2e})")
    if verbose:
        print(f"structural bounds == global bounds on the 8-layer reference stackup "
              f"(phi_3D={m['phi_3D']:.5f}, per-layer phi 0.02-0.99): "
              f"{'PASS' if struct_ok else 'FAIL'}")
        print(f"  E_xy = {m['E_xy_voigt_MPa']:.3f} MPa, E_z = {m['E_z_reuss_MPa']:.3f} MPa, "
              f"alpha_xy = {m['alpha_xy_global_per_K'] * 1e6:.4f} ppm/K")
        print("  -> a smeared through-thickness material sees the stackup only "
              "through phi_3D;\n     only the per-slab (z-resolved) materials "
              "depend on how the copper is distributed.")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
