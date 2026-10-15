"""
Unit tests for pcbh.homogenization: pure-phase limits, slab-count invariance,
the structural==global bounds identity, and a hand-computed closed-form
check of a 3-slab (bottom Cu / FR4 core / top Cu) laminate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbh.homogenization import SlabSpec, laminate_bounds, slab_properties
from pcbh.materials import PhaseProps

_MATERIALS = {
    "copper": {"E_GPa": 120.0, "nu": 0.34, "CTE_per_K": 17e-6},
    "FR4": {"E_GPa": 20.0, "nu": 0.20,
            "CTE_inplane_per_K": 14e-6, "CTE_outofplane_per_K": 60e-6},
}


def _props() -> PhaseProps:
    return PhaseProps.from_materials(_MATERIALS)


def test_pure_phases_exact():
    # E_xy/E_z/alpha_z are exact (no intermediate multiply-then-divide by the
    # same quantity); alpha_xy involves (E_d*a)/E_d, which IEEE 754 does not
    # guarantee bit-identical to `a` -- allow a tiny (1-ULP-scale) tolerance.
    p = _props()
    d = slab_properties(0.0, p)
    assert d["E_xy_MPa"] == p.E_d_MPa
    assert d["E_z_MPa"] == p.E_d_MPa
    assert abs(d["alpha_xy_per_K"] - p.alpha_d_ip) < 1e-18
    assert d["alpha_z_per_K"] == p.alpha_d_op

    cu = slab_properties(1.0, p)
    assert cu["E_xy_MPa"] == p.E_cu_MPa
    assert cu["E_z_MPa"] == p.E_cu_MPa
    assert abs(cu["alpha_xy_per_K"] - p.alpha_cu) < 1e-18
    assert cu["alpha_z_per_K"] == p.alpha_cu


def test_slab_count_invariance_at_uniform_phi():
    """A uniform-phi laminate homogenizes to the same result independently
    of how many slabs it is cut into."""
    p = _props()
    phi = 0.37
    one = laminate_bounds([SlabSpec(1.0, phi, "copper", "one")], p)
    many = laminate_bounds([SlabSpec(0.1, phi, "copper", f"s{i}") for i in range(10)], p)
    for key in ("phi_3D", "E_xy_voigt_MPa", "E_z_reuss_MPa",
                "alpha_xy_global_per_K", "alpha_xy_struct_per_K", "alpha_z_reuss_per_K"):
        assert abs(one[key] - many[key]) <= 1e-12 * max(abs(one[key]), 1e-30)


def test_structural_bounds_equal_global_bounds_on_8layer_stackup():
    """The module docstring's proof: the layering-aware ('structural')
    bounds are algebraically identical to the geometry-blind ('global')
    ones, for any number of slabs -- verified here on an asymmetric 8-copper
    -layer stackup with per-layer phi spanning 0.02 to 0.99."""
    p = _props()
    cu_t = [0.035, 0.038, 0.035, 0.025, 0.025, 0.035, 0.038, 0.035]
    di_t = [0.065, 0.065, 0.070, 0.062, 0.070, 0.065, 0.065]
    phis = [0.02, 0.99, 0.10, 0.95, 0.97, 0.05, 0.88, 0.30]
    slabs: list[SlabSpec] = []
    for i, (t, ph) in enumerate(zip(cu_t, phis)):
        slabs.append(SlabSpec(t, ph, "copper", f"cu{i}"))
        if i < len(di_t):
            slabs.append(SlabSpec(di_t[i], 0.0, "dielectric", f"d{i}"))
    m = laminate_bounds(slabs, p)
    assert abs(m["E_xy_voigt_MPa"] - m["E_voigt_global_MPa"]) / m["E_voigt_global_MPa"] < 1e-12
    assert abs(m["E_z_reuss_MPa"] - m["E_reuss_global_MPa"]) / m["E_reuss_global_MPa"] < 1e-12
    assert (abs(m["alpha_xy_struct_per_K"] - m["alpha_xy_global_per_K"])
            / m["alpha_xy_global_per_K"] < 1e-12)


def test_3slab_laminate_matches_hand_computed_values():
    """Hand-computed (independent of this module) reference values for a
    3-slab [bottom Cu / FR4 core / top Cu] laminate at phi_bot=0.3,
    phi_top=0.7, t_cu=0.035 mm, t_fr4=1.53 mm -- the 2-layer board's
    geometry, now expressed as 3 slabs instead of a dedicated 2-layer
    formula set."""
    p = _props()
    t_cu, t_fr4 = 0.035, 1.53
    phi_bot, phi_top = 0.3, 0.7
    slabs = [
        SlabSpec(t_cu, phi_bot, "copper", "bot"),
        SlabSpec(t_fr4, 0.0, "dielectric", "core"),
        SlabSpec(t_cu, phi_top, "copper", "top"),
    ]
    got = laminate_bounds(slabs, p)

    expected = {
        "phi_3D": 0.021875000000000002,
        "E_xy_voigt_MPa": 22187.5,
        "E_z_reuss_MPa": 20371.35278514589,
        "E_voigt_global_MPa": 22187.5,
        "E_reuss_global_MPa": 20371.35278514589,
        "alpha_xy_global_per_K": 1.435492957746479e-05,
        "alpha_z_reuss_per_K": 5.9059375000000004e-05,
    }
    for key, val in expected.items():
        assert got[key] == val or abs(got[key] - val) <= 1e-9 * abs(val), (key, got[key], val)

    bot_slab, top_slab = got["slabs"][0], got["slabs"][2]
    assert abs(bot_slab["E_xy_MPa"] - 50000.0) < 1e-6
    assert abs(bot_slab["E_z_MPa"] - 26666.666666666668) < 1e-6
    assert abs(top_slab["E_xy_MPa"] - 90000.0) < 1e-6
    assert abs(top_slab["E_z_MPa"] - 47999.99999999999) < 1e-6
