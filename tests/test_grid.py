"""
Unit tests for pcbh.grid: conservation, region_of_element arithmetic,
material dedup, round-trip JSON I/O, and equivalence with the
pre-restructure quadtree/coarsegrain golden output on tests/data/synth4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbh import grid
from pcbh.materials import PhaseProps
from pcbh.stackup import load_board

ROOT = Path(__file__).resolve().parent.parent
SYNTH4 = ROOT / "tests" / "data" / "synth4"

_MATERIALS = {
    "copper": {"E_GPa": 120.0, "nu": 0.34, "CTE_per_K": 17e-6},
    "FR4": {"E_GPa": 20.0, "nu": 0.20,
            "CTE_inplane_per_K": 14e-6, "CTE_outofplane_per_K": 60e-6},
}


def _props() -> PhaseProps:
    return PhaseProps.from_materials(_MATERIALS)


def _discretized_board(resolution: float = 10.0):
    """Discretize synth4 into a fresh temp mesh/ dir-free in-memory board
    (bypasses discretize.py's CLI/file writes -- rasterizes directly)."""
    import json as _json

    from pcbh import gerber
    from pcbh.raster import rasterize_layer
    from pcbh.stackup import BoardStackup, Slab

    meta = _json.loads((SYNTH4 / "metadata.json").read_text())
    W, H = meta["board"]["width_mm"], meta["board"]["height_mm"]
    nx, ny = round(W * resolution), round(H * resolution)

    slabs = []
    for rec in sorted(meta["stackup"]["layers"], key=lambda r: r["index"]):
        slabs.append(Slab(index=rec["index"], kind=rec["kind"], name=rec["name"],
                          thickness_mm=rec["thickness_mm"], z0_mm=rec["z0_mm"],
                          z1_mm=rec["z1_mm"], layer_name=rec.get("layer_name")))
    board = BoardStackup(sample_id=meta["sample_id"], width_mm=W, height_mm=H, slabs=slabs)

    rasters = {}
    for layer in meta["layers"]:
        geo = gerber.parse_gerber(SYNTH4 / layer["file"])
        rasters[layer["name"]] = rasterize_layer(geo, W, H, nx, ny).astype(bool)
    for s in board.copper_slabs:
        s.raster = rasters[s.layer_name]
    return board


def test_region_of_element_exact_and_coarser_and_misaligned():
    # nx == gx: identity mapping
    assert grid.region_of_element(7, 3, 20, 20, 20, 20) == (7, 3)
    # nx == 5*gx: 5 elements per region, grouped correctly
    assert grid.region_of_element(0, 0, 100, 100, 20, 20) == (0, 0)
    assert grid.region_of_element(4, 4, 100, 100, 20, 20) == (0, 0)
    assert grid.region_of_element(5, 5, 100, 100, 20, 20) == (1, 1)
    assert grid.region_of_element(99, 99, 100, 100, 20, 20) == (19, 19)
    # nx == 3*gx + 1 (does not divide evenly): still in range, monotonic
    gi_prev = -1
    for ix in range(31):
        gi, gj = grid.region_of_element(ix, 0, 31, 31, 10, 10)
        assert 0 <= gi < 10 and 0 <= gj < 10
        assert gi >= gi_prev
        gi_prev = gi


def test_conservation_on_synth4_exact_division():
    board = _discretized_board(resolution=10.0)
    p = _props()
    model = grid.build(board, gx=10, gy=10, region_model="zlayers", p=p)
    for s in board.copper_slabs:
        weighted = float(np.mean(model.layer_phi[s.index]))
        whole = float(s.raster.mean())
        assert abs(weighted - whole) < 1e-12, (s.name, weighted, whole)


def test_material_dedup_uniform_board_is_one_material_per_slab():
    """A perfectly uniform (all-copper) board needs exactly one material
    per copper slab plus the pure dielectric, however fine the grid."""
    from pcbh.stackup import BoardStackup, Slab

    raster = np.ones((20, 20), dtype=bool)
    slabs = [
        Slab(0, "copper", "bot", 0.035, 0.0, 0.035, layer_name="bot", raster=raster),
        Slab(1, "dielectric", "core", 1.53, 0.035, 1.565),
        Slab(2, "copper", "top", 0.035, 1.565, 1.6, layer_name="top", raster=raster),
    ]
    board = BoardStackup(sample_id="uniform", width_mm=20.0, height_mm=15.0, slabs=slabs)
    model = grid.build(board, gx=8, gy=8, region_model="zlayers", p=_props())
    # Only the pre-seeded pure "copper" + "dielectric" materials are used --
    # no extra per-region materials needed since every region is phi=1.0.
    assert len(model.materials) == 2
    assert set(np.unique(model.layer_material[0])) == {"copper"}
    assert set(np.unique(model.layer_material[2])) == {"copper"}


def test_grid_matches_pre_restructure_coarsegrain_golden():
    """Region-by-region phi equivalence against coarsegrain_nlayer.py's
    pre-restructure output on the same board/grid (tests/golden/
    synth4_cg_10x10.json), captured before the quadtree/coarsegrain code
    was deleted."""
    golden = json.loads((ROOT / "tests" / "golden" / "synth4_cg_10x10.json").read_text())
    board = _discretized_board(resolution=10.0)
    model = grid.build(board, gx=10, gy=10, region_model="zlayers", p=_props())

    dx, dy = board.width_mm / 10, board.height_mm / 10
    max_diff = 0.0
    for rec in golden["regions"]:
        x0, y0, x1, y1 = rec["bbox_mm"]
        gi, gj = int(((x0 + x1) / 2) // dx), int(((y0 + y1) / 2) // dy)
        for slab_str, val in rec["phi"].items():
            got = float(model.layer_phi[int(slab_str)][gj, gi])
            max_diff = max(max_diff, abs(got - val))
    # golden's phi values were saved rounded to 6 decimals (coarsegrain_
    # nlayer.py's _PHI_DECIMALS); half a ULP at 6 decimals is 5e-7.
    assert max_diff < 6e-7


def test_save_load_round_trip(tmp_path):
    board = _discretized_board(resolution=10.0)
    model = grid.build(board, gx=5, gy=5, region_model="zlayers", p=_props())
    out = tmp_path / "homogenization.json"
    grid.save(model, out)
    back = grid.load(out)
    assert back.gx == model.gx and back.gy == model.gy
    assert back.region_model == model.region_model
    for idx in model.layer_phi:
        assert np.allclose(back.layer_phi[idx], model.layer_phi[idx], atol=1e-6)
        assert np.array_equal(back.layer_material[idx], model.layer_material[idx])
