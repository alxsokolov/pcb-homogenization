"""
Gerber and ODB++ parsing + rasterization, checked against copper fractions
captured from the pre-restructure code (tests/golden/copper_fractions.json,
tests/golden/*_cg_*.json).

The Gerber check uses tests/data/synth4 (committed to the repo -- its four
Gerbers are copies of rv3032_example's own F_Cu.gbr/B_Cu.gbr, see
RESTRUCTURING_PLAN.md Phase 0). The ODB++ check needs data/samples/
odb_kitchen_sink and data/samples/hella, which are NOT committed (data/ is
gitignored -- partner-provided files live on the MikroVAL project share) and
is skipped when they are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcbh.gerber import parse_gerber
from pcbh.odb import parse_odb_attrlist, parse_odb_features
from pcbh.raster import rasterize_layer

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads((ROOT / "tests" / "golden" / "copper_fractions.json").read_text())


def _copper_fraction(path: Path, width_mm: float, height_mm: float, nx: int, ny: int) -> float:
    geo = parse_gerber(path)
    mask = rasterize_layer(geo, width_mm, height_mm, nx, ny)
    return float(mask.mean())


def test_gerber_rasterization_matches_golden_per_layer_fractions():
    # Captured from coarsegrain_nlayer.py's "whole-layer" column on
    # rv3032_example at the 200x150 raster discretize.py produces by
    # default at --resolution 10 on a 20x15mm board (tests/golden/
    # rv3032_cg_20x20.json), before the restructure.
    gdir = ROOT / "tests" / "data" / "synth4" / "gerber"
    f_cu = _copper_fraction(gdir / "F_Cu.gbr", 20.0, 15.0, 200, 150)
    b_cu = _copper_fraction(gdir / "B_Cu.gbr", 20.0, 15.0, 200, 150)
    assert abs(f_cu - 0.183767) < 5e-5
    assert abs(b_cu - 0.866100) < 5e-5


def test_gerber_combined_or_matches_discretize_golden():
    gdir = ROOT / "tests" / "data" / "synth4" / "gerber"
    f_geo = parse_gerber(gdir / "F_Cu.gbr")
    b_geo = parse_gerber(gdir / "B_Cu.gbr")
    f_mask = rasterize_layer(f_geo, 20.0, 15.0, 200, 150)
    b_mask = rasterize_layer(b_geo, 20.0, 15.0, 200, 150)
    combined = (f_mask.astype(bool) | b_mask.astype(bool))
    assert abs(float(combined.mean()) - GOLDEN["rv3032_example"]["copper_fraction"]) < 5e-5


_odb_kitchen_sink = ROOT / "data" / "samples" / "odb_kitchen_sink"
_hella = ROOT / "data" / "samples" / "hella"


@pytest.mark.skipif(not _odb_kitchen_sink.is_dir(), reason="data/ not present (gitignored)")
def test_odb_kitchen_sink_rasterization_matches_golden():
    meta = json.loads((_odb_kitchen_sink / "metadata.json").read_text())
    w, h = meta["board"]["width_mm"], meta["board"]["height_mm"]
    origin = tuple(meta.get("odb", {}).get("origin_offset_mm", (0.0, 0.0)))
    nx, ny = GOLDEN["odb_kitchen_sink"]["grid"]["nx"], GOLDEN["odb_kitchen_sink"]["grid"]["ny"]
    combined = None
    for layer in meta["layers"]:
        path = _odb_kitchen_sink / layer["file"]
        geo = parse_odb_features(path, origin=origin)
        mask = rasterize_layer(geo, w, h, nx, ny).astype(bool)
        combined = mask if combined is None else (combined | mask)
    assert abs(float(combined.mean()) - GOLDEN["odb_kitchen_sink"]["copper_fraction"]) < 5e-4


@pytest.mark.skipif(not _hella.is_dir(), reason="data/ not present (gitignored)")
def test_odb_attrlist_whitespace_fix_changes_hella_copper_thickness():
    """D11: parse_odb_attrlist now strips whitespace around '=', so
    hella's '.layer_dielectric = 0.065'-style line (with surrounding
    spaces) is read correctly instead of landing under a key with a
    trailing space. This changes hella's resolved copper thickness from
    the pre-fix 0.035 mm default to 0.036614 mm (see CHANGELOG)."""
    attrs = parse_odb_attrlist(_hella / "odb" / "steps" / "pcb" / "layers" / "top" / "attrlist")
    assert "layer_dielectric" in attrs
    assert abs(attrs["layer_dielectric"] - 0.036614) < 1e-6
