#!/usr/bin/env python3
"""
Uniform in-plane homogenization grid: cuts the board plane into a gx x gy
grid of equal-size regions, independent of the FE mesh resolution the
board is later solved at, and resolves each region's copper into
materials.

This is the middle of the pipeline. The material space map
(pcbh.stackup: layout rasters x stackup slabs) is continuous; this step
chooses the in-plane regions it is averaged over, and how each region's
copper becomes a material. The result is a complete, inspectable
description of the board model -- region geometry, per-layer copper
fractions, and the material constants -- written before MAPDL is involved
at all, so it can be checked and plotted on its own.

Region model (how each region's copper becomes material):
    zlayers      (default) each copper slab gets its own orthotropic
                 Voigt-Reuss material from that slab's own phi in that
                 region; dielectric slabs stay plain dielectric.
    smeared      one smeared orthotropic material for the whole board
                 thickness in that region, from the region's phi_3D.
    projection   black/white: each copper slab in the region is resolved
                 to pure copper or pure dielectric by phi >= threshold.

Why zlayers is the default: the smeared model's bounds depend on the
stackup only through phi_3D -- algebraically identical to the geometry-
blind global bounds, for any stackup (see pcbh.homogenization's module
docstring). So smearing discards the layer structure entirely. On a
2-layer board that costs little, since copper is a few percent of the
thickness; on a many-layer board copper can be a third of the thickness
with per-layer fractions spanning the whole [0,1] range, so smearing
throws away most of what distinguishes the board.

Materials are deduplicated by rounded copper fraction, so a 50x50 grid
over eight copper layers yields a few hundred distinct materials rather
than 20000.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pcbh.homogenization import SlabSpec, laminate_bounds, orthotropic_slab, orthotropic_smeared
from pcbh.materials import PhaseProps
from pcbh.stackup import BoardStackup, layer_phi_grid

SCHEMA_VERSION = 1

# Copper fractions are rounded to this many decimals before being used as a
# material-table key. 1e-6 is far below any physically meaningful difference
# in phi and well below the raster's own quantization, so this only collapses
# duplicates.
_PHI_DECIMALS = 6


# ---------------------------------------------------------------------------
# Material deduplication
# ---------------------------------------------------------------------------


class MaterialTable:
    """Deduplicating material store: identical constants share one key."""

    def __init__(self) -> None:
        self._by_key: dict[tuple, str] = {}
        self.materials: dict[str, dict] = {}

    def add(self, dedup_key: tuple, build, name: str | None = None) -> str:
        existing = self._by_key.get(dedup_key)
        if existing is not None:
            return existing
        mat_name = name or f"m{len(self.materials)}"
        self._by_key[dedup_key] = mat_name
        self.materials[mat_name] = build()
        return mat_name

    def __len__(self) -> int:
        return len(self.materials)


# ---------------------------------------------------------------------------
# Grid model
# ---------------------------------------------------------------------------


@dataclass
class GridModel:
    sample_id: str
    width_mm: float
    height_mm: float
    total_thickness_mm: float
    gx: int
    gy: int
    region_model: str                       # "zlayers" | "smeared" | "projection"
    projection_threshold: float | None
    materials: dict[str, dict]              # name -> MAPDL MP label dict
    layer_phi: dict[int, np.ndarray] = field(default_factory=dict)       # slab idx -> (gy, gx)
    layer_material: dict[int, np.ndarray] = field(default_factory=dict)  # slab idx -> (gy, gx) of names
    column_material: np.ndarray | None = None                            # (gy, gx) of names ("smeared")
    phi_3D: np.ndarray | None = None                                     # (gy, gx)
    stackup_records: list[dict] = field(default_factory=list)

    @property
    def dx_mm(self) -> float:
        return self.width_mm / self.gx

    @property
    def dy_mm(self) -> float:
        return self.height_mm / self.gy


def region_of_element(ix: int, iy: int, nx: int, ny: int, gx: int, gy: int) -> tuple[int, int]:
    """
    The homogenization-grid cell (gi, gj) containing FE mesh cell (ix, iy)
    of an nx x ny mesh, by pure integer arithmetic -- no bbox search needed
    since both grids are uniform and share an origin.
    """
    gi = min((ix * gx) // nx, gx - 1)
    gj = min((iy * gy) // ny, gy - 1)
    return gi, gj


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build(board: BoardStackup, gx: int, gy: int, region_model: str,
         p: PhaseProps, projection_threshold: float = 0.5) -> GridModel:
    """Resolve every region's copper into materials, over a uniform gx x gy
    in-plane grid. See module docstring for the three region models."""
    if gx < 1 or gy < 1:
        raise ValueError(f"gx/gy must be >= 1, got {gx}, {gy}")
    if region_model not in ("zlayers", "smeared", "projection"):
        raise ValueError(f"Unknown region_model {region_model!r}")

    table = MaterialTable()
    # Pre-seeded so it's always present under this name: pcbh.mesh's
    # mat_resolver() falls back to materials["dielectric"] for any slab not
    # in layer_material (i.e. every dielectric slab), even on a board with
    # no copper region at phi=0. Dedup key ("phase", 0.0) is the same one
    # a phi=0.0 copper-slab region resolves to below, so the two never
    # duplicate.
    table.add(("phase", 0.0), lambda: orthotropic_slab(0.0, p), name="dielectric")

    layer_phi: dict[int, np.ndarray] = {}
    for s in board.copper_slabs:
        if s.raster is None:
            raise ValueError(f"Copper slab {s.name!r} has no raster attached -- "
                             "run discretize.py first")
        layer_phi[s.index] = layer_phi_grid(s.raster, gx, gy)

    Lz = board.total_thickness_mm
    phi_3D = np.zeros((gy, gx), dtype=np.float64)
    for s in board.copper_slabs:
        phi_3D += s.thickness_mm * layer_phi[s.index]
    phi_3D /= Lz

    layer_material: dict[int, np.ndarray] = {}
    column_material: np.ndarray | None = None

    if region_model == "smeared":
        column_material = np.empty((gy, gx), dtype=object)
        for j in range(gy):
            for i in range(gx):
                phi = round(float(phi_3D[j, i]), _PHI_DECIMALS)
                name = table.add(("smeared", phi), _smeared_builder(Lz, phi, p))
                column_material[j, i] = name
    else:
        for s in board.copper_slabs:
            mat_names = np.empty((gy, gx), dtype=object)
            phi_grid = layer_phi[s.index]
            for j in range(gy):
                for i in range(gx):
                    phi = float(phi_grid[j, i])
                    if region_model == "projection":
                        phi_eff = 1.0 if phi >= projection_threshold else 0.0
                    else:
                        phi_eff = phi
                    phi_key = round(phi_eff, _PHI_DECIMALS)
                    # Consistent dedup key with the pre-seeded dielectric
                    # material above, and with every other slab/region at
                    # the same phi: a region landing at exactly phi=0.0 (or,
                    # for zlayers, phi=1.0) reuses "dielectric"/"copper"
                    # instead of being built again under an "m<n>" name.
                    name_hint = ("copper" if phi_key == 1.0 else
                                "dielectric" if phi_key == 0.0 else None)
                    name = table.add(("phase", phi_key),
                                     lambda pk=phi_key: orthotropic_slab(pk, p), name=name_hint)
                    mat_names[j, i] = name
            layer_material[s.index] = mat_names

    stackup_records = [
        {"index": s.index, "kind": s.kind, "name": s.name, "layer_name": s.layer_name,
         "thickness_mm": s.thickness_mm, "z0_mm": s.z0_mm, "z1_mm": s.z1_mm}
        for s in board.slabs
    ]

    return GridModel(
        sample_id=board.sample_id, width_mm=board.width_mm, height_mm=board.height_mm,
        total_thickness_mm=Lz, gx=gx, gy=gy, region_model=region_model,
        projection_threshold=projection_threshold if region_model == "projection" else None,
        materials=table.materials, layer_phi=layer_phi, layer_material=layer_material,
        column_material=column_material, phi_3D=phi_3D, stackup_records=stackup_records,
    )


def _smeared_builder(Lz: float, phi: float, p: PhaseProps):
    """One smeared material for the whole laminate thickness at copper
    volume fraction phi -- built via laminate_bounds() on a single
    pseudo-slab [thickness Lz, phi] rather than orthotropic_slab(phi, p)
    directly, so the EZ/alpha_z fields come out as the laminate's Reuss/
    series combination (orthotropic_smeared's contract), not a single-slab
    Voigt/Reuss pair -- the two are numerically different even though both
    depend on the stackup only through phi_3D (see pcbh.homogenization).
    """
    def _build():
        bounds = laminate_bounds([SlabSpec(Lz, phi, "copper", "smeared")], p)
        return orthotropic_smeared(bounds)
    return _build


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

OUTPUT_NAME = "homogenization.json"


def save(model: GridModel, path: Path) -> None:
    out = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": model.sample_id,
        "generated_by": "homogenize.py",
        "board": {"width_mm": model.width_mm, "height_mm": model.height_mm,
                  "total_thickness_mm": model.total_thickness_mm},
        "grid": {"gx": model.gx, "gy": model.gy,
                 "dx_mm": round(model.dx_mm, 8), "dy_mm": round(model.dy_mm, 8)},
        "region_model": model.region_model,
        "projection_threshold": model.projection_threshold,
        "material_units": "MAPDL mm-N-MPa (E and G in MPa, CTE in 1/K)",
        "stackup": model.stackup_records,
        "materials": model.materials,
        "layers": [
            {"slab_index": idx, "layer_name": next(
                (r["layer_name"] for r in model.stackup_records if r["index"] == idx), None),
             "phi": np.round(model.layer_phi[idx], _PHI_DECIMALS).tolist(),
             "material": model.layer_material[idx].tolist()}
            for idx in sorted(model.layer_material)
        ],
        "column_material": (model.column_material.tolist()
                            if model.column_material is not None else None),
        "phi_3D": np.round(model.phi_3D, _PHI_DECIMALS).tolist(),
        "n_materials": len(model.materials),
    }
    path.write_text(json.dumps(out, indent=2) + "\n")


def load(path: Path) -> GridModel:
    doc = json.loads(Path(path).read_text())
    gx, gy = doc["grid"]["gx"], doc["grid"]["gy"]
    layer_phi = {rec["slab_index"]: np.array(rec["phi"], dtype=np.float64) for rec in doc["layers"]}
    layer_material = {rec["slab_index"]: np.array(rec["material"], dtype=object)
                      for rec in doc["layers"]}
    column_material = (np.array(doc["column_material"], dtype=object)
                       if doc["column_material"] is not None else None)
    phi_3D = np.array(doc["phi_3D"], dtype=np.float64)
    return GridModel(
        sample_id=doc["sample_id"], width_mm=doc["board"]["width_mm"],
        height_mm=doc["board"]["height_mm"],
        total_thickness_mm=doc["board"]["total_thickness_mm"],
        gx=gx, gy=gy, region_model=doc["region_model"],
        projection_threshold=doc["projection_threshold"],
        materials=doc["materials"], layer_phi=layer_phi, layer_material=layer_material,
        column_material=column_material, phi_3D=phi_3D, stackup_records=doc["stackup"],
    )


# ---------------------------------------------------------------------------
# MAPDL material resolution (used by pcbh.mesh)
# ---------------------------------------------------------------------------


def mat_resolver(model: GridModel, mat_ids: dict[str, int], nx: int, ny: int):
    """
    Returns a callable (ix, iy, slab_index) -> MAT id, closed over the grid
    model and the FE mesh resolution it is being applied at -- the mesh
    builder calls this once per (in-plane cell, z-row) rather than doing a
    bbox search, since both grids are uniform.
    """
    copper_slabs = set(model.layer_material) if model.region_model != "smeared" else None

    def resolve(ix: int, iy: int, slab_index: int) -> int:
        gi, gj = region_of_element(ix, iy, nx, ny, model.gx, model.gy)
        if model.region_model == "smeared":
            return mat_ids[model.column_material[gj, gi]]
        if slab_index in model.layer_material:
            return mat_ids[model.layer_material[slab_index][gj, gi]]
        return mat_ids["dielectric"]

    return resolve
