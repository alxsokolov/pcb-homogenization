#!/usr/bin/env python3
"""
Sample -> physical stackup + per-layer copper rasters.

The single place the pipeline reads a sample's geometry from. Turns
<sample_dir>/metadata.json into an ordered list of z slabs (bottom face at
z = 0, ascending) and attaches each copper slab's raster from
<sample_dir>/mesh/ (written by discretize.py).

This is the "geometrical space map of materials according to layout and
stackup" step: layout (the rasters) x stackup (the z slabs) fully determines
which material sits at any (x, y, z) in the board, before any mesh or
homogenization enters the picture. is_copper_at() evaluates exactly that
map, and every later step is a discretization of it.

A 2-copper-layer board is not a special case: it is just a 3-slab stackup
(bottom Cu / dielectric / top Cu). metadata.json schema_version 2 is the
only schema this module reads -- see import_board.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2


@dataclass
class Slab:
    """One physical layer of the laminate. z0 < z1, z = 0 at the bottom face."""
    index: int
    kind: str                       # "copper" | "dielectric"
    name: str                       # source layer name (e.g. ODB++ "mid-layer_1")
    thickness_mm: float
    z0_mm: float
    z1_mm: float
    layer_name: str | None = None   # copper only: raster key, e.g. "In1_Cu"
    material: str | None = None
    dielectric_type: str | None = None
    raster: np.ndarray | None = None    # copper only, (ny, nx), attached by attach_rasters()

    @property
    def is_copper(self) -> bool:
        return self.kind == "copper"

    @property
    def z_mid_mm(self) -> float:
        return 0.5 * (self.z0_mm + self.z1_mm)


@dataclass
class BoardStackup:
    sample_id: str
    width_mm: float
    height_mm: float
    slabs: list[Slab] = field(default_factory=list)      # ascending z
    dielectric_material: str = "FR4"

    @property
    def copper_slabs(self) -> list[Slab]:
        return [s for s in self.slabs if s.is_copper]

    @property
    def n_copper(self) -> int:
        return len(self.copper_slabs)

    @property
    def total_thickness_mm(self) -> float:
        return self.slabs[-1].z1_mm if self.slabs else 0.0

    def z_interfaces_mm(self) -> list[float]:
        """Ascending physical-layer boundaries, [0, ..., total_thickness]."""
        return [0.0] + [s.z1_mm for s in self.slabs]

    def slab_at_z(self, z_mm: float) -> Slab:
        """The slab containing z (half-open [z0, z1), clamped at the top face)."""
        for s in self.slabs:
            if s.z0_mm <= z_mm < s.z1_mm:
                return s
        return self.slabs[-1] if z_mm >= self.slabs[-1].z0_mm else self.slabs[0]

    # -- the material space map ---------------------------------------------

    def is_copper_at(self, x_mm: float, y_mm: float, z_mm: float) -> bool:
        """
        Evaluate the board's material map at a point: True for copper, False
        for dielectric. The full 3D geometry, with no discretization applied.
        """
        s = self.slab_at_z(z_mm)
        if not s.is_copper or s.raster is None:
            return False
        return bool(raster_lookup(s.raster, x_mm, y_mm, self.width_mm, self.height_mm))

    def copper_fraction_3d(self) -> float:
        """Thickness-weighted copper volume fraction of the whole board."""
        if self.total_thickness_mm <= 0:
            return 0.0
        num = sum(s.thickness_mm * float(s.raster.mean())
                  for s in self.copper_slabs if s.raster is not None)
        return num / self.total_thickness_mm

    def describe(self) -> str:
        lines = [f"{self.sample_id}: {self.width_mm:.3f} x {self.height_mm:.3f} mm, "
                 f"{self.total_thickness_mm * 1e3:.1f} um, {self.n_copper} copper layers"]
        w = max([len(s.name) for s in self.slabs] + [4])
        for s in reversed(self.slabs):
            phi = (f"phi={float(s.raster.mean()):.4f}"
                   if s.is_copper and s.raster is not None else "")
            lines.append(f"  [{s.index:>2}] {s.kind:<10} {s.name:<{w}} "
                         f"z {s.z0_mm:7.4f}-{s.z1_mm:7.4f}  "
                         f"{s.thickness_mm * 1e3:6.1f} um  {phi}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Raster lookup -- shared convention
# ---------------------------------------------------------------------------

def raster_lookup(raster: np.ndarray, x_mm: float, y_mm: float,
                  width_mm: float, height_mm: float) -> bool:
    """
    True if (x, y) in mm falls on a copper pixel.

    Raster convention, matching discretize.py's saved .npy arrays: row index
    scales with y (row 0 -> y near 0), column index with x, threshold 0.5.
    Only the exported PNG previews flip Y for display; the arrays themselves
    do not.
    """
    ny, nx = raster.shape
    ix = min(max(int(x_mm / width_mm * nx), 0), nx - 1)
    iy = min(max(int(y_mm / height_mm * ny), 0), ny - 1)
    return bool(raster[iy, ix] > 0.5)


def raster_area_mean(raster: np.ndarray, width_mm: float, height_mm: float,
                     x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Mean of the raster pixels whose footprint overlaps the mm-space box
    [x0,x1) x [y0,y1) -- i.e. that box's copper area fraction.
    """
    ny, nx = raster.shape
    ix0 = max(0, int(np.floor(x0 / width_mm * nx)))
    ix1 = min(nx, max(ix0 + 1, int(np.ceil(x1 / width_mm * nx))))
    iy0 = max(0, int(np.floor(y0 / height_mm * ny)))
    iy1 = min(ny, max(iy0 + 1, int(np.ceil(y1 / height_mm * ny))))
    return float(raster[iy0:iy1, ix0:ix1].mean())


def layer_phi_grid(raster: np.ndarray, gx: int, gy: int) -> np.ndarray:
    """
    Copper area fraction of one copper layer's raster, averaged onto a
    uniform gx (columns) x gy (rows) homogenization grid -- the grid
    pcbh.grid.build() resolves materials over, independent of both the
    raster's own pixel resolution and any later FE mesh resolution.

    When the raster divides evenly (nx % gx == 0 and ny % gy == 0) this is
    an exact reshape-and-mean: every pixel contributes to exactly one grid
    cell, so the area-weighted grid mean reproduces the whole-layer mean to
    machine precision (no pixel is ever double-counted). Otherwise falls
    back to raster_area_mean() per cell (the same floor/ceil pixel-slice
    rule discretize.py's callers have always used), which double-counts
    boundary pixels between adjacent cells by a small amount -- a warning is
    printed once by the caller (pcbh.grid.build()), not here, so a batch
    build over many layers prints it once rather than once per layer.
    """
    ny, nx = raster.shape
    if nx % gx == 0 and ny % gy == 0:
        return raster.reshape(gy, ny // gy, gx, nx // gx).mean(axis=(1, 3)).astype(np.float64)
    width_mm, height_mm = float(nx), float(ny)   # pixel units; cell size is just nx/gx, ny/gy
    dx, dy = nx / gx, ny / gy
    out = np.empty((gy, gx), dtype=np.float64)
    for j in range(gy):
        y0, y1 = j * dy, (j + 1) * dy
        for i in range(gx):
            x0, x1 = i * dx, (i + 1) * dx
            out[j, i] = raster_area_mean(raster, width_mm, height_mm, x0, y0, x1, y1)
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _slabs_from_metadata(stackup: dict) -> list[Slab]:
    slabs = []
    for rec in sorted(stackup["layers"], key=lambda r: r["index"]):
        slabs.append(Slab(
            index=int(rec["index"]),
            kind=rec["kind"],
            name=rec["name"],
            thickness_mm=float(rec["thickness_mm"]),
            z0_mm=float(rec["z0_mm"]),
            z1_mm=float(rec["z1_mm"]),
            layer_name=rec.get("layer_name"),
            material=rec.get("material"),
            dielectric_type=rec.get("dielectric_type"),
        ))
    return slabs


def load_board(sample_dir: Path, load_rasters: bool = True) -> BoardStackup:
    """
    Read <sample_dir>/metadata.json into a BoardStackup, optionally attaching
    each copper slab's raster from <sample_dir>/mesh/.
    """
    meta_path = Path(sample_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Not found: {meta_path}")
    meta = json.loads(meta_path.read_text())

    if meta.get("schema_version") != SCHEMA_VERSION or "layers" not in meta.get("stackup", {}):
        raise ValueError(
            f"{meta_path}: not a schema_version {SCHEMA_VERSION} sample (got "
            f"{meta.get('schema_version')!r}). Re-import with import_board.py, or migrate an "
            "older sample with tools/migrate_metadata.py."
        )
    stk = meta["stackup"]
    slabs = _slabs_from_metadata(stk)

    if not slabs:
        raise ValueError(f"{meta_path}: stackup contains no layers")
    bad = [s.name for s in slabs if s.thickness_mm <= 0]
    if bad:
        raise ValueError(f"{meta_path}: zero/negative thickness for layer(s) {bad} -- "
                         "re-import with an explicit --stackup override")

    board = BoardStackup(
        sample_id=meta.get("sample_id", Path(sample_dir).name),
        width_mm=float(meta["board"]["width_mm"]),
        height_mm=float(meta["board"]["height_mm"]),
        slabs=slabs,
        dielectric_material=stk.get("dielectric_material", "FR4"),
    )
    if load_rasters:
        attach_rasters(sample_dir, board)
    return board


def attach_rasters(sample_dir: Path, board: BoardStackup) -> None:
    """
    Attach each copper slab's raster from <sample_dir>/mesh/, using
    discretization.json's layer_maps (keyed by the layer name in
    metadata.json, e.g. "F_Cu"/"In3_Cu").
    """
    mesh_dir = Path(sample_dir) / "mesh"
    disc_path = mesh_dir / "discretization.json"
    if not disc_path.exists():
        raise FileNotFoundError(f"Run discretize.py first -- not found: {disc_path}")
    disc = json.loads(disc_path.read_text())
    layer_maps = disc.get("layers", {})

    missing = []
    shapes = set()
    for s in board.copper_slabs:
        info = layer_maps.get(s.layer_name)
        if info is None:
            missing.append(s.layer_name)
            continue
        path = mesh_dir / info["file"]
        if not path.exists():
            missing.append(f"{s.layer_name} ({path.name})")
            continue
        s.raster = np.load(path)
        shapes.add(s.raster.shape)

    if missing:
        raise FileNotFoundError(
            f"No raster for copper layer(s) {missing} in {disc_path}. "
            f"Available: {sorted(layer_maps)}. Re-run discretize.py on this sample."
        )
    if len(shapes) > 1:
        raise ValueError(f"Copper layer rasters have differing shapes {sorted(shapes)} -- "
                         "re-run discretize.py so every layer is on one grid")
