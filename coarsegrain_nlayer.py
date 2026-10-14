#!/usr/bin/env python3
"""
Coarse-grain an n-layer PCB: discretize the board plane into regions, then
resolve each region's copper into materials.

This is the middle of the n-layer pipeline. The material space map
(stackup_nlayer.py: layout rasters x stackup slabs) is continuous; this step
chooses the in-plane regions it is averaged over, and how each region's
copper becomes a material. The result is a complete, inspectable description
of the board model -- region geometry, per-layer copper fractions, and the
material constants -- written before MAPDL is involved at all, so it can be
checked and plotted on its own.

The two choices are independent:

  --mode          how the board plane is cut into regions
                    grid      NX x NY uniform rectangles. Pass the FE mesh's
                              own --nx/--ny to make regions coincide with
                              elements (no in-plane averaging beyond the
                              element itself); pass fewer for regions
                              uniformly coarser than the elements.
                    quadtree  adaptive: refine only where a copper/dielectric
                              boundary runs, so detail is spent where the
                              layout is actually heterogeneous. --max-level
                              refines every mixed region to that depth and
                              2:1-balances; --max-leaves instead spends a
                              fixed region budget on the most heterogeneous
                              regions first.

  --region-model  how each region's copper becomes material
                    homogenized_zlayers  (default) each copper slab gets its
                              own orthotropic Voigt-Reuss material from that
                              slab's own phi in that region; dielectric slabs
                              stay plain dielectric.
                    homogenized          one smeared orthotropic material for
                              the whole board thickness, from the region's
                              phi_3D.
                    projection           black/white: each copper slab in the
                              region is resolved to pure copper or pure
                              dielectric by phi >= --projection-threshold.

Why homogenized_zlayers is the default here, unlike the 2-layer pipeline:
the smeared model depends on the stackup only through phi_3D -- the
"structural" Voigt/Reuss bounds are algebraically identical to the
geometry-blind global ones (proof and numerical check in
homogenization_nlayer.py). So smearing discards the layer structure entirely.
On a 2-layer board that costs little, since copper is ~4% of the thickness.
On the 8-layer reference board copper is 36% of the thickness and the
per-layer fractions range from 0.67 to 0.86, so smearing throws away most of
what distinguishes the board.

Output: <sample_dir>/mesh/coarsegrain_nlayer.json

    regions[]   bbox_mm, level, phi per copper slab, phi_3D, and the material
                key for each copper slab (or one column material, smeared)
    materials{} the deduplicated material table, MAPDL MP labels in mm-N-MPa
                units, ready for mapdl.mp()
    stackup[]   the slab list the regions were built against

Materials are deduplicated by copper fraction, so a 50x50 grid over eight
layers yields a few hundred distinct materials rather than 20000.

Unlike the 2-layer path, no copper "dilution" factor is needed: the FE mesh
bands z by physical slab, so every copper layer is resolved by at least one
element row and its volume is represented at its true thickness. (The 2-layer
model needed dilution because --cu-penetration-elements deliberately smeared
35 um of copper over a much thicker core element.)

Usage:
    python coarsegrain_nlayer.py data/samples/ublox_nlayer --mode grid --nx 50 --ny 50
    python coarsegrain_nlayer.py data/samples/ublox_nlayer --mode quadtree --max-level 6
    python coarsegrain_nlayer.py data/samples/ublox_nlayer --mode quadtree --max-leaves 256
    python coarsegrain_nlayer.py data/samples/ublox_nlayer --mode grid --nx 50 --ny 50 \\
        --region-model projection
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from homogenization import load_materials
from homogenization_nlayer import (
    PhaseProps,
    SlabSpec,
    laminate_bounds,
    orthotropic_slab,
    orthotropic_smeared,
)
from stackup_nlayer import BoardStackup, load_board_stackup, raster_area_mean

SCHEMA_VERSION = 1
OUTPUT_NAME = "coarsegrain_nlayer.json"

# A region's copper fraction counts as pure (not worth refining) within this
# of 0 or 1 -- matches quadtree._EPS_FRAC.
_EPS_FRAC = 1e-9

# Copper fractions are rounded to this many decimals before being used as a
# material-table key. 1e-6 is far below any physically meaningful difference
# in phi and well below the raster's own quantization, so this only collapses
# duplicates.
_PHI_DECIMALS = 6


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


@dataclass
class Region:
    """One in-plane coarse-graining region, with its per-copper-slab phi."""
    x0: float
    y0: float
    x1: float
    y1: float
    level: int = 0
    phis: dict[int, float] = field(default_factory=dict)   # slab index -> phi
    children: "list[Region] | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def impurity(self) -> float:
        """
        Phase variance of the most heterogeneous copper layer in this region:
        phi*(1-phi) peaks at phi=0.5 and is zero for a pure layer.

        Taken as a max over layers, not a sum or a mean: a region containing
        a boundary on even one layer still needs refining, and a mean would
        let seven uniform planes mask one detailed signal layer -- exactly
        the situation on a multilayer board, where most layers are mostly
        plane. Deliberately not area-weighted, for the reason given in
        quadtree._impurity_score(): area otherwise dominates and the splitter
        just quarters uniformly instead of concentrating on heterogeneity.
        """
        if not self.phis:
            return 0.0
        return max(phi * (1.0 - phi) for phi in self.phis.values())

    def is_mixed(self) -> bool:
        return any(_EPS_FRAC < phi < 1.0 - _EPS_FRAC for phi in self.phis.values())


def _measure(board: BoardStackup, x0: float, y0: float, x1: float, y1: float,
             level: int) -> Region:
    """Build a region and fill in every copper slab's area fraction over it."""
    phis = {
        s.index: raster_area_mean(s.raster, board.width_mm, board.height_mm, x0, y0, x1, y1)
        for s in board.copper_slabs if s.raster is not None
    }
    return Region(x0, y0, x1, y1, level=level, phis=phis)


def _split(node: Region, board: BoardStackup) -> None:
    if node.children is not None:
        return
    xm, ym = (node.x0 + node.x1) / 2, (node.y0 + node.y1) / 2
    lv = node.level + 1
    node.children = [
        _measure(board, node.x0, node.y0, xm, ym, lv),
        _measure(board, xm, node.y0, node.x1, ym, lv),
        _measure(board, node.x0, ym, xm, node.y1, lv),
        _measure(board, xm, ym, node.x1, node.y1, lv),
    ]


def collect_leaves(node: Region, out: list[Region] | None = None) -> list[Region]:
    if out is None:
        out = []
    if node.is_leaf:
        out.append(node)
    else:
        for c in node.children:
            collect_leaves(c, out)
    return out


def _find_leaf(root: Region, x: float, y: float) -> Region:
    node = root
    while not node.is_leaf:
        xm, ym = (node.x0 + node.x1) / 2, (node.y0 + node.y1) / 2
        node = node.children[(1 if x >= xm else 0) + (2 if y >= ym else 0)]
    return node


def build_grid(board: BoardStackup, nx: int, ny: int) -> Region:
    """
    Uniform nx x ny regions, as a flat one-level forest: a non-leaf root whose
    children are the grid cells. Same representation as the adaptive tree, so
    everything downstream reads one flat leaf list either way.
    """
    root = Region(0.0, 0.0, board.width_mm, board.height_mm, level=0)
    dx, dy = board.width_mm / nx, board.height_mm / ny
    root.children = [
        _measure(board, i * dx, j * dy, (i + 1) * dx, (j + 1) * dy, 1)
        for j in range(ny) for i in range(nx)
    ]
    return root


def _recursive_split(node: Region, board: BoardStackup, max_level: int) -> None:
    if node.level >= max_level or not node.is_mixed():
        return
    _split(node, board)
    for c in node.children:
        _recursive_split(c, board, max_level)


def balance_2to1(root: Region, board: BoardStackup, max_level: int) -> None:
    """
    Split leaves whose neighbour is more than one level finer, until adjacent
    leaves differ by at most one level. Neighbours are found by point-location
    descent at edge midpoints (O(n log n) per pass), as in quadtree.py.
    """
    eps = min(board.width_mm, board.height_mm) * 1e-6
    while True:
        to_split = []
        for leaf in collect_leaves(root):
            xmid, ymid = (leaf.x0 + leaf.x1) / 2, (leaf.y0 + leaf.y1) / 2
            probes = []
            if leaf.x1 < board.width_mm - eps:
                probes.append((leaf.x1 + eps, ymid))
            if leaf.x0 > eps:
                probes.append((leaf.x0 - eps, ymid))
            if leaf.y1 < board.height_mm - eps:
                probes.append((xmid, leaf.y1 + eps))
            if leaf.y0 > eps:
                probes.append((xmid, leaf.y0 - eps))
            if any(_find_leaf(root, px, py).level - leaf.level > 1 for px, py in probes):
                to_split.append(leaf)
        if not to_split:
            return
        split_any = False
        for leaf in to_split:
            if leaf.level < max_level:
                _split(leaf, board)
                split_any = True
        if not split_any:
            return      # everything that wants splitting is already at max_level


def build_quadtree_boundary(board: BoardStackup, max_level: int) -> tuple[Region, int]:
    """Refine every region straddling a copper boundary on any layer, then
    2:1-balance. Returns (root, leaf count before balancing)."""
    root = _measure(board, 0.0, 0.0, board.width_mm, board.height_mm, 0)
    _recursive_split(root, board, max_level)
    n_before = len(collect_leaves(root))
    balance_2to1(root, board, max_level)
    return root, n_before


def build_quadtree_budget(board: BoardStackup, max_leaves: int,
                          max_level: int) -> tuple[Region, int]:
    """
    Spend a fixed region budget greedily: repeatedly split the single most
    heterogeneous region (by Region.impurity()) until the budget runs out or
    nothing is worth splitting. Not 2:1-balanced -- balancing fights a hard
    region-count cap, and nothing downstream builds a conforming mesh on
    these regions.
    """
    root = _measure(board, 0.0, 0.0, board.width_mm, board.height_mm, 0)
    n_leaves = 1
    counter = itertools.count()       # tie-break so heapq never compares Regions
    heap: list[tuple[float, int, Region]] = []

    def maybe_push(node: Region) -> None:
        if node.level >= max_level:
            return
        score = node.impurity()
        if score > 0.0:
            heapq.heappush(heap, (-score, next(counter), node))

    maybe_push(root)
    while heap and n_leaves + 3 <= max_leaves:
        node = heapq.heappop(heap)[-1]
        _split(node, board)
        n_leaves += 3                  # one leaf -> four children
        for c in node.children:
            maybe_push(c)
    return root, n_leaves


# ---------------------------------------------------------------------------
# Region -> materials
# ---------------------------------------------------------------------------


class MaterialTable:
    """Deduplicating material store: identical constants share one key."""

    def __init__(self) -> None:
        self._by_key: dict[tuple, str] = {}
        self.materials: dict[str, dict] = {}

    def add(self, dedup_key: tuple, build) -> str:
        name = self._by_key.get(dedup_key)
        if name is None:
            name = f"m{len(self.materials)}"
            self._by_key[dedup_key] = name
            self.materials[name] = build()
        return name

    def __len__(self) -> int:
        return len(self.materials)


def _slab_specs(board: BoardStackup, phis: dict[int, float]) -> list[SlabSpec]:
    return [
        SlabSpec(thickness_mm=s.thickness_mm,
                 phi=phis.get(s.index, 0.0) if s.is_copper else 0.0,
                 kind=s.kind, name=s.name)
        for s in board.slabs
    ]


def resolve_region_materials(region: Region, board: BoardStackup, p: PhaseProps,
                             region_model: str, table: MaterialTable,
                             projection_threshold: float) -> dict:
    """
    Turn one region's copper fractions into material assignments, writing any
    new materials into `table`. Returns the region's JSON record.
    """
    bounds = laminate_bounds(_slab_specs(board, region.phis), p)
    rec: dict = {
        "bbox_mm": [round(region.x0, 6), round(region.y0, 6),
                    round(region.x1, 6), round(region.y1, 6)],
        "level": region.level,
        "phi": {str(i): round(phi, _PHI_DECIMALS) for i, phi in sorted(region.phis.items())},
        "phi_3D": round(bounds["phi_3D"], _PHI_DECIMALS),
    }

    if region_model == "homogenized":
        # One material for the whole column. Keyed on phi_3D alone, which is
        # genuinely all this model depends on.
        key = ("smeared", round(bounds["phi_3D"], _PHI_DECIMALS))
        rec["column_material"] = table.add(key, lambda: orthotropic_smeared(bounds))
        return rec

    # Per-copper-slab material (z-resolved), homogenized or black/white.
    slab_mats: dict[str, str] = {}
    for s in board.copper_slabs:
        phi = region.phis.get(s.index, 0.0)
        if region_model == "projection":
            is_cu = phi >= projection_threshold
            phi_eff = 1.0 if is_cu else 0.0
        else:
            phi_eff = phi
        phi_key = round(phi_eff, _PHI_DECIMALS)
        slab_mats[str(s.index)] = table.add(
            ("slab", phi_key), lambda pk=phi_key: orthotropic_slab(pk, p))
    rec["slab_materials"] = slab_mats
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Coarse-grain an n-layer PCB into regions with resolved materials.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--mode", choices=["grid", "quadtree"], default="grid",
                    help="How to cut the board plane into regions (default: grid)")
    ap.add_argument("--nx", type=int, default=50, help="--mode grid: regions along x")
    ap.add_argument("--ny", type=int, default=50, help="--mode grid: regions along y")
    ap.add_argument("--max-level", type=int, default=6,
                    help="--mode quadtree: maximum depth (root=0). With --max-leaves this "
                         "is only a safety cap.")
    ap.add_argument("--max-leaves", type=int, default=None,
                    help="--mode quadtree: cap the region count, spending the budget on the "
                         "most heterogeneous regions first instead of refining every mixed "
                         "region to --max-level")
    ap.add_argument("--region-model",
                    choices=["homogenized_zlayers", "homogenized", "projection"],
                    default="homogenized_zlayers",
                    help="How each region's copper becomes material (default: "
                         "homogenized_zlayers -- see this script's --help epilog for why)")
    ap.add_argument("--projection-threshold", type=float, default=0.5,
                    help="--region-model projection: phi at or above which a region's layer "
                         "is taken as solid copper")
    ap.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    args = ap.parse_args()

    if args.mode == "grid" and (args.nx < 1 or args.ny < 1):
        sys.exit("--nx/--ny must be >= 1")
    if args.mode == "quadtree" and args.max_leaves is not None and args.max_leaves < 4:
        sys.exit("--max-leaves must be >= 4 (one split produces four regions)")
    if args.mode == "grid" and args.max_leaves is not None:
        sys.exit("--max-leaves applies to --mode quadtree; --mode grid is sized by --nx/--ny")
    if not 0.0 < args.projection_threshold <= 1.0:
        sys.exit("--projection-threshold must be in (0, 1]")

    try:
        board = load_board_stackup(args.sample_dir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))

    print(board.describe())
    print(f"\n3D copper volume fraction: {board.copper_fraction_3d():.6f}")
    if board.schema == "legacy2":
        print("[note] legacy 2-layer sample -- the n-layer path reduces exactly to the "
              "2-layer one here, which makes this a useful cross-check.")

    p = PhaseProps.from_materials(load_materials(args.materials))

    # -- discretize the plane into regions --
    if args.mode == "grid":
        print(f"\nRegions: uniform {args.nx} x {args.ny} grid "
              f"({args.nx * args.ny} regions, "
              f"{board.width_mm / args.nx:.4f} x {board.height_mm / args.ny:.4f} mm each) …")
        root = build_grid(board, args.nx, args.ny)
        n_before = args.nx * args.ny
        criterion = "uniform_grid"
        disc = {"mode": "grid", "nx": args.nx, "ny": args.ny}
    elif args.max_leaves is not None:
        print(f"\nRegions: quadtree, budget max_leaves={args.max_leaves}, "
              f"max_level={args.max_level} …")
        root, n_before = build_quadtree_budget(board, args.max_leaves, args.max_level)
        criterion = "greedy_impurity_budget"
        disc = {"mode": "quadtree", "max_leaves": args.max_leaves,
                "max_level": args.max_level, "balanced_2to1": False}
    else:
        print(f"\nRegions: quadtree, refine every mixed region to max_level={args.max_level}, "
              f"then 2:1-balance …")
        root, n_before = build_quadtree_boundary(board, args.max_level)
        criterion = "boundary_straddle_then_2to1_balance"
        disc = {"mode": "quadtree", "max_level": args.max_level, "balanced_2to1": True}

    leaves = collect_leaves(root)
    levels = [r.level for r in leaves]
    print(f"  {len(leaves)} regions (before balancing / at budget stop: {n_before})")
    if args.mode == "quadtree":
        print(f"  level range {min(levels)}-{max(levels)}")
        for lv in range(min(levels), max(levels) + 1):
            cnt = sum(1 for l in levels if l == lv)
            if cnt:
                print(f"    level {lv}: {cnt} regions")

    # -- resolve materials --
    table = MaterialTable()
    records = [resolve_region_materials(r, board, p, args.region_model, table,
                                        args.projection_threshold)
               for r in leaves]
    n_pairs = (len(leaves) if args.region_model == "homogenized"
               else len(leaves) * board.n_copper)
    print(f"\nRegion model: {args.region_model}")
    print(f"  {n_pairs} material assignments -> {len(table)} distinct materials "
          f"({n_pairs / max(len(table), 1):.1f}x deduplication)")

    stackup_records = [
        {"index": s.index, "kind": s.kind, "name": s.name, "layer_name": s.layer_name,
         "thickness_mm": s.thickness_mm, "z0_mm": s.z0_mm, "z1_mm": s.z1_mm}
        for s in board.slabs
    ]
    out = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": board.sample_id,
        "generated_by": "coarsegrain_nlayer.py",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "board": {"width_mm": board.width_mm, "height_mm": board.height_mm,
                  "total_thickness_mm": board.total_thickness_mm},
        "stackup_schema": board.schema,
        "discretization": {**disc, "criterion": criterion},
        "region_model": args.region_model,
        "projection_threshold": (args.projection_threshold
                                 if args.region_model == "projection" else None),
        "material_units": "MAPDL mm-N-MPa (E and G in MPa, CTE in 1/K)",
        "stackup": stackup_records,
        "copper_slab_indices": [s.index for s in board.copper_slabs],
        "n_regions": len(records),
        "n_materials": len(table),
        "materials": table.materials,
        "regions": records,
    }

    mesh_dir = Path(args.sample_dir) / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    out_path = mesh_dir / OUTPUT_NAME
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")

    # Per-layer summary, and with it a conservation check: the area-weighted
    # mean of the region fractions must reproduce the whole-layer copper
    # fraction, since coarse-graining only averages copper, it does not
    # create or destroy it. (An unweighted mean would not -- quadtree regions
    # differ in area by orders of magnitude -- so weighting is what makes
    # this a check rather than a coincidence.) Small residuals are expected:
    # a region's phi comes from the pixels whose footprint overlaps it, so
    # boundary pixels are counted in both neighbours, exactly as in
    # quadtree._area_mean().
    print("\nPer-copper-layer copper fraction:")
    print(f"  {'slab':>5} {'layer':<10} {'area-weighted':>13} {'whole-layer':>12} "
          f"{'delta':>9}   {'min':>6} {'max':>6}")
    areas = np.array([r.area for r in leaves])
    total_area = areas.sum()
    worst_delta = 0.0
    for s in board.copper_slabs:
        vals = np.array([r.phis[s.index] for r in leaves])
        weighted = float((vals * areas).sum() / total_area)
        whole = float(s.raster.mean())
        delta = weighted - whole
        worst_delta = max(worst_delta, abs(delta))
        print(f"  [{s.index:>3}] {s.layer_name or s.name:<10} {weighted:13.6f} "
              f"{whole:12.6f} {delta:+9.2e}   {vals.min():6.4f} {vals.max():6.4f}")
    if worst_delta > 5e-3:
        print(f"  [warn] worst area-weighted deviation {worst_delta:.2e} is larger than "
              f"expected from boundary-pixel double counting -- check the region geometry",
              file=sys.stderr)


if __name__ == "__main__":
    main()
