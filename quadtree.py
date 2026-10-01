#!/usr/bin/env python3
"""
Quadtree coarse-graining of a discretized PCB, with per-leaf Voigt-Reuss
homogenization.

Milestone 1 of the goal-oriented adaptive coarse-graining pipeline: builds a
2:1-balanced quadtree over the board plane, splitting leaves that straddle a
copper/FR4 boundary (pure-copper and pure-FR4 leaves are already exact under
Voigt-Reuss, so they don't need refining), then homogenizes each leaf
independently. No thermomechanical solve and no solder-joint QoI yet — this
only produces the coarse-graining map and its per-leaf effective properties.

Reads the per-layer material maps produced by discretize.py.

Output written to <sample_dir>/mesh/quadtree.json:
    max_level, criterion, board dims, and one record per leaf:
    bbox_mm, level, phi_top, phi_bot, plus the homogenized properties from
    homogenization.voigt_reuss_bounds().

Usage:
    python quadtree.py data/samples/dummy_test
    python quadtree.py data/samples/dummy_test --max-level 8
    python quadtree.py data/samples/dummy_test --max-leaves 16
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from homogenization import load_materials, voigt_reuss_bounds

_EPS_FRAC = 1e-9


# ---------------------------------------------------------------------------
# Quadtree data structure
# ---------------------------------------------------------------------------

@dataclass
class QuadNode:
    x0: float
    y0: float
    x1: float
    y1: float
    level: int
    phi_top: float = 0.0
    phi_bot: float = 0.0
    children: "list[QuadNode] | None" = None
    props: dict | None = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


def _bbox_to_pixel_slice(map_arr: np.ndarray, Lx: float, Ly: float,
                         x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    """Pixel index range [iy0:iy1, ix0:ix1] covering the mm-space bbox.

    map_arr follows discretize.py's raw-array convention: row index scales
    directly with y (row 0 -> y near 0), column index scales with x -- the
    same convention simulate_tensile.py's assign_materials() relies on. Only
    the exported PNG preview flips Y for display; the .npy arrays do not.
    """
    ny_map, nx_map = map_arr.shape
    ix0 = max(0, int(np.floor(x0 / Lx * nx_map)))
    ix1 = min(nx_map, max(ix0 + 1, int(np.ceil(x1 / Lx * nx_map))))
    iy0 = max(0, int(np.floor(y0 / Ly * ny_map)))
    iy1 = min(ny_map, max(iy0 + 1, int(np.ceil(y1 / Ly * ny_map))))
    return iy0, iy1, ix0, ix1


def _area_mean(map_arr: np.ndarray, Lx: float, Ly: float,
               x0: float, y0: float, x1: float, y1: float) -> float:
    """Mean of map_arr pixels whose footprint overlaps [x0,x1) x [y0,y1) (mm)."""
    iy0, iy1, ix0, ix1 = _bbox_to_pixel_slice(map_arr, Lx, Ly, x0, y0, x1, y1)
    return float(map_arr[iy0:iy1, ix0:ix1].mean())


def _make_leaf(top_map, bot_map, Lx, Ly, x0, y0, x1, y1, level) -> QuadNode:
    return QuadNode(
        x0, y0, x1, y1, level=level,
        phi_top=_area_mean(top_map, Lx, Ly, x0, y0, x1, y1),
        phi_bot=_area_mean(bot_map, Lx, Ly, x0, y0, x1, y1),
    )


def _is_mixed(node: QuadNode) -> bool:
    def mixed(phi: float) -> bool:
        return _EPS_FRAC < phi < 1.0 - _EPS_FRAC
    return mixed(node.phi_top) or mixed(node.phi_bot)


def _split(node: QuadNode, top_map, bot_map, Lx, Ly) -> None:
    """Turn a leaf into an internal node with 4 children (SW, SE, NW, NE)."""
    if node.children is not None:
        return
    xm, ym = (node.x0 + node.x1) / 2, (node.y0 + node.y1) / 2
    node.children = [
        _make_leaf(top_map, bot_map, Lx, Ly, node.x0, node.y0, xm, ym, node.level + 1),
        _make_leaf(top_map, bot_map, Lx, Ly, xm, node.y0, node.x1, ym, node.level + 1),
        _make_leaf(top_map, bot_map, Lx, Ly, node.x0, ym, xm, node.y1, node.level + 1),
        _make_leaf(top_map, bot_map, Lx, Ly, xm, ym, node.x1, node.y1, node.level + 1),
    ]


def _recursive_split(node: QuadNode, top_map, bot_map, Lx, Ly, max_level) -> None:
    if node.level >= max_level or not _is_mixed(node):
        return
    _split(node, top_map, bot_map, Lx, Ly)
    for c in node.children:
        _recursive_split(c, top_map, bot_map, Lx, Ly, max_level)


def build_quadtree(top_map: np.ndarray, bot_map: np.ndarray,
                   Lx: float, Ly: float, max_level: int) -> QuadNode:
    """Root covers the full board; splits where copper/FR4 boundaries mix."""
    root = _make_leaf(top_map, bot_map, Lx, Ly, 0.0, 0.0, Lx, Ly, level=0)
    _recursive_split(root, top_map, bot_map, Lx, Ly, max_level)
    return root


def _impurity_score(node: QuadNode) -> float:
    """
    Phase variance: phi*(1-phi) peaks at phi=0.5 (most heterogeneous) and is
    zero for a pure leaf. A cheap geometric stand-in for a real error
    indicator (e.g. DWR eta_K later).

    Deliberately NOT area-weighted: an area*impurity score was tried first,
    but since most of a real board is fairly evenly mixed at a coarse
    view, area dominates and the greedy splitter just quarters uniformly
    breadth-first (empirically: 16/16 leaves landed at exactly level 2 on
    both test samples) instead of concentrating depth on the locally most
    heterogeneous regions -- the opposite of what a leaf-count budget is
    for. Pure impurity (no area factor) produces genuinely non-uniform
    depth (verified: levels 1-4 in the same 16-leaf budget).
    """
    return max(node.phi_top * (1.0 - node.phi_top), node.phi_bot * (1.0 - node.phi_bot))


def build_quadtree_by_budget(top_map: np.ndarray, bot_map: np.ndarray,
                             Lx: float, Ly: float,
                             max_leaves: int, max_level: int = 12) -> QuadNode:
    """
    Greedy leaf-count-budget coarse-graining: repeatedly split the single
    most heterogeneous leaf (by _impurity_score) until either the leaf
    budget is exhausted or nothing left is worth splitting. Directly caps
    the total leaf count, unlike build_quadtree()'s depth-first "split
    every mixed leaf" which has no leaf-count ceiling short of max_level.
    """
    root = _make_leaf(top_map, bot_map, Lx, Ly, 0.0, 0.0, Lx, Ly, level=0)
    n_leaves = 1

    counter = itertools.count()  # tie-breaker so heapq never compares QuadNodes
    heap: list[tuple[float, int, QuadNode]] = []

    def maybe_push(node: QuadNode) -> None:
        if node.level >= max_level:
            return
        score = _impurity_score(node)
        if score <= 0.0:
            return
        heapq.heappush(heap, (-score, next(counter), node))

    maybe_push(root)

    while heap and n_leaves < max_leaves:
        _neg_score, _tie, node = heapq.heappop(heap)
        _split(node, top_map, bot_map, Lx, Ly)
        n_leaves += 3  # one leaf -> four children, net +3
        for c in node.children:
            maybe_push(c)

    return root


def collect_leaves(node: QuadNode, out: list | None = None) -> list[QuadNode]:
    if out is None:
        out = []
    if node.is_leaf:
        out.append(node)
    else:
        for c in node.children:
            collect_leaves(c, out)
    return out


def _find_leaf(root: QuadNode, x: float, y: float) -> QuadNode:
    node = root
    while not node.is_leaf:
        xm, ym = (node.x0 + node.x1) / 2, (node.y0 + node.y1) / 2
        idx = (1 if x >= xm else 0) + (2 if y >= ym else 0)
        node = node.children[idx]
    return node


def balance_2to1(root: QuadNode, top_map, bot_map, Lx, Ly, max_level: int) -> None:
    """
    Iteratively split leaves whose neighbor is more than one level finer,
    until every pair of adjacent leaves differs by at most one level.

    Neighbors are located by point-location descent from the root at each
    leaf's edge midpoints (+- a small epsilon) rather than pairwise bbox
    comparison -- O(n * max_level) per pass instead of O(n^2).
    """
    eps = min(Lx, Ly) * 1e-6
    while True:
        leaves = collect_leaves(root)
        to_split = []
        for leaf in leaves:
            xmid, ymid = (leaf.x0 + leaf.x1) / 2, (leaf.y0 + leaf.y1) / 2
            probes = []
            if leaf.x1 < Lx - eps:
                probes.append((leaf.x1 + eps, ymid))
            if leaf.x0 > eps:
                probes.append((leaf.x0 - eps, ymid))
            if leaf.y1 < Ly - eps:
                probes.append((xmid, leaf.y1 + eps))
            if leaf.y0 > eps:
                probes.append((xmid, leaf.y0 - eps))
            for px, py in probes:
                if _find_leaf(root, px, py).level - leaf.level > 1:
                    to_split.append(leaf)
                    break
        if not to_split:
            break
        for leaf in to_split:
            if leaf.level < max_level:
                _split(leaf, top_map, bot_map, Lx, Ly)


# ---------------------------------------------------------------------------
# Per-leaf homogenization
# ---------------------------------------------------------------------------

def homogenize_leaves(root: QuadNode, materials: dict,
                      t_cu_mm: float, t_fr4_mm: float) -> None:
    """Populate leaf.props with Voigt-Reuss homogenized properties."""
    E_cu = materials["copper"]["E_GPa"] * 1e9
    E_fr4 = materials["FR4"]["E_GPa"] * 1e9
    alpha_cu = materials["copper"].get("CTE_per_K", 17e-6)
    alpha_fr4_ip = materials["FR4"].get("CTE_inplane_per_K", materials["FR4"].get("CTE_per_K", 14e-6))
    alpha_fr4_op = materials["FR4"].get("CTE_outofplane_per_K", materials["FR4"].get("CTE_per_K", 60e-6))

    for leaf in collect_leaves(root):
        leaf.props = voigt_reuss_bounds(
            leaf.phi_top, leaf.phi_bot,
            E_cu=E_cu, E_fr4=E_fr4,
            t_cu=t_cu_mm, t_fr4=t_fr4_mm,
            alpha_cu=alpha_cu, alpha_fr4_ip=alpha_fr4_ip, alpha_fr4_op=alpha_fr4_op,
        )


def to_records(root: QuadNode) -> list[dict]:
    records = []
    for leaf in collect_leaves(root):
        rec = {
            "bbox_mm": [round(leaf.x0, 6), round(leaf.y0, 6), round(leaf.x1, 6), round(leaf.y1, 6)],
            "level": leaf.level,
            "phi_top": round(leaf.phi_top, 6),
            "phi_bot": round(leaf.phi_bot, 6),
        }
        if leaf.props:
            rec.update(leaf.props)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Sample I/O + CLI
# ---------------------------------------------------------------------------

def _load_layer_maps(sample_dir: Path) -> dict[str, np.ndarray]:
    mesh_dir = sample_dir / "mesh"
    disc_path = mesh_dir / "discretization.json"
    if not disc_path.exists():
        sys.exit(f"Run discretize.py first — not found: {disc_path}")
    with open(disc_path) as f:
        disc = json.load(f)

    layer_maps: dict[str, np.ndarray] = {}
    for _lname, linfo in disc.get("layer_maps", {}).items():
        pos = linfo.get("position")
        fpath = mesh_dir / linfo["file"]
        if pos in ("top", "bottom") and fpath.exists():
            layer_maps[pos] = np.load(fpath)

    if not layer_maps:
        combined = np.load(mesh_dir / "material_map.npy")
        layer_maps = {"top": combined, "bottom": combined}
    return layer_maps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a 2:1-balanced quadtree coarse-graining with per-leaf Voigt-Reuss homogenization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--max-level", type=int, default=6,
                        help="Maximum quadtree depth (root=0). In --max-leaves mode this is "
                             "just a safety cap, not the primary control.")
    parser.add_argument("--max-leaves", type=int, default=None,
                        help="Cap total leaf count via greedy impurity-priority splitting "
                             "instead of depth-first 'split every mixed leaf'. Skips 2:1 "
                             "balancing (which would fight a hard leaf-count cap).")
    parser.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    args = parser.parse_args()

    if not args.sample_dir.exists():
        sys.exit(f"Not found: {args.sample_dir}")

    with open(args.sample_dir / "metadata.json") as f:
        meta = json.load(f)

    layer_maps = _load_layer_maps(args.sample_dir)
    mat = load_materials(args.materials)

    Lx = meta["board"]["width_mm"]
    Ly = meta["board"]["height_mm"]
    t_cu = meta["stackup"]["copper_thickness_mm"]
    t_fr4 = meta["stackup"]["dielectric_thickness_mm"]

    if args.max_leaves is not None:
        print(f"Board {Lx:.1f} x {Ly:.1f} mm — building quadtree "
             f"(budget: max_leaves={args.max_leaves}, max_level={args.max_level}) …")
        root = build_quadtree_by_budget(layer_maps["top"], layer_maps["bottom"], Lx, Ly,
                                        args.max_leaves, args.max_level)
        n_before = len(collect_leaves(root))
        print("Skipping 2:1 balancing (--max-leaves mode) — not yet needed since nothing "
             "downstream builds a conforming FE mesh on this quadtree.")
        criterion = "greedy_impurity_budget"
    else:
        print(f"Board {Lx:.1f} x {Ly:.1f} mm — building quadtree (max_level={args.max_level}) …")
        root = build_quadtree(layer_maps["top"], layer_maps["bottom"], Lx, Ly, args.max_level)
        n_before = len(collect_leaves(root))
        balance_2to1(root, layer_maps["top"], layer_maps["bottom"], Lx, Ly, args.max_level)
        criterion = "boundary_straddle_then_2to1_balance"

    homogenize_leaves(root, mat, t_cu, t_fr4)

    leaves = collect_leaves(root)
    levels = [leaf.level for leaf in leaves]
    print(f"Leaves: {len(leaves)}  (before 2:1 balancing / at budget stop: {n_before})")
    print(f"Level range: {min(levels)}–{max(levels)}")
    for lv in range(min(levels), max(levels) + 1):
        cnt = sum(1 for l in levels if l == lv)
        if cnt:
            print(f"  level {lv}: {cnt} leaves")

    mesh_dir = args.sample_dir / "mesh"
    out = {
        "max_level": args.max_level,
        "max_leaves": args.max_leaves,
        "criterion": criterion,
        "board": {"width_mm": Lx, "height_mm": Ly},
        "n_leaves": len(leaves),
        "leaves": to_records(root),
    }
    out_path = mesh_dir / "quadtree.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
