#!/usr/bin/env python3
"""
Homogenize a board's copper into materials over a uniform in-plane grid.

See pcbh.grid's module docstring for the three region models. The grid
resolution (--gx/--gy) is independent of the FE mesh resolution the board is
later solved at (simulate_board.py/simulate_assembly.py's --nx/--ny) --
pass the mesh's own --nx/--ny here to make regions coincide with elements
(no in-plane averaging beyond the element itself), or fewer for regions
uniformly coarser than the elements.

Output: <sample_dir>/mesh/homogenization.json

Usage:
    python homogenize.py data/samples/my_board --gx 50 --gy 50
    python homogenize.py data/samples/my_board --gx 50 --gy 50 --model smeared
    python homogenize.py data/samples/my_board --gx 50 --gy 50 --model projection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from pcbh import grid
from pcbh.materials import PhaseProps, load_materials
from pcbh.stackup import load_board


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Homogenize a board's copper into materials over a uniform in-plane grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--gx", type=int, required=True, help="Grid regions along x")
    ap.add_argument("--gy", type=int, required=True, help="Grid regions along y")
    ap.add_argument("--model", choices=["zlayers", "smeared", "projection"], default="zlayers",
                    help="How each region's copper becomes material (default: zlayers -- see "
                         "this script's module docstring for why)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="--model projection: phi at or above which a region's layer is "
                         "taken as solid copper")
    ap.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    args = ap.parse_args()

    if not 0.0 < args.threshold <= 1.0:
        sys.exit("--threshold must be in (0, 1]")

    try:
        board = load_board(args.sample_dir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))

    print(board.describe())
    print(f"\n3D copper volume fraction: {board.copper_fraction_3d():.6f}")

    for s in board.copper_slabs:
        if s.raster.shape[1] % args.gx or s.raster.shape[0] % args.gy:
            print(f"[warn] raster shape {s.raster.shape} is not an exact multiple of the "
                  f"{args.gx} x {args.gy} grid -- region phi falls back to a per-cell pixel-"
                  "slice mean, which double-counts boundary pixels between adjacent regions "
                  "by a small amount.", file=sys.stderr)
            break

    p = PhaseProps.from_materials(load_materials(args.materials))
    print(f"\nRegions: uniform {args.gx} x {args.gy} grid "
          f"({args.gx * args.gy} regions, "
          f"{board.width_mm / args.gx:.4f} x {board.height_mm / args.gy:.4f} mm each) …")
    model = grid.build(board, args.gx, args.gy, args.model, p, args.threshold)

    n_pairs = (args.gx * args.gy if args.model == "smeared"
               else args.gx * args.gy * board.n_copper)
    print(f"\nRegion model: {args.model}")
    print(f"  {n_pairs} material assignments -> {len(model.materials)} distinct materials "
          f"({n_pairs / max(len(model.materials), 1):.1f}x deduplication)")

    mesh_dir = Path(args.sample_dir) / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    out_path = mesh_dir / grid.OUTPUT_NAME
    grid.save(model, out_path)
    print(f"\nWrote {out_path}")

    # Conservation check: the area-weighted mean of the region fractions
    # must reproduce the whole-layer copper fraction, since homogenizing
    # only averages copper, it does not create or destroy it. On an exact
    # grid division (raster evenly divisible by gx/gy) this should be at
    # machine precision; otherwise small residuals are expected (boundary
    # pixels counted in both neighbouring regions -- see the warning above).
    print("\nPer-copper-layer copper fraction:")
    print(f"  {'slab':>5} {'layer':<10} {'area-weighted':>13} {'whole-layer':>12} {'delta':>9}")
    worst_delta = 0.0
    for s in board.copper_slabs:
        weighted = float(np.mean(model.layer_phi[s.index]))
        whole = float(s.raster.mean())
        delta = weighted - whole
        worst_delta = max(worst_delta, abs(delta))
        print(f"  [{s.index:>3}] {s.layer_name or s.name:<10} {weighted:13.6f} "
              f"{whole:12.6f} {delta:+9.2e}")
    if worst_delta > 5e-3:
        print(f"  [warn] worst area-weighted deviation {worst_delta:.2e} is larger than "
              f"expected -- check the raster/grid resolution", file=sys.stderr)


if __name__ == "__main__":
    main()
