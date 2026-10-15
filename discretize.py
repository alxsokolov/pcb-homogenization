#!/usr/bin/env python3
"""
Rasterize every copper layer of a sample (Gerber or ODB++, per
metadata.json's layers[].format) onto a uniform pixel grid.

Cell values:
    1.0  →  copper
    0.0  →  dielectric

Output written to <sample_dir>/mesh/:
    raster_<layer>.npy      per-copper-layer boolean map (e.g. raster_F_Cu.npy)
    discretization.json     grid metadata for homogenize.py / simulate_*.py

Usage:
    python discretize.py data/samples/rv3032_example
    python discretize.py data/samples/rv3032_example --resolution 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pcbh import gerber, odb
from pcbh.raster import rasterize_layer

SCHEMA_VERSION = 2


def discretize_sample(sample_dir: Path, resolution_px_per_mm: float = 10.0) -> dict:
    """Parse + rasterize every copper layer in a sample; write mesh/ outputs.
    Returns the per-layer copper fraction, keyed by layer name."""
    meta_path = sample_dir / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"Not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        sys.exit(f"{meta_path}: schema_version {meta.get('schema_version')!r}, "
                 f"expected {SCHEMA_VERSION} -- re-import with import_board.py, or migrate "
                 "an older sample with tools/migrate_metadata.py")

    W = meta["board"]["width_mm"]
    H = meta["board"]["height_mm"]
    nx = max(1, round(W * resolution_px_per_mm))
    ny = max(1, round(H * resolution_px_per_mm))
    origin = tuple(meta.get("odb", {}).get("origin_offset_mm", (0.0, 0.0)))

    mesh_dir = sample_dir / "mesh"
    mesh_dir.mkdir(exist_ok=True)

    layer_info: dict[str, dict] = {}
    fractions: dict[str, float] = {}

    for layer in meta["layers"]:
        name = layer["name"]
        src_path = sample_dir / layer["file"]
        if not src_path.exists():
            print(f"  [warn] {src_path} not found, skipping", file=sys.stderr)
            continue

        if layer.get("format") == "odb":
            print(f"  Parsing   {src_path} (ODB++) …")
            geo = odb.parse_odb_features(src_path, origin=origin)
            if geo.dropped:
                dropped_str = ", ".join(f"{n}x {k}" for k, n in geo.dropped.items())
                print(f"  [warn] {src_path.name}: skipped unsupported records: {dropped_str}",
                      file=sys.stderr)
        else:
            print(f"  Parsing   {src_path.name} …")
            geo = gerber.parse_gerber(src_path)
            if geo.dropped:
                dropped_str = ", ".join(f"{n}x {k}" for k, n in geo.dropped.items())
                print(f"  [warn] {src_path.name}: skipped unsupported aperture macros: "
                      f"{dropped_str}", file=sys.stderr)

        print(f"  Rasterizing {name} → {nx}×{ny} grid …")
        mask = rasterize_layer(geo, W, H, nx, ny)

        fname = f"raster_{name}.npy"
        np.save(mesh_dir / fname, mask.astype(bool))
        fraction = float(mask.mean())
        layer_info[name] = {"file": fname, "stackup_index": layer["stackup_index"],
                            "copper_fraction": round(fraction, 6)}
        fractions[name] = fraction
        print(f"  Saved  {fname}  copper_fraction={fraction:.4f}")

    disc = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": meta["sample_id"],
        "resolution_px_per_mm": resolution_px_per_mm,
        "grid": dict(nx=nx, ny=ny, dx_mm=round(W / nx, 8), dy_mm=round(H / ny, 8)),
        "board": dict(width_mm=W, height_mm=H),
        "layers": layer_info,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    (mesh_dir / "discretization.json").write_text(json.dumps(disc, indent=2) + "\n")
    print(f"  Saved  discretization.json")
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize every copper layer (Gerber or ODB++) to a binary material map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path,
                        help="Sample directory produced by import_board.py")
    parser.add_argument("--resolution", type=float, default=10.0,
                        help="Grid resolution in pixels per mm")
    args = parser.parse_args()

    if not args.sample_dir.exists():
        sys.exit(f"Not found: {args.sample_dir}")

    print(f"Discretizing {args.sample_dir}  @ {args.resolution} px/mm …")
    discretize_sample(args.sample_dir, resolution_px_per_mm=args.resolution)


if __name__ == "__main__":
    main()
