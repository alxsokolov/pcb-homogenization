#!/usr/bin/env python3
"""
Import an n-layer Gerber board into the pcb_homogenization sample format.

n-layer counterpart of import_real_board.py, which takes exactly two Gerbers
(--f-cu/--b-cu). Here any number of copper layers is accepted, given top
first, and the stackup comes from a YAML file.

Gerber carries no stackup information whatsoever -- it is artwork, one file
per layer, with no notion of layer order, thickness or material. (ODB++ does;
see odb_stackup.py.) So for Gerber the stackup is an input, not something to
be recovered, and --stackup is required for more than two copper layers.

Stackup YAML. Thicknesses in mm, ordered top to bottom, matching the
--copper order. A scalar applies to every layer of that kind:

    copper_thickness_mm: 0.035            # all copper layers 35 um
    dielectric_thickness_mm: [0.065, 0.2, 0.065]   # one per gap, top first
    dielectric_material: FR4

A board with n copper layers has n-1 dielectric gaps between them, so a list
must have exactly that many entries. Outer solder mask is not part of the
structural laminate and is not represented.

As in import_real_board.py, every absolute coordinate is shifted so the
board bounding box starts at (0, 0) -- the convention discretize.py assumes,
since a source board's own coordinates are normally an arbitrary offset from
the CAD tool's global origin. The bounding box must be supplied with --bbox
(e.g. read off Edge_Cuts.gbr), again because Gerber does not identify its own
outline.

Layer names follow the KiCad convention the rest of the pipeline uses: F_Cu
(top), In1_Cu .. In<n-2>_Cu, B_Cu (bottom). The metadata.json written is the
same n-layer schema import_odb_board_nlayer.py produces, so discretize.py,
coarsegrain_nlayer.py and simulate_pyansys_nlayer.py treat the two inputs
identically.

Usage:
    python import_gerber_board_nlayer.py --sample-id my4layer \\
        --copper F_Cu.gbr In1_Cu.gbr In2_Cu.gbr B_Cu.gbr \\
        --bbox 0 0 100 80 --stackup configs/stackup_my4layer.yaml

    # 2 copper layers: --stackup optional, falls back to the flat defaults
    python import_gerber_board_nlayer.py --sample-id my2layer \\
        --copper F_Cu.gbr B_Cu.gbr --bbox 0 0 20 15
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# _shift_gerber is the 2-layer importer's coordinate-shift routine; reused
# rather than duplicated, since the shift is identical per layer regardless
# of how many layers there are. Importing it runs only regex compilation --
# that module's main() is __main__-guarded.
from import_real_board import _shift_gerber

DEFAULT_COPPER_THICKNESS_MM = 0.035
DEFAULT_DIELECTRIC_THICKNESS_MM = 1.53


def _copper_layer_names(n: int) -> list[str]:
    if n == 2:
        return ["F_Cu", "B_Cu"]
    return ["F_Cu"] + [f"In{k}_Cu" for k in range(1, n - 1)] + ["B_Cu"]


def _positions(n: int) -> list[str]:
    if n == 2:
        return ["top", "bottom"]
    return ["top"] + [f"inner_{k}" for k in range(1, n - 1)] + ["bottom"]


def _as_list(value, count: int, label: str, default: float) -> list[float]:
    """Expand a scalar to `count` entries, or validate a list's length."""
    if value is None:
        return [default] * count
    if isinstance(value, (int, float)):
        return [float(value)] * count
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            sys.exit(f"--stackup: {label} has {len(value)} entries, expected {count}")
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            sys.exit(f"--stackup: {label} contains a non-numeric value: {value!r}")
    sys.exit(f"--stackup: {label} must be a number or a list of numbers, got {value!r}")


def _load_stackup(path: Path | None, n_copper: int) -> tuple[list[float], list[float], str]:
    """Return (copper thicknesses, dielectric thicknesses, dielectric material),
    each ordered top to bottom."""
    n_diel = n_copper - 1
    doc: dict = {}
    if path is not None:
        if not path.exists():
            sys.exit(f"Not found: {path}")
        try:
            import yaml
        except ImportError:
            sys.exit("--stackup needs PyYAML (pip install pyyaml)")
        doc = yaml.safe_load(path.read_text()) or {}
        if not isinstance(doc, dict):
            sys.exit(f"{path}: expected a mapping at the top level")
        doc = doc.get("stackup", doc)

    cu = _as_list(doc.get("copper_thickness_mm"), n_copper,
                  "copper_thickness_mm", DEFAULT_COPPER_THICKNESS_MM)
    di = _as_list(doc.get("dielectric_thickness_mm"), n_diel,
                  "dielectric_thickness_mm", DEFAULT_DIELECTRIC_THICKNESS_MM)
    material = str(doc.get("dielectric_material", "FR4"))

    for label, vals in (("copper_thickness_mm", cu), ("dielectric_thickness_mm", di)):
        bad = [v for v in vals if v <= 0]
        if bad:
            sys.exit(f"--stackup: {label} must be strictly positive, got {bad}")
    return cu, di, material


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copper", type=Path, nargs="+", required=True, metavar="GBR",
                    help="Copper-layer Gerbers, TOP FIRST then inner layers in stackup "
                         "order, bottom last")
    ap.add_argument("--bbox", type=float, nargs=4, required=True,
                    metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                    help="Board bounding box in the source Gerber coordinate system (mm)")
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--output-root", type=Path, default=Path("data/samples"))
    ap.add_argument("--stackup", type=Path, default=None,
                    help="Stackup YAML (see this script's --help). Required for more than "
                         "two copper layers; optional for two, which fall back to "
                         f"{DEFAULT_COPPER_THICKNESS_MM} mm copper / "
                         f"{DEFAULT_DIELECTRIC_THICKNESS_MM} mm dielectric.")
    args = ap.parse_args()

    n_cu = len(args.copper)
    if n_cu < 2:
        sys.exit(f"--copper needs at least 2 layers, got {n_cu}")
    missing = [str(p) for p in args.copper if not p.exists()]
    if missing:
        sys.exit(f"Not found: {', '.join(missing)}")
    if n_cu > 2 and args.stackup is None:
        sys.exit(f"{n_cu} copper layers given but no --stackup. Gerber carries no stackup "
                 "information, so the layer thicknesses have to be supplied explicitly "
                 "(see --help for the YAML format).")

    xmin, ymin, xmax, ymax = args.bbox
    width_mm, height_mm = xmax - xmin, ymax - ymin
    if width_mm <= 0 or height_mm <= 0:
        sys.exit(f"Bad --bbox: width={width_mm}, height={height_mm} "
                 "(expected xmin ymin xmax ymax)")

    cu_t, di_t, diel_material = _load_stackup(args.stackup, n_cu)
    names = _copper_layer_names(n_cu)
    positions = _positions(n_cu)

    sample_dir = args.output_root / args.sample_id
    gerber_dir = sample_dir / "gerber"
    for d in (gerber_dir, sample_dir / "mesh", sample_dir / "results"):
        d.mkdir(parents=True, exist_ok=True)

    for src, name in zip(args.copper, names):
        dst = gerber_dir / f"{name}.gbr"
        if src.resolve() == dst.resolve():
            print(f"  Skipped {dst} (source and destination are the same file)")
            continue
        dst.write_text(_shift_gerber(src.read_text(), dx_mm=-xmin, dy_mm=-ymin))
        print(f"  Wrote {dst}  (from {src.name}, shifted by {-xmin:+.4f}, {-ymin:+.4f} mm)")

    if args.stackup is not None:
        shutil.copy2(args.stackup, sample_dir / "stackup.yaml")
        print(f"  Copied {sample_dir / 'stackup.yaml'}")

    # Build the ordered slab list, bottom (z=0) to top -- the n-layer schema's
    # convention. The inputs are top-first, so walk them in reverse.
    cu_bottom_first = list(reversed(cu_t))
    di_bottom_first = list(reversed(di_t))
    names_bottom_first = list(reversed(names))

    stackup_records: list[dict] = []
    z = 0.0
    for k in range(n_cu):
        t = cu_bottom_first[k]
        stackup_records.append({
            "index": len(stackup_records), "kind": "copper",
            "name": names_bottom_first[k], "layer_name": names_bottom_first[k],
            "thickness_mm": t, "z0_mm": round(z, 9), "z1_mm": round(z + t, 9),
            "thickness_source": "stackup_yaml" if args.stackup else "default",
        })
        z += t
        if k < n_cu - 1:
            td = di_bottom_first[k]
            stackup_records.append({
                "index": len(stackup_records), "kind": "dielectric",
                "name": f"dielectric_{n_cu - 1 - k}",
                "thickness_mm": td, "z0_mm": round(z, 9), "z1_mm": round(z + td, 9),
                "material": diel_material,
                "thickness_source": "stackup_yaml" if args.stackup else "default",
            })
            z += td
    total_t = round(z, 9)
    index_of_name = {r["name"]: r["index"] for r in stackup_records if r["kind"] == "copper"}

    layer_records = [
        {"name": name, "file": f"gerber/{name}.gbr", "type": "copper",
         "position": position, "thickness_mm": t,
         "stackup_index": index_of_name[name]}
        for name, position, t in zip(names, positions, cu_t)
    ]

    stackup = {
        "n_copper_layers": n_cu,
        "n_dielectric_layers": n_cu - 1,
        "dielectric_material": diel_material.upper(),
        "total_thickness_mm": total_t,
        "layers": stackup_records,
    }
    if n_cu == 2:
        # Flat 2-layer keys as well, so a 2-layer board imported here still
        # runs through the existing 2-layer pipeline unchanged.
        stackup["copper_thickness_mm"] = cu_t[0]
        stackup["dielectric_thickness_mm"] = di_t[0]

    meta = {
        "sample_id": args.sample_id,
        "board": {"width_mm": round(width_mm, 6), "height_mm": round(height_mm, 6)},
        "layers": layer_records,
        "stackup": stackup,
        "generation": {
            "script": "import_gerber_board_nlayer.py",
            "type": "gerber_import_nlayer",
            "source_bbox_mm": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            "source_files": [str(p) for p in args.copper],
            "stackup_file": str(args.stackup) if args.stackup else None,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        },
    }
    (sample_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nWrote {sample_dir / 'metadata.json'}")
    print(f"Board size: {width_mm:.3f} x {height_mm:.3f} mm, {total_t * 1e3:.1f} um thick")
    print(f"Copper layers ({n_cu}), top to bottom:")
    for rec in layer_records:
        print(f"  {rec['name']:<8} {rec['position']:<9} {rec['thickness_mm'] * 1e3:6.1f} um")
    print(f"\nNext: python discretize.py {sample_dir}")


if __name__ == "__main__":
    main()
