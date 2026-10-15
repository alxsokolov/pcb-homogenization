#!/usr/bin/env python3
"""
Import a PCB board -- Gerber or ODB++ -- into the pcb-homogenization sample
format (<sample_dir>/metadata.json + copied source files).

A 2-layer board is not a special case here: both subcommands always write
the full stackup slab list (metadata.json schema_version 2 -- see
RESTRUCTURING_PLAN.md §4.1), bottom (z=0) to top. A 2-copper-layer board is
simply a 3-slab stackup [bottom Cu / dielectric / top Cu].

Layer naming follows the KiCad convention the rest of the pipeline uses:
F_Cu (top), In1_Cu .. In<n-2>_Cu (inner, counted from the top), B_Cu
(bottom).

Usage:
    python import_board.py gerber --sample-id my_board \\
        --copper F_Cu.gbr B_Cu.gbr --bbox XMIN YMIN XMAX YMAX

    python import_board.py gerber --sample-id my_4layer \\
        --copper F_Cu.gbr In1_Cu.gbr In2_Cu.gbr B_Cu.gbr \\
        --bbox 0 0 100 80 --stackup configs/stackup_my4layer.yaml

    python import_board.py odb --sample-id my_board --odb-dir path/to/extracted/job
    python import_board.py odb --sample-id my_board --odb-dir path/to/job \\
        --stackup configs/stackup_my_board.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from pcbh import odb, odb_stackup
from pcbh.gerber import shift_coordinates
from pcbh.stackup import SCHEMA_VERSION

_OUTLINE_NAME_RE = re.compile(r'edge.?cuts|profile|outline|board.?outline', re.I)
_DRILL_TYPES = ("DRILL", "ROUT")

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


def _write_sample(sample_dir: Path, sample_id: str, width_mm: float, height_mm: float,
                  layer_records: list[dict], stackup_slabs: list[dict],
                  dielectric_material: str, source: dict, extra_top: dict | None = None) -> None:
    """Write metadata.json (schema_version 2) and print a summary table."""
    total_t = round(stackup_slabs[-1]["z1_mm"], 9) if stackup_slabs else 0.0
    meta = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "board": {"width_mm": round(width_mm, 6), "height_mm": round(height_mm, 6)},
        "layers": layer_records,
        "stackup": {
            "dielectric_material": dielectric_material.upper(),
            "total_thickness_mm": total_t,
            "layers": stackup_slabs,
        },
        "source": source,
        "generation": {
            "script": "import_board.py",
            "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        },
    }
    if extra_top:
        meta.update(extra_top)

    (sample_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nWrote {sample_dir / 'metadata.json'}")
    print(f"Board size: {width_mm:.3f} x {height_mm:.3f} mm, {total_t * 1e3:.1f} um thick")
    print(f"Copper layers ({len(layer_records)}), top to bottom:")
    for rec in layer_records:
        slab = next(s for s in stackup_slabs if s["index"] == rec["stackup_index"])
        print(f"  {rec['name']:<8} {rec['position']:<9} {slab['thickness_mm'] * 1e3:6.1f} um")
    print(f"\nNext: python discretize.py {sample_dir}")


# ---------------------------------------------------------------------------
# gerber
# ---------------------------------------------------------------------------

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


def _load_gerber_stackup(path: Path | None, n_copper: int) -> tuple[list[float], list[float], str]:
    """
    Return (copper thicknesses, dielectric thicknesses, dielectric material),
    each ordered top to bottom. Gerber carries no stackup information
    whatsoever -- it is artwork, one file per layer, with no notion of layer
    order, thickness or material -- so for more than 2 copper layers this is
    an input, not something to be recovered; see --stackup's --help text.

    Stackup YAML, thicknesses in mm, ordered top to bottom, matching the
    --copper order. A scalar applies to every layer of that kind:

        copper_thickness_mm: 0.035            # all copper layers 35 um
        dielectric_thickness_mm: [0.065, 0.2, 0.065]   # one per gap, top first
        dielectric_material: FR4

    A board with n copper layers has n-1 dielectric gaps between them, so a
    list must have exactly that many entries.
    """
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


def cmd_gerber(args: argparse.Namespace) -> None:
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

    cu_t, di_t, diel_material = _load_gerber_stackup(args.stackup, n_cu)
    names = _copper_layer_names(n_cu)
    positions = _positions(n_cu)

    sample_dir = args.output_root / args.sample_id
    gerber_dir = sample_dir / "gerber"
    gerber_dir.mkdir(parents=True, exist_ok=True)

    for src, name in zip(args.copper, names):
        dst = gerber_dir / f"{name}.gbr"
        if src.resolve() == dst.resolve():
            print(f"  Skipped {dst} (source and destination are the same file)")
            continue
        dst.write_text(shift_coordinates(src.read_text(), dx_mm=-xmin, dy_mm=-ymin))
        print(f"  Wrote {dst}  (from {src.name}, shifted by {-xmin:+.4f}, {-ymin:+.4f} mm)")

    if args.stackup is not None:
        shutil.copy2(args.stackup, sample_dir / "stackup.yaml")
        print(f"  Copied {sample_dir / 'stackup.yaml'}")

    # Build the ordered slab list, bottom (z=0) to top -- the schema's
    # convention. The inputs are top-first, so walk them in reverse.
    cu_bottom_first = list(reversed(cu_t))
    di_bottom_first = list(reversed(di_t))
    names_bottom_first = list(reversed(names))

    stackup_slabs: list[dict] = []
    z = 0.0
    for k in range(n_cu):
        t = cu_bottom_first[k]
        stackup_slabs.append({
            "index": len(stackup_slabs), "kind": "copper",
            "name": names_bottom_first[k], "layer_name": names_bottom_first[k],
            "thickness_mm": t, "z0_mm": round(z, 9), "z1_mm": round(z + t, 9),
        })
        z += t
        if k < n_cu - 1:
            td = di_bottom_first[k]
            stackup_slabs.append({
                "index": len(stackup_slabs), "kind": "dielectric",
                "name": f"dielectric_{n_cu - 1 - k}",
                "thickness_mm": td, "z0_mm": round(z, 9), "z1_mm": round(z + td, 9),
                "material": diel_material,
            })
            z += td
    index_of_name = {s["name"]: s["index"] for s in stackup_slabs if s["kind"] == "copper"}

    layer_records = [
        {"name": name, "file": f"gerber/{name}.gbr", "format": "gerber",
         "position": position, "stackup_index": index_of_name[name]}
        for name, position, t in zip(names, positions, cu_t)
    ]

    source = {
        "format": "gerber",
        "bbox_mm": [xmin, ymin, xmax, ymax],
        "files": [str(p) for p in args.copper],
        "stackup_file": str(args.stackup) if args.stackup else None,
    }
    _write_sample(sample_dir, args.sample_id, width_mm, height_mm,
                 layer_records, stackup_slabs, diel_material, source)


# ---------------------------------------------------------------------------
# odb
# ---------------------------------------------------------------------------

def _find_step_dir(odb_dir: Path, step: str | None) -> tuple[str, Path]:
    steps_dir = odb_dir / 'steps'
    if not steps_dir.is_dir():
        sys.exit(f'Not found: {steps_dir}  (expected an extracted ODB++ job root)')
    if step:
        step_dir = steps_dir / step
        if not step_dir.is_dir():
            sys.exit(f'Not found: {step_dir}')
        return step, step_dir
    candidates = sorted(d for d in steps_dir.iterdir() if d.is_dir())
    if not candidates:
        sys.exit(f'No step directories found under {steps_dir}')
    return candidates[0].name, candidates[0]


def _load_stackup_overrides(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    if not path.exists():
        sys.exit(f'Not found: {path}')
    try:
        import yaml
    except ImportError:
        sys.exit('--stackup needs PyYAML (pip install pyyaml)')
    doc = yaml.safe_load(path.read_text()) or {}
    layers = doc.get('layers', doc)
    if not isinstance(layers, dict):
        sys.exit(f'{path}: expected a mapping of layer name -> thickness_mm '
                 '(optionally under a top-level "layers:" key)')
    out: dict[str, float] = {}
    for k, v in layers.items():
        try:
            out[str(k).lower()] = float(v)
        except (TypeError, ValueError):
            sys.exit(f'{path}: thickness for layer {k!r} is not a number: {v!r}')
    return out


def cmd_odb(args: argparse.Namespace) -> None:
    odb_dir = args.odb_dir
    matrix_path = odb_dir / 'matrix' / 'matrix'
    if not matrix_path.exists():
        sys.exit(f'Not found: {matrix_path}')

    step_name, step_dir = _find_step_dir(odb_dir, args.step)
    layers_dir = step_dir / 'layers'
    odb_layers = odb.parse_odb_matrix(matrix_path)

    overrides = _load_stackup_overrides(args.stackup)
    try:
        stk = odb_stackup.resolve_stackup(
            odb_dir, step_dir, overrides=overrides,
            default_copper_thickness_mm=args.default_copper_thickness_mm)
    except ValueError as e:
        sys.exit(str(e))

    print(f'Step {step_name!r} — resolved stackup:')
    print(odb_stackup.format_stackup(stk))
    print()
    for w in stk.warnings:
        print(f'  [warn] {w}', file=sys.stderr)
    if stk.total_thickness_mm <= 0:
        sys.exit('Resolved stackup has zero total thickness — supply --stackup overrides.')

    copper = stk.copper_layers                      # ascending z (bottom first)
    copper_top_first = list(reversed(copper))        # matrix row order (top first)
    n_cu = len(copper)

    # Via copper is not modelled; make the omission visible rather than silent.
    drills = [l for l in odb_layers if l.type in _DRILL_TYPES]
    if drills:
        print(f'  [note] {len(drills)} drill/rout layer(s) in this job are NOT imported — '
              f'via copper is not represented in the layer-plane model:', file=sys.stderr)
        for l in drills:
            span = f' ({l.start_name} -> {l.end_name})' if l.start_name or l.end_name else ''
            print(f'           {l.name}{span}', file=sys.stderr)

    # --- board outline / bounding box ---
    profile_path = step_dir / 'profile'
    outline_layer = next((l for l in odb_layers if _OUTLINE_NAME_RE.search(l.name)), None)
    if profile_path.exists():
        outline_source = 'profile'
        outline_geo = odb.parse_odb_features(profile_path)
    elif outline_layer is not None:
        outline_source = outline_layer.name
        outline_geo = odb.parse_odb_features(layers_dir / outline_layer.name / 'features')
    else:
        sys.exit('No outline found: no steps/<step>/profile file, and no matrix layer '
                 'named EDGE.CUTS / PROFILE / OUTLINE')
    bbox = odb.layer_bbox_mm(outline_geo)
    if bbox is None:
        sys.exit(f'Outline source {outline_source!r} has no geometry to compute a bounding box from')
    xmin, ymin, xmax, ymax = bbox
    width_mm, height_mm = xmax - xmin, ymax - ymin
    if width_mm <= 0 or height_mm <= 0:
        sys.exit(f'Bad outline bounding box: width={width_mm}, height={height_mm}')

    # --- copy the trimmed job subtree ---
    sample_dir = args.output_root / args.sample_id
    odb_out = sample_dir / 'odb'
    for d in [odb_out, sample_dir / 'mesh', sample_dir / 'results']:
        d.mkdir(parents=True, exist_ok=True)

    (odb_out / 'matrix').mkdir(exist_ok=True)
    shutil.copy2(matrix_path, odb_out / 'matrix' / 'matrix')

    (odb_out / 'misc').mkdir(exist_ok=True)
    for fname in ('info', 'attrlist'):
        src = odb_dir / 'misc' / fname
        if src.exists():
            shutil.copy2(src, odb_out / 'misc' / fname)

    step_out = odb_out / 'steps' / step_name
    step_out.mkdir(parents=True, exist_ok=True)
    for fname in ('stephdr', 'attrlist'):
        src = step_dir / fname
        if src.exists():
            shutil.copy2(src, step_out / fname)

    copy_names = [l.name for l in stk.layers]
    if outline_layer is not None:
        copy_names.append(outline_layer.name)
    for name in copy_names:
        src, dst = layers_dir / name, step_out / 'layers' / name
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ('features', 'attrlist'):
            if (src / fname).exists():
                shutil.copy2(src / fname, dst / fname)
        print(f'  Copied  odb/steps/{step_name}/layers/{name}/')

    if profile_path.exists():
        shutil.copy2(profile_path, step_out / 'profile')
        print(f'  Copied  odb/steps/{step_name}/profile')

    # --- metadata.json ---
    names = _copper_layer_names(n_cu)
    positions = _positions(n_cu)
    index_of = {id(l): i for i, l in enumerate(stk.layers)}

    layer_records = []
    for name, position, l in zip(names, positions, copper_top_first):
        layer_records.append({
            "name": name,
            "file": f'odb/steps/{step_name}/layers/{l.name}/features',
            "format": "odb",
            "position": position,
            "odb_layer_name": l.name,
            "odb_layer_type": l.odb_type,
            "stackup_index": index_of[id(l)],
        })

    stackup_slabs = []
    for i, l in enumerate(stk.layers):
        rec = {
            "index": i,
            "kind": l.kind,
            "name": l.name,
            "thickness_mm": l.thickness_mm,
            "z0_mm": l.z0_mm,
            "z1_mm": l.z1_mm,
            "thickness_source": l.thickness_source,
        }
        if l.kind == "copper":
            rec["layer_name"] = names[copper_top_first.index(l)]
        if l.dielectric_type:
            rec["dielectric_type"] = l.dielectric_type
        if l.material:
            rec["material"] = l.material
        if l.dielectric_constant is not None:
            rec["dielectric_constant"] = l.dielectric_constant
        if l.loss_tangent is not None:
            rec["loss_tangent"] = l.loss_tangent
        stackup_slabs.append(rec)

    diel = [l for l in stk.layers if not l.is_copper]
    diel_material = next((l.material for l in diel if l.material), None) or "FR4"

    source = {
        "format": "odb",
        "bbox_mm": [round(xmin, 6), round(ymin, 6), round(xmax, 6), round(ymax, 6)],
        "files": [str(odb_dir)],
        "stackup_file": str(args.stackup) if args.stackup else None,
    }
    extra_top = {
        "odb": {
            "job_step": step_name,
            "outline_source": outline_source,
            "origin_offset_mm": [round(xmin, 6), round(ymin, 6)],
            "drill_layers_not_imported": [l.name for l in drills],
        },
    }
    if stk.board_thickness_attr_mm is not None:
        extra_top["odb"]["board_thickness_attr_mm"] = stk.board_thickness_attr_mm

    _write_sample(sample_dir, args.sample_id, width_mm, height_mm,
                 layer_records, stackup_slabs, diel_material, source, extra_top)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="format", required=True)

    g = sub.add_parser("gerber", help="Import from Gerber (any number of copper layers)",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=_load_gerber_stackup.__doc__)
    g.add_argument("--copper", type=Path, nargs="+", required=True, metavar="GBR",
                   help="Copper-layer Gerbers, TOP FIRST then inner layers in stackup "
                        "order, bottom last")
    g.add_argument("--bbox", type=float, nargs=4, required=True,
                   metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="Board bounding box in the source Gerber coordinate system (mm)")
    g.add_argument("--sample-id", required=True)
    g.add_argument("--output-root", type=Path, default=Path("data/samples"))
    g.add_argument("--stackup", type=Path, default=None,
                   help="Stackup YAML (see above). Required for more than two copper "
                        f"layers; optional for two, which fall back to "
                        f"{DEFAULT_COPPER_THICKNESS_MM} mm copper / "
                        f"{DEFAULT_DIELECTRIC_THICKNESS_MM} mm dielectric.")
    g.set_defaults(func=cmd_gerber)

    o = sub.add_parser("odb", help="Import from an extracted ODB++ job (all copper layers)")
    o.add_argument("--odb-dir", type=Path, required=True,
                   help="Extracted ODB++ job root (contains matrix/, steps/, misc/)")
    o.add_argument("--sample-id", required=True)
    o.add_argument("--output-root", type=Path, default=Path("data/samples"))
    o.add_argument("--step", help="Step name to import (default: first found under steps/)")
    o.add_argument("--stackup", type=Path, default=None,
                   help="YAML mapping ODB++ layer name -> thickness in mm, overriding "
                        "whatever the job itself carries (for boards whose ODB++ has no "
                        "usable stackup)")
    o.add_argument("--default-copper-thickness-mm", type=float,
                   default=odb_stackup.DEFAULT_COPPER_THICKNESS_MM,
                   help="Copper thickness for a conductive layer with no .copper_weight")
    o.set_defaults(func=cmd_odb)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
