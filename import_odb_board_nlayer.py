#!/usr/bin/env python3
"""
Import an ODB++ job into the pcb_homogenization sample format, keeping ALL
copper layers (n-layer counterpart of import_odb_board.py).

Where the 2-layer importer picks the outermost pair of SIGNAL layers and sums
whatever dielectric sits between them into one core, this one keeps every
conductive layer (SIGNAL / POWER_GROUND / MIXED) and every dielectric between
the outermost pair as its own slab with its own thickness, resolved by
odb_stackup.resolve_stackup() -- see that module for where ODB++ actually
stores stackup information and the three decoding rules involved.

The written metadata.json is a superset of the 2-layer schema:

  layers[]            one entry per copper layer, in ODB++ matrix row order
                      (top first), each with "stackup_index" pointing into
                      stackup.layers. discretize.py consumes this list
                      unchanged and writes one <name>_map.npy per layer, so
                      rasterization needs no n-layer variant.
  stackup.layers[]    the full laminate, ordered bottom (z=0) to top, each
                      slab with kind/thickness/z0/z1/material. This is what
                      the n-layer homogenization and FE scripts read.
  stackup.copper_thickness_mm / .dielectric_thickness_mm
                      emitted ONLY for a 2-copper-layer board, so such a
                      sample still runs through the existing 2-layer
                      pipeline. For n > 2 they are deliberately absent:
                      there is no single meaningful value, and omitting them
                      makes the 2-layer scripts fail loudly on an n-layer
                      sample instead of silently building a wrong model.

Layer naming follows the KiCad convention the rest of the pipeline uses:
F_Cu (top), In1_Cu .. In<n-2>_Cu (inner, counted from the top), B_Cu
(bottom).

Boards whose ODB++ carries no usable stackup (no DIELECTRIC rows, or
thicknesses that are plainly not the board's) need their slab thicknesses
supplied from outside the job: --stackup <yaml> maps ODB++ layer names to
thicknesses in mm and overrides whatever was found.

    # stackup.yaml
    layers:
      top: 0.035
      dielectric_1: 0.2
      bottom: 0.035

Only what the pipeline consumes is copied into <sample_dir>/odb/: every
conductive layer's features + attrlist, the dielectrics' attrlist, the board
outline, matrix/matrix, misc/info and misc/attrlist. Solder mask/paste/
silkscreen/drill layers are left behind -- re-run against the full job if
they are ever needed.

Vias are NOT represented. On a thin multilayer board (the JODY-W6 reference
job is 0.728 mm over 8 layers, with blind and stacked micro vias between
every adjacent layer pair) via copper carries real through-thickness
stiffness that this layer-plane-only model does not see. The drill layers are
reported at import so the omission is at least visible.

Usage:
    python import_odb_board_nlayer.py --odb-dir path/to/extracted/job --sample-id my_board
    python import_odb_board_nlayer.py --odb-dir path/to/job --sample-id my_board \\
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

import odb_parser
import odb_stackup

_OUTLINE_NAME_RE = re.compile(r'edge.?cuts|profile|outline|board.?outline', re.I)
_DRILL_TYPES = ("DRILL", "ROUT")


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


def _copper_layer_names(n: int) -> list[str]:
    """KiCad-convention names for n copper layers, top first."""
    if n == 2:
        return ['F_Cu', 'B_Cu']
    return ['F_Cu'] + [f'In{k}_Cu' for k in range(1, n - 1)] + ['B_Cu']


def _positions(n: int) -> list[str]:
    """Human-readable positions, top first. Unique per layer, so they stay
    usable as discretization.json's layer_maps keys."""
    if n == 2:
        return ['top', 'bottom']
    return ['top'] + [f'inner_{k}' for k in range(1, n - 1)] + ['bottom']


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--odb-dir', type=Path, required=True,
                        help='Extracted ODB++ job root (contains matrix/, steps/, misc/)')
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--output-root', type=Path, default=Path('data/samples'))
    parser.add_argument('--step', help='Step name to import (default: first found under steps/)')
    parser.add_argument('--stackup', type=Path, default=None,
                        help='YAML mapping ODB++ layer name -> thickness in mm, overriding '
                             'whatever the job itself carries (for boards whose ODB++ has no '
                             'usable stackup)')
    parser.add_argument('--default-copper-thickness-mm', type=float,
                        default=odb_stackup.DEFAULT_COPPER_THICKNESS_MM,
                        help='Copper thickness for a conductive layer with no .copper_weight')
    parser.add_argument('--source', default=None,
                        help='Attribution note for metadata.json (e.g. an origin URL)')
    args = parser.parse_args()

    odb_dir = args.odb_dir
    matrix_path = odb_dir / 'matrix' / 'matrix'
    if not matrix_path.exists():
        sys.exit(f'Not found: {matrix_path}')

    step_name, step_dir = _find_step_dir(odb_dir, args.step)
    layers_dir = step_dir / 'layers'
    odb_layers = odb_parser.parse_odb_matrix(matrix_path)

    # --- stackup ---
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
    copper_top_first = list(reversed(copper))       # matrix row order (top first)
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
    # Prefer steps/<step>/profile (the standard ODB++ location, and the only
    # option for jobs whose outline is merely implied by a DRILL/ROUT layer);
    # fall back to a name-matched matrix layer.
    profile_path = step_dir / 'profile'
    outline_layer = next((l for l in odb_layers if _OUTLINE_NAME_RE.search(l.name)), None)
    if profile_path.exists():
        outline_source = 'profile'
        outline_geo = odb_parser.parse_odb_features(profile_path)
    elif outline_layer is not None:
        outline_source = outline_layer.name
        outline_geo = odb_parser.parse_odb_features(layers_dir / outline_layer.name / 'features')
    else:
        sys.exit('No outline found: no steps/<step>/profile file, and no matrix layer '
                 'named EDGE.CUTS / PROFILE / OUTLINE')
    bbox = odb_parser.layer_bbox_mm(outline_geo)
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

    # Conductive layers need features (rasterized by discretize.py);
    # dielectrics only need attrlist (their geometry is never rasterized --
    # a dielectric slab is full-area by definition).
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
            "type": "copper",
            "position": position,
            "thickness_mm": l.thickness_mm,
            "format": "odb",
            "odb_layer_name": l.name,
            "odb_layer_type": l.odb_type,
            # index into stackup.layers (ascending z) -- the unambiguous link
            # between a raster and its physical slab
            "stackup_index": index_of[id(l)],
        })

    stackup_records = []
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
            # back-link to layers[] so the FE script can find this slab's raster
            rec["layer_name"] = names[copper_top_first.index(l)]
        if l.dielectric_type:
            rec["dielectric_type"] = l.dielectric_type
        if l.material:
            rec["material"] = l.material
        if l.dielectric_constant is not None:
            rec["dielectric_constant"] = l.dielectric_constant
        if l.loss_tangent is not None:
            rec["loss_tangent"] = l.loss_tangent
        stackup_records.append(rec)

    diel = [l for l in stk.layers if not l.is_copper]
    diel_material = next((l.material for l in diel if l.material), None)
    stackup = {
        "n_copper_layers": n_cu,
        "n_dielectric_layers": len(diel),
        "dielectric_material": (diel_material or "FR4").upper(),
        "total_thickness_mm": round(stk.total_thickness_mm, 9),
        "layers": stackup_records,
    }
    if stk.board_thickness_attr_mm is not None:
        stackup["board_thickness_attr_mm"] = stk.board_thickness_attr_mm
    if n_cu == 2:
        # Flat 2-layer keys, so a 2-copper-layer board imported with this
        # script still runs through the existing 2-layer pipeline unchanged.
        # Omitted for n > 2 on purpose -- see the module docstring.
        stackup["copper_thickness_mm"] = copper[0].thickness_mm
        stackup["dielectric_thickness_mm"] = round(sum(l.thickness_mm for l in diel), 9)

    meta = {
        "sample_id": args.sample_id,
        "board": {"width_mm": round(width_mm, 6), "height_mm": round(height_mm, 6)},
        "layers": layer_records,
        "stackup": stackup,
        "odb": {
            "job_step": step_name,
            "outline_source": outline_source,
            "origin_offset_mm": [round(xmin, 6), round(ymin, 6)],
            "drill_layers_not_imported": [l.name for l in drills],
        },
        "generation": {
            "script": "import_odb_board_nlayer.py",
            "type": "odb_import_nlayer",
            "source": args.source,
            "stackup_overrides": overrides or None,
            "stackup_warnings": stk.warnings or None,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        },
    }
    (sample_dir / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'\nWrote {sample_dir / "metadata.json"}')
    print(f'Board size: {width_mm:.3f} x {height_mm:.3f} mm, '
          f'{stk.total_thickness_mm * 1e3:.1f} um thick')
    print(f'Copper layers ({n_cu}), top to bottom:')
    for rec in layer_records:
        print(f'  {rec["name"]:<8} {rec["position"]:<9} {rec["odb_layer_name"]:<14} '
              f'{rec["thickness_mm"] * 1e3:6.1f} um')
    print(f'\nNext: python discretize.py {sample_dir}')


if __name__ == '__main__':
    main()
