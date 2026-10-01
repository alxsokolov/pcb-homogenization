#!/usr/bin/env python3
"""
Import an ODB++ job into the pcb_homogenization sample format.

Reads an extracted ODB++ job directory (matrix/, misc/, steps/<step>/...),
picks the outermost pair of SIGNAL layers as the top/bottom copper layers
(this pipeline is 2-layer-only, same as generate_gerber.py / import_real_board.py),
sums the DIELECTRIC layers between them for the core thickness, and copies the
relevant matrix/layer files into <sample_dir>/odb/ alongside a metadata.json
matching the schema generate_gerber.py produces -- so discretize.py and every
downstream script treat an ODB++ sample exactly like a Gerber one, just with
each copper layer's "file" pointing at an ODB++ features file and "format": "odb".

Only the layers discretize.py and homogenization.py actually consume are
copied into <sample_dir>/odb/ (the two signal layers, any core/prepreg
dielectric layers between them, and the board outline) -- solder mask/paste/
silkscreen/drill/etc. are left out to keep the sample small; re-run against
the full job directory if those are needed later. The board outline is read
from steps/<step>/profile when present (the standard ODB++ location, and the
only option for jobs where the outline isn't a separately named layer -- e.g.
it's only implied by a DRILL/ROUT layer), falling back to a matrix layer
named EDGE.CUTS/PROFILE/OUTLINE/BOARD OUTLINE otherwise.

Usage:
    python import_odb_board.py --odb-dir path/to/extracted/job --sample-id my_board
    python import_odb_board.py --odb-dir path/to/job --sample-id my_board \\
        --source "https://github.com/FixturFab/ODB2kicad samples/odb-kitchen-sink"
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

_OUTLINE_NAME_RE = re.compile(r'edge.?cuts|profile|outline|board.?outline', re.I)


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


def _copper_thickness_mm(attrs: dict) -> float:
    """Prefer the explicit stackup thickness; fall back to copper weight (1 oz = 0.035 mm)."""
    if 'layer_dielectric' in attrs:
        return float(attrs['layer_dielectric'])
    if 'copper_weight' in attrs:
        return round(float(attrs['copper_weight']) * 0.035, 6)
    return 0.035


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--odb-dir', type=Path, required=True,
                        help='Extracted ODB++ job root (contains matrix/, steps/, misc/)')
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--output-root', type=Path, default=Path('data/samples'))
    parser.add_argument('--step', help='Step name to import (default: first found under steps/)')
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

    signal = [l for l in odb_layers if l.type == 'SIGNAL']
    if len(signal) < 2:
        sys.exit(f'Expected >= 2 SIGNAL layers in matrix, found {len(signal)}')
    top_layer, bottom_layer = signal[0], signal[-1]
    if len(signal) > 2:
        print(f'  [warn] {len(signal)} SIGNAL layers found; this pipeline is 2-layer-only -- '
              f'using {top_layer.name!r} (top) and {bottom_layer.name!r} (bottom), '
              f'skipping {len(signal) - 2} inner layer(s)', file=sys.stderr)
    dielectric = [l for l in odb_layers
                 if l.type == 'DIELECTRIC' and top_layer.row < l.row < bottom_layer.row]

    # Board bounding box, read straight from the outline geometry (no manual
    # --bbox flag needed, unlike import_real_board.py's Gerber path). Prefer
    # the dedicated `profile` file (the standard ODB++ location for the
    # physical board outline) when present -- this also covers jobs where
    # the outline is only implied by a DRILL/ROUT layer rather than a
    # separately named outline/document layer. Fall back to a name-matched
    # layer otherwise.
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

    cu_t_top = _copper_thickness_mm(odb_parser.parse_odb_attrlist(layers_dir / top_layer.name / 'attrlist'))
    cu_t_bot = _copper_thickness_mm(odb_parser.parse_odb_attrlist(layers_dir / bottom_layer.name / 'attrlist'))

    diel_t = 0.0
    diel_material = None
    diel_detail = []
    for d in dielectric:
        attrs = odb_parser.parse_odb_attrlist(layers_dir / d.name / 'attrlist')
        t = float(attrs.get('layer_dielectric', 0.0))
        diel_t += t
        diel_material = diel_material or attrs.get('material')
        diel_detail.append({'name': d.name, 'thickness_mm': t, **attrs})

    # --- copy the trimmed job subtree ---
    sample_dir = args.output_root / args.sample_id
    odb_out = sample_dir / 'odb'
    for d in [odb_out, sample_dir / 'mesh', sample_dir / 'results']:
        d.mkdir(parents=True, exist_ok=True)

    (odb_out / 'matrix').mkdir(exist_ok=True)
    shutil.copy2(matrix_path, odb_out / 'matrix' / 'matrix')

    misc_info = odb_dir / 'misc' / 'info'
    if misc_info.exists():
        (odb_out / 'misc').mkdir(exist_ok=True)
        shutil.copy2(misc_info, odb_out / 'misc' / 'info')

    step_out = odb_out / 'steps' / step_name
    step_out.mkdir(parents=True, exist_ok=True)
    stephdr = step_dir / 'stephdr'
    if stephdr.exists():
        shutil.copy2(stephdr, step_out / 'stephdr')

    outline_layers = [outline_layer] if outline_layer is not None else []
    for l in [top_layer, bottom_layer] + outline_layers + dielectric:
        src, dst = layers_dir / l.name, step_out / 'layers' / l.name
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ('features', 'attrlist'):
            if (src / fname).exists():
                shutil.copy2(src / fname, dst / fname)
        print(f'  Copied  odb/steps/{step_name}/layers/{l.name}/')

    if profile_path.exists():
        shutil.copy2(profile_path, step_out / 'profile')
        print(f'  Copied  odb/steps/{step_name}/profile')

    def _rel(layer_name: str) -> str:
        return f'odb/steps/{step_name}/layers/{layer_name}/features'

    meta = {
        "sample_id": args.sample_id,
        "board": {"width_mm": round(width_mm, 6), "height_mm": round(height_mm, 6)},
        "layers": [
            {"name": "F_Cu", "file": _rel(top_layer.name), "type": "copper", "position": "top",
             "thickness_mm": cu_t_top, "format": "odb", "odb_layer_name": top_layer.name},
            {"name": "B_Cu", "file": _rel(bottom_layer.name), "type": "copper", "position": "bottom",
             "thickness_mm": cu_t_bot, "format": "odb", "odb_layer_name": bottom_layer.name},
        ],
        "stackup": {
            "n_copper_layers": 2,
            "copper_thickness_mm": cu_t_top,
            "dielectric_material": (diel_material or "FR4").upper(),
            "dielectric_thickness_mm": round(diel_t, 6),
            "total_thickness_mm": round(cu_t_top + cu_t_bot + diel_t, 6),
            "dielectric_layers": diel_detail,
        },
        "odb": {
            "job_step": step_name,
            "outline_source": outline_source,
            "origin_offset_mm": [round(xmin, 6), round(ymin, 6)],
        },
        "generation": {
            "script": "import_odb_board.py",
            "type": "odb_import",
            "source": args.source,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        },
    }
    (sample_dir / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Wrote {sample_dir / "metadata.json"}')
    print(f'Board size: {width_mm:.3f} x {height_mm:.3f} mm')
    print(f'Copper: top={top_layer.name} ({cu_t_top} mm)  bottom={bottom_layer.name} ({cu_t_bot} mm)')
    print(f'Dielectric: {diel_t:.4f} mm total ({len(dielectric)} layer(s))')


if __name__ == '__main__':
    main()
