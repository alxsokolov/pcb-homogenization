#!/usr/bin/env python3
"""
Import a real (externally authored) 2-layer PCB into the pcb_homogenization
sample format.

Copies the F_Cu/B_Cu Gerbers into <sample_dir>/gerber/, shifting every
absolute coordinate so the board's bounding box starts at (0,0) -- the
convention discretize.py assumes, since the source board's own coordinate
system is normally an arbitrary offset from the CAD tool's global origin --
and writes a matching metadata.json.

The board bounding box must be supplied manually (e.g. read off the
Edge_Cuts.gbr extents by inspection) -- no component-placement parsing is
attempted here. This only prepares the copper geometry for discretize.py.

Usage:
    python import_real_board.py \\
        --f-cu path/to/Board-F_Cu.gbr --b-cu path/to/Board-B_Cu.gbr \\
        --bbox 122 -105 142 -90 \\
        --sample-id rv3032_real
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_FORMAT_RE     = re.compile(r'%FSLAX(\d)(\d)Y(\d)(\d)%')
_COORD_LINE_RE = re.compile(r'D0[123]\*?\s*$')


def _shift_gerber(text: str, dx_mm: float, dy_mm: float) -> str:
    """
    Shift absolute X/Y coordinates in move/draw/flash command lines (D01/02/03)
    by (dx_mm, dy_mm). Leaves %...% extended-parameter lines (aperture defs,
    format spec, attributes) and arc I/J relative offsets untouched.
    """
    dec_digits = 6
    m = _FORMAT_RE.search(text)
    if m:
        dec_digits = int(m.group(2))
    scale  = 10 ** dec_digits
    dx_raw = round(dx_mm * scale)
    dy_raw = round(dy_mm * scale)

    def shift_line(line: str) -> str:
        if line.lstrip().startswith('%') or not _COORD_LINE_RE.search(line):
            return line
        line = re.sub(r'X(-?\d+)', lambda mo: f'X{int(mo.group(1)) + dx_raw}', line)
        line = re.sub(r'Y(-?\d+)', lambda mo: f'Y{int(mo.group(1)) + dy_raw}', line)
        return line

    return '\n'.join(shift_line(l) for l in text.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--f-cu', type=Path, required=True, help='Source F.Cu (top copper) Gerber')
    parser.add_argument('--b-cu', type=Path, required=True, help='Source B.Cu (bottom copper) Gerber')
    parser.add_argument('--bbox', type=float, nargs=4, required=True,
                        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                        help='Board bounding box in the source Gerber coordinate system (mm)')
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--output-root', type=Path, default=Path('data/samples'))
    parser.add_argument('--cu-thickness-mm', type=float, default=0.035)
    parser.add_argument('--fr4-thickness-mm', type=float, default=1.53)
    args = parser.parse_args()

    xmin, ymin, xmax, ymax = args.bbox
    width_mm, height_mm = xmax - xmin, ymax - ymin
    if width_mm <= 0 or height_mm <= 0:
        sys.exit(f"Bad --bbox: width={width_mm}, height={height_mm} (expected xmin ymin xmax ymax)")

    sample_dir = args.output_root / args.sample_id
    gerber_dir = sample_dir / 'gerber'
    gerber_dir.mkdir(parents=True, exist_ok=True)

    for src, dst_name in [(args.f_cu, 'F_Cu.gbr'), (args.b_cu, 'B_Cu.gbr')]:
        shifted = _shift_gerber(src.read_text(), dx_mm=-xmin, dy_mm=-ymin)
        (gerber_dir / dst_name).write_text(shifted)
        print(f'  Wrote {gerber_dir / dst_name}  (shifted by {-xmin:+.4f}, {-ymin:+.4f} mm)')

    total_t = round(2 * args.cu_thickness_mm + args.fr4_thickness_mm, 6)
    meta = {
        "sample_id": args.sample_id,
        "board": {"width_mm": round(width_mm, 6), "height_mm": round(height_mm, 6)},
        "layers": [
            {"name": "F_Cu", "file": "gerber/F_Cu.gbr", "type": "copper", "position": "top",
             "thickness_mm": args.cu_thickness_mm},
            {"name": "B_Cu", "file": "gerber/B_Cu.gbr", "type": "copper", "position": "bottom",
             "thickness_mm": args.cu_thickness_mm},
        ],
        "stackup": {
            "n_copper_layers": 2,
            "copper_thickness_mm": args.cu_thickness_mm,
            "dielectric_material": "FR4",
            "dielectric_thickness_mm": args.fr4_thickness_mm,
            "total_thickness_mm": total_t,
        },
        "generation": {
            "script": "import_real_board.py",
            "type": "real_board_import",
            "source_bbox_mm": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        },
    }
    (sample_dir / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'  Wrote {sample_dir / "metadata.json"}')
    print(f'Board size: {width_mm:.3f} x {height_mm:.3f} mm')


if __name__ == '__main__':
    main()
