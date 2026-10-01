#!/usr/bin/env python3
"""
Discretize PCB copper layers (Gerber or ODB++) into a 2D binary material map.

Reads a sample directory (produced by generate_gerber.py, import_real_board.py,
or import_odb_board.py), parses every copper layer — RS-274X Gerber or ODB++
feature files, per each layer's "format" in metadata.json — rasterizes it onto
a uniform pixel grid, and merges all layers into a single material indicator
array. The two formats share the LayerGeometry model and rasterizer in
raster_geometry.py, so downstream output is identical either way.

Cell values:
    1.0  →  copper
    0.0  →  FR4 dielectric

Output written to <sample_dir>/mesh/:
    material_map.npy         float32 array of shape (ny, nx)
    material_map.png         grayscale image: white=copper, black=FR4  (Y flipped)
    discretization.json      grid metadata for downstream FEniCSx scripts

Usage:
    python discretize.py data/samples/dummy_test
    python discretize.py data/samples/dummy_test --resolution 10
    python discretize.py data/samples/dummy_test --resolution 20 --no-png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import odb_parser
from raster_geometry import Aperture, LayerGeometry, rasterize_layer


# ---------------------------------------------------------------------------
# RS-274X Gerber parser
# ---------------------------------------------------------------------------

_FORMAT_RE  = re.compile(r'%FSLAX(\d)(\d)Y(\d)(\d)%')
_UNIT_RE    = re.compile(r'%MO(MM|IN)%')
_ADD_RE     = re.compile(r'%ADD(\d+)(RoundRect|[CRO]),([^%]+)%')
_ADD_ANY_RE = re.compile(r'%ADD(\d+)([A-Za-z][A-Za-z0-9]*),')  # for logging unsupported macro apertures
_COORD_RE   = re.compile(r'(?:X(-?\d+))?(?:Y(-?\d+))?D(\d+)$')


def _tokenize(text: str) -> list[str]:
    """Split RS-274X Gerber source into individual command tokens (no trailing *)."""
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in ' \t\r\n':
            i += 1
        elif c == '%':
            j = text.find('%', i + 1)
            if j == -1:
                break
            for part in text[i + 1:j].split('*'):
                p = part.strip()
                if p:
                    tokens.append('%' + p + '%')
            i = j + 1
        else:
            j = text.find('*', i)
            if j == -1:
                break
            cmd = text[i:j].strip()
            if cmd:
                tokens.append(cmd)
            i = j + 1
    return tokens


def parse_gerber(path: Path) -> LayerGeometry:
    """Parse a RS-274X Gerber file and return geometry with coordinates in mm."""
    tokens     = _tokenize(path.read_text())
    dec_digits = 6       # default for X46Y46 → scale = 1e6
    units_mm   = True
    apertures: dict[int, Aperture] = {}
    unsupported: dict[int, str] = {}  # aperture id -> macro name, when _ADD_RE didn't match
    geo        = LayerGeometry()

    in_region    = False
    region_pts: list[tuple[float, float]] = []
    cur_x = cur_y = 0.0
    cur_ap_id: int | None = None

    def to_mm(raw: str) -> float:
        val = int(raw) / (10 ** dec_digits)
        return val if units_mm else val * 25.4

    for tok in tokens:
        # ── Extended parameters ───────────────────────────────────────────
        if tok.startswith('%'):
            m = _FORMAT_RE.match(tok)
            if m:
                dec_digits = int(m.group(2))
                continue

            m = _UNIT_RE.match(tok)
            if m:
                units_mm = (m.group(1) == 'MM')
                continue

            m = _ADD_RE.match(tok)
            if m:
                params = [float(p) for p in m.group(3).split('X') if p]
                apertures[int(m.group(1))] = Aperture(kind=m.group(2), params=params)
            else:
                m_any = _ADD_ANY_RE.match(tok)
                if m_any:
                    unsupported[int(m_any.group(1))] = m_any.group(2)
            continue  # skip TF/TA/TO/LP etc.

        # ── Strip leading G-code prefix (handles e.g. "G01X100Y200D01") ──
        g_match = re.match(r'^G(\d+)(.*)', tok)
        if g_match:
            g    = int(g_match.group(1))
            rest = g_match.group(2)
            if g == 4:   # comment
                continue
            if g == 36:
                in_region = True
                region_pts = []
            elif g == 37:
                if in_region and len(region_pts) >= 3:
                    geo.regions.append(np.array(region_pts, dtype=float))
                in_region = False
                region_pts = []
            # G01/G02/G03 set interpolation mode; no extra state needed for now
            if not rest:
                continue
            tok = rest   # fall through to process any trailing coordinate

        # ── Bare D-code (aperture select, e.g. "D10") ────────────────────
        if re.fullmatch(r'D\d+', tok):
            d = int(tok[1:])
            if d >= 10:
                cur_ap_id = d
            continue

        # ── End of file ───────────────────────────────────────────────────
        if tok == 'M02':
            break

        # ── Coordinate + D operation ──────────────────────────────────────
        m = _COORD_RE.match(tok)
        if not m:
            continue

        x_raw, y_raw, d_code = m.group(1), m.group(2), int(m.group(3))
        x = to_mm(x_raw) if x_raw is not None else cur_x
        y = to_mm(y_raw) if y_raw is not None else cur_y

        if d_code == 2:                            # move
            if in_region:
                region_pts = [(x, y)]
            cur_x, cur_y = x, y

        elif d_code == 1:                          # linear draw
            if in_region:
                region_pts.append((x, y))
            else:
                ap = apertures.get(cur_ap_id)  # type: ignore[arg-type]
                if ap:
                    geo.strokes.append({'p0': (cur_x, cur_y), 'p1': (x, y), 'aperture': ap})
                elif cur_ap_id in unsupported:
                    geo.dropped[unsupported[cur_ap_id]] = geo.dropped.get(unsupported[cur_ap_id], 0) + 1
            cur_x, cur_y = x, y

        elif d_code == 3:                          # flash
            ap = apertures.get(cur_ap_id)  # type: ignore[arg-type]
            if ap:
                geo.flashes.append({'x': x, 'y': y, 'aperture': ap})
            elif cur_ap_id in unsupported:
                geo.dropped[unsupported[cur_ap_id]] = geo.dropped.get(unsupported[cur_ap_id], 0) + 1
            cur_x, cur_y = x, y

    return geo


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_png(material: np.ndarray, path: Path) -> None:
    """Grayscale PNG: white=copper, black=FR4. Flipped so Y=0 is bottom."""
    try:
        from PIL import Image
        img = Image.fromarray((material[::-1] * 255).astype(np.uint8), mode='L')
        img.save(path)
    except ImportError:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.imshow(material[::-1], cmap='gray', vmin=0, vmax=1, origin='upper')
        ax.axis('off')
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def _save_json(path: Path, **kwargs) -> None:
    path.write_text(json.dumps(kwargs, indent=2) + '\n')


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def discretize_sample(
    sample_dir: Path,
    resolution_px_per_mm: float = 10.0,
    save_png: bool = True,
) -> np.ndarray:
    """Parse + rasterize all copper layers in a sample; write mesh/ outputs."""
    meta_path = sample_dir / 'metadata.json'
    if not meta_path.exists():
        sys.exit(f"Not found: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    W = meta['board']['width_mm']
    H = meta['board']['height_mm']
    nx = max(1, round(W * resolution_px_per_mm))
    ny = max(1, round(H * resolution_px_per_mm))

    combined = np.zeros((ny, nx), dtype=bool)
    layer_map_info = {}   # {layer_name: {file, position}} — written to discretization.json
    origin_mm = tuple(meta.get('odb', {}).get('origin_offset_mm', (0.0, 0.0)))

    mesh_dir = sample_dir / 'mesh'
    mesh_dir.mkdir(exist_ok=True)

    for layer in meta['layers']:
        if layer['type'] != 'copper':
            continue
        layer_path = sample_dir / layer['file']
        if not layer_path.exists():
            print(f'  [warn] {layer_path} not found, skipping', file=sys.stderr)
            continue
        fmt = layer.get('format', 'gerber')
        print(f'  Parsing   {layer_path.name}  ({fmt}) …')
        if fmt == 'odb':
            geo = odb_parser.parse_odb_features(layer_path, origin=origin_mm)
        else:
            geo = parse_gerber(layer_path)
        if geo.dropped:
            dropped_str = ', '.join(f'{n}x {name}' for name, n in geo.dropped.items())
            print(f'  [warn] {layer_path.name}: skipped unsupported shapes: {dropped_str}',
                  file=sys.stderr)
        n_shapes = len(geo.regions) + len(geo.flashes) + len(geo.strokes) + len(geo.surfaces)
        print(f'  Rasterizing {n_shapes} shape(s) → {nx}×{ny} grid …')
        layer_mask = rasterize_layer(geo, W, H, nx, ny)
        combined |= layer_mask.astype(bool)

        # Save individual layer map so simulate_tensile.py can assign each
        # copper layer its own pattern rather than using the combined OR map.
        lname    = layer['name']           # e.g. "F_Cu", "B_Cu"
        lfile    = f'{lname}_map.npy'
        position = layer.get('position', 'unknown')
        np.save(mesh_dir / lfile, layer_mask)
        layer_map_info[lname] = {'file': lfile, 'position': position}
        print(f'  Saved  {lfile}  (position={position})')

    material = combined.astype(np.float32)

    np.save(mesh_dir / 'material_map.npy', material)
    print(f'  Saved  material_map.npy  shape={material.shape}  (combined OR of all Cu layers)')

    if save_png:
        _save_png(material, mesh_dir / 'material_map.png')
        print(f'  Saved  material_map.png')

    _save_json(
        mesh_dir / 'discretization.json',
        sample_id=meta['sample_id'],
        resolution_px_per_mm=resolution_px_per_mm,
        grid=dict(nx=nx, ny=ny,
                  dx_mm=round(W / nx, 8),
                  dy_mm=round(H / ny, 8)),
        board=dict(width_mm=W, height_mm=H),
        material_map_file='material_map.npy',
        material_map_png='material_map.png',
        layer_maps=layer_map_info,
        copper_fraction=round(float(material.mean()), 6),
        timestamp=datetime.now(tz=timezone.utc).isoformat(timespec='seconds'),
    )
    print(f'  Saved  discretization.json')
    print(f'  Copper fraction: {material.mean():.4f}')
    return material


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Rasterize Gerber copper layers to a binary material map.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('sample_dir', type=Path,
                        help='Sample directory produced by generate_gerber.py')
    parser.add_argument('--resolution', type=float, default=10.0,
                        help='Grid resolution in pixels per mm')
    parser.add_argument('--no-png', action='store_true', help='Skip PNG output')
    args = parser.parse_args()

    if not args.sample_dir.exists():
        sys.exit(f'Not found: {args.sample_dir}')

    print(f'Discretizing {args.sample_dir}  @ {args.resolution} px/mm …')
    discretize_sample(args.sample_dir,
                      resolution_px_per_mm=args.resolution,
                      save_png=not args.no_png)


if __name__ == '__main__':
    main()
