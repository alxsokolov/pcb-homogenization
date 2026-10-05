#!/usr/bin/env python3
"""
Discretize Gerber copper layers into a 2D binary material map.

Reads a sample directory (produced by import_real_board.py), parses every copper layer's RS-274X Gerber file,
rasterizes it onto a uniform pixel grid, and merges all layers into a single
material indicator array.

Cell values:
    1.0  →  copper
    0.0  →  FR4 dielectric

Output written to <sample_dir>/mesh/:
    material_map.npy         float32 array of shape (ny, nx), combined OR of all Cu layers
    <layer>_map.npy          per-copper-layer boolean map (e.g. F_Cu_map.npy)
    discretization.json      grid metadata for quadtree.py / simulate_pyansys_*.py

Usage:
    python discretize.py data/samples/rv3032_example
    python discretize.py data/samples/rv3032_example --resolution 40
    python discretize.py data/samples/rv3032_example --resolution 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath


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


@dataclass
class _Aperture:
    kind: str              # 'C' circle | 'R' rectangle | 'O' oblong | 'RoundRect' macro
    params: list[float]    # C:[diam], R/O:[width, height]
                           # RoundRect:[radius, x1,y1,x2,y2,x3,y3,x4,y4, rotation_deg]
                           # (KiCad's standard RoundRect aperture macro: a rectangle with
                           # rounded corners, given as its 4 corner offsets + a rotation;
                           # the corner radius is not applied here — the shape is
                           # rasterized as the sharp-cornered quadrilateral, a negligible
                           # approximation for homogenization purposes)

    @property
    def width(self) -> float:
        return self.params[0]

    @property
    def height(self) -> float:
        return self.params[1] if len(self.params) > 1 else self.params[0]


@dataclass
class GerberGeometry:
    """Raw geometry from one RS-274X layer (all coordinates in mm)."""
    regions: list[np.ndarray] = field(default_factory=list)  # closed polygons (N,2)
    flashes: list[dict]       = field(default_factory=list)  # {x, y, aperture}
    strokes: list[dict]       = field(default_factory=list)  # {p0, p1, aperture}
    dropped: dict[str, int]   = field(default_factory=dict)  # {macro_name: count} skipped shapes


def parse_gerber(path: Path) -> GerberGeometry:
    """Parse a RS-274X Gerber file and return geometry with coordinates in mm."""
    tokens     = _tokenize(path.read_text())
    dec_digits = 6       # default for X46Y46 → scale = 1e6
    units_mm   = True
    apertures: dict[int, _Aperture] = {}
    unsupported: dict[int, str] = {}  # aperture id -> macro name, when _ADD_RE didn't match
    geo        = GerberGeometry()

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
                apertures[int(m.group(1))] = _Aperture(kind=m.group(2), params=params)
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
# Geometry → polygon list
# ---------------------------------------------------------------------------

def _circle_poly(cx: float, cy: float, r: float, n: int = 32) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def _rect_poly(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    hw, hh = w / 2, h / 2
    return np.array([[cx - hw, cy - hh], [cx + hw, cy - hh],
                     [cx + hw, cy + hh], [cx - hw, cy + hh]], dtype=float)


def _roundrect_poly(cx: float, cy: float, params: list[float]) -> np.ndarray:
    """
    KiCad's RoundRect aperture macro: [radius, x1,y1,x2,y2,x3,y3,x4,y4, rotation_deg].
    The 4 (xi,yi) are corner offsets already in board orientation; rotation_deg
    (observed as 0 in practice) is applied on top for robustness. Corner
    rounding is ignored — rasterized as the sharp-cornered quadrilateral.
    """
    _radius, *corners_and_rot = params
    rot_deg = corners_and_rot[8] if len(corners_and_rot) > 8 else 0.0
    pts = np.array(corners_and_rot[:8], dtype=float).reshape(4, 2)
    if rot_deg:
        th = np.radians(rot_deg)
        c, s = np.cos(th), np.sin(th)
        pts = pts @ np.array([[c, -s], [s, c]]).T
    return pts + np.array([cx, cy])


def _capsule_poly(p0: tuple, p1: tuple, width: float, n_cap: int = 10) -> np.ndarray:
    """Convert a stroke segment + aperture width to a stadium (capsule) polygon."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L  = np.hypot(dx, dy)
    r  = width / 2
    if L < 1e-9:
        return _circle_poly(x0, y0, r)
    tx, ty =  dx / L,  dy / L   # unit tangent
    nx, ny = -ty,  tx           # unit normal (left of tangent)

    pts = []
    # Right side (p0 → p1) and end-cap at p1
    pts.append([x0 - nx * r, y0 - ny * r])
    pts.append([x1 - nx * r, y1 - ny * r])
    for k in range(1, n_cap):
        th = np.pi * k / n_cap
        pts.append([x1 + r * ( tx * np.sin(th) - nx * np.cos(th)),
                    y1 + r * ( ty * np.sin(th) - ny * np.cos(th))])
    # Left side (p1 → p0) and end-cap at p0
    pts.append([x1 + nx * r, y1 + ny * r])
    pts.append([x0 + nx * r, y0 + ny * r])
    for k in range(1, n_cap):
        th = np.pi * k / n_cap
        pts.append([x0 + r * (-tx * np.sin(th) + nx * np.cos(th)),
                    y0 + r * (-ty * np.sin(th) + ny * np.cos(th))])
    return np.array(pts, dtype=float)


def geometry_to_polygons(geo: GerberGeometry) -> list[np.ndarray]:
    """Flatten all geometry types (regions, flashes, strokes) into polygon arrays."""
    polys: list[np.ndarray] = list(geo.regions)

    for fl in geo.flashes:
        x, y, ap = fl['x'], fl['y'], fl['aperture']
        if ap.kind == 'C':
            polys.append(_circle_poly(x, y, ap.width / 2))
        elif ap.kind == 'RoundRect':
            polys.append(_roundrect_poly(x, y, ap.params))
        else:
            polys.append(_rect_poly(x, y, ap.width, ap.height))

    for st in geo.strokes:
        ap = st['aperture']
        w  = ap.width if ap.kind == 'C' else max(ap.width, ap.height)
        polys.append(_capsule_poly(st['p0'], st['p1'], w))

    return polys


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def rasterize(
    polygons: list[np.ndarray],
    width_mm: float,
    height_mm: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Rasterize a list of closed polygons onto an (ny × nx) grid.
    Samples at cell centres of a uniform pixel lattice.
    Returns float32 array with values in {0.0, 1.0}.
    """
    xs = np.linspace(width_mm  / nx / 2, width_mm  - width_mm  / nx / 2, nx)
    ys = np.linspace(height_mm / ny / 2, height_mm - height_mm / ny / 2, ny)
    xx, yy  = np.meshgrid(xs, ys)
    pts_flat = np.column_stack([xx.ravel(), yy.ravel()])

    mask = np.zeros(ny * nx, dtype=bool)
    for poly in polygons:
        if len(poly) < 3:
            continue
        mask |= MplPath(poly).contains_points(pts_flat)

    return mask.reshape(ny, nx).astype(np.float32)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, **kwargs) -> None:
    path.write_text(json.dumps(kwargs, indent=2) + '\n')


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def discretize_sample(
    sample_dir: Path,
    resolution_px_per_mm: float = 10.0,
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

    mesh_dir = sample_dir / 'mesh'
    mesh_dir.mkdir(exist_ok=True)

    for layer in meta['layers']:
        if layer['type'] != 'copper':
            continue
        gbr_path = sample_dir / layer['file']
        if not gbr_path.exists():
            print(f'  [warn] {gbr_path} not found, skipping', file=sys.stderr)
            continue
        print(f'  Parsing   {gbr_path.name} …')
        geo   = parse_gerber(gbr_path)
        if geo.dropped:
            dropped_str = ', '.join(f'{n}x {name}' for name, n in geo.dropped.items())
            print(f'  [warn] {gbr_path.name}: skipped unsupported aperture macros: {dropped_str}',
                  file=sys.stderr)
        polys = geometry_to_polygons(geo)
        print(f'  Rasterizing {len(polys)} polygon(s) → {nx}×{ny} grid …')
        layer_mask = rasterize(polys, W, H, nx, ny)
        combined |= layer_mask.astype(bool)

        # Save individual layer map so the FE models can assign each copper
        # layer its own pattern rather than using the combined OR map.
        lname    = layer['name']           # e.g. "F_Cu", "B_Cu"
        lfile    = f'{lname}_map.npy'
        position = layer.get('position', 'unknown')
        np.save(mesh_dir / lfile, layer_mask)
        layer_map_info[lname] = {'file': lfile, 'position': position}
        print(f'  Saved  {lfile}  (position={position})')

    material = combined.astype(np.float32)

    np.save(mesh_dir / 'material_map.npy', material)
    print(f'  Saved  material_map.npy  shape={material.shape}  (combined OR of all Cu layers)')

    _save_json(
        mesh_dir / 'discretization.json',
        sample_id=meta['sample_id'],
        resolution_px_per_mm=resolution_px_per_mm,
        grid=dict(nx=nx, ny=ny,
                  dx_mm=round(W / nx, 8),
                  dy_mm=round(H / ny, 8)),
        board=dict(width_mm=W, height_mm=H),
        material_map_file='material_map.npy',
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
                        help='Sample directory produced by import_real_board.py')
    parser.add_argument('--resolution', type=float, default=10.0,
                        help='Grid resolution in pixels per mm')
    args = parser.parse_args()

    if not args.sample_dir.exists():
        sys.exit(f'Not found: {args.sample_dir}')

    print(f'Discretizing {args.sample_dir}  @ {args.resolution} px/mm …')
    discretize_sample(args.sample_dir, resolution_px_per_mm=args.resolution)


if __name__ == '__main__':
    main()
