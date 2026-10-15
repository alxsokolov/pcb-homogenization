#!/usr/bin/env python3
"""
ODB++ job parser: matrix (layer stackup), layer feature files, and attrlist
stackup attributes -- the ODB++ analogue of pcbh.gerber's RS-274X parser.
Produces the same pcbh.raster.LayerGeometry that Gerber layers produce, so
discretize.py rasterizes both formats identically.

Covers the subset of the ODB++ Design Format actually needed here: the
`matrix/matrix` structured-text layer table, and per-layer `features` /
`attrlist` files under `steps/<step>/layers/<layer>/`. Feature records
supported: pad (P), line (L), arc (A, flattened to a straight stroke -- the
same curvature approximation pcbh.gerber already makes for Gerber G02/G03),
and surface (S ... SE, with OB/OS/OC outer-boundary and CB/CS/CC hole
segments; OC/CC arc segments are likewise flattened to straight edges).

Reference: ODB++ Design Format Specification (odbplusplus.com/design) and
the field layouts documented by the open-source ODBPy parser
(github.com/ulikoehler/ODBPy, Apache-2.0), used here to confirm record
column order -- this module is an independent implementation against the
spec, not a port of ODBPy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pcbh.raster import Aperture, LayerGeometry, Surface

# ---------------------------------------------------------------------------
# matrix/matrix — structured-text layer table
# ---------------------------------------------------------------------------


@dataclass
class OdbLayer:
    name: str                 # lowercased, e.g. "f.cu"
    row: int
    context: str               # "BOARD" | "MISC"
    type: str                  # "SIGNAL" | "DIELECTRIC" | "DOCUMENT" | ...
    dielectric_type: str | None = None   # "CORE" | "PREPREG", DIELECTRIC layers only
    start_name: str | None = None        # DRILL layers: span start
    end_name: str | None = None          # DRILL layers: span end


def parse_odb_matrix(path: Path) -> list[OdbLayer]:
    """Parse `matrix/matrix` and return its LAYER {...} blocks in ROW order."""
    text = path.read_text()
    layers: list[OdbLayer] = []
    for block in re.finditer(r'LAYER\s*\{([^}]*)\}', text):
        attrs: dict[str, str] = {}
        for line in block.group(1).splitlines():
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, _, v = line.partition('=')
            attrs[k.strip()] = v.strip()
        if 'NAME' not in attrs:
            continue
        layers.append(OdbLayer(
            name=attrs['NAME'].lower(),
            row=int(attrs.get('ROW', 0)),
            context=attrs.get('CONTEXT', ''),
            type=attrs.get('TYPE', ''),
            dielectric_type=attrs.get('DIELECTRIC_TYPE') or None,
            start_name=(attrs.get('START_NAME') or None) and attrs['START_NAME'].lower(),
            end_name=(attrs.get('END_NAME') or None) and attrs['END_NAME'].lower(),
        ))
    layers.sort(key=lambda l: l.row)
    return layers


# ---------------------------------------------------------------------------
# attrlist — flat `.key=value` stackup / material attributes
# ---------------------------------------------------------------------------

# Whitespace- and CRLF-tolerant, and aware of a leading UNITS=MM|INCH
# declaration that scales length-valued attributes to mm. A pre-restructure
# version of this parser split on '=' without stripping whitespace, so
# ".layer_dielectric = 0.065" (spaces around the '=', as several exporters
# write it) landed under the key "layer_dielectric " -- with a trailing
# space -- and every .get() for it missed; only jobs writing
# ".layer_dielectric=1.51" with no spaces parsed correctly. Fixed here: both
# forms parse the same way, and it changes the resolved stackup for any job
# that hit the bug (see CHANGELOG).
_ATTR_RE = re.compile(r'^\.([A-Za-z_][\w]*)\s*=\s*(.*)$')
_UNITS_RE = re.compile(r'^UNITS\s*=\s*(MM|INCH)\s*$', re.I)

_MM_PER_INCH = 25.4

# Attributes whose value is a length in the file's own UNITS.
_LENGTH_ATTRS = frozenset({"layer_dielectric", "board_thickness"})


def parse_odb_attrlist(path: Path) -> dict[str, float | str]:
    """
    Parse an ODB++ `.key = value` attrlist, tolerating whitespace around the
    '=' and CRLF line endings. Numeric values come back as float.

    A leading `UNITS=MM|INCH` line applies to every length-valued attribute
    in the file; values are converted to mm so callers never deal with
    inches. Only the attributes that are lengths are converted
    (_LENGTH_ATTRS) -- .copper_weight is a weight (oz) or an explicit
    micrometre figure, never a file-unit length, and .dielectric_constant
    and .loss_tangent are dimensionless.
    """
    if not path.exists():
        return {}
    out: dict[str, float | str] = {}
    scale = 1.0
    for raw in path.read_text(errors="replace").replace("\r", "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m_units = _UNITS_RE.match(line)
        if m_units:
            scale = _MM_PER_INCH if m_units.group(1).upper() == "INCH" else 1.0
            continue
        m = _ATTR_RE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        try:
            num = float(val)
        except ValueError:
            out[key] = val
            continue
        out[key] = num * scale if key in _LENGTH_ATTRS else num
    return out


# ---------------------------------------------------------------------------
# Symbol names → Aperture
# ---------------------------------------------------------------------------

# ODB++ symbol sizes are always given in thousandths of the file's UNITS
# (mm here), regardless of the coordinate scale used elsewhere in the file.
_SYM_SCALE = 1000.0

_SYM_ROUND    = re.compile(r'^r(\d+\.?\d*)$')
_SYM_SQUARE   = re.compile(r'^sq(\d+\.?\d*)$')
_SYM_RECT     = re.compile(r'^rect(\d+\.?\d*)x(\d+\.?\d*)(?:xr\d+\.?\d*)?$')  # trailing xr<radius> (rounded corner) ignored, same approximation as Gerber RoundRect
_SYM_OVAL     = re.compile(r'^oval?(?:al)?(\d+\.?\d*)x(\d+\.?\d*)$')
_SYM_DONUT    = re.compile(r'^donut_r(\d+\.?\d*)x(\d+\.?\d*)$')  # outer x inner diameter; rasterized as a solid outer-diameter disc (inner hole ignored)


def parse_odb_symbol(name: str) -> Aperture | None:
    """Map an ODB++ symbol name (e.g. 'r600.0', 'rect800.0x950.0xr200.0') to an Aperture."""
    name = name.strip()
    m = _SYM_ROUND.match(name)
    if m:
        return Aperture(kind='C', params=[float(m.group(1)) / _SYM_SCALE])
    m = _SYM_SQUARE.match(name)
    if m:
        d = float(m.group(1)) / _SYM_SCALE
        return Aperture(kind='R', params=[d, d])
    m = _SYM_RECT.match(name)
    if m:
        return Aperture(kind='R', params=[float(m.group(1)) / _SYM_SCALE, float(m.group(2)) / _SYM_SCALE])
    m = _SYM_OVAL.match(name)
    if m:
        return Aperture(kind='O', params=[float(m.group(1)) / _SYM_SCALE, float(m.group(2)) / _SYM_SCALE])
    m = _SYM_DONUT.match(name)
    if m:
        return Aperture(kind='C', params=[float(m.group(1)) / _SYM_SCALE])
    return None


# ---------------------------------------------------------------------------
# features — layer geometry (pads, lines, arcs, surfaces)
# ---------------------------------------------------------------------------

_UNITS_FEATURES_RE = re.compile(r'^(?:UNITS\s*=\s*|U\s+)(INCH|MM)\b', re.I)
_DEF_RE  = re.compile(r'^([$@&])(\d+)\s+(.*)$')
_PAD_RE  = re.compile(
    r'^P\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-1\s+\d+\s+[\d.]+|\d+)\s+([PN])\s+(\d+)\s+'
    r'([0-7]|[89]\s+-?[\d.]+)(?:\s*;.*)?$'
)
_LINE_RE = re.compile(
    r'^L\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+([PN])\s+(\d+)(?:\s*;.*)?$'
)
_ARC_RE  = re.compile(
    r'^A\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+'
    r'(\d+)\s+([PN])\s+(\d+)(?:\s+([YN]))?(?:\s*;.*)?$'
)

# Old-style orientation codes 0-7 map to (mirror, angle); 8/9 <angle> is the
# new-style form (8 = not mirrored, 9 = mirrored about X), used directly.
_ORIENT_OLD = {0: 0, 1: 90, 2: 180, 3: 270, 4: 0, 5: 90, 6: 180, 7: 270}


def _resolve_symbol(idx: str, symbols: dict[int, Aperture | None]) -> Aperture | None:
    """apt_def is either a bare symbol index or the long '-1 sym resize' form."""
    idx = idx.strip()
    if idx.startswith('-1'):
        _, sym, _resize = idx.split()
        return symbols.get(int(sym))
    return symbols.get(int(idx))


def parse_odb_features(path: Path, origin: tuple[float, float] = (0.0, 0.0)) -> LayerGeometry:
    """
    Parse an ODB++ `features` (or `profile`) file and return a LayerGeometry
    with coordinates in mm, shifted by -origin (so the sample's board
    bounding box can be normalized to start at (0,0), matching
    import_board.py's convention for Gerber-sourced samples).

    Coordinates are scaled to mm according to the file's own `UNITS=INCH|MM`
    (features files) or `U INCH|MM` (profile files) declaration; MM is
    assumed if no such declaration is present. Note this does NOT apply to
    symbol/aperture sizes (see `_SYM_SCALE`), which ODB++ always expresses in
    thousandths of mm regardless of the coordinate UNITS declaration.
    """
    ox, oy = origin
    geo = LayerGeometry()
    symbols: dict[int, Aperture | None] = {}
    scale = 1.0  # mm per file coordinate unit, set by a UNITS/U declaration

    lines = [l.rstrip('\n') for l in path.read_text().splitlines()]

    surf: Surface | None = None
    surf_pts: list[tuple[float, float]] = []
    surf_marker = 'I'   # the active OB boundary's I(sland)/H(ole) tag, set when it opens

    def flush_boundary():
        nonlocal surf_pts
        pts = [(x - ox, y - oy) for x, y in surf_pts]
        surf_pts = []
        return np.array(pts, dtype=float) if len(pts) >= 3 else None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#') or line.startswith('F '):
            continue

        m = _UNITS_FEATURES_RE.match(line)
        if m:
            scale = 25.4 if m.group(1).upper() == 'INCH' else 1.0
            continue

        m = _DEF_RE.match(line)
        if m:
            marker, idx, rest = m.groups()
            if marker == '$':
                symbols[int(idx)] = parse_odb_symbol(rest)
            continue  # @ (attribute names) / & (attribute strings) not needed for rasterization

        # ── Surfaces: S <pol> <dcode>  (OB/OS/OC ... OE)+  SE
        # Each surface record can carry several OB...OE boundaries in a row,
        # not just one -- confirmed against real-world files (e.g. a board
        # profile: one island outline plus several hole cutouts, all as
        # separate OB...OE blocks with no intervening S/SE). Each OB line's
        # 3rd token is 'I' (island, additive) or 'H' (hole, subtractive);
        # legacy CB...CE blocks are always holes. Unconditionally overwriting
        # surf.outer on every OE (as earlier versions of this parser did)
        # silently keeps only the last boundary in the record and drops the
        # rest -- confirmed to corrupt real board outlines this way.
        if line.startswith('S ') or line == 'S':
            surf = Surface(outer=np.empty((0, 2)), holes=[])
            continue
        if line == 'SE':
            if surf is not None and surf.outer.size:
                geo.surfaces.append(surf)
            surf = None
            continue
        if line.startswith('OB '):
            parts = line.split()
            surf_pts = [(float(parts[1]) * scale, float(parts[2]) * scale)]
            surf_marker = parts[3] if len(parts) > 3 else 'I'
            continue
        if line.startswith('CB '):
            parts = line.split()
            surf_pts = [(float(parts[1]) * scale, float(parts[2]) * scale)]
            surf_marker = 'H'
            continue
        if line.startswith('OS ') or line.startswith('CS '):
            parts = line.split()
            surf_pts.append((float(parts[1]) * scale, float(parts[2]) * scale))
            continue
        if line.startswith('OC ') or line.startswith('CC '):
            # Arc boundary segment: x y cx cy dir -- flattened to a straight edge to the endpoint.
            parts = line.split()
            surf_pts.append((float(parts[1]) * scale, float(parts[2]) * scale))
            continue
        if line in ('OE', 'CE'):
            poly = flush_boundary()
            if surf is not None and poly is not None:
                if surf_marker == 'H':
                    surf.holes.append(poly)
                elif surf.outer.size:
                    # A second island in the same surface record (rare, but
                    # seen in practice) -- flush what we have as its own
                    # surface and start a fresh one for this island, rather
                    # than overwrite or merge it with unrelated holes.
                    geo.surfaces.append(surf)
                    surf = Surface(outer=poly, holes=[])
                else:
                    surf.outer = poly
            continue

        # ── Pad (flash) ──────────────────────────────────────────────────
        m = _PAD_RE.match(line)
        if m:
            x, y, apt_def, _pol, _dcode, orient = m.groups()
            ap = _resolve_symbol(apt_def, symbols)
            if ap is None:
                geo.dropped['unresolved_symbol'] = geo.dropped.get('unresolved_symbol', 0) + 1
                continue
            if ' ' in orient:                       # new-style "8 <angle>" / "9 <angle>"
                code, angle = orient.split()
                angle_deg, mirror = float(angle), (code == '9')
            else:                                    # old-style single digit 0-7
                code = int(orient)
                angle_deg, mirror = _ORIENT_OLD[code], code >= 4
            geo.flashes.append({
                'x': float(x) * scale - ox, 'y': float(y) * scale - oy, 'aperture': ap,
                'rotation': angle_deg, 'mirror': mirror,
            })
            continue

        # ── Line (stroke) ────────────────────────────────────────────────
        m = _LINE_RE.match(line)
        if m:
            xs, ys, xe, ye, sym, _pol, _dcode = m.groups()
            ap = symbols.get(int(sym))
            if ap is None:
                geo.dropped['unresolved_symbol'] = geo.dropped.get('unresolved_symbol', 0) + 1
                continue
            geo.strokes.append({
                'p0': (float(xs) * scale - ox, float(ys) * scale - oy),
                'p1': (float(xe) * scale - ox, float(ye) * scale - oy),
                'aperture': ap,
            })
            continue

        # ── Arc (flattened to a straight stroke, same curvature approximation as Gerber G02/G03) ──
        m = _ARC_RE.match(line)
        if m:
            xs, ys, xe, ye, _cx, _cy, sym, _pol, _dcode, _dirn = m.groups()
            ap = symbols.get(int(sym))
            if ap is None:
                geo.dropped['unresolved_symbol'] = geo.dropped.get('unresolved_symbol', 0) + 1
                continue
            geo.strokes.append({
                'p0': (float(xs) * scale - ox, float(ys) * scale - oy),
                'p1': (float(xe) * scale - ox, float(ye) * scale - oy),
                'aperture': ap,
            })
            continue

        geo.dropped['unparsed_record'] = geo.dropped.get('unparsed_record', 0) + 1

    return geo


def layer_bbox_mm(geo: LayerGeometry) -> tuple[float, float, float, float] | None:
    """Bounding box (xmin, ymin, xmax, ymax) of all geometry in a LayerGeometry, or None if empty."""
    xs: list[float] = []
    ys: list[float] = []
    for st in geo.strokes:
        xs += [st['p0'][0], st['p1'][0]]
        ys += [st['p0'][1], st['p1'][1]]
    for fl in geo.flashes:
        xs.append(fl['x'])
        ys.append(fl['y'])
    for poly in geo.regions:
        xs += list(poly[:, 0])
        ys += list(poly[:, 1])
    for surf in geo.surfaces:
        xs += list(surf.outer[:, 0])
        ys += list(surf.outer[:, 1])
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)
