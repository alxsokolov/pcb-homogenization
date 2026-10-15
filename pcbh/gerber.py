#!/usr/bin/env python3
"""
RS-274X Gerber parser: produces the same pcbh.raster.LayerGeometry that
pcbh.odb's ODB++ parser produces, so discretize.py rasterizes both formats
identically.

Also carries shift_coordinates(), used by import_board.py to normalize a
source board's absolute coordinates so its bounding box starts at (0, 0) --
the convention discretize.py assumes, since a source board's own coordinate
system is normally an arbitrary offset from the CAD tool's global origin.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pcbh.raster import Aperture, LayerGeometry

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
# Coordinate shift, for import_board.py's "normalize to (0,0)" step
# ---------------------------------------------------------------------------

_COORD_LINE_RE = re.compile(r'D0[123]\*?\s*$')


def shift_coordinates(text: str, dx_mm: float, dy_mm: float) -> str:
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
