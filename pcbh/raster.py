#!/usr/bin/env python3
"""
Shared 2D geometry model + rasterizer for PCB layer inputs.

Both Gerber (pcbh.gerber) and ODB++ (pcbh.odb) layer parsers build a
LayerGeometry from their own file format, then hand it to rasterize_layer()
here to produce the same (ny, nx) boolean copper/no-copper grid. Keeping the
model + rasterizer format-agnostic is what lets discretize.py treat a Gerber
layer and an ODB++ layer identically from this point on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath


@dataclass
class Aperture:
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
class Surface:
    """A filled polygon with optional holes (ODB++ S...SE / Gerber region-with-cutout)."""
    outer: np.ndarray                            # closed polygon (N,2)
    holes: list[np.ndarray] = field(default_factory=list)


@dataclass
class LayerGeometry:
    """Raw geometry from one copper layer, regardless of source format (coords in mm)."""
    regions:  list[np.ndarray] = field(default_factory=list)  # closed polygons (N,2), no holes
    flashes:  list[dict]       = field(default_factory=list)  # {x, y, aperture}
    strokes:  list[dict]       = field(default_factory=list)  # {p0, p1, aperture}
    surfaces: list[Surface]    = field(default_factory=list)  # polygons with holes
    dropped:  dict[str, int]   = field(default_factory=dict)  # {name: count} skipped shapes


# ---------------------------------------------------------------------------
# Geometry → polygon list
# ---------------------------------------------------------------------------

def circle_poly(cx: float, cy: float, r: float, n: int = 32) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def rect_poly(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    hw, hh = w / 2, h / 2
    return np.array([[cx - hw, cy - hh], [cx + hw, cy - hh],
                     [cx + hw, cy + hh], [cx - hw, cy + hh]], dtype=float)


def roundrect_poly(cx: float, cy: float, params: list[float]) -> np.ndarray:
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


def capsule_poly(p0: tuple, p1: tuple, width: float, n_cap: int = 10) -> np.ndarray:
    """Convert a stroke segment + aperture width to a stadium (capsule) polygon."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L  = np.hypot(dx, dy)
    r  = width / 2
    if L < 1e-9:
        return circle_poly(x0, y0, r)
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


def _oriented_poly(poly: np.ndarray, cx: float, cy: float, angle_deg: float, mirror_x: bool) -> np.ndarray:
    """Rotate/mirror a polygon about (cx, cy) -- used for ODB++ pad flashes, which (unlike
    Gerber flashes) carry an explicit per-instance orientation."""
    pts = poly - np.array([cx, cy])
    if mirror_x:
        pts[:, 0] *= -1
    if angle_deg:
        th = np.radians(angle_deg)
        c, s = np.cos(th), np.sin(th)
        pts = pts @ np.array([[c, -s], [s, c]]).T
    return pts + np.array([cx, cy])


def geometry_to_polygons(geo: LayerGeometry) -> list[np.ndarray]:
    """Flatten regions, flashes, and strokes (everything but hole-bearing surfaces) into polygons."""
    polys: list[np.ndarray] = list(geo.regions)

    for fl in geo.flashes:
        x, y, ap = fl['x'], fl['y'], fl['aperture']
        if ap.kind == 'C':
            polys.append(circle_poly(x, y, ap.width / 2))
            continue
        if ap.kind == 'RoundRect':
            poly = roundrect_poly(x, y, ap.params)
        else:
            poly = rect_poly(x, y, ap.width, ap.height)
        angle_deg, mirror_x = fl.get('rotation', 0.0), fl.get('mirror', False)
        if angle_deg or mirror_x:
            poly = _oriented_poly(poly, x, y, angle_deg, mirror_x)
        polys.append(poly)

    for st in geo.strokes:
        ap = st['aperture']
        w  = ap.width if ap.kind == 'C' else max(ap.width, ap.height)
        polys.append(capsule_poly(st['p0'], st['p1'], w))

    return polys


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def _grid_points(width_mm: float, height_mm: float, nx: int, ny: int) -> np.ndarray:
    xs = np.linspace(width_mm  / nx / 2, width_mm  - width_mm  / nx / 2, nx)
    ys = np.linspace(height_mm / ny / 2, height_mm - height_mm / ny / 2, ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def rasterize(
    polygons: list[np.ndarray],
    width_mm: float,
    height_mm: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Rasterize a list of closed (hole-free) polygons onto an (ny × nx) grid.
    Samples at cell centres of a uniform pixel lattice.
    Returns float32 array with values in {0.0, 1.0}.
    """
    pts_flat = _grid_points(width_mm, height_mm, nx, ny)
    mask = np.zeros(ny * nx, dtype=bool)
    for poly in polygons:
        if len(poly) < 3:
            continue
        mask |= MplPath(poly).contains_points(pts_flat)
    return mask.reshape(ny, nx).astype(np.float32)


def rasterize_layer(
    geo: LayerGeometry,
    width_mm: float,
    height_mm: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """
    Rasterize a full LayerGeometry (regions/flashes/strokes OR-ed together,
    plus each hole-bearing surface as outer-minus-holes) onto an (ny × nx) grid.
    """
    pts_flat = _grid_points(width_mm, height_mm, nx, ny)
    mask = np.zeros(ny * nx, dtype=bool)

    for poly in geometry_to_polygons(geo):
        if len(poly) < 3:
            continue
        mask |= MplPath(poly).contains_points(pts_flat)

    for surf in geo.surfaces:
        if len(surf.outer) < 3:
            continue
        surf_mask = MplPath(surf.outer).contains_points(pts_flat)
        for hole in surf.holes:
            if len(hole) < 3:
                continue
            surf_mask &= ~MplPath(hole).contains_points(pts_flat)
        mask |= surf_mask

    return mask.reshape(ny, nx).astype(np.float32)
