#!/usr/bin/env python3
"""
ODB++ physical stackup resolution (n-layer).

Reads the layer stackup -- order, kind, and *thickness* of every conductive
and dielectric layer -- out of an extracted ODB++ job. This is the piece the
2-layer importer (import_odb_board.py) only needed a degenerate version of:
there, "the stackup" was one copper thickness plus the sum of whatever sat
between the two outer layers. For an n-layer board the individual thicknesses
and their z order are the whole point, so they get resolved properly here.

Where ODB++ keeps stackup information, in the order this module tries:

  matrix/matrix
      Layer order and kind: ROW (top -> bottom), TYPE (SIGNAL /
      POWER_GROUND / MIXED / DIELECTRIC / SOLDER_MASK / ...), and for
      DIELECTRIC rows DIELECTRIC_TYPE (CORE | PREPREG) and DIELECTRIC_NAME.
      Identity and order only -- no thicknesses.

  steps/<step>/layers/<layer>/attrlist
      Where the numbers actually live: .layer_dielectric, .copper_weight,
      .dielectric_constant, .loss_tangent, .material. (Attribute
      *definitions* -- type and unit -- are in misc/attrlist, misc/sysattr
      and misc/userattr; the values are per layer.)

  misc/attrlist
      Job-level attributes, including .board_thickness -- used here purely
      as a cross-check on the resolved stackup, and .primary_side.

  matrix/stackup.xml
      Optional, added in ODB++ 8.1 Update 3: the dedicated stackup file
      (StackupFile/EdaData/Stackup/Group/Layer/SpecRef ->
      Specs/Spec/Material/Default_Thickness), meant to eventually supersede
      the matrix for this purpose. It exists mainly for rigid-flex, where a
      single linear matrix cannot express the stackup at all. Neither of the
      jobs this was developed against has one, so it is detected and
      reported but not parsed -- see _warn_unparsed_stackup_xml().

Three decoding rules, each established against the JODY-W6 (Altium, 8 layers)
and Hella (Allegro-style) jobs:

  1. .layer_dielectric on a CONDUCTIVE layer is the thickness of the
     dielectric BELOW it, not that layer's copper thickness. In JODY-W6
     every signal row equals the DIELECTRIC row beneath it (TOP_LAYER
     0.065 = DIELECTRIC_1, MID-LAYER_1 0.065 = DIELECTRIC_2, ...), and
     BOTTOM_LAYER, which has nothing below it, carries a leftover 0.02.
     Summing .layer_dielectric over every row double-counts each dielectric
     (0.944 mm against a real 0.728 mm board). Copper thickness therefore
     comes from .copper_weight; a conductive layer's own .layer_dielectric
     is used only as a fallback source for the dielectric beneath it, when
     that dielectric's own attrlist is missing.

  2. .copper_weight units vary by exporter. Altium writes micrometres
     (35.001, 38.001, 25.001); Allegro writes ounces (1). Real copper runs
     0.5-3 oz = 17-105 um, so a value above _CU_WEIGHT_OZ_MAX is read as um
     and anything at or below it as oz (x 0.035 mm/oz). The two ranges do
     not overlap.

  3. The resolved total must match misc/attrlist .board_thickness. For
     JODY-W6: 8 copper layers (0.266008 mm, from .copper_weight as um) +
     7 dielectrics (0.462 mm, from their own .layer_dielectric) = 0.728008
     mm against .board_thickness = 0.72801226 -- agreement to 4 nm, which
     is what confirms rules 1 and 2.

Not every job carries a stackup at all: the Hella board has no DIELECTRIC
rows in its matrix, .copper_weight = 1 oz, and .layer_dielectric = 0.036614
on its top layer -- a 37 um core, which is not a believable board. Boards
like that need their dielectric thicknesses supplied from outside the job;
resolve_stackup() takes an `overrides` mapping for exactly that and reports
every layer whose thickness it had to guess.

Used by import_odb_board_nlayer.py. The 2-layer import_odb_board.py does not
use this module and is unaffected by it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import odb_parser

# ODB++ layer TYPEs that carry copper we homogenize. POWER_GROUND and MIXED
# are copper layers just as much as SIGNAL is -- a plane layer is still
# copper, and on a real multilayer board the inner planes are usually the
# *most* copper-rich layers, so dropping them would bias the homogenization
# badly.
CONDUCTIVE_TYPES = ("SIGNAL", "POWER_GROUND", "MIXED")
DIELECTRIC_TYPES = ("DIELECTRIC",)

# .copper_weight above this is read as micrometres, at or below as ounces.
# Real PCB copper is 0.5-3 oz = 17-105 um, so nothing legitimate lands near
# the boundary from either side.
_CU_WEIGHT_OZ_MAX = 10.0
_MM_PER_OZ = 0.035

# Default copper thickness when a conductive layer carries no .copper_weight
# at all (1 oz, the commonest outer-layer finish).
DEFAULT_COPPER_THICKNESS_MM = 0.035

# Tolerance on the .board_thickness cross-check before it is reported as a
# mismatch: 2 um, comfortably above the rounding in the source files and far
# below any real stackup discrepancy.
_BOARD_THICKNESS_TOL_MM = 2e-3


# ---------------------------------------------------------------------------
# attrlist parsing
# ---------------------------------------------------------------------------

# NOTE: deliberately NOT odb_parser.parse_odb_attrlist(). That one splits on
# '=' without stripping whitespace, so Altium's ".layer_dielectric = 0.065"
# (spaces around the '=') lands in the dict under the key
# "layer_dielectric " -- with a trailing space -- and every .get() for it
# misses. Jobs that write ".layer_dielectric=1.51" with no spaces (the
# odb-kitchen-sink sample) parse fine, which is why the 2-layer path never
# noticed. Fixing it there would change the 2-layer samples' resolved
# thicknesses, so the whitespace-tolerant version lives here instead and the
# 2-layer path is left exactly as it is.
_ATTR_RE = re.compile(r'^\.([A-Za-z_][\w]*)\s*=\s*(.*)$')
_UNITS_RE = re.compile(r'^UNITS\s*=\s*(MM|INCH)\s*$', re.I)

_MM_PER_INCH = 25.4


def parse_attrlist(path: Path) -> dict[str, float | str]:
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


# Attributes whose value is a length in the file's own UNITS.
_LENGTH_ATTRS = frozenset({"layer_dielectric", "board_thickness"})


def _copper_thickness_mm(attrs: dict) -> tuple[float, str]:
    """
    Copper thickness of one conductive layer, from .copper_weight.

    Returns (thickness_mm, source) where source is one of "copper_weight_um",
    "copper_weight_oz" or "default" -- callers report the guessed ones.

    .layer_dielectric is deliberately NOT consulted: on a conductive layer it
    describes the dielectric below (rule 1 in the module docstring), so using
    it here would report a 65 um "copper" layer for a 35 um one.
    """
    w = attrs.get("copper_weight")
    if isinstance(w, (int, float)) and w > 0:
        if w > _CU_WEIGHT_OZ_MAX:
            return round(float(w) / 1000.0, 9), "copper_weight_um"
        return round(float(w) * _MM_PER_OZ, 9), "copper_weight_oz"
    return DEFAULT_COPPER_THICKNESS_MM, "default"


# ---------------------------------------------------------------------------
# Resolved stackup model
# ---------------------------------------------------------------------------


@dataclass
class StackupLayer:
    """
    One physical slab of the laminate, in z order (z0 < z1, z=0 at the board's
    bottom face -- the convention the FE mesh uses).
    """
    kind: str                       # "copper" | "dielectric"
    name: str                       # ODB++ layer name, lowercased
    thickness_mm: float
    z0_mm: float = 0.0
    z1_mm: float = 0.0
    odb_row: int = 0
    odb_type: str = ""              # SIGNAL | POWER_GROUND | MIXED | DIELECTRIC
    dielectric_type: str | None = None      # CORE | PREPREG
    material: str | None = None
    dielectric_constant: float | None = None
    loss_tangent: float | None = None
    thickness_source: str = ""      # how thickness_mm was obtained

    @property
    def is_copper(self) -> bool:
        return self.kind == "copper"


@dataclass
class Stackup:
    """A resolved n-layer stackup, ordered bottom (z=0) to top."""
    layers: list[StackupLayer] = field(default_factory=list)
    board_thickness_attr_mm: float | None = None     # misc/attrlist .board_thickness
    warnings: list[str] = field(default_factory=list)

    @property
    def copper_layers(self) -> list[StackupLayer]:
        return [l for l in self.layers if l.is_copper]

    @property
    def n_copper(self) -> int:
        return len(self.copper_layers)

    @property
    def total_thickness_mm(self) -> float:
        return sum(l.thickness_mm for l in self.layers)

    def z_interfaces_mm(self) -> list[float]:
        """Ascending z breakpoints: [0, z1_0, z1_1, ..., total]."""
        return [0.0] + [l.z1_mm for l in self.layers]


def _assign_z(layers: list[StackupLayer]) -> None:
    z = 0.0
    for l in layers:
        l.z0_mm = round(z, 9)
        z += l.thickness_mm
        l.z1_mm = round(z, 9)


def _warn_unparsed_stackup_xml(odb_dir: Path, warnings: list[str]) -> None:
    """
    matrix/stackup.xml (ODB++ 8.1u3+) is the dedicated stackup file and, where
    present, is more authoritative than matrix + attrlist -- especially for
    rigid-flex, where a linear matrix cannot express the stackup. It is not
    parsed here (no job on hand has one), so its presence is reported rather
    than silently ignored: a board that ships one may well disagree with what
    this module derives from the matrix.
    """
    p = odb_dir / "matrix" / "stackup.xml"
    if p.exists():
        warnings.append(
            f"{p} exists but is not parsed -- the stackup below was derived from "
            "matrix/matrix + per-layer attrlist instead. For a rigid-flex job the "
            "two can disagree; check it by hand."
        )


def resolve_stackup(
    odb_dir: Path,
    step_dir: Path,
    overrides: dict[str, float] | None = None,
    default_copper_thickness_mm: float = DEFAULT_COPPER_THICKNESS_MM,
) -> Stackup:
    """
    Resolve the full n-layer stackup of an extracted ODB++ job.

    odb_dir   job root (contains matrix/, steps/, misc/)
    step_dir  the step whose layers/ holds the per-layer attrlists
    overrides {layer_name_lowercased: thickness_mm} -- wins over anything
              found in the job, for boards whose ODB++ carries no usable
              stackup (see the module docstring on the Hella board).

    Only the conductive and dielectric layers between (and including) the
    outermost pair of conductive layers become part of the stackup; solder
    mask, paste, silkscreen, drill and document layers are not structural
    here and are skipped.
    """
    overrides = {k.lower(): v for k, v in (overrides or {}).items()}
    warnings: list[str] = []
    _warn_unparsed_stackup_xml(odb_dir, warnings)

    matrix = odb_parser.parse_odb_matrix(odb_dir / "matrix" / "matrix")
    layers_dir = step_dir / "layers"

    structural = [l for l in matrix
                  if l.type in CONDUCTIVE_TYPES + DIELECTRIC_TYPES]
    conductive_rows = [l for l in structural if l.type in CONDUCTIVE_TYPES]
    if len(conductive_rows) < 2:
        raise ValueError(
            f"Expected >= 2 conductive layers ({'/'.join(CONDUCTIVE_TYPES)}) in "
            f"{odb_dir / 'matrix' / 'matrix'}, found {len(conductive_rows)}"
        )

    # Trim to the outermost conductive pair: a DIELECTRIC row above the top
    # copper or below the bottom copper is a mask/coverlay, not part of the
    # structural laminate.
    first, last = conductive_rows[0].row, conductive_rows[-1].row
    structural = [l for l in structural if first <= l.row <= last]

    # attrlists, keyed by matrix row, read once.
    attrs_of = {l.row: parse_attrlist(layers_dir / l.name / "attrlist")
                for l in structural}

    # Rule 1: a conductive layer's .layer_dielectric describes the dielectric
    # BELOW it. Build that fallback map (dielectric row -> thickness from the
    # conductive row immediately above it) before resolving thicknesses, so a
    # dielectric whose own attrlist is missing or silent can still be filled.
    diel_from_above: dict[int, float] = {}
    for prev, cur in zip(structural, structural[1:]):
        if cur.type in DIELECTRIC_TYPES and prev.type in CONDUCTIVE_TYPES:
            t = attrs_of[prev.row].get("layer_dielectric")
            if isinstance(t, (int, float)) and t > 0:
                diel_from_above[cur.row] = float(t)

    resolved: list[StackupLayer] = []
    for l in structural:
        attrs = attrs_of[l.row]
        is_cu = l.type in CONDUCTIVE_TYPES

        if l.name in overrides:
            thickness, source = overrides[l.name], "override"
        elif is_cu:
            thickness, source = _copper_thickness_mm(attrs)
            if source == "default":
                thickness = default_copper_thickness_mm
        else:
            t = attrs.get("layer_dielectric")
            if isinstance(t, (int, float)) and t > 0:
                thickness, source = float(t), "layer_dielectric"
            elif l.row in diel_from_above:
                thickness, source = diel_from_above[l.row], "layer_dielectric_of_layer_above"
            else:
                thickness, source = 0.0, "missing"

        if source in ("default", "missing"):
            warnings.append(
                f"layer {l.name!r} ({l.type}): no usable thickness in the job "
                f"-- using {thickness} mm ({source}). Pass an explicit override."
            )

        material = attrs.get("material")
        resolved.append(StackupLayer(
            kind="copper" if is_cu else "dielectric",
            name=l.name,
            thickness_mm=round(float(thickness), 9),
            odb_row=l.row,
            odb_type=l.type,
            dielectric_type=l.dielectric_type,
            material=str(material) if isinstance(material, str) and material else None,
            dielectric_constant=(float(attrs["dielectric_constant"])
                                 if isinstance(attrs.get("dielectric_constant"), (int, float))
                                 else None),
            loss_tangent=(float(attrs["loss_tangent"])
                          if isinstance(attrs.get("loss_tangent"), (int, float)) else None),
            thickness_source=source,
        ))

    # ODB++ matrix ROW order runs top -> bottom; the FE mesh and every
    # homogenization formula here put z=0 at the board's bottom face, so
    # reverse into ascending z before assigning z0/z1.
    resolved.reverse()
    _assign_z(resolved)

    misc_attrs = parse_attrlist(odb_dir / "misc" / "attrlist")
    board_t = misc_attrs.get("board_thickness")
    board_t = float(board_t) if isinstance(board_t, (int, float)) and board_t > 0 else None

    stk = Stackup(layers=resolved, board_thickness_attr_mm=board_t, warnings=warnings)

    # Rule 3: cross-check against the job's own .board_thickness. A mismatch
    # means a thickness was read from the wrong attribute or in the wrong
    # unit, which is exactly the failure mode rules 1 and 2 exist to avoid --
    # so it is reported loudly rather than passed over.
    if board_t is not None:
        delta = stk.total_thickness_mm - board_t
        if abs(delta) > _BOARD_THICKNESS_TOL_MM:
            warnings.append(
                f"resolved total thickness {stk.total_thickness_mm:.6f} mm disagrees with "
                f"misc/attrlist .board_thickness {board_t:.6f} mm by {delta * 1e3:+.1f} um "
                "-- a thickness is probably being read from the wrong attribute or unit."
            )
    return stk


def format_stackup(stk: Stackup) -> str:
    """Human-readable stackup table, bottom row first (ascending z)."""
    def kind_of(l: StackupLayer) -> str:
        return l.kind if l.dielectric_type is None else f"{l.kind}/{l.dielectric_type.lower()}"

    nw = max([len(l.name) for l in stk.layers] + [4])
    kw = max([len(kind_of(l)) for l in stk.layers] + [4])
    lines = [f"{'z0 (mm)':>9} {'z1 (mm)':>9} {'t (um)':>8} {'kind':<{kw}} "
             f"{'name':<{nw}} {'Dk':>4} source"]
    for l in reversed(stk.layers):       # print top-of-board first, as a stackup is drawn
        # Dk is only meaningful for the dielectrics; the exporter writes 1 on
        # copper layers, which is noise here.
        dk = (f"{l.dielectric_constant:.1f}"
              if l.dielectric_constant and not l.is_copper else "")
        lines.append(f"{l.z0_mm:9.4f} {l.z1_mm:9.4f} {l.thickness_mm * 1e3:8.1f} "
                     f"{kind_of(l):<{kw}} {l.name:<{nw}} {dk:>4} {l.thickness_source}")
    pad = f"{'':9} {'':9}"
    lines.append(f"{pad} {stk.total_thickness_mm * 1e3:8.1f}  total "
                 f"({stk.n_copper} copper, {len(stk.layers) - stk.n_copper} dielectric)")
    if stk.board_thickness_attr_mm is not None:
        lines.append(f"{pad} {stk.board_thickness_attr_mm * 1e3:8.1f}  "
                     f".board_thickness (misc/attrlist)")
    return "\n".join(lines)
