#!/usr/bin/env python3
"""
Import a partner-provided ANSYS component archive (.cdb) into an existing
MAPDL session, without colliding with whatever's already there (the board,
built directly via mapdl.n()/mapdl.e() in simulate_pyansys_benchmark.py),
apply a solder material from a standalone APDL macro (.mac), and attach the
component to the board via MPC bonded contact -- reusing the TARGE170/
CONTA174 element-type template the archive already ships with, rather than
building a contact pair from scratch.

Background (from inspecting data/components/cr0805/DemonstratorB_CR0805_Export.cdb
directly -- see its MANIFEST.md for the full writeup, and both confirmed
against a real local MAPDL 2025 R2 session, not just read off the file):

  - Node/element IDs: the archive's own IDs are internal to that file and
    collide with the board's (both start low) unless shifted -- NUMOFF
    does **not** work for this, despite first appearances. Patching
    NUMOFF,NODE/ELEM in the archive text looked like it worked as long as
    the board stayed small, but a later, larger board (>=11953 elements or
    >=52372 nodes -- this archive's own fixed node/element ID range,
    confirmed empirically) proved otherwise: the component landed at the
    *exact same* absolute IDs (nodes 52372-78731, elements 11953-17276)
    regardless of what NUMOFF,NODE/ELEM was patched to, silently
    overwriting whichever board elements happened to already occupy that
    range. Root cause: NOOFFSET,NODE/ELEM (needed below so CDREAD doesn't
    auto-renumber the *board's* pre-existing data on collision, see next
    paragraph) suppresses *all* offsetting of that item during CDREAD, not
    just the automatic kind -- including an explicit NUMOFF set beforehand.
    So CDREAD can never be trusted to offset NODE/ELEM here. Instead,
    _renumber_archive_ids() shifts every node ID (NBLOCK, BFBLOCK) and
    every element ID + node connectivity (EBLOCK, and CMBLOCK's
    EVALELEMENTS element-range list) by rewriting the archive's own
    fixed-width ID fields directly, before the text ever reaches CDREAD --
    the same text-patching approach already used elsewhere in this module
    for MAT/TYPE (below) and for UNBL-tagged MPTEMP/MPDATA
    (reapply_component_materials()). This also means a second (or third)
    component placement is safe: each call computes its own offsets from
    the session's current max IDs, so multiple placed instances each land
    in their own disjoint range, never at the archive's fixed originals.

  - Material IDs: NUMOFF,MAT does **not** work the same way -- verified
    empirically (MPLIST/ESEL-by-MAT after import both show the archive's
    materials landed at their literal original IDs 1-10, unshifted,
    regardless of what NUMOFF,MAT was patched to). Apparently NUMOFF only
    biases NBLOCK/EBLOCK's own dedicated fast-read path, not the plain
    MP/MPDATA/TB command arguments (or EBLOCK's own material-attribute
    field) a coded DB archive uses -- those are explicit literal numbers,
    not sequentially-assigned IDs. Same story for element TYPE. Since
    rewriting every literal MAT/TYPE reference scattered through
    MP/MPDATA/TB and every EBLOCK row would be far more invasive and
    fragile than patching two header lines, the fix here goes the other
    way: remap_board_materials() moves the *board's* two materials (which
    we do fully control) up to a safe high ID range before the component
    is imported, leaving the component's own materials exactly as
    authored. Element TYPE needs no such remap: the component's own
    TARGE170/CONTA174 sit at 10/11, and its own SOLID186 redefinition at
    TYPE 1 is identical to the board's, so nothing actually collides.

  - The archive's ETBLOCK defines TARGE170/CONTA174 with KEYOPT(2)=2 (MPC
    algorithm) and KEYOPT(12)=5 (Bonded, always) already set -- i.e.
    exactly "MPC bonded contact" -- but no contact elements are actually
    meshed in the file. generate_bonded_contact() meshes them ourselves
    (via ESURF) using that same type definition (TYPE 10/11, used as-is),
    so the KEYOPT settings the partner configured are the ones actually
    used.

The exact ESURF face-selection logic (picking precisely the solder's
bottom face and the board's top face, and nothing else) is the part most
worth double-checking if the contact pair ever looks wrong (zero elements
on one side, or elements on unexpected faces).

  - Material *properties* (as opposed to material *IDs*, above): this
    machine's local MAPDL install is 2025 R1.02, but the archive's own
    header says it was written by 2025 R2 -- and its MPTEMP/MPDATA lines
    use an extended, "UNBL"-tagged argument format
    (``MPDATA,UNBL,<table_id>,<Lab>,<mat>,<stloc>,<C1..C6>`` and
    ``MPTEMP,UNBL,<table_id>,<stloc>,<T1..T6>``, i.e. two extra leading
    fields before the classic Lab/MAT/STLOC form) that this R1 CDREAD does
    not understand. Confirmed empirically (scratch_debug_mplist.py): right
    after CDREAD, MPLIST on the archive's own MAT 2 reported it completely
    undefined in one probe, while a later STATIC solve claimed MAT 2 had
    EX/EY/EZ/NUXY/NUYZ/NUXZ all individually set (making it look
    orthotropic and short exactly the GXY/GYZ/GXZ it'd then need) -- but
    grepping the raw file directly shows MAT 2 only ever authors EX/NUXY/
    ALPX (a plain isotropic definition, no orthotropic fields at all).
    Both symptoms are CDREAD misparsing the UNBL-tagged lines, not real
    gaps in the partner's data. reapply_component_materials() below
    sidesteps this entirely by parsing those lines with our own code
    (which does understand the UNBL format, having been written against
    this exact file) and reissuing them as plain MP/MPTEMP/MPDATA calls
    that any MAPDL release can read correctly, for whichever MAT ids are
    actually referenced by imported elements (empirically 2, 6, 9 for the
    CR0805 archive -- 1, 3, 4, 5, 7, 8, 10 exist in its material table but
    aren't used by any of its elements). MAT 9's TB,CREE/TBDATA creep
    definition is untouched by any of this -- those lines use the classic,
    non-UNBL syntax already, and apply_solder_material() re-supplies MAT
    9's elastic/CTE properties from a standalone, non-UNBL .mac regardless.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_VALID_ROTATIONS_DEG = (0, 90)


@dataclass
class Offsets:
    node: int
    elem: int


@dataclass
class ComponentHandle:
    """IDs actually added by import_component() -- computed from a live
    before/after diff of the MAPDL database, not derived from the archive's
    own (sparse, non-contiguous) internal numbering."""
    node_ids: np.ndarray
    elem_ids: np.ndarray
    offsets: Offsets
    origin_xy: tuple[float, float]
    """The archive's own (X, Y) bounding-box center, as authored -- NOT
    (0, 0). Confirmed empirically (see scratch_debug_targe.py) that the
    CR0805 archive's node coordinates are still wherever they sat on the
    partner's original demonstrator board (bbox center (10.0, 10.0) for
    that file), not pre-centered at the origin as place_component() used
    to assume. place_component() subtracts this before rotating/
    translating so (x_mm, y_mm) lands the component's actual footprint
    center where asked, instead of silently offsetting by however far the
    archive's own coordinates happened to sit from (0, 0)."""


def compute_safe_offsets(mapdl) -> Offsets:
    """Current session's max node/element IDs, so a subsequently-imported
    archive's own (low, internal) node/element IDs can be shifted clear of
    them via NUMOFF before CDREAD."""
    mapdl.allsel()
    node_max = int(mapdl.mesh.nnum_all.max()) if mapdl.mesh.nnum_all.size else 0
    elem_max = int(mapdl.mesh.enum_all.max()) if mapdl.mesh.enum_all.size else 0
    return Offsets(node=node_max, elem=elem_max)


def _board_mat_cmname(new_id: int) -> str:
    return f"BRDMAT{new_id}"


def remap_board_materials(mapdl, mat_id_map: dict[int, int]) -> None:
    """Moves the board's own materials to new (safely high, clear of any
    component archive's own 1-10-ish range) IDs -- MPCOPY duplicates the
    linear MP property table at the new ID, EMODIF repoints the board's
    elements at it, so nothing about the underlying values changes, only
    which MAT ID they live at. Call this *before* import_component(), so
    the component's own (unshiftable, see module docstring) MAT 1/2 don't
    silently overwrite the board's real copper/FR4 properties.

    Also stashes each remapped group as a named element component (CM),
    keyed by the new MAT id -- see reassert_board_materials()'s docstring
    for why this is needed on top of the EMODIF here, not a belt-and-
    braces duplicate of it.
    """
    for old_id, new_id in mat_id_map.items():
        mapdl.mpcopy(old_id, new_id)
        mapdl.esel("S", "MAT", "", old_id)
        mapdl.cm(_board_mat_cmname(new_id), "ELEM")
        mapdl.emodif("ALL", "MAT", new_id)
    mapdl.allsel()


def reassert_board_materials(mapdl, new_mat_ids) -> None:
    """Re-applies EMODIF,MAT for the board's remapped element groups, by
    the named component remap_board_materials() stashed for each one --
    NOT by re-selecting on the (new) MAT id itself.

    Needed because CDREAD silently reverts a *subset* of the board's
    already-remapped elements back toward whichever MAT ids the archive's
    own coded MP/MPDATA/TB definitions actually touch -- confirmed
    empirically (scratch_debug_targe.py) for the CR0805 archive: after
    remap_board_materials({1: 9001, 2: 9002}) + import_component(), roughly
    2354 of the board's 9654 FR4 (MAT 9002) elements came back reporting
    MAT 2/6/9 (exactly the CR0805 archive's own ceramic/nickel/solder
    material ids -- the ones it actually defines property data for, not
    the ones left as unreferenced carry-over entries), while all 5346
    copper (MAT 9001) elements happened to survive intact in this
    particular run. Re-selecting by current MAT after that point can't
    distinguish a genuinely-reverted board element from one of the
    archive's own legitimate MAT-2/6/9 elements -- both now read the same
    MAT id -- so the fix re-selects by the CM snapshot taken before CDREAD
    ran (an explicit element-ID membership list, unaffected by whatever
    CDREAD did to the MAT field afterwards) and reasserts MAT on exactly
    that original set. Call this once, right after import_component().
    """
    for new_id in new_mat_ids:
        mapdl.cmsel("S", _board_mat_cmname(new_id))
        mapdl.emodif("ALL", "MAT", new_id)
    mapdl.allsel()


def _parse_unbl_value_fields(fields: list[str]) -> list[float]:
    return [float(f) for f in (raw.strip() for raw in fields) if f]


def _parse_unbl_mp_tables(
    cdb_text: str,
) -> tuple[dict[int, dict[int, float]], dict[tuple[int, str], dict[int, tuple[int, float]]]]:
    """Parses the archive's own ``MPTEMP,UNBL,...``/``MPDATA,UNBL,...`` lines
    (see module docstring for why CDREAD can't be trusted to do this itself
    on a 2025 R1 install) directly from the raw CDB text.

    Returns (table_temps, mat_prop_data):
      - table_temps[table_id][stloc] = temperature, from lines like
        ``MPTEMP,UNBL,<table_id>,<stloc>,<T1..T6>``.
      - mat_prop_data[(mat, lab)][stloc] = (table_id, value), from lines
        like ``MPDATA,UNBL,<table_id>,<lab>,<mat>,<stloc>,<C1..C6>``.
    table_id is a temperature-table reference, reused across materials that
    share the same temperature points (e.g. table 1 is a generic
    room-temperature-only entry reused by most of this archive's
    materials) -- it is NOT the same number as the material id, except
    coincidentally for materials with their own dedicated (non-shared)
    temperature curve.
    """
    table_temps: dict[int, dict[int, float]] = {}
    mat_prop_data: dict[tuple[int, str], dict[int, tuple[int, float]]] = {}

    for line in cdb_text.splitlines():
        line = line.strip()
        if line.startswith("MPTEMP,UNBL"):
            fields = line.split(",")
            table_id = int(fields[2])
            stloc = int(fields[3])
            temps = _parse_unbl_value_fields(fields[4:])
            table = table_temps.setdefault(table_id, {})
            for i, t in enumerate(temps):
                table[stloc + i] = t
        elif line.startswith("MPDATA,UNBL"):
            fields = line.split(",")
            table_id = int(fields[2])
            lab = fields[3].strip()
            mat_id = int(fields[4])
            stloc = int(fields[5])
            values = _parse_unbl_value_fields(fields[6:])
            entries = mat_prop_data.setdefault((mat_id, lab), {})
            for i, v in enumerate(values):
                entries[stloc + i] = (table_id, v)

    return table_temps, mat_prop_data


def _material_curve(
    mat_id: int,
    lab: str,
    table_temps: dict[int, dict[int, float]],
    mat_prop_data: dict[tuple[int, str], dict[int, tuple[int, float]]],
) -> list[tuple[float, float]]:
    entries = mat_prop_data.get((mat_id, lab))
    if not entries:
        return []
    curve = [(table_temps[table_id][stloc], value) for stloc, (table_id, value) in entries.items()]
    curve.sort()
    return curve


def reapply_component_materials(mapdl, cdb_path: Path, mat_ids) -> None:
    """Clears (MPDELE) and cleanly re-defines, from our own parse of the
    archive's raw ``UNBL``-tagged MPTEMP/MPDATA lines, the linear material
    properties for whichever of the archive's own MAT ids are given --
    intended for exactly the MAT ids actually referenced by imported
    elements (empirically 2, 6, 9 for the CR0805 archive; see module
    docstring for why the properties CDREAD itself loaded for these can't
    be trusted).

    MPDELE,"ALL" wipes *every* linear property label for that MAT id first
    (not just the ones about to be reapplied), since CDREAD's misparse
    could in principle have scattered a value onto any label/material
    slot, not only the ones the archive legitimately authors for that MAT
    -- this guarantees a clean slate rather than layering correct data on
    top of unknown garbage. TB,CREEP data (MAT 9's creep table) is
    untouched -- it uses the classic, non-UNBL syntax and isn't affected.

    Call this once, right after import_component() (order relative to
    reassert_board_materials() doesn't matter -- they touch disjoint MAT
    ids). apply_solder_material() should still be called afterwards for
    MAT 9, since it re-supplies MAT 9's EX/PRXY/CTEX from the standalone
    .mac on top of whatever this function reconstructs from the archive.
    """
    table_temps, mat_prop_data = _parse_unbl_mp_tables(cdb_path.read_text())
    labs_by_mat: dict[int, set[str]] = {}
    for mat_id, lab in mat_prop_data:
        labs_by_mat.setdefault(mat_id, set()).add(lab)

    for mat_id in mat_ids:
        mapdl.mpdele("ALL", mat_id, mat_id)
        for lab in sorted(labs_by_mat.get(mat_id, ())):
            curve = _material_curve(mat_id, lab, table_temps, mat_prop_data)
            if not curve:
                continue
            temps = [t for t, _ in curve]
            values = [v for _, v in curve]
            # MPTEMP's temperature table is a shared scratch table reused
            # across calls (STLOC only overwrites specific slots, it doesn't
            # resize the table) -- without erasing it first, a shorter
            # curve here would leave the *previous* lab's higher-index
            # entries in place past this curve's own length, and MAPDL
            # validates the whole table's ordering, not just the slots this
            # call touched. Confirmed empirically: MAT 2's single-point
            # ALPX/EX/NUXY curves after MAT 9's 100-point ALPX curve (mat
            # ids are processed in ascending order, so MAT 9 last isn't the
            # trigger, but any short-after-long pair is) raised "Temperatures
            # for material properties must be in ascending order" on the
            # very next MPTEMP call. Bare MPTEMP with no arguments erases
            # the table -- same idiom SAC305_MikroVal.mac itself uses before
            # each of its own property blocks.
            mapdl.mptemp()
            for start in range(0, len(temps), 6):
                t_chunk = temps[start : start + 6]
                v_chunk = values[start : start + 6]
                mapdl.mptemp(start + 1, *t_chunk)
                mapdl.mpdata(lab, mat_id, start + 1, *v_chunk)


def _line_ending(line: str) -> str:
    stripped = line.rstrip("\r\n")
    return line[len(stripped):]


def _header_fields(line: str) -> list[str]:
    """Comma-separated fields of a block header/CMBLOCK line, comment and
    line ending stripped (CMBLOCK's often carries a trailing "! ..." note)."""
    return [p.strip() for p in line.split("!")[0].strip().rstrip(",").split(",")]


def _shift_field(line: str, start: int, width: int, delta: int) -> str:
    """Re-renders the width-char integer field at [start:start+width],
    shifted by delta, right-justified in the same width -- the rest of the
    line (other fields, line ending) is untouched."""
    old = int(line[start:start + width])
    new = old + delta
    field = f"{new:{width}d}"
    if len(field) > width:
        raise ValueError(f"Shifted ID {new} (from {old} + {delta}) no longer fits "
                         f"the archive's {width}-char field -- offset too large.")
    return line[:start] + field + line[start + width:]


def _split_fields(line: str, width: int) -> tuple[list[str], str]:
    """A fixed-width data line's integer fields (as ints) and its line
    ending, for a line containing only width-char fields with no header."""
    ending = _line_ending(line)
    body = line[:len(line) - len(ending)] if ending else line
    n = len(body) // width
    return [int(body[k * width:(k + 1) * width]) for k in range(n)], ending


def _render_fields(values: list[int], width: int, ending: str) -> str:
    rendered = []
    for v in values:
        field = f"{v:{width}d}"
        if len(field) > width:
            raise ValueError(f"Shifted ID {v} no longer fits the archive's "
                             f"{width}-char field -- offset too large.")
        rendered.append(field)
    return "".join(rendered) + ending


def _shift_all_fields(line: str, width: int, delta: int, *, signed_magnitude: bool = False) -> tuple[str, int]:
    """Shifts every width-char fixed field on a data line (no header
    fields mixed in) by delta. signed_magnitude=True shifts |value| and
    keeps the sign (CMBLOCK's range-pair encoding: a negative value closes
    the range opened by the preceding positive one). Returns (new_line,
    n_fields)."""
    values, ending = _split_fields(line, width)
    shifted = [(-1 if v < 0 else 1) * (abs(v) + delta) if signed_magnitude else v + delta
               for v in values]
    return _render_fields(shifted, width, ending), len(values)


def _renumber_nblock_or_bfblock(lines: list[str], i: int, count: int, node_offset: int) -> tuple[list[str], int]:
    """NBLOCK/BFBLOCK: one data line per entry, node ID in the first
    9-char field. `count` data lines follow the format-descriptor line."""
    out = [lines[i]]  # format-descriptor line, unchanged
    i += 1
    for _ in range(count):
        out.append(_shift_field(lines[i], 0, 9, node_offset))
        i += 1
    return out, i


def _renumber_eblock(lines: list[str], i: int, n_records: int,
                     node_offset: int, elem_offset: int) -> tuple[list[str], int]:
    """EBLOCK, format (19i10): each record is a header line (10 attribute
    fields, then the element ID, then up to 8 node IDs = 19 fields) plus as
    many pure-node-ID continuation lines (width 10, up to 19 fields each)
    as the record's own NUMNODES (field 8, 0-based) needs beyond those
    first 8."""
    width = 10
    out = [lines[i]]  # format-descriptor line, unchanged
    i += 1
    for _ in range(n_records):
        fields, ending = _split_fields(lines[i], width)
        numnodes = fields[8]
        fields[10] += elem_offset
        n_header_nodes = min(8, len(fields) - 11)
        for k in range(11, 11 + n_header_nodes):
            fields[k] += node_offset
        out.append(_render_fields(fields, width, ending))
        i += 1
        remaining = numnodes - n_header_nodes
        while remaining > 0:
            shifted, n = _shift_all_fields(lines[i], width, node_offset)
            out.append(shifted)
            remaining -= n
            i += 1
    return out, i


def _renumber_cmblock_elem(lines: list[str], i: int, count: int, elem_offset: int) -> tuple[list[str], int]:
    """CMBLOCK,<name>,ELEM: a range-pair list (negative = end of the range
    opened by the preceding positive value), format (8i10)."""
    width = 10
    out = [lines[i]]  # format-descriptor line, unchanged
    i += 1
    remaining = count
    while remaining > 0:
        shifted, n = _shift_all_fields(lines[i], width, elem_offset, signed_magnitude=True)
        out.append(shifted)
        remaining -= n
        i += 1
    return out, i


def _renumber_archive_ids(cdb_text: str, offsets: Offsets) -> str:
    """Shifts every node ID (NBLOCK, BFBLOCK) and every element ID + node
    connectivity (EBLOCK, CMBLOCK's element-range lists) in the archive by
    offsets.node/offsets.elem, by rewriting the fixed-width ID fields
    directly -- see the module docstring for why this replaces NUMOFF
    entirely rather than patching its two header lines (that only ever
    appeared to work; NOOFFSET,NODE/ELEM, needed to stop CDREAD from
    renumbering the *board's* pre-existing data instead, silently
    suppresses NUMOFF's effect too).

    Each block's record layout was reverse-engineered from
    data/components/cr0805/DemonstratorB_CR0805_Export.cdb directly (see
    its MANIFEST.md); a differently-exported archive could in principle
    lay these out differently, but CDWRITE's own format is standardized
    enough that this should hold for any 2025 R1/R2 CDB.
    """
    lines = cdb_text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        header = lines[i]
        stripped = header.strip()
        if stripped.startswith("NBLOCK,") or stripped.startswith("BFBLOCK,"):
            fields = _header_fields(stripped)
            count = int(fields[-2] if stripped.startswith("BFBLOCK,") else fields[-1])
            out.append(header)
            block_lines, i = _renumber_nblock_or_bfblock(lines, i + 1, count, offsets.node)
            out.extend(block_lines)
            continue
        if stripped.startswith("EBLOCK,"):
            count = int(_header_fields(stripped)[-1])
            out.append(header)
            block_lines, i = _renumber_eblock(lines, i + 1, count, offsets.node, offsets.elem)
            out.extend(block_lines)
            continue
        if stripped.startswith("CMBLOCK,") and _header_fields(stripped)[2].upper() == "ELEM":
            count = int(_header_fields(stripped)[-1])
            out.append(header)
            block_lines, i = _renumber_cmblock_elem(lines, i + 1, count, offsets.elem)
            out.extend(block_lines)
            continue
        out.append(header)
        i += 1
    return "".join(out)


def import_component(mapdl, cdb_path: Path, offsets: Offsets) -> ComponentHandle:
    """Reads a partner-exported .cdb archive into the current MAPDL
    database, node/element IDs shifted clear of everything already in the
    session. Material IDs are NOT shifted (see module docstring) -- call
    remap_board_materials() beforehand to clear the low ID range instead.
    """
    mapdl.prep7()
    mapdl.allsel()
    nodes_before = set(mapdl.mesh.nnum_all.tolist())
    elems_before = set(mapdl.mesh.enum_all.tolist())
    patched_text = _renumber_archive_ids(cdb_path.read_text(), offsets)
    with tempfile.NamedTemporaryFile("w", suffix=".cdb", delete=False) as f:
        f.write(patched_text)
        patched_path = Path(f.name)
    # CDREAD has its own automatic collision-avoidance: "If a set of data
    # exists prior to the CDREAD operation, that data set is offset upward
    # to allow the new data to fit without overlap" (ANSYS CDREAD docs).
    # Confirmed empirically that this fires regardless of whether the
    # incoming data actually needs the room -- it silently renumbered the
    # *board's* own nodes/elements (already clear of the archive by our own
    # _renumber_archive_ids() above), which is not what we want (the
    # board's own node IDs are referenced elsewhere, e.g. the 3-2-1 fixture
    # applied before this call). NOOFFSET tells CDREAD not to touch
    # pre-existing NODE/ELEM numbering -- safe here because the incoming
    # archive's IDs have already been shifted clear of the board's in the
    # text itself, so there's nothing left for CDREAD's own offsetting to
    # protect against. (It also means NODE/ELEM never depended on NUMOFF in
    # the first place by the time CDREAD sees this text -- see module
    # docstring for why NUMOFF,NODE/ELEM can't be trusted here at all: this
    # same NOOFFSET call silently suppresses it too, not just the automatic
    # offsetting described above.)
    #
    # This same auto-offset also fires for MAT/TYPE (confirmed empirically
    # via scratch_debug_targe.py: without the NOOFFSET,MAT/TYPE calls
    # below, the board's post-remap_board_materials() element MAT
    # attributes came back corrupted after CDREAD -- e.g. element 1's MAT
    # silently shifted 9001->9011 (the archive's own NUMOFF,MAT=10 staging
    # value, applied to *pre-existing* data despite the module docstring's
    # finding that it does nothing for the archive's *own* MAT numbers),
    # while other board elements' MAT reverted to their pre-remap ID
    # entirely -- either way, this must be suppressed the same way NODE/
    # ELEM's auto-offset is, and for the same reason: remap_board_materials
    # already guarantees no overlap, so there is nothing for CDREAD to
    # protect by renumbering. REAL/CSYS get the same treatment for
    # consistency even though no collision with the board has been
    # observed for those (the board never assigns non-default REAL/CSYS).
    mapdl.nooffset("NODE")
    mapdl.nooffset("ELEM")
    mapdl.nooffset("MAT")
    mapdl.nooffset("TYPE")
    mapdl.nooffset("REAL")
    mapdl.nooffset("CSYS")
    try:
        mapdl.cdread("DB", str(patched_path))
    finally:
        patched_path.unlink(missing_ok=True)
    # The archive ends with its own FINISH command, which drops the session
    # out of PREP7 -- re-enter before returning control.
    mapdl.prep7()
    # The archive carries full solid-model geometry (keypoints/lines/...)
    # that its mesh nodes are associated with, which blocks direct
    # coordinate redefinition ("Node N is attached to KEYPOINT/LINE M and
    # cannot be altered" -- confirmed empirically). We only want the mesh,
    # not the CAD geometry, and place_component() needs to move nodes
    # directly -- MODMSH,NOCHECK turns off that solid-model consistency
    # check so node/element modification commands work directly (NOT the
    # same as MODMSH,DETACH, which -- also tested -- still left some nodes
    # protected).
    mapdl.modmsh("NOCHECK")
    mapdl.allsel()
    # The archive also carries a BFBLOCK fixing every one of its nodes at
    # 20C (its export temperature). Nodal BF temperatures override BFUNIF,
    # so left in place the component and solder never follow the thermal
    # profile -- they sit at 20C for the whole solve while only the board
    # cycles, and with TREF=180C they start with a 160 K thermal mismatch
    # (seen as >200 MPa solder stress at a constant 180C hold). Delete them
    # so the uniform profile temperature applies to the whole assembly.
    mapdl.bfdele("ALL", "TEMP")

    nodes_after = set(mapdl.mesh.nnum_all.tolist())
    elems_after = set(mapdl.mesh.enum_all.tolist())
    new_node_ids = np.array(sorted(nodes_after - nodes_before), dtype=np.int64)
    new_elem_ids = np.array(sorted(elems_after - elems_before), dtype=np.int64)
    if new_node_ids.size == 0:
        raise RuntimeError(f"CDREAD of {cdb_path} added no new nodes -- import failed.")
    if new_elem_ids.size == 0:
        raise RuntimeError(f"CDREAD of {cdb_path} added no new elements -- import failed "
                           "(this is the ID-collision failure mode _renumber_archive_ids() "
                           "exists to prevent; if you see this, something about the board's "
                           "current node/element IDs still overlaps the archive's own).")

    mapdl.nsel("S", "NODE", "", int(new_node_ids.min()), int(new_node_ids.max()))
    raw_coords = np.asarray(mapdl.mesh.nodes)
    origin_xy = (
        float((raw_coords[:, 0].min() + raw_coords[:, 0].max()) / 2.0),
        float((raw_coords[:, 1].min() + raw_coords[:, 1].max()) / 2.0),
    )
    mapdl.allsel()

    return ComponentHandle(node_ids=new_node_ids, elem_ids=new_elem_ids, offsets=offsets,
                           origin_xy=origin_xy)


def apply_solder_material(mapdl, mac_path: Path, mat_id: int) -> None:
    """Sets the .mac's required `matID` parameter and runs it, so the
    authoritative SAC305 properties (EX/PRXY/CTEX + Garofalo TB,CREEP) are
    (re-)applied on top of whatever the .cdb already baked in for that
    material ID."""
    mapdl.run(f"matID = {mat_id}")
    mapdl.input(str(mac_path))


def solder_bottom_z(mapdl, solder_mat_id: int) -> float:
    """Minimum Z (in whatever frame the component's nodes are currently in)
    across all nodes belonging to elements of the given solder material --
    the face that needs to land on the board's top surface."""
    mapdl.allsel()
    mapdl.esel("S", "MAT", "", solder_mat_id)
    mapdl.nsle("S")
    coords = np.asarray(mapdl.mesh.nodes)
    if coords.size == 0:
        raise RuntimeError(f"No nodes found for solder MAT {solder_mat_id} -- "
                           "check the material ID.")
    z_min = float(coords[:, 2].min())
    mapdl.allsel()
    return z_min


def place_component(mapdl, handle: ComponentHandle, x_mm: float, y_mm: float,
                    rotation_deg: int, board_top_z_mm: float, solder_mat_id: int) -> None:
    """Rigid in-place transform of an already-imported component's nodes:
    an optional 0/90 deg in-plane rotation about the component's own local
    origin (the only two orientations a 2-terminal chip component can
    actually sit in), then translation. (x_mm, y_mm) is where the
    component's own local (0, 0) lands on the board -- manual placement
    only, no auto-detection from Gerber/production data (yet). Z-shift is
    computed automatically so the solder's lowest face sits exactly on the
    board's top surface, since getting that gap wrong even slightly would
    silently break the contact.

    "Component's own local (0, 0)" is handle.origin_xy, not the archive's
    raw (0, 0) -- confirmed empirically (scratch_debug_targe.py) that the
    CR0805 archive's node coordinates are still wherever the part sat on
    the partner's original demonstrator board (bbox center (10.0, 10.0),
    not (0, 0)), so the raw coordinates are recentered on handle.origin_xy
    before rotating/translating. Skipping this silently placed the
    component ~10mm off from (x_mm, y_mm) -- past the edge of this repo's
    20x15mm sample board entirely, which is what left the board-side
    TARGE170 contact selection empty (nothing at the component's assumed
    XY window actually overlapped the board).
    """
    if rotation_deg not in _VALID_ROTATIONS_DEG:
        raise ValueError(f"rotation_deg must be one of {_VALID_ROTATIONS_DEG}, got {rotation_deg}")

    z_bottom = solder_bottom_z(mapdl, solder_mat_id)
    dz = board_top_z_mm - z_bottom

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    origin_x, origin_y = handle.origin_xy

    mapdl.allsel()
    mapdl.nsel("S", "NODE", "", int(handle.node_ids.min()), int(handle.node_ids.max()))
    ids = mapdl.mesh.nnum
    coords = np.asarray(mapdl.mesh.nodes)
    x, y, z = coords[:, 0] - origin_x, coords[:, 1] - origin_y, coords[:, 2]
    x_new = x * cos_t - y * sin_t + x_mm
    y_new = x * sin_t + y * cos_t + y_mm
    z_new = z + dz

    # Nodes are redefined one at a time (N command has no "move this whole
    # set rigidly" form), so for most of the loop a given element has a mix
    # of already-moved and not-yet-moved nodes -- MAPDL checks element
    # Jacobians eagerly as nodes change and will raise on that transient,
    # geometrically-meaningless mid-update shape. The rigid transform
    # preserves every element's true shape once ALL its nodes are updated,
    # so these are false alarms for THIS specific operation.
    #
    # This eager per-node shape/Jacobian check (SHPP) is also the dominant
    # cost of this loop, not just the ~20811 individual N commands
    # themselves -- confirmed empirically (the earlier, SHPP-enabled
    # version of this loop produced a 75+ MB error log from repeated
    # Jacobian-violation messages on a full run, one recomputation-and-
    # possible-log-write per affected element per node move). SHPP,OFF
    # disables that checking for the duration of the loop; SHPP,ON restores
    # it afterwards so later, legitimate mesh operations still get checked
    # normally -- ignore_errors=True stays on too, as a safety net, but
    # with shape checking off there should be nothing left for it to catch.
    #
    # Chunked (rather than one non_interactive block for all ~20k nodes) so
    # there's a checkpoint to print progress from -- a single giant block
    # returns nothing to Python until every node in it is done, which for
    # the CR0805's ~20811 nodes reads as a long, silent hang with no way to
    # tell it apart from a real stall short of tailing MAPDL's own .out/.err
    # files directly (see simulate_pyansys_assembly.py's note on this).
    _PLACEMENT_CHUNK_SIZE = 2000
    n_total = len(ids)
    prior_ignore_errors = mapdl.ignore_errors
    mapdl.ignore_errors = True
    mapdl.shpp("OFF")
    try:
        for start in range(0, n_total, _PLACEMENT_CHUNK_SIZE):
            chunk = slice(start, start + _PLACEMENT_CHUNK_SIZE)
            with mapdl.non_interactive:
                for nid, xi, yi, zi in zip(ids[chunk], x_new[chunk], y_new[chunk], z_new[chunk]):
                    mapdl.n(int(nid), float(xi), float(yi), float(zi))
            print(f"  place_component: moved {min(start + _PLACEMENT_CHUNK_SIZE, n_total)}/{n_total} nodes",
                 flush=True)
    finally:
        mapdl.shpp("ON")
        mapdl.ignore_errors = prior_ignore_errors
    # Commands queued via non_interactive skip PyMAPDL's usual post-command
    # cache invalidation (confirmed empirically: mapdl.mesh.nodes/nnum kept
    # returning pre-transform coordinates after this loop, while a direct
    # mapdl.get_value() on the same node showed the database was actually
    # correct) -- force the mesh cache to refetch so callers relying on
    # mapdl.mesh.* after this see the real, moved positions.
    mapdl._reset_cache()
    mapdl.allsel()


def generate_bonded_contact(mapdl, board_top_z_mm: float, solder_mat_id: int,
                            targe_type_id: int, conta_type_id: int,
                            board_pad_mat_id: int, board_mat_ids: tuple[int, int],
                            xy_margin_mm: float = 0.3,
                            z_tol_mm: float = 1e-3) -> tuple[int, int]:
    """Bonded MPC contact between the board's top surface and the
    component's solder-bottom faces, using the TARGE170/CONTA174 type IDs
    carried over from the imported archive (already configured for
    KEYOPT(2)=2 MPC / KEYOPT(12)=5 Bonded-always). One combined contact
    pair covers both solder joints -- two disconnected islands in one
    contact pair is fine. Must be called after place_component(), which is
    what guarantees the solder-bottom face sits exactly at board_top_z_mm.
    board_mat_ids must be the board's *post-remap_board_materials* IDs.
    Returns (n_targe_elements, n_conta_elements).
    """
    mapdl.allsel()

    # Component side (CONTA174): solder elements' bottom face only --
    # ESEL narrows to the solder material, NSLE+NSEL narrows further to
    # just the nodes sitting on the bottom-face plane, and ESURF then only
    # skins faces whose nodes are entirely within that node selection.
    mapdl.esel("S", "MAT", "", solder_mat_id)
    mapdl.nsle("S")
    mapdl.nsel("R", "LOC", "Z", board_top_z_mm - z_tol_mm, board_top_z_mm + z_tol_mm)
    footprint = np.asarray(mapdl.mesh.nodes)
    if footprint.size == 0:
        raise RuntimeError("No solder-bottom nodes found at the board's top surface -- "
                           "check placement (solder may not actually be touching the board).")
    x_min, x_max = float(footprint[:, 0].min()), float(footprint[:, 0].max())
    y_min, y_max = float(footprint[:, 1].min()), float(footprint[:, 1].max())

    mapdl.esel("S", "MAT", "", solder_mat_id)
    mapdl.nsle("S")
    mapdl.nsel("R", "LOC", "Z", board_top_z_mm - z_tol_mm, board_top_z_mm + z_tol_mm)
    mapdl.type(conta_type_id)
    mapdl.mat(solder_mat_id)
    mapdl.real(1)
    mapdl.esurf()
    mapdl.esel("S", "TYPE", "", conta_type_id)
    n_conta = int(mapdl.mesh.enum.size)

    # Board side (TARGE170): top-surface elements under the solder
    # footprint (XY window with a small margin -- standard practice for
    # the target surface to be slightly larger than the contact source).
    # board_mat_ids selects the board's own (post-remap) elements (copper +
    # FR4, both can appear at the top surface depending on the raster);
    # board_pad_mat_id below is just the attribute pointer stamped on the
    # new TARGE170 elements, unrelated to the selection.
    mapdl.esel("S", "MAT", "", min(board_mat_ids), max(board_mat_ids))
    mapdl.esel("R", "CENT", "X", x_min - xy_margin_mm, x_max + xy_margin_mm)
    mapdl.esel("R", "CENT", "Y", y_min - xy_margin_mm, y_max + xy_margin_mm)
    mapdl.nsle("S")
    mapdl.nsel("R", "LOC", "Z", board_top_z_mm - z_tol_mm, board_top_z_mm + z_tol_mm)
    mapdl.type(targe_type_id)
    mapdl.mat(board_pad_mat_id)
    mapdl.real(1)
    mapdl.esurf()
    mapdl.esel("S", "TYPE", "", targe_type_id)
    n_targe = int(mapdl.mesh.enum.size)

    mapdl.allsel()
    return n_targe, n_conta
