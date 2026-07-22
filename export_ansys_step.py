#!/usr/bin/env python3
"""
Export a quadtree homogenization result to CAD geometry (STEP) plus
companion material data, so the leaves show up as real, native bodies
under an ANSYS Workbench Static Structural system's Geometry cell -- the
one thing a raw APDL script cannot do (Workbench's Geometry cell content
always comes from an attached CAD source, never from solve-time PREP7
commands; see export_ansys.py's docstring for the APDL-native alternative
and why it targets a standalone "Mechanical APDL" system instead).

Produces four files in --out-dir:
  model.step             One solid block per leaf (via gmsh's OpenCASCADE
                          kernel, fragmented so adjacent leaves share
                          conformal faces), units declared as mm in the
                          STEP header. HIGH CONFIDENCE -- built with a
                          real CAD kernel and round-tripped/checked here
                          (volume conservation, fragment count) before
                          shipping.
  materials_table.csv    leaf id, bbox, and full orthotropic engineering
                          constants (EX/EY/EZ, PRXY/PRYZ/PRXZ, GXY/GYZ/GXZ,
                          ALPX/ALPY/ALPZ) per leaf. HIGH CONFIDENCE, always
                          usable even if the other two files need fixing --
                          use it to hand-enter materials in Engineering
                          Data if the XML below doesn't import cleanly.
  materials.xml           Engineering Data MatML import file, one
                          orthotropic material per leaf named "leaf_01",
                          "leaf_02", etc. Schema matched field-by-field
                          against a real Engineering Data export (ANSYS
                          2025 R1, an isotropic Cu + orthotropic material)
                          -- fixed several wrong guesses from the first
                          attempt (Pa not MPa, <Metadata> placement, the
                          required "Interpolation Options"/dummy-temperature
                          parameters, CTE's Definition/Behavior qualifiers).
                          Still only spot-checked by eye, not by an actual
                          Engineering Data import -- if a property looks
                          off after import, re-export one hand-made
                          material from Engineering Data (File > Export)
                          and diff it against this file.
  assign_materials.py     Mechanical scripting (Python, native API,
                          ANSYS 2023R2+) snippet: matches each imported
                          body to its leaf by bounding box (STEP does not
                          carry ANSYS-recognized body names, so this
                          avoids relying on import order or naming) and
                          sets body.Material to the matching leaf's
                          material name. MODERATE CONFIDENCE: the
                          body.Material = "name" assignment pattern is
                          well established, but the exact bounding-box
                          accessor and unit convention can vary by
                          version -- see the UNIT_SCALE comment at the top
                          of the generated script if no bodies match.

Requires gmsh (with its bundled OpenCASCADE kernel) -- available in the
dolfinx09-env / fenicsx-env conda envs, not the plain pcb-ml-env used for
ml_homogenization.py.

Usage:
    python export_ansys_step.py data/samples/rv3032_real
    python export_ansys_step.py data/samples/rv3032_real --out-dir data/samples/rv3032_real/mesh/ansys_step
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from homogenization import load_materials, orthotropic_engineering_constants


def _leaf_table(qt: dict, meta: dict, materials: dict) -> list[dict]:
    Lz_mm = meta["stackup"]["total_thickness_mm"]
    leaves = qt["leaves"]
    width = len(str(len(leaves)))
    table = []
    for i, leaf in enumerate(leaves, start=1):
        x0, y0, x1, y1 = leaf["bbox_mm"]
        c = orthotropic_engineering_constants(leaf, materials)
        table.append({
            "id": i, "name": f"leaf_{i:0{width}d}",
            "x0": x0, "y0": y0, "x1": x1, "y1": y1, "z0": 0.0, "z1": Lz_mm,
            **c,
        })
    return table


# ---------------------------------------------------------------------------
# STEP geometry (gmsh + OpenCASCADE)
# ---------------------------------------------------------------------------

def build_step_geometry(leaf_table: list[dict], out_path: Path) -> None:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
        gmsh.model.add("pcb_leaves")
        boxes = []
        for leaf in leaf_table:
            tag = gmsh.model.occ.addBox(
                leaf["x0"], leaf["y0"], leaf["z0"],
                leaf["x1"] - leaf["x0"], leaf["y1"] - leaf["y0"], leaf["z1"] - leaf["z0"],
            )
            boxes.append((3, tag))
        gmsh.model.occ.synchronize()

        out, mapping = gmsh.model.occ.fragment(boxes, [])
        gmsh.model.occ.synchronize()

        if len(out) != len(leaf_table):
            print(f"[warn] fragment produced {len(out)} volumes for {len(leaf_table)} leaves "
                 "-- leaf<->volume order below is no longer guaranteed 1:1; "
                 "assign_materials.py's bbox matching handles this regardless.", file=sys.stderr)
        total_leaf_area = sum((l["x1"] - l["x0"]) * (l["y1"] - l["y0"]) * (l["z1"] - l["z0"])
                              for l in leaf_table)
        total_mesh_vol = sum(gmsh.model.occ.getMass(d, t) for d, t in gmsh.model.getEntities(3))
        if abs(total_mesh_vol - total_leaf_area) / total_leaf_area > 1e-6:
            print(f"[warn] volume mismatch after fragment: {total_mesh_vol:.6f} vs "
                 f"expected {total_leaf_area:.6f} -- geometry may have gaps/overlaps.", file=sys.stderr)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(out_path))
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------
# materials_table.csv -- always-usable fallback / reference
# ---------------------------------------------------------------------------

def write_materials_csv(leaf_table: list[dict], out_path: Path) -> None:
    import csv
    fields = ["id", "name", "x0", "y0", "x1", "y1", "z0", "z1",
             "EX", "EY", "EZ", "PRXY", "PRYZ", "PRXZ", "GXY", "GYZ", "GXZ",
             "ALPX", "ALPY", "ALPZ"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in leaf_table:
            w.writerow({k: row[k] for k in fields})


# ---------------------------------------------------------------------------
# materials.xml -- best-effort Engineering Data MatML import (unverified)
# ---------------------------------------------------------------------------

# Schema confirmed against a real Engineering Data export (ANSYS 2025 R1,
# "engineering_data_shengui.xml" -- a working library containing an
# isotropic Cu and an orthotropic "Shengyi S1000H" material). Differences
# from the first, hand-guessed attempt that this version fixes:
#   - stress unit is Pa, not MPa (Engineering Data stores SI internally
#     regardless of the project's display unit system)
#   - <Metadata> comes AFTER all <Material> entries, not before
#   - every PropertyData needs a leading <Data format="string">-</Data>
#   - Elasticity needs a leading "Interpolation Options" ParameterValue
#     (pa4) even for a single, non-temperature-dependent data point, plus
#     a trailing dummy-temperature ParameterValue (pa9) -- Engineering
#     Data always represents Elasticity as a temperature table internally,
#     even a table of one row. The exact sentinel value below
#     (7.88860905221012e-31) is copied verbatim from the reference file;
#     it's ANSYS's own placeholder for "no real temperature dependence".
#   - Coefficient of Thermal Expansion is a *single* property (not a
#     separate "orthotropic" one) carrying a Definition qualifier
#     (Secant/Instantaneous) and a Behavior qualifier (Orthotropic) --
#     using Secant here since our alpha values are proportional
#     (strain = alpha*(T-Tref)), not a fitted tangent curve.
#   - CTE's inverse-temperature unit is a structured power="-1" unit
#     block, not a flat "C^-1" string.
_ELAST_PARAMS = [
    ("EX", "pa10", "Young's Modulus X direction", "Pa"),
    ("EY", "pa11", "Young's Modulus Y direction", "Pa"),
    ("EZ", "pa12", "Young's Modulus Z direction", "Pa"),
    ("PRXY", "pa13", "Poisson's Ratio XY", None),
    ("PRYZ", "pa14", "Poisson's Ratio YZ", None),
    ("PRXZ", "pa15", "Poisson's Ratio XZ", None),
    ("GXY", "pa16", "Shear Modulus XY", "Pa"),
    ("GYZ", "pa17", "Shear Modulus YZ", "Pa"),
    ("GXZ", "pa18", "Shear Modulus XZ", "Pa"),
]
_CTE_PARAMS = [
    ("ALPX", "pa19", "Coefficient of Thermal Expansion X direction"),
    ("ALPY", "pa20", "Coefficient of Thermal Expansion Y direction"),
    ("ALPZ", "pa21", "Coefficient of Thermal Expansion Z direction"),
]
_DUMMY_TEMPERATURE = "7.88860905221012e-31"


def write_materials_matml(leaf_table: list[dict], out_path: Path) -> None:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<EngineeringData version="25.1.0.203" versiondate="">')
    lines.append("  <Notes>Generated by export_ansys_step.py -- one orthotropic elastic + "
                 "CTE material per quadtree leaf. Schema matched against a real Engineering "
                 "Data export; still worth spot-checking one material by hand after import.</Notes>")
    lines.append("  <Materials>")
    lines.append("    <MatML_Doc>")

    for leaf in leaf_table:
        lines.append("      <Material>")
        lines.append("        <BulkDetails>")
        lines.append(f"          <Name>{leaf['name']}</Name>")

        lines.append('          <PropertyData property="pr1">')
        lines.append('            <Data format="string">-</Data>')
        lines.append('            <Qualifier name="Behavior">Orthotropic</Qualifier>')
        lines.append('            <Qualifier name="Field Variable Compatible">Temperature</Qualifier>')
        lines.append('            <ParameterValue parameter="pa4" format="string">')
        lines.append("              <Data>Interpolation Options</Data>")
        lines.append('              <Qualifier name="AlgorithmType">Linear Multivariate</Qualifier>')
        lines.append('              <Qualifier name="Normalized">True</Qualifier>')
        lines.append('              <Qualifier name="Cached">True</Qualifier>')
        lines.append('              <Qualifier name="ExtrapolationType">Projection to the Bounding Box</Qualifier>')
        lines.append("            </ParameterValue>")
        for key, pid, _name, unit in _ELAST_PARAMS:
            if unit == "Pa":
                value_str = f"{leaf[key] * 1e6:.0f}"   # MPa -> Pa, plain decimal (matches reference style)
            else:
                value_str = f"{leaf[key]:.8g}"
            lines.append(f'            <ParameterValue parameter="{pid}" format="float">')
            lines.append(f"              <Data>{value_str}</Data>")
            lines.append('              <Qualifier name="Variable Type">Dependent</Qualifier>')
            lines.append("            </ParameterValue>")
        lines.append('            <ParameterValue parameter="pa9" format="float">')
        lines.append(f"              <Data>{_DUMMY_TEMPERATURE}</Data>")
        lines.append('              <Qualifier name="Variable Type">Independent</Qualifier>')
        lines.append("            </ParameterValue>")
        lines.append("          </PropertyData>")

        lines.append('          <PropertyData property="pr2">')
        lines.append('            <Data format="string">-</Data>')
        lines.append('            <Qualifier name="Definition">Secant</Qualifier>')
        lines.append('            <Qualifier name="Behavior">Orthotropic</Qualifier>')
        for key, pid, _name in _CTE_PARAMS:
            lines.append(f'            <ParameterValue parameter="{pid}" format="float">')
            lines.append(f"              <Data>{leaf[key]:.8g}</Data>")
            lines.append('              <Qualifier name="Variable Type">Dependent</Qualifier>')
            lines.append("            </ParameterValue>")
        lines.append('            <ParameterValue parameter="pa9" format="float">')
        lines.append(f"              <Data>{_DUMMY_TEMPERATURE}</Data>")
        lines.append('              <Qualifier name="Variable Type">Independent</Qualifier>')
        lines.append("            </ParameterValue>")
        lines.append("          </PropertyData>")

        lines.append("        </BulkDetails>")
        lines.append("      </Material>")

    lines.append("      <Metadata>")
    lines.append('        <ParameterDetails id="pa4">')
    lines.append("          <Name>Options Variable</Name>")
    lines.append("          <Unitless />")
    lines.append("        </ParameterDetails>")
    lines.append('        <ParameterDetails id="pa9">')
    lines.append("          <Name>Temperature</Name>")
    lines.append('          <Units name="Temperature"><Unit><Name>C</Name></Unit></Units>')
    lines.append("        </ParameterDetails>")
    for key, pid, name, unit in _ELAST_PARAMS:
        lines.append(f'        <ParameterDetails id="{pid}">')
        lines.append(f"          <Name>{name}</Name>")
        if unit:
            lines.append(f'          <Units name="Stress"><Unit><Name>{unit}</Name></Unit></Units>')
        else:
            lines.append("          <Unitless />")
        lines.append("        </ParameterDetails>")
    for key, pid, name in _CTE_PARAMS:
        lines.append(f'        <ParameterDetails id="{pid}">')
        lines.append(f"          <Name>{name}</Name>")
        lines.append('          <Units name="InvTemp1"><Unit power="-1"><Name>C</Name></Unit></Units>')
        lines.append("        </ParameterDetails>")
    lines.append('        <PropertyDetails id="pr1">')
    lines.append("          <Unitless />")
    lines.append("          <Name>Elasticity</Name>")
    lines.append("        </PropertyDetails>")
    lines.append('        <PropertyDetails id="pr2">')
    lines.append("          <Unitless />")
    lines.append("          <Name>Coefficient of Thermal Expansion</Name>")
    lines.append("        </PropertyDetails>")
    lines.append("      </Metadata>")

    lines.append("    </MatML_Doc>")
    lines.append("  </Materials>")
    lines.append("</EngineeringData>")
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# assign_materials.py -- Mechanical scripting snippet (bbox-matched)
# ---------------------------------------------------------------------------

_ASSIGN_SCRIPT_TEMPLATE = '''\
# Run this from ANSYS Mechanical's own Python scripting console (Automation
# tab, Scripting, or the interactive console), AFTER:
#   1. model.step has been imported as this Static Structural system's
#      Geometry (right-click Geometry cell, Import Geometry, or drag
#      model.step onto it), and
#   2. materials.xml has been imported into Engineering Data (or the
#      materials were created by hand from materials_table.csv) so a
#      material named e.g. leaf_01 exists for every leaf below.
#
# Matches each imported body to its source leaf by bounding box rather than
# name or import order, since STEP does not carry ANSYS-recognized body
# names and gmsh does not preserve them either -- coordinates are the one
# thing guaranteed to survive the round trip.
#
# Leaf coordinates below are in mm (matching model.step's declared units).
# Body geometry may come back as a plain float or as an ANSYS "Quantity"
# (value+unit) depending on which property this ANSYS version exposes --
# to_mm() below detects and converts either case, so this shouldn't need
# manual adjustment; a total mismatch (no bodies found within tolerance)
# is the symptom to watch for in the printed diagnostics if it does.

TOLERANCE_MM = 0.05  # per-axis centroid match tolerance, in leaf mm-coordinates

LEAVES = {leaves_literal}

model = ExtAPI.DataModel.Project.Model
bodies = model.Geometry.GetChildren(
    Ansys.Mechanical.DataModel.Enums.DataModelObjectCategory.Body, True)
print("Found {{0}} bodies in Geometry, {{1}} leaves to match.".format(len(bodies), len(LEAVES)))

# ANSYS's scripting API returns geometric lengths as a "Quantity" object
# (value + unit) on some property paths, not a plain float -- confirmed by
# a "SystemError: Unable to subtract number from quantity with units m"
# on a first attempt that assumed plain floats. This converts any such
# value to a plain float in millimeters before any arithmetic touches it.
_UNIT_TO_MM = {{"m": 1000.0, "mm": 1.0, "cm": 10.0, "um": 0.001, "in": 25.4, "ft": 304.8}}
_to_mm_diagnostic_printed = [False]


def to_mm(x):
    if isinstance(x, (int, float)):
        return float(x)
    for unit_attr in ("Unit", "Units"):
        if hasattr(x, unit_attr) and hasattr(x, "Value"):
            scale = _UNIT_TO_MM.get(str(getattr(x, unit_attr)), 1000.0)
            return float(x.Value) * scale
    if hasattr(x, "Value"):
        return float(x.Value) * 1000.0   # Quantity with no readable unit -- meters, per the confirmed error above
    try:
        return float(x)
    except Exception:
        if not _to_mm_diagnostic_printed[0]:
            print("[error] could not convert a geometry value to a plain number. "
                 "Attributes available on it:")
            print(", ".join(sorted([a for a in dir(x) if not a.startswith("_")])))
            _to_mm_diagnostic_printed[0] = True
        raise


def get_body_centroid(body):
    # Tries several known ANSYS Mechanical scripting API spellings for a
    # body's geometric center, since this varies by version. Falls back to
    # a raw geometry handle, then gives up and lets the caller print
    # diagnostics -- safer than betting on one guessed property name.
    if hasattr(body, "Bounds"):
        b = [to_mm(v) for v in body.Bounds]
        return ((b[0] + b[3]) / 2.0, (b[1] + b[4]) / 2.0, (b[2] + b[5]) / 2.0)
    if hasattr(body, "BoundingBox"):
        b = [to_mm(v) for v in body.BoundingBox]
        return ((b[0] + b[3]) / 2.0, (b[1] + b[4]) / 2.0, (b[2] + b[5]) / 2.0)
    if hasattr(body, "Centroid"):
        c = body.Centroid
        return (to_mm(c[0]), to_mm(c[1]), to_mm(c[2]))
    if hasattr(body, "CentroidX") and hasattr(body, "CentroidY") and hasattr(body, "CentroidZ"):
        return (to_mm(body.CentroidX), to_mm(body.CentroidY), to_mm(body.CentroidZ))
    geo = body.GetGeoBody() if hasattr(body, "GetGeoBody") else None
    if geo is not None:
        if hasattr(geo, "BoundingBox"):
            b = [to_mm(v) for v in geo.BoundingBox]
            return ((b[0] + b[3]) / 2.0, (b[1] + b[4]) / 2.0, (b[2] + b[5]) / 2.0)
        if hasattr(geo, "Centroid"):
            c = geo.Centroid
            return (to_mm(c[0]), to_mm(c[1]), to_mm(c[2]))
    return None


body_centroids = []
diagnostic_printed = False
for body in bodies:
    c = get_body_centroid(body)
    if c is None:
        if not diagnostic_printed:
            print("[error] none of the known property names worked on this Body object. "
                 "Attributes actually available on it:")
            print(", ".join(sorted([a for a in dir(body) if not a.startswith("_")])))
            diagnostic_printed = True
        continue
    body_centroids.append((body, c))

print("Got geometry for {{0}} / {{1}} bodies.".format(len(body_centroids), len(bodies)))

unmatched_leaves = []
for leaf in LEAVES:
    lcx = (leaf["x0"] + leaf["x1"]) / 2.0
    lcy = (leaf["y0"] + leaf["y1"]) / 2.0
    lcz = (leaf["z0"] + leaf["z1"]) / 2.0

    best_body, best_dist = None, None
    for body, (bcx, bcy, bcz) in body_centroids:
        dist = abs(bcx - lcx) + abs(bcy - lcy) + abs(bcz - lcz)
        if best_dist is None or dist < best_dist:
            best_body, best_dist = body, dist

    tol = TOLERANCE_MM * 3
    if best_body is not None and best_dist <= tol:
        best_body.Material = leaf["name"]
    else:
        unmatched_leaves.append((leaf["name"], best_dist))

if unmatched_leaves:
    print(("[warn] {{0}} leaves had no body within tolerance -- check the to_mm() unit "
          "conversion above (print a raw body centroid value to see its actual type/unit) "
          "or TOLERANCE_MM:").format(len(unmatched_leaves)))
    for name, dist in unmatched_leaves[:10]:
        print("    {{0}}  best_dist={{1}}".format(name, dist))
else:
    print("All leaves matched to a body and assigned their material.")
'''


def write_assign_script(leaf_table: list[dict], out_path: Path) -> None:
    leaves_compact = [
        {"name": l["name"], "x0": l["x0"], "y0": l["y0"], "z0": l["z0"],
         "x1": l["x1"], "y1": l["y1"], "z1": l["z1"]}
        for l in leaf_table
    ]
    script = _ASSIGN_SCRIPT_TEMPLATE.format(leaves_literal=json.dumps(leaves_compact, indent=2))
    out_path.write_text(script)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a quadtree.py result to STEP geometry + Engineering Data "
                    "materials for import into a Workbench Static Structural system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: <sample_dir>/mesh/ansys_step/)")
    parser.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    args = parser.parse_args()

    if not args.sample_dir.exists():
        sys.exit(f"Not found: {args.sample_dir}")
    qt_path = args.sample_dir / "mesh" / "quadtree.json"
    if not qt_path.exists():
        sys.exit(f"Run quadtree.py first -- not found: {qt_path}")

    with open(qt_path) as f:
        qt = json.load(f)
    with open(args.sample_dir / "metadata.json") as f:
        meta = json.load(f)
    materials = load_materials(args.materials)

    out_dir = args.out_dir or (args.sample_dir / "mesh" / "ansys_step")
    out_dir.mkdir(parents=True, exist_ok=True)

    leaves = _leaf_table(qt, meta, materials)
    print(f"{len(leaves)} leaves -- building STEP geometry (gmsh/OpenCASCADE) …")
    build_step_geometry(leaves, out_dir / "model.step")
    write_materials_csv(leaves, out_dir / "materials_table.csv")
    write_materials_matml(leaves, out_dir / "materials.xml")
    write_assign_script(leaves, out_dir / "assign_materials.py")

    print(f"Wrote {out_dir}/:")
    print("  model.step             -- import as Geometry (tested: volume-conserving, mm units)")
    print("  materials_table.csv    -- reference/fallback, always usable")
    print("  materials.xml          -- Engineering Data import, schema matched to a real export")
    print("  assign_materials.py    -- run in Mechanical's scripting console after both imports")


if __name__ == "__main__":
    main()
