#!/usr/bin/env python3
"""
Board + CR0805 resistor + SAC305 solder assembly, via PyMAPDL: builds the
board exactly as simulate_pyansys_benchmark.py does (untouched, reused
directly), imports the partner-provided CR0805 component archive
(data/components/cr0805/DemonstratorB_CR0805_Export.cdb) without
remeshing it, places it manually at a given board (x, y) with the solder
automatically dropped onto the board's top surface, attaches it to the
board via MPC bonded contact (component_import.generate_bonded_contact,
reusing the TARGE170/CONTA174 KEYOPT template already in the archive), and
applies the SAC305 solder material from
data/materials/sac305/SAC305_MikroVal.mac.

Solves the full reflow-to-thermal-cycling temperature history from
--thermal-profile (default configs/thermal_profile.yaml): one ramped
static load step per consecutive pair of (time, temperature) points in
that profile (see thermal_profile.py), with TREF set to the profile's
first point (the reflow peak -- the stress-free state solder solidifies
at). Creep is OFF by default (RATE,OFF, issued right after entering
/SOLU) so this is an elastic-only check of the assembly -- the solder's
TB,CREEP table from --solder-mac stays defined either way, just not
evaluated unless --creep is given (pass it for a real thermal-cycling
fatigue study, since that's the whole point of a multi-segment profile
like this one rather than a single ramp).
Results (.db + .rst) are copied to <sample_dir>/results/ so they can be
opened in a separately-launched, standalone ANSYS Mechanical APDL GUI
(mapdl.open_gui() itself is NOT used here -- see --inspect's note below on
why it crashes in this environment), or read directly and plotted via
plot_creep_strain_history.py without a live MAPDL session at all.

Placement: by default the component is centred on the PAD_1/PAD_2 midpoint
from solder_reliability.py and rotated onto the pad axis; --place-x-mm/
--place-y-mm/--rotation-deg override that. Reading pads from Gerber/production
data automatically is a likely follow-up, not built here.

--inspect stops right after placement (before contact generation/solve)
and opens an interactive PyVista/VTK plot (mapdl.eplot()) of the live
database, so placement/orientation/solder-to-board gap can be checked
visually before committing to the (currently unreliable -- see git log/
conversation notes on the static solve grinding on distorted elements
near the top of the component's ID range) solve step.

This uses PyMAPDL's own VTK-based eplot(), NOT mapdl.open_gui() (the
legacy Win32 APDL GUI) -- confirmed empirically that open_gui() crashes
on this machine (an "Intel(r) Visual Fortran run-time error" dialog right
as its OpenGL window tries to initialize), consistent with running over
an Amazon DCV remote session rather than a GPU-accelerated local one.
open_gui() also kills and relaunches the whole MAPDL session just to
attach the legacy GUI to a saved database -- eplot() does neither, so
it's both more reliable here and lighter-weight.

Usage:
    python simulate_pyansys_assembly.py data/samples/rv3032_example
    python simulate_pyansys_assembly.py data/samples/rv3032_example --place-x-mm 15 --place-y-mm 12.375 --rotation-deg 90
    python simulate_pyansys_assembly.py data/samples/rv3032_example --inspect
    python simulate_pyansys_assembly.py data/samples/rv3032_example --creep --thermal-profile configs/thermal_profile.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from component_import import (
    apply_solder_material,
    compute_safe_offsets,
    generate_bonded_contact,
    import_component,
    place_component,
    reapply_component_materials,
    reassert_board_materials,
    remap_board_materials,
)
from homogenization import load_materials
from quadtree import _load_layer_maps
from simulate_pyansys_benchmark import (
    _CU_MAT_ID,
    _FR4_MAT_ID,
    apply_321_fixture,
    apply_homogenized_board,
    apply_homogenized_board_zlayers,
    assign_material_properties,
    build_mesh_and_materials,
)
from solder_reliability import PAD_1_XY_MM, PAD_2_XY_MM
from thermal_profile import apply_thermal_profile, load_thermal_profile, plot_thermal_profile

# From inspecting DemonstratorB_CR0805_Export.cdb directly (see
# data/components/cr0805/MANIFEST.md for the full derivation): the file's
# own MAT/TYPE IDs, used as-is -- confirmed empirically that NUMOFF does
# NOT shift these for this archive format (see component_import.py's
# module docstring), so there's no offset arithmetic to apply here.
_CR0805_SOLDER_MAT = 9
_CR0805_TARGE_TYPE = 10
_CR0805_CONTA_TYPE = 11

# MAT ids from the archive's own material table that are actually
# referenced by imported elements -- determined empirically (ESEL by MAT
# across the imported element range); 1, 3, 4, 5, 7, 8, 10 exist in the
# archive's material table but aren't used by any of its elements, so they
# don't need reapply_component_materials()'s CDREAD-misparse workaround.
_CR0805_USED_MAT_IDS = (2, 6, 9)

# Where the board's own copper/FR4 materials get relocated before the
# component is imported, since the component's own MAT 1-10 would
# otherwise silently overwrite them (see remap_board_materials()).
_BOARD_CU_MAT_REMAPPED = 9001
_BOARD_FR4_MAT_REMAPPED = 9002

_DEFAULT_CDB_PATH = Path("data/components/cr0805/DemonstratorB_CR0805_Export.cdb")
_DEFAULT_MAC_PATH = Path("data/materials/sac305/SAC305_MikroVal.mac")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Board + CR0805 + SAC305 solder assembly (MPC bonded contact), via PyMAPDL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--nz-per-layer", type=int, default=2)
    parser.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    parser.add_argument("--cdb", type=Path, default=_DEFAULT_CDB_PATH)
    parser.add_argument("--solder-mac", type=Path, default=_DEFAULT_MAC_PATH)
    parser.add_argument("--place-x-mm", type=float, default=None,
                        help="Component's local (0,0) target X on the board; defaults to the "
                             "PAD_1/PAD_2 midpoint from solder_reliability.py")
    parser.add_argument("--place-y-mm", type=float, default=None,
                        help="Component's local (0,0) target Y on the board; see --place-x-mm")
    parser.add_argument("--rotation-deg", type=int, choices=(0, 90), default=None,
                        help="In-plane rotation about the component's local origin. Default: "
                             "aligned with the PAD_1/PAD_2 axis from solder_reliability.py "
                             "(the archive's component lies along x, so pads along y -> 90)")
    parser.add_argument("--inspect", action="store_true",
                        help="Open an interactive VTK plot (mapdl.eplot()) right after "
                             "placement, then exit without generating contact or solving "
                             "(rerun without --inspect once placement looks right)")
    parser.add_argument("--thermal-profile", type=Path, default=Path("configs/thermal_profile.yaml"),
                        help="Time/temperature profile YAML (see thermal_profile.py) -- one ramped "
                             "load step per consecutive pair of points; the first point's "
                             "temperature is used as TREF (stress-free reference)")
    parser.add_argument("--creep", action="store_true",
                        help="Leave creep active (RATE,ON, as authored by --solder-mac) instead "
                             "of the default elastic-only check (RATE,OFF)")
    parser.add_argument("--creep-ratio-limit", type=float, default=None,
                        help="Force a creep ratio limit (CUTCONTROL,CRPLIMIT,<value>,1; only with "
                             "--creep): max creep-strain-increment/elastic-strain ratio per "
                             "substep before MAPDL cuts the time step back. Default: not issued, "
                             "so MAPDL's automatic time stepping applies (no creep ratio control "
                             "for implicit creep). The trailing 1 is required for implicit creep "
                             "-- without it MAPDL silently ignores the limit.")
    parser.add_argument("--mapdl-dir", type=Path, default=Path("mapdl_run"),
                        help="MAPDL working directory (job files, incl. file0.out/file0.err for "
                             "live progress -- see the tail command printed at startup). Fixed "
                             "and persistent, unlike launch_mapdl()'s default random temp "
                             "directory, which also gets deleted on exit.")
    parser.add_argument("--board-model", choices=["projection", "homogenized", "homogenized_zlayers"],
                        default="projection",
                        help="projection: each board element is pure copper or FR4 from the raster "
                             "at its centre. homogenized: each board element gets the orthotropic "
                             "Voigt-Reuss material of the quadtree leaf containing it, smeared through "
                             "the full thickness. homogenized_zlayers: z-resolved -- each of the two "
                             "copper layers gets its OWN per-leaf homogenized material (from phi_bot / "
                             "phi_top alone), the FR4 core stays plain FR4 untouched (it never carries "
                             "copper). Both homogenized modes need mesh/quadtree.json from quadtree.py.")
    parser.add_argument("--nproc", type=int, default=os.cpu_count(),
                        help="Distributed MAPDL (DMP) core count -- defaults to all logical cores "
                             "on this machine (launch_mapdl()'s own default is just 2). Your ANSYS "
                             "license may cap usable HPC parallelism below this; MAPDL falls back "
                             "on its own if so, which isn't a bug in this script.")
    args = parser.parse_args()

    sample_dir = args.sample_dir
    if not sample_dir.exists():
        sys.exit(f"Not found: {sample_dir}")
    if not args.cdb.exists():
        sys.exit(f"Not found: {args.cdb}")
    if not args.solder_mac.exists():
        sys.exit(f"Not found: {args.solder_mac}")
    if not args.thermal_profile.exists():
        sys.exit(f"Not found: {args.thermal_profile}")
    if (args.board_model in ("homogenized", "homogenized_zlayers")
            and not (sample_dir / "mesh" / "quadtree.json").exists()):
        sys.exit(f"--board-model {args.board_model} needs {sample_dir / 'mesh' / 'quadtree.json'} "
                 "-- run quadtree.py first")
    # Resolve to absolute before any MAPDL command (e.g. apply_solder_material's
    # /INPUT) sees them -- confirmed empirically that a relative path here
    # silently fails ("File ... does not exist. The /INPUT command is
    # ignored.") on a worker rank once truly distributed across multiple
    # processes (--nproc > 2), since a worker's own working directory isn't
    # guaranteed to match the one this script itself was launched from. Only
    # matters for paths a raw MAPDL command reads server-side; --materials/
    # --thermal-profile are read client-side in Python and are unaffected,
    # but resolving them too is harmless and one less thing to reason about.
    args.cdb = args.cdb.resolve()
    args.solder_mac = args.solder_mac.resolve()
    args.materials = args.materials.resolve()
    args.thermal_profile = args.thermal_profile.resolve()
    with open(sample_dir / "metadata.json") as f:
        meta = json.load(f)

    layer_maps = _load_layer_maps(sample_dir)
    materials = load_materials(args.materials)
    thermal_profile = load_thermal_profile(args.thermal_profile)
    print(f"Thermal profile: {thermal_profile.time_s.size} points, "
         f"{thermal_profile.time_s[0]:.0f}-{thermal_profile.time_s[-1]:.0f} s, "
         f"reference (reflow) temp={thermal_profile.reference_temp_c:.1f} C")
    # Record the applied load next to the results before solving: a copy of
    # the profile (plot_cycle_creep_increment.py takes the cycle boundaries
    # from it) and its plot, so the load can be checked while MAPDL runs.
    results_dir = sample_dir / "results"
    results_dir.mkdir(exist_ok=True)
    shutil.copyfile(args.thermal_profile, results_dir / "thermal_profile.yaml")
    plot_thermal_profile(thermal_profile, results_dir / "thermal_profile.png")
    print(f"Wrote {results_dir / 'thermal_profile.yaml'}\nWrote {results_dir / 'thermal_profile.png'}\n")

    Lx_mm = meta["board"]["width_mm"]
    Ly_mm = meta["board"]["height_mm"]
    z_cu_bot_mm = meta["stackup"]["copper_thickness_mm"]
    z_cu_top_mm = z_cu_bot_mm + meta["stackup"]["dielectric_thickness_mm"]
    Lz_mm = meta["stackup"]["total_thickness_mm"]

    place_x = args.place_x_mm if args.place_x_mm is not None else (PAD_1_XY_MM[0] + PAD_2_XY_MM[0]) / 2
    place_y = args.place_y_mm if args.place_y_mm is not None else (PAD_1_XY_MM[1] + PAD_2_XY_MM[1]) / 2
    if args.rotation_deg is None:
        # the archive's component has its length along x; turn it onto the pad axis
        pads_along_y = abs(PAD_1_XY_MM[1] - PAD_2_XY_MM[1]) > abs(PAD_1_XY_MM[0] - PAD_2_XY_MM[0])
        args.rotation_deg = 90 if pads_along_y else 0

    print(f"Board {Lx_mm:.1f} x {Ly_mm:.1f} mm, top surface z={Lz_mm:.4f} mm")
    print(f"Placing component local (0,0) at ({place_x:.3f}, {place_y:.3f}) mm, "
         f"rotation={args.rotation_deg} deg\n")

    args.mapdl_dir.mkdir(parents=True, exist_ok=True)
    mapdl_run_location = str(args.mapdl_dir.resolve())

    from ansys.mapdl.core import launch_mapdl
    # override=True: deletes a stale file.lock left behind if a previous
    # run into this same --mapdl-dir crashed/was killed without exiting
    # cleanly (a fixed, reused directory makes that collision far more
    # likely than launch_mapdl()'s default fresh-random-temp-dir behavior).
    # nproc=args.nproc: distributed-memory-parallel (DMP) MAPDL across
    # this many cores, instead of launch_mapdl()'s own default of just 2.
    mapdl = launch_mapdl(run_location=mapdl_run_location, override=True, nproc=args.nproc)
    print(f"MAPDL working directory: {mapdl.directory}")
    print(f"MAPDL running distributed across {args.nproc} core(s) "
         f"(this machine has {os.cpu_count()} logical cores available; "
         f"your ANSYS license may cap usable HPC parallelism lower -- check the "
         f"'PARALLEL PROCESSES REQUESTED'/'CORES REQUESTED' lines MAPDL prints at "
         f"startup against what actually got used if unsure)")
    print(f"  To watch solver progress live (substeps/equilibrium iterations, warnings) "
         f"independently of this script's own prints, tail that directory's job files in "
         f"a separate terminal, e.g.:\n"
         f"    Get-Content -Path \"{mapdl.directory}\\{mapdl.jobname}0.out\" -Wait -Tail 50\n"
         f"Unlike the default temp directory, this one is NOT deleted when the script exits, "
         f"so the log stays available for review afterwards too.\n")
    try:
        node_ids = build_mesh_and_materials(
            mapdl, layer_maps, Lx_mm, Ly_mm, z_cu_bot_mm, z_cu_top_mm, Lz_mm,
            args.nx, args.ny, args.nz_per_layer)
        assign_material_properties(mapdl, materials)
        apply_321_fixture(mapdl, node_ids)
        print(f"Board: {mapdl.mesh.nnum_all.size} nodes, {mapdl.mesh.enum_all.size} elements\n")

        remap_board_materials(mapdl, {
            _CU_MAT_ID: _BOARD_CU_MAT_REMAPPED,
            _FR4_MAT_ID: _BOARD_FR4_MAT_REMAPPED,
        })
        print(f"Board materials remapped: copper {_CU_MAT_ID}->{_BOARD_CU_MAT_REMAPPED}, "
             f"FR4 {_FR4_MAT_ID}->{_BOARD_FR4_MAT_REMAPPED} (clears the low MAT range "
             f"the component's own materials need)\n")

        offsets = compute_safe_offsets(mapdl)
        print(f"Node/element offsets -- node:{offsets.node} elem:{offsets.elem}")

        handle = import_component(mapdl, args.cdb, offsets)
        print(f"Imported component: {handle.node_ids.size} nodes "
             f"(IDs {handle.node_ids.min()}-{handle.node_ids.max()}), "
             f"{handle.elem_ids.size} elements "
             f"(IDs {handle.elem_ids.min()}-{handle.elem_ids.max()})")

        reassert_board_materials(mapdl, (_BOARD_CU_MAT_REMAPPED, _BOARD_FR4_MAT_REMAPPED))
        print("Board materials reasserted post-import (CDREAD partially reverts them -- see "
             "reassert_board_materials()'s docstring)\n")

        reapply_component_materials(mapdl, args.cdb, _CR0805_USED_MAT_IDS)
        print(f"Component materials {_CR0805_USED_MAT_IDS} re-derived directly from the archive's "
             f"own UNBL-tagged MPTEMP/MPDATA lines (this R1 install's CDREAD misparses that "
             f"2025 R2 format -- see reapply_component_materials()'s docstring)\n")

        solder_mat_id = _CR0805_SOLDER_MAT
        targe_type_id = _CR0805_TARGE_TYPE
        conta_type_id = _CR0805_CONTA_TYPE
        print(f"Solder MAT={solder_mat_id}  TARGE170 TYPE={targe_type_id}  "
             f"CONTA174 TYPE={conta_type_id}\n")

        apply_solder_material(mapdl, args.solder_mac, solder_mat_id)
        place_component(mapdl, handle, place_x, place_y, args.rotation_deg, Lz_mm, solder_mat_id)

        if args.inspect:
            print("Opening interactive plot for inspection -- close the plot window to return here.")
            mapdl.allsel()
            mapdl.eplot(show_edges=True, title="Board + placed component (close window to exit)")
            print("Plot closed. Exiting without generating contact/solving "
                 "(rerun without --inspect to continue).")
            return

        n_targe, n_conta = generate_bonded_contact(
            mapdl, board_top_z_mm=Lz_mm, solder_mat_id=solder_mat_id,
            targe_type_id=targe_type_id, conta_type_id=conta_type_id,
            board_pad_mat_id=_BOARD_CU_MAT_REMAPPED,
            board_mat_ids=(_BOARD_CU_MAT_REMAPPED, _BOARD_FR4_MAT_REMAPPED))
        print(f"Contact pair: {n_targe} TARGE170 + {n_conta} CONTA174 elements")
        if n_targe == 0 or n_conta == 0:
            sys.exit("Contact pair has an empty side -- assembly is not connected, aborting "
                     "before solve.")

        if args.board_model == "homogenized":
            # last step before /SOLU: component import and contact generation
            # above both select the board by its copper/FR4 MAT IDs
            summary = apply_homogenized_board(
                mapdl, sample_dir / "mesh" / "quadtree.json", materials,
                (_BOARD_CU_MAT_REMAPPED, _BOARD_FR4_MAT_REMAPPED))
            print(f"Board homogenized from mesh/quadtree.json: {summary['n_elements']} board "
                  f"elements -> {summary['n_materials']} orthotropic materials (MAT "
                  f"{summary['mat_ids'][0]}-{summary['mat_ids'][1]}, from "
                  f"{summary['n_leaves']} leaves)\n")
        elif args.board_model == "homogenized_zlayers":
            summary = apply_homogenized_board_zlayers(
                mapdl, sample_dir / "mesh" / "quadtree.json", materials,
                (_BOARD_CU_MAT_REMAPPED, _BOARD_FR4_MAT_REMAPPED), z_cu_bot_mm, z_cu_top_mm)
            print(f"Board homogenized per z-layer from mesh/quadtree.json: {summary['n_elements']} "
                  f"copper-layer elements -> {summary['n_materials']} orthotropic materials (MAT "
                  f"{summary['mat_ids'][0]}-{summary['mat_ids'][1]}, from {summary['n_leaves_used']}/"
                  f"{summary['n_leaves']} leaves used, bottom+top sides); FR4 core rows left as "
                  f"plain FR4 (MAT {_BOARD_FR4_MAT_REMAPPED})\n")

        mapdl.allsel()
        mapdl.finish()
        mapdl.slashsolu()
        if args.creep:
            mapdl.rate("on")
            # Only forced when asked for -- see --creep-ratio-limit's help
            # text. Option 1 = implicit creep (the SAC305 law is implicit);
            # without it MAPDL reports "CRITERION FOR CREEP RATIO CONTROL =
            # none" and the limit has no effect.
            if args.creep_ratio_limit is not None:
                mapdl.cutcontrol("CRPLIMIT", args.creep_ratio_limit, 1)
                print(f"Creep ON for this solve (CRPLIMIT={args.creep_ratio_limit}, forced)\n")
            else:
                print("Creep ON for this solve (creep ratio control: MAPDL automatic)\n")
        else:
            # RATE,OFF deactivates implicit creep integration session-wide;
            # the solder's TB,CREEP table (from --solder-mac) stays defined,
            # it's just not evaluated -- so this is an elastic-only check
            # without needing a separate creep-free material variant.
            mapdl.rate("off")
            print("Creep OFF for this solve\n")
        mapdl.antype("STATIC")
        # Write every substep's results, not just each load step's last one
        # (MAPDL's OUTRES default) -- needed for any post-hoc time-history
        # plot (e.g. plot_creep_strain_history.py) to have more than one
        # point per segment to plot.
        mapdl.outres("ALL", "ALL")
        # One load step per consecutive pair of profile points -- each has
        # its own explicit end TIME, so AUTOTS always has a real target to
        # stop at. See thermal_profile.apply_thermal_profile().
        apply_thermal_profile(mapdl, thermal_profile)
        mapdl.finish()
        print(f"\nThermal profile solve completed: {thermal_profile.time_s.size} points over "
             f"{thermal_profile.time_s[-1]:.0f} s, reference temp "
             f"{thermal_profile.reference_temp_c:.1f} C "
             f"({'creep active' if args.creep else 'elastic-only, creep off'}).")

        mapdl.save()
        db_src = Path(mapdl.directory) / f"{mapdl.jobname}.db"
        rst_src = Path(mapdl.result_file) if mapdl.result_file else None
        db_dst = results_dir / "pyansys_assembly.db"
        rst_dst = results_dir / "pyansys_assembly.rst"
        shutil.copyfile(db_src, db_dst)
        print(f"Wrote {db_dst}")
        if rst_src and rst_src.is_file():
            shutil.copyfile(rst_src, rst_dst)
            print(f"Wrote {rst_dst}")
        else:
            print(f"[warn] No .rst result file found at {rst_src} -- only the .db was copied.")
        print(f"Open {db_dst.name} in a standalone ANSYS Mechanical APDL session "
             f"(File > Resume from...) to inspect in the GUI.")
    finally:
        mapdl.exit()


if __name__ == "__main__":
    main()
