#!/usr/bin/env python3
"""
n-layer PCB board model in MAPDL (board only), via PyMAPDL.

The last step of the n-layer pipeline: build the FE mesh, assign each element
the material the coarse-graining step resolved for it, apply a thermal load
and solve. Board only -- no component, no solder, no creep. Component
placement sits on top of this and is a separate concern.

    import_odb_board_nlayer.py   ODB++ -> sample + resolved stackup
    discretize.py                copper rasters (unchanged, already n-layer)
    coarsegrain_nlayer.py        regions x homogenize-or-project -> materials
    simulate_pyansys_nlayer.py   mesh + material assignment + solve   <- here

Mesh: a structured hex grid, nx x ny in plane, with z banded by PHYSICAL
LAYER -- every copper and dielectric slab from the stackup gets its own
--nz-per-layer element rows, so every slab boundary is an element boundary
and no element ever straddles two slabs. This is what makes the n-layer
copper volume exact and removes any need for the 2-layer path's copper
"dilution" correction: a 25 um copper layer is represented by elements 25 um
thick, not smeared into a core element many times thicker.

The cost is aspect ratio. On the 8-layer reference board a 25 um slab against
a 0.3 mm in-plane element is 12:1, and SOLID186 handles that (the 2-layer
model already runs 35 um copper rows against 0.4 mm elements), but it is the
reason element counts climb fast with --nx/--ny: 15 slabs x nz_per_layer z
rows, not 3.

Material assignment, from mesh/coarsegrain_nlayer.json:

  homogenized_zlayers   each copper slab's elements get that (region, slab)
                        material; dielectric slabs get plain dielectric
  homogenized           every element in a region's column gets the one
                        smeared material, dielectrics included
  projection            as homogenized_zlayers, but each (region, slab) was
                        already resolved to pure copper or pure dielectric

  --board-model raster  ignores the coarse-grain file and paints each element
                        pure copper or dielectric straight from the raster at
                        its own centroid -- the full-resolution reference the
                        coarse-grained models are meant to approximate

MAT is set at element creation (mapdl.mat() before each group's mapdl.e()
calls) rather than by a post-mesh EMODIF pass, as in
simulate_pyansys_benchmark.py.

Load and boundary conditions: a 3-2-1 fixture, which removes all six
rigid-body modes without restraining free thermal expansion, and a single
static load step to ref_temp + delta_t. Linear-elastic materials have no path
dependence, so a ramp and a single end state give the same answer; that only
changes once a creep/plastic solder is in the model, which is out of scope
here.

Quantities of interest, all board-level and so comparable across board
models and against the analytical bounds:

  alpha_eff_x/y   effective in-plane CTE, from the 3-2-1-fixtured corner
                  displacements. Compared automatically against
                  homogenization_nlayer's analytical laminate alpha_xy, which
                  is a real check on the whole chain -- a free thermal
                  expansion of a symmetric laminate should reproduce it.
  warpage         peak-to-valley of the top surface's out-of-plane
                  displacement after removing the best-fit plane (so the
                  fixture's own tilt does not count as warpage)
  von Mises       top-surface field, plotted

Requires `ansys-mapdl-core` and a reachable MAPDL install/license.

Usage:
    python coarsegrain_nlayer.py data/samples/ublox_nlayer --mode grid --nx 40 --ny 40
    python simulate_pyansys_nlayer.py data/samples/ublox_nlayer --nx 40 --ny 40

    python simulate_pyansys_nlayer.py data/samples/ublox_nlayer --nx 40 --ny 40 \\
        --board-model raster --tag raster
    python simulate_pyansys_nlayer.py data/samples/ublox_nlayer --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coarsegrain_nlayer import OUTPUT_NAME as COARSEGRAIN_NAME
from homogenization import load_materials
from homogenization_nlayer import PhaseProps, SlabSpec, laminate_bounds, pure_phase_material
from stackup_nlayer import BoardStackup, load_board_stackup, raster_lookup

_ELEM_TYPE_ID = 1
_COPPER_MAT_ID = 1
_DIELECTRIC_MAT_ID = 2

# Process names mapdl.exit() leaves running on this machine -- the same
# backstop simulate_pyansys_assembly.py carries, and for the same reason:
# exit() talks to the gRPC server (rank 0) over its channel, but the other
# DMP ranks and the license client do not reliably go down with it, and if
# the server already crashed the call cannot reach it at all. The leftover
# processes keep file.lock held in --mapdl-dir, so the NEXT run fails to
# launch ("Unable to remove lock file") even though it asked for override.
# Call this alongside mapdl.exit(), not instead of it.
_MAPDL_PROCESS_NAMES = {"ansys.exe", "ansys251.exe"}
# Coarse-grained materials are numbered from here, well clear of the two
# pure-phase IDs so a mixed run can carry both.
_HOMOG_MAT_BASE = 1001

# Above this, warn before generating -- element creation goes through APDL
# commands, so a few million elements is a long wait before the solver even
# starts.
_ELEMENT_WARN_LIMIT = 1_500_000


# ---------------------------------------------------------------------------
# z grid: banded by physical layer
# ---------------------------------------------------------------------------


def _force_kill_mapdl_processes(run_location: str) -> None:
    """
    Kill MAPDL processes still pinned to this run's working directory. Scoped
    by working directory (not by name alone) so a MAPDL session the user has
    open elsewhere is never touched. Same implementation as
    simulate_pyansys_assembly._force_kill_mapdl_processes().
    """
    try:
        import psutil
    except ImportError:
        print("[warn] psutil not installed -- cannot clean up leftover MAPDL processes; "
              "a stale file.lock may block the next run.", file=sys.stderr)
        return
    run_location = str(Path(run_location).resolve())
    killed = []
    for proc in psutil.process_iter(["pid", "name"]):
        if (proc.info.get("name") or "").lower() not in _MAPDL_PROCESS_NAMES:
            continue
        try:
            if proc.cwd() != run_location:
                continue
            proc.kill()
            killed.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if killed:
        print(f"[cleanup] Force-killed {len(killed)} leftover MAPDL process(es) "
              f"pinned to {run_location}: {killed}")


@dataclass
class ZGrid:
    """z node coordinates plus, for every element row, which slab it is in."""
    z_pts: np.ndarray            # (nzn,) ascending node z coordinates
    slab_of_row: list[int]       # (nz,) slab index for each element row

    @property
    def nz(self) -> int:
        return len(self.slab_of_row)


def build_z_grid(board: BoardStackup, nz_per_layer: int) -> ZGrid:
    """
    Subdivide every stackup slab into nz_per_layer element rows. Slab
    interfaces are therefore always node planes, so no element spans two
    physical layers and each slab's thickness is represented exactly.
    """
    if nz_per_layer < 1:
        raise ValueError("nz_per_layer must be >= 1")
    z_pts: list[float] = [0.0]
    slab_of_row: list[int] = []
    for s in board.slabs:
        edges = np.linspace(s.z0_mm, s.z1_mm, nz_per_layer + 1)
        z_pts.extend(edges[1:].tolist())
        slab_of_row.extend([s.index] * nz_per_layer)
    return ZGrid(np.asarray(z_pts, dtype=float), slab_of_row)


# ---------------------------------------------------------------------------
# Coarse-graining file -> per-cell region lookup
# ---------------------------------------------------------------------------


@dataclass
class CoarseGrain:
    region_model: str
    bboxes: np.ndarray                    # (n_regions, 4) x0 y0 x1 y1
    materials: dict[str, dict]            # name -> MP label dict
    slab_materials: list[dict[int, str]]  # per region: slab index -> material name
    column_materials: list[str | None]    # per region: smeared material name
    discretization: dict
    source: Path

    @property
    def n_regions(self) -> int:
        return len(self.bboxes)


def load_coarsegrain(sample_dir: Path) -> CoarseGrain:
    path = Path(sample_dir) / "mesh" / COARSEGRAIN_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Not found: {path} -- run coarsegrain_nlayer.py first, or pass "
            "--board-model raster to paint straight from the rasters.")
    doc = json.loads(path.read_text())
    regions = doc["regions"]
    if not regions:
        raise ValueError(f"{path} contains no regions")
    return CoarseGrain(
        region_model=doc["region_model"],
        bboxes=np.array([r["bbox_mm"] for r in regions], dtype=float),
        materials=doc["materials"],
        slab_materials=[{int(k): v for k, v in r.get("slab_materials", {}).items()}
                        for r in regions],
        column_materials=[r.get("column_material") for r in regions],
        discretization=doc.get("discretization", {}),
        source=path,
    )


def region_of_cells(cg: CoarseGrain, cx: np.ndarray, cy: np.ndarray,
                    chunk: int = 4096) -> np.ndarray:
    """
    Region index for each in-plane cell centroid: the first region whose bbox
    contains it. Done once for the nx*ny grid rather than per element row,
    since the regions are purely in-plane.

    Chunked over cells so the (cells x regions) boolean never has to exist in
    full -- a 200x200 mesh against 16k regions would otherwise be 640 M
    entries.
    """
    b = cg.bboxes
    out = np.full(cx.size, -1, dtype=np.int64)
    for start in range(0, cx.size, chunk):
        sl = slice(start, min(start + chunk, cx.size))
        inside = ((cx[sl, None] >= b[None, :, 0]) & (cx[sl, None] <= b[None, :, 2])
                  & (cy[sl, None] >= b[None, :, 1]) & (cy[sl, None] <= b[None, :, 3]))
        hit = inside.any(axis=1)
        out[sl] = np.where(hit, inside.argmax(axis=1), -1)
    if (out < 0).any():
        n = int((out < 0).sum())
        raise RuntimeError(
            f"{n} of {cx.size} element centroids lie in no region of {cg.source.name} -- "
            "was it generated for this board?")
    return out


# ---------------------------------------------------------------------------
# Material registration
# ---------------------------------------------------------------------------


def register_materials(mapdl, cg: CoarseGrain | None, p: PhaseProps) -> dict[str, int]:
    """
    Write every material the run needs into MAPDL and return {name: MAT id}.
    The two pure phases are always defined (the dielectric slabs need one,
    and --board-model raster needs both).
    """
    pure = {"copper": pure_phase_material(True, p),
            "dielectric": pure_phase_material(False, p)}
    ids = {"copper": _COPPER_MAT_ID, "dielectric": _DIELECTRIC_MAT_ID}

    # MP is a PREP7 command: issued at BEGIN level MAPDL ignores it with
    # "MP is not a recognized BEGIN command", which PyMAPDL then raises on.
    # This runs before build_mesh(), so PREP7 has to be entered here.
    mapdl.prep7()
    with mapdl.non_interactive:
        for name, mat_id in ids.items():
            for label, value in pure[name].items():
                mapdl.mp(label, mat_id, value)
        if cg is not None:
            for k, name in enumerate(sorted(cg.materials, key=lambda s: int(s[1:]))):
                mat_id = _HOMOG_MAT_BASE + k
                ids[name] = mat_id
                for label, value in cg.materials[name].items():
                    mapdl.mp(label, mat_id, value)
    return ids


# ---------------------------------------------------------------------------
# Mesh + material assignment
# ---------------------------------------------------------------------------


def build_mesh(mapdl, board: BoardStackup, zgrid: ZGrid, nx: int, ny: int,
               cg: CoarseGrain | None, mat_ids: dict[str, int],
               board_model: str, elem_type: str = "SOLID186") -> dict:
    """
    Generate nodes and elements, each element created under the MAT its
    (region, slab) or raster lookup resolves to. Returns the node IDs the
    boundary conditions and post-processing need, including a 2D node-ID
    grid for each of the 6 outer faces (x0/xL/y0/yL/z0/zL) -- what a thermal
    flux/fixed-temperature boundary condition is applied to in
    simulate_pyansys_thermal_nlayer.py. elem_type swaps in SOLID70 for that
    script; the mesh and material assignment are otherwise identical.
    """
    x_pts = np.linspace(0.0, board.width_mm, nx + 1)
    y_pts = np.linspace(0.0, board.height_mm, ny + 1)
    z_pts = zgrid.z_pts
    nxn, nyn, nzn = len(x_pts), len(y_pts), len(z_pts)
    nz = zgrid.nz

    def nid(i, j, k) -> int:
        return (i * nyn + j) * nzn + k + 1        # 1-based APDL node numbering

    mapdl.prep7()
    mapdl.et(_ELEM_TYPE_ID, elem_type)
    mapdl.type(_ELEM_TYPE_ID)

    n_nodes, n_elems = nxn * nyn * nzn, nx * ny * nz
    print(f"Generating {n_nodes} nodes / {n_elems} elements "
          f"({nx} x {ny} x {nz}, {len(board.slabs)} slabs) …")
    with mapdl.non_interactive:
        for i in range(nxn):
            for j in range(nyn):
                for k in range(nzn):
                    mapdl.n(nid(i, j, k), x_pts[i], y_pts[j], z_pts[k])

    # In-plane cell centroids, and the region each falls in.
    xc = (x_pts[:-1] + x_pts[1:]) / 2.0
    yc = (y_pts[:-1] + y_pts[1:]) / 2.0
    cx, cy = np.meshgrid(xc, yc, indexing="ij")
    cx_f, cy_f = cx.ravel(), cy.ravel()
    region_of = (region_of_cells(cg, cx_f, cy_f) if cg is not None
                 else np.zeros(cx_f.size, dtype=np.int64))

    slab_by_index = {s.index: s for s in board.slabs}

    # Group every element by MAT id first, then emit each group under one
    # mapdl.mat() -- far fewer command switches than setting MAT per element.
    by_mat: dict[int, list[list[int]]] = {}
    for k in range(nz):
        # Which slab this row belongs to is already exact: z is banded by
        # slab, so a row never straddles two. No centroid test needed.
        slab = slab_by_index[zgrid.slab_of_row[k]]
        for cell, (xcc, ycc) in enumerate(zip(cx_f, cy_f)):
            i, j = divmod(cell, ny)
            if board_model == "raster":
                is_cu = slab.is_copper and slab.raster is not None and raster_lookup(
                    slab.raster, xcc, ycc, board.width_mm, board.height_mm)
                mat = mat_ids["copper"] if is_cu else mat_ids["dielectric"]
            elif cg.region_model == "homogenized":
                # Smeared: the whole column, dielectric rows included, is one
                # material -- there is no per-slab distinction to make.
                mat = mat_ids[cg.column_materials[region_of[cell]]]
            elif slab.is_copper:
                mat = mat_ids[cg.slab_materials[region_of[cell]][slab.index]]
            else:
                mat = mat_ids["dielectric"]
            by_mat.setdefault(mat, []).append([
                nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                nid(i, j, k + 1), nid(i + 1, j, k + 1),
                nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
            ])

    print(f"  {len(by_mat)} distinct materials across {n_elems} elements")
    with mapdl.non_interactive:
        for mat, cells in by_mat.items():
            mapdl.mat(mat)
            for nodes8 in cells:
                mapdl.e(*nodes8)

    # Top-surface node ids, for the warpage field, plus a 2D node-ID grid for
    # every outer face -- unused by the mechanical BCs/post-processing above,
    # but what the thermal script's flux/fixed-temperature loads select.
    top_nodes = [[nid(i, j, nzn - 1) for j in range(nyn)] for i in range(nxn)]
    face_grids = {
        "x0_nodes": [[nid(0, j, k) for k in range(nzn)] for j in range(nyn)],
        "xL_nodes": [[nid(nxn - 1, j, k) for k in range(nzn)] for j in range(nyn)],
        "y0_nodes": [[nid(i, 0, k) for k in range(nzn)] for i in range(nxn)],
        "yL_nodes": [[nid(i, nyn - 1, k) for k in range(nzn)] for i in range(nxn)],
        "z0_nodes": [[nid(i, j, 0) for j in range(nyn)] for i in range(nxn)],
        "zL_nodes": top_nodes,
    }
    return {
        "corner_000": nid(0, 0, 0),
        "corner_Lx00": nid(nxn - 1, 0, 0),
        "corner_0Ly0": nid(0, nyn - 1, 0),
        "corner_LxLy0": nid(nxn - 1, nyn - 1, 0),
        "top_nodes": top_nodes,
        "n_nodes": n_nodes,
        "n_elements": n_elems,
        "n_materials_used": len(by_mat),
        **face_grids,
    }


# ---------------------------------------------------------------------------
# BCs, solve
# ---------------------------------------------------------------------------


def apply_321_fixture(mapdl, node_ids: dict) -> None:
    """3-2-1 fixture: removes all 6 rigid-body modes without restraining free
    thermal expansion. Same scheme as simulate_pyansys_benchmark.py, so the
    n-layer and 2-layer models' displacements are directly comparable."""
    n0, n1, n2 = node_ids["corner_000"], node_ids["corner_Lx00"], node_ids["corner_0Ly0"]
    mapdl.d(n0, "UX", 0)
    mapdl.d(n0, "UY", 0)
    mapdl.d(n0, "UZ", 0)
    mapdl.d(n1, "UY", 0)
    mapdl.d(n1, "UZ", 0)
    mapdl.d(n2, "UZ", 0)


def solve_thermal_step(mapdl, ref_temp_c: float, delta_t: float) -> None:
    mapdl.finish()
    mapdl.slashsolu()
    mapdl.allsel()
    mapdl.tref(ref_temp_c)
    mapdl.bfunif("TEMP", ref_temp_c + delta_t)
    mapdl.antype("STATIC")
    mapdl.solve()
    mapdl.finish()


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _fit_plane_residual(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """w minus its best-fit plane -- warpage with rigid tilt removed."""
    A = np.column_stack([x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    return w - A @ coef


def evaluate(mapdl, board: BoardStackup, node_ids: dict,
             delta_t: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """Effective in-plane CTE, warpage, and the nodal von Mises field."""
    mapdl.post1()
    mapdl.set(1, 1)

    def disp(node_id: int) -> np.ndarray:
        return np.array([mapdl.get_value("NODE", node_id, "U", c) for c in ("X", "Y", "Z")])

    # With the 3-2-1 fixture anchored at corner_000, the x-neighbour corner's
    # UX is the board's free in-plane expansion over its full width.
    u_x = disp(node_ids["corner_Lx00"])
    u_y = disp(node_ids["corner_0Ly0"])
    alpha_x = float(u_x[0] / (delta_t * board.width_mm)) if delta_t else float("nan")
    alpha_y = float(u_y[1] / (delta_t * board.height_mm)) if delta_t else float("nan")

    mapdl.allsel()
    coords = np.asarray(mapdl.mesh.nodes)
    uz_all = np.asarray(mapdl.post_processing.nodal_displacement("Z"))
    eqv = np.asarray(mapdl.post_processing.nodal_eqv_stress())

    top_mask = np.isclose(coords[:, 2], board.total_thickness_mm)
    xt, yt, wt = coords[top_mask, 0], coords[top_mask, 1], uz_all[top_mask]
    resid = _fit_plane_residual(xt, yt, wt)
    result = {
        "delta_t_K": delta_t,
        "alpha_eff_x_per_K": alpha_x,
        "alpha_eff_y_per_K": alpha_y,
        "u_corner_Lx_mm": u_x.tolist(),
        "u_corner_Ly_mm": u_y.tolist(),
        "warpage_pv_um": float((resid.max() - resid.min()) * 1e3),
        "warpage_rms_um": float(np.sqrt(np.mean(resid ** 2)) * 1e3),
        "von_mises_max_MPa": float(eqv.max()),
        "von_mises_top_max_MPa": float(eqv[top_mask].max()),
        "n_top_nodes": int(top_mask.sum()),
    }
    return result, coords, eqv


def plot_von_mises_top(coords: np.ndarray, eqv: np.ndarray, board: BoardStackup,
                       out_path: Path, title_suffix: str = "") -> None:
    top = np.isclose(coords[:, 2], board.total_thickness_mm)
    x, y, s = coords[top, 0], coords[top, 1], eqv[top]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    tpc = ax.tricontourf(x, y, s, levels=20, cmap="inferno")
    ax.set_xlim(0, board.width_mm)
    ax.set_ylim(0, board.height_mm)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Top-surface von Mises{title_suffix}")
    fig.colorbar(tpc, ax=ax, label="von Mises [MPa]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analytical_reference(board: BoardStackup, p: PhaseProps) -> dict:
    """
    Whole-board analytical laminate properties at the true per-layer copper
    fractions -- the independent yardstick for the solved alpha_eff.
    """
    slabs = [SlabSpec(s.thickness_mm,
                      float(s.raster.mean()) if (s.is_copper and s.raster is not None) else 0.0,
                      s.kind, s.name)
             for s in board.slabs]
    b = laminate_bounds(slabs, p)
    return {"phi_3D": b["phi_3D"],
            "alpha_xy_per_K": b["alpha_xy_global_per_K"],
            "alpha_z_per_K": b["alpha_z_reuss_per_K"],
            "E_xy_MPa": b["E_xy_voigt_MPa"],
            "E_z_MPa": b["E_z_reuss_MPa"]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="n-layer PCB board model in MAPDL (board only), via PyMAPDL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--nz-per-layer", type=int, default=1,
                    help="Element rows per PHYSICAL layer. 1 already resolves every slab "
                         "exactly, since z is banded by slab; an n-layer stackup has many "
                         "more slabs than the 2-layer path's 3, so raising this multiplies "
                         "the element count fast.")
    ap.add_argument("--board-model", choices=["coarsegrain", "raster"], default="coarsegrain",
                    help="coarsegrain: take materials from mesh/coarsegrain_nlayer.json "
                         "(whatever --region-model it was built with). raster: ignore that "
                         "file and paint each element pure copper/dielectric from the raster "
                         "at its centroid -- the full-resolution reference.")
    ap.add_argument("--delta-t", type=float, default=100.0, help="Thermal step [K]")
    ap.add_argument("--ref-temp-c", type=float, default=20.0)
    ap.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    ap.add_argument("--tag", default=None,
                    help="Suffix for the results file names, to keep several board models "
                         "side by side (default: the board/region model)")
    ap.add_argument("--mapdl-dir", type=Path, default=Path("mapdl_run_nlayer"),
                    help="MAPDL working directory. Put it on a local disk -- MAPDL writes "
                         "multi-GB scratch there and a network share slows the solve badly.")
    ap.add_argument("--nproc", type=int, default=None,
                    help="MAPDL core count (default: launch_mapdl()'s own)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the mesh, materials and analytical reference, then stop "
                         "without launching MAPDL")
    args = ap.parse_args()

    sample_dir = args.sample_dir
    if not sample_dir.exists():
        sys.exit(f"Not found: {sample_dir}")
    if args.nx < 1 or args.ny < 1:
        sys.exit("--nx/--ny must be >= 1")

    try:
        board = load_board_stackup(sample_dir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    p = PhaseProps.from_materials(load_materials(args.materials))

    cg = None
    if args.board_model == "coarsegrain":
        try:
            cg = load_coarsegrain(sample_dir)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(str(e))

    zgrid = build_z_grid(board, args.nz_per_layer)
    n_elems = args.nx * args.ny * zgrid.nz
    n_nodes = (args.nx + 1) * (args.ny + 1) * len(zgrid.z_pts)

    print(board.describe())
    print(f"\nMesh: {args.nx} x {args.ny} x {zgrid.nz} = {n_elems} elements, {n_nodes} nodes")
    print(f"  in-plane element {board.width_mm / args.nx:.4f} x "
          f"{board.height_mm / args.ny:.4f} mm")
    thin = min(board.slabs, key=lambda s: s.thickness_mm)
    dz_thin = thin.thickness_mm / args.nz_per_layer
    print(f"  thinnest slab {thin.name!r}: {thin.thickness_mm * 1e3:.1f} um -> "
          f"{args.nz_per_layer} row(s) of {dz_thin * 1e3:.1f} um "
          f"(in-plane/through-thickness aspect ratio "
          f"{(board.width_mm / args.nx) / dz_thin:.0f}:1)")
    if cg is not None:
        print(f"\nBoard model: coarsegrain / {cg.region_model}  "
              f"({cg.n_regions} regions, {len(cg.materials)} materials, "
              f"{cg.discretization.get('mode', '?')} discretization)")
        if cg.discretization.get("mode") == "grid":
            gnx, gny = cg.discretization.get("nx"), cg.discretization.get("ny")
            if (gnx, gny) == (args.nx, args.ny):
                print("  regions coincide with elements (one region per element)")
            else:
                print(f"  regions are {gnx} x {gny}, mesh is {args.nx} x {args.ny} -- "
                      f"{'coarser' if (gnx or 0) < args.nx else 'finer'} than the mesh")
    else:
        print("\nBoard model: raster (pure copper/dielectric per element, full resolution)")

    ref = analytical_reference(board, p)
    print(f"\nAnalytical laminate reference (whole board, true per-layer phi):")
    print(f"  phi_3D     = {ref['phi_3D']:.6f}")
    print(f"  alpha_xy   = {ref['alpha_xy_per_K'] * 1e6:.4f} ppm/K")
    print(f"  alpha_z    = {ref['alpha_z_per_K'] * 1e6:.4f} ppm/K")
    print(f"  E_xy/E_z   = {ref['E_xy_MPa']:.0f} / {ref['E_z_MPa']:.0f} MPa")

    if n_elems > _ELEMENT_WARN_LIMIT:
        print(f"\n[warn] {n_elems} elements is a lot -- nodes and elements are generated "
              f"through APDL commands, so this will take a long time before the solver "
              f"even starts. Consider lowering --nx/--ny or --nz-per-layer.", file=sys.stderr)

    if args.dry_run:
        print("\n--dry-run: stopping before MAPDL.")
        return

    tag = args.tag or (cg.region_model if cg is not None else "raster")
    args.mapdl_dir.mkdir(parents=True, exist_ok=True)
    run_location = str(args.mapdl_dir.resolve())

    from ansys.mapdl.core import launch_mapdl
    kwargs = {"run_location": run_location, "override": True}
    if args.nproc:
        kwargs["nproc"] = args.nproc
    mapdl = launch_mapdl(**kwargs)
    print(f"\nMAPDL working directory: {mapdl.directory}")
    try:
        mat_ids = register_materials(mapdl, cg, p)
        print(f"Registered {len(mat_ids)} materials in MAPDL")
        node_ids = build_mesh(mapdl, board, zgrid, args.nx, args.ny, cg, mat_ids,
                              args.board_model)
        apply_321_fixture(mapdl, node_ids)
        print(f"Solving: TREF={args.ref_temp_c} C, uniform "
              f"{args.ref_temp_c + args.delta_t} C (delta_t={args.delta_t} K) …")
        solve_thermal_step(mapdl, args.ref_temp_c, args.delta_t)
        result, coords, eqv = evaluate(mapdl, board, node_ids, args.delta_t)
    finally:
        try:
            mapdl.exit()
        except Exception as e:                       # noqa: BLE001 - best effort
            print(f"[warn] mapdl.exit() failed: {e}", file=sys.stderr)
        # Always, whether the run succeeded or raised: exit() alone leaves
        # DMP ranks holding file.lock, which breaks the next launch.
        _force_kill_mapdl_processes(run_location)

    result.update({
        "sample_id": board.sample_id,
        "board_model": args.board_model,
        "region_model": cg.region_model if cg is not None else None,
        "discretization": cg.discretization if cg is not None else None,
        "mesh": {"nx": args.nx, "ny": args.ny, "nz": zgrid.nz,
                 "nz_per_layer": args.nz_per_layer,
                 "n_elements": node_ids["n_elements"], "n_nodes": node_ids["n_nodes"],
                 "n_materials_used": node_ids["n_materials_used"]},
        "stackup": {"n_copper_layers": board.n_copper,
                    "total_thickness_mm": board.total_thickness_mm},
        "analytical_reference": ref,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    })

    # The solved free expansion should sit near the analytical in-plane CTE.
    # Treat a gap as information, not as an error: alpha_eff is measured on
    # the BOTTOM face (where the 3-2-1 fixture is anchored), so on an
    # asymmetric stackup -- which any real multilayer board is; the reference
    # 8-layer board runs phi 0.67 at the bottom against 0.86 at the top --
    # thermal bending makes the bottom-face expansion differ from the
    # laminate average by design. A gap of tens of percent points at the
    # material assignment, stackup or fixture; a few percent on an
    # asymmetric board is expected.
    a_ref = ref["alpha_xy_per_K"]
    for axis in ("x", "y"):
        a = result[f"alpha_eff_{axis}_per_K"]
        rel = (a - a_ref) / abs(a_ref) if a_ref else float("nan")
        result[f"alpha_eff_{axis}_vs_analytical_rel"] = rel
    phi_bot = next((float(s.raster.mean()) for s in board.copper_slabs
                    if s.raster is not None), None)
    phi_top = next((float(s.raster.mean()) for s in reversed(board.copper_slabs)
                    if s.raster is not None), None)
    asymmetric = (phi_bot is not None and phi_top is not None
                  and abs(phi_top - phi_bot) > 0.02)
    print(f"\nEffective in-plane CTE, bottom face (analytical laminate "
          f"{a_ref * 1e6:.4f} ppm/K):")
    for axis in ("x", "y"):
        a = result[f"alpha_eff_{axis}_per_K"]
        print(f"  alpha_eff_{axis} = {a * 1e6:9.4f} ppm/K  "
              f"({result[f'alpha_eff_{axis}_vs_analytical_rel'] * 100:+.2f} %)")
    result["stackup_asymmetric"] = bool(asymmetric)
    if asymmetric:
        print(f"  [note] asymmetric stackup (bottom phi={phi_bot:.3f}, top phi={phi_top:.3f}): "
              f"the board bends, so a bottom-face value offset from the laminate\n"
              f"         average is physical, not an error. The warpage below is that bending.")
    print(f"\nWarpage (top surface, best-fit plane removed):")
    print(f"  peak-to-valley = {result['warpage_pv_um']:.3f} um")
    print(f"  RMS            = {result['warpage_rms_um']:.3f} um")
    print(f"von Mises max = {result['von_mises_max_MPa']:.3f} MPa "
          f"(top surface {result['von_mises_top_max_MPa']:.3f} MPa)")

    results_dir = sample_dir / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"pyansys_nlayer_{tag}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {out_path}")

    plot_path = results_dir / f"von_mises_top_nlayer_{tag}.png"
    plot_von_mises_top(coords, eqv, board, plot_path,
                       f" — {board.n_copper} layers, {tag}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
