#!/usr/bin/env python3
"""
MAPDL mesh construction shared by simulate_board.py and simulate_assembly.py:
a structured hex grid banded by PHYSICAL LAYER in z, with each in-plane cell
and z-row's material resolved from a pcbh.grid.GridModel.

Mesh: nx x ny in plane; z is subdivided --nz-per-layer times PER STACKUP
SLAB, so every slab interface is a node plane and no element ever straddles
two physical layers -- every copper layer is represented at its true
thickness, with no "dilution" correction needed anywhere in this pipeline.

The cost is aspect ratio: a 25 um copper slab against a 0.3 mm in-plane
element is 12:1, which is why element counts climb fast with --nx/--ny on a
board with many thin layers.

Board materials are registered from BOARD_MAT_BASE (9001) upward, well clear
of a component archive's own MAT 1-10 (see pcbh.component) -- so, unlike the
pre-restructure 2-layer path, nothing needs remapping before a component is
imported. The board's elements are also tagged with a named element
component ("BOARD"), which is what lets the board be re-selected after
import_component() without depending on its MAT id range at all.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pcbh.grid import GridModel
from pcbh.materials import PhaseProps
from pcbh.stackup import BoardStackup

_ELEM_TYPE_ID = 1

BOARD_MAT_BASE = 9001
BOARD_CM = "BOARD"

# Above this, warn before generating -- element creation goes through APDL
# commands, so a few million elements is a long wait before the solver even
# starts.
ELEMENT_WARN_LIMIT = 1_500_000

# Process names mapdl.exit() leaves running on this machine -- confirmed
# empirically (Windows, distributed/DMP launch_mapdl(nproc>1)): exit() talks
# to the gRPC server (rank 0) over its channel, but the other DMP ranks and
# the standalone license client don't reliably go down with it, and if the
# server itself already crashed the call can't reach it at all. Leftover
# processes keep file.lock held in --mapdl-dir, so the NEXT run fails to
# launch ("Unable to remove lock file") even though it asked for override.
# Call this alongside mapdl.exit(), not instead of it.
_MAPDL_PROCESS_NAMES = {"ansys.exe", "ansys251.exe"}


def _force_kill_mapdl_processes(run_location: str) -> None:
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


def launch(run_location: Path, nproc: int | None = None):
    """launch_mapdl() with override=True (clears a stale file.lock from a
    previous crashed run into the same, fixed, reused directory) and an
    explicit nproc if given."""
    from ansys.mapdl.core import launch_mapdl
    run_location.mkdir(parents=True, exist_ok=True)
    kwargs = {"run_location": str(run_location.resolve()), "override": True}
    if nproc:
        kwargs["nproc"] = nproc
    return launch_mapdl(**kwargs)


def shutdown(mapdl, run_location: Path) -> None:
    """mapdl.exit() plus the process-cleanup backstop, always both,
    regardless of whether the run succeeded or raised -- see
    _force_kill_mapdl_processes()'s docstring for why exit() alone isn't
    enough."""
    try:
        mapdl.exit()
    except Exception as e:                        # noqa: BLE001 - best effort
        print(f"[warn] mapdl.exit() raised ({e}); force-killing leftover processes instead.")
    _force_kill_mapdl_processes(str(run_location))


# ---------------------------------------------------------------------------
# z grid: banded by physical layer
# ---------------------------------------------------------------------------


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
# Material registration
# ---------------------------------------------------------------------------


def register_materials(mapdl, model: GridModel, p: PhaseProps) -> dict[str, int]:
    """Write every material model.materials needs into MAPDL, from
    BOARD_MAT_BASE upward, and return {name: MAT id}."""
    mapdl.prep7()
    mat_ids: dict[str, int] = {}
    with mapdl.non_interactive:
        for k, name in enumerate(sorted(model.materials)):
            mat_id = BOARD_MAT_BASE + k
            mat_ids[name] = mat_id
            for label, value in model.materials[name].items():
                mapdl.mp(label, mat_id, value)
    return mat_ids


# ---------------------------------------------------------------------------
# Mesh + material assignment
# ---------------------------------------------------------------------------


@dataclass
class MeshInfo:
    corner_000: int
    corner_Lx00: int
    corner_0Ly0: int
    corner_LxLy0: int
    top_nodes: list[list[int]]
    x0_nodes: list[list[int]]
    xL_nodes: list[list[int]]
    y0_nodes: list[list[int]]
    yL_nodes: list[list[int]]
    z0_nodes: list[list[int]]
    zL_nodes: list[list[int]]
    n_nodes: int
    n_elements: int
    n_materials_used: int
    board_cm: str = BOARD_CM


def build_mesh(mapdl, board: BoardStackup, zgrid: ZGrid, nx: int, ny: int,
               mat_of_cell, mat_ids: dict[str, int],
               elem_type: str = "SOLID186") -> MeshInfo:
    """
    Generate nodes and elements, each element created under the MAT its
    (in-plane cell, z-row) resolves to via mat_of_cell(ix, iy, slab_index)
    (a pcbh.grid.mat_resolver() closure). Tags every element with the
    "BOARD" element component once meshing is done, so the board can be
    re-selected later (component import, contact generation) independent of
    its MAT id range. elem_type swaps in SOLID70 for simulate_board.py's
    --analysis k; the mesh and material assignment are otherwise identical.
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

    by_mat: dict[int, list[list[int]]] = {}
    for k in range(nz):
        slab_index = zgrid.slab_of_row[k]
        for i in range(nx):
            for j in range(ny):
                mat = mat_of_cell(i, j, slab_index)
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

    mapdl.esel("S", "TYPE", "", _ELEM_TYPE_ID)
    mapdl.cm(BOARD_CM, "ELEM")
    mapdl.allsel()

    top_nodes = [[nid(i, j, nzn - 1) for j in range(nyn)] for i in range(nxn)]
    return MeshInfo(
        corner_000=nid(0, 0, 0), corner_Lx00=nid(nxn - 1, 0, 0),
        corner_0Ly0=nid(0, nyn - 1, 0), corner_LxLy0=nid(nxn - 1, nyn - 1, 0),
        top_nodes=top_nodes,
        x0_nodes=[[nid(0, j, k) for k in range(nzn)] for j in range(nyn)],
        xL_nodes=[[nid(nxn - 1, j, k) for k in range(nzn)] for j in range(nyn)],
        y0_nodes=[[nid(i, 0, k) for k in range(nzn)] for i in range(nxn)],
        yL_nodes=[[nid(i, nyn - 1, k) for k in range(nzn)] for i in range(nxn)],
        z0_nodes=[[nid(i, j, 0) for j in range(nyn)] for i in range(nxn)],
        zL_nodes=top_nodes,
        n_nodes=n_nodes, n_elements=n_elems, n_materials_used=len(by_mat),
    )


def apply_321_fixture(mapdl, mesh: MeshInfo) -> None:
    """3-2-1 fixture: removes all 6 rigid-body modes without restraining
    free thermal expansion."""
    n0, n1, n2 = mesh.corner_000, mesh.corner_Lx00, mesh.corner_0Ly0
    mapdl.d(n0, "UX", 0)
    mapdl.d(n0, "UY", 0)
    mapdl.d(n0, "UZ", 0)
    mapdl.d(n1, "UY", 0)
    mapdl.d(n1, "UZ", 0)
    mapdl.d(n2, "UZ", 0)
