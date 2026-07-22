#!/usr/bin/env python3
"""
FEniCSx linear elasticity for PCB homogenization: 3 tensile tests.

Creates a layered hexahedral mesh matching the PCB stackup, assigns isotropic
elastic properties from configs/materials.yaml per cell (copper or FR4),
solves one displacement-controlled tensile test per coordinate direction, and
extracts effective Young's moduli and Poisson ratios.

Stackup modelled (3 layers):
    z = 0               … z_cu_bot      → bottom copper layer (Cu where map=1, FR4 otherwise)
    z = z_cu_bot        … z_cu_top      → FR4 core (always FR4)
    z = z_cu_top        … Lz            → top copper layer (same map as bottom)

Output in <sample_dir>/results/:
    homogenized.json                — effective E, ν, and CTE (always written)

With --save-xdmf (optional, for ParaView debugging):
    material.xdmf                   — material label per cell (1=Cu, 0=FR4)
    tensile_{x,y,z}_disp.xdmf      — displacement field [m]
    tensile_{x,y,z}_stress.xdmf    — von Mises stress field [Pa]
    cte_disp.xdmf                   — CTE free-expansion displacement [m]
    cte_alpha_ip.xdmf               — in-plane CTE field [1/K] (verify Cu assignment)

Usage:
    python simulate_tensile.py data/samples/example_test
    python simulate_tensile.py data/samples/example_test --nx 50 --ny 50 --nz-per-layer 2
    python simulate_tensile.py data/samples/example_test --materials configs/materials.yaml
    python simulate_tensile.py data/samples/example_test --save-xdmf
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ufl
import basix
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, la
from dolfinx.fem.petsc import assemble_matrix, apply_lifting, set_bc
from dolfinx.io import XDMFFile
from dolfinx.mesh import (
    compute_midpoints,
    create_mesh,
    locate_entities_boundary,
)
from dolfinx.fem import locate_dofs_topological, locate_dofs_geometrical

from homogenization import load_materials, voigt_reuss_bounds


# ---------------------------------------------------------------------------
# Layered hexahedral mesh with non-uniform z-spacing
# ---------------------------------------------------------------------------

def create_layered_mesh(comm, Lx: float, Ly: float, z_layers: list[float],
                        nx: int, ny: int, nz_per_layer: int):
    """
    Structured hex mesh with nz_per_layer elements in each physical z-layer.

    z_layers: boundary z-coordinates, e.g. [0, z_cu_bot, z_cu_top, Lz].
    Returns (msh, z_node_coords).
    """
    # Build z-node array: nz_per_layer sub-segments per physical layer
    z_segs = [np.linspace(z_layers[i], z_layers[i + 1], nz_per_layer + 1)
              for i in range(len(z_layers) - 1)]
    z_pts = np.concatenate([s[:-1] for s in z_segs] + [z_segs[-1][-1:]])

    x_pts = np.linspace(0., Lx, nx + 1)
    y_pts = np.linspace(0., Ly, ny + 1)
    nyn, nzn = len(y_pts), len(z_pts)
    nz = nzn - 1

    xx, yy, zz = np.meshgrid(x_pts, y_pts, z_pts, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)

    # Basix hexahedron vertex order (reference coords in {0,1}^3):
    # 0:(0,0,0) 1:(1,0,0) 2:(0,1,0) 3:(1,1,0) 4:(0,0,1) 5:(1,0,1) 6:(0,1,1) 7:(1,1,1)
    def nid(i, j, k):
        return (i * nyn + j) * nzn + k

    cells = np.array([
        [nid(i,   j,   k),   nid(i+1, j,   k),
         nid(i,   j+1, k),   nid(i+1, j+1, k),
         nid(i,   j,   k+1), nid(i+1, j,   k+1),
         nid(i,   j+1, k+1), nid(i+1, j+1, k+1)]
        for i in range(nx) for j in range(ny) for k in range(nz)
    ], dtype=np.int64)

    domain = ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))
    msh = create_mesh(comm, cells, points, domain)
    return msh, z_pts


# ---------------------------------------------------------------------------
# Material assignment
# ---------------------------------------------------------------------------

def assign_materials(msh, V0,
                     layer_maps: dict[str, np.ndarray],
                     Lx: float, Ly: float, z_cu_bot: float, z_cu_top: float,
                     mat: dict) -> tuple[fem.Function, fem.Function, np.ndarray]:
    """
    Populate DG0 lambda/mu functions and return a material label array.

    layer_maps: {'bottom': array, 'top': array} — per-layer rasters so that
                the bottom Cu layer and top Cu layer can carry different Gerber
                patterns (B.Cu vs F.Cu).  If a position key is missing, that
                layer is treated as all-FR4.

    Copper cells (label=1): in a copper z-layer AND the corresponding layer
                            map value > 0.5.
    FR4 cells  (label=0): all others (FR4 core always, copper layer where map=0).
    """
    tdim    = msh.topology.dim
    n_cells = msh.topology.index_map(tdim).size_local
    msh.topology.create_connectivity(tdim, 0)
    mid = compute_midpoints(msh, tdim, np.arange(n_cells, dtype=np.int32))

    E_cu,  nu_cu  = mat["copper"]["E_GPa"] * 1e9, mat["copper"]["nu"]
    E_fr4, nu_fr4 = mat["FR4"]["E_GPa"]    * 1e9, mat["FR4"]["nu"]

    E_arr  = np.full(n_cells, E_fr4,  dtype=np.float64)
    nu_arr = np.full(n_cells, nu_fr4, dtype=np.float64)
    label  = np.zeros(n_cells, dtype=np.float64)

    for c in range(n_cells):
        xc, yc, zc = mid[c]
        if zc < z_cu_bot:
            m = layer_maps.get('bottom')
        elif zc > z_cu_top:
            m = layer_maps.get('top')
        else:
            m = None                                 # FR4 core — always FR4

        if m is not None:
            ny_map, nx_map = m.shape
            ix = min(int(xc / Lx * nx_map), nx_map - 1)
            iy = min(int(yc / Ly * ny_map), ny_map - 1)
            if m[iy, ix] > 0.5:
                E_arr[c], nu_arr[c], label[c] = E_cu, nu_cu, 1.

    lam = fem.Function(V0)
    mu  = fem.Function(V0)
    lam.x.array[:] = (E_arr * nu_arr) / ((1 + nu_arr) * (1 - 2 * nu_arr))
    mu.x.array[:]  = E_arr / (2 * (1 + nu_arr))
    return lam, mu, label


def assign_alpha(msh, V0,
                 layer_maps: dict[str, np.ndarray],
                 Lx: float, Ly: float, z_cu_bot: float, z_cu_top: float,
                 mat: dict) -> tuple[fem.Function, fem.Function]:
    """
    Populate DG0 CTE component functions: alpha_ip (in-plane x,y) and alpha_op (out-of-plane z).

    Copper is isotropic: alpha_ip = alpha_op = CTE_per_K.
    FR4 is anisotropic:  alpha_ip = CTE_inplane_per_K, alpha_op = CTE_outofplane_per_K.
    """
    tdim    = msh.topology.dim
    n_cells = msh.topology.index_map(tdim).size_local
    mid     = compute_midpoints(msh, tdim, np.arange(n_cells, dtype=np.int32))

    a_cu   = mat["copper"].get("CTE_per_K", 17e-6)
    a_fr4_ip = mat["FR4"].get("CTE_inplane_per_K",    mat["FR4"].get("CTE_per_K", 14e-6))
    a_fr4_op = mat["FR4"].get("CTE_outofplane_per_K", mat["FR4"].get("CTE_per_K", 60e-6))

    ip_arr = np.full(n_cells, a_fr4_ip, dtype=np.float64)  # default: FR4 in-plane
    op_arr = np.full(n_cells, a_fr4_op, dtype=np.float64)  # default: FR4 out-of-plane

    for c in range(n_cells):
        xc, yc, zc = mid[c]
        if zc < z_cu_bot:
            m = layer_maps.get('bottom')
        elif zc > z_cu_top:
            m = layer_maps.get('top')
        else:
            m = None

        if m is not None:
            ny_map, nx_map = m.shape
            ix = min(int(xc / Lx * nx_map), nx_map - 1)
            iy = min(int(yc / Ly * ny_map), ny_map - 1)
            if m[iy, ix] > 0.5:
                ip_arr[c] = a_cu
                op_arr[c] = a_cu

    alpha_ip = fem.Function(V0)
    alpha_op = fem.Function(V0)
    alpha_ip.x.array[:] = ip_arr
    alpha_op.x.array[:] = op_arr
    return alpha_ip, alpha_op


# ---------------------------------------------------------------------------
# Boundary conditions for displacement-controlled tensile test
# ---------------------------------------------------------------------------

# For each loading direction: three point constraints that eliminate the
# remaining rigid body modes after the two face constraints are applied.
# Format: list of ((px,py,pz), component_index)
_RBC = {
    0: [((0, 0, 0), 1), ((0, 0, 0), 2), ((0, 0, 1), 1)],   # Lz placeholder for z-coord
    1: [((0, 0, 0), 0), ((0, 0, 0), 2), ((0, 0, 1), 0)],
    2: [((0, 0, 0), 0), ((0, 0, 0), 1), ((1, 0, 0), 1)],   # Lx placeholder for x-coord
}


def build_bcs(msh, V, direction: int, delta: float,
              Lx: float, Ly: float, Lz: float) -> list:
    L     = (Lx, Ly, Lz)[direction]
    fdim  = msh.topology.dim - 1

    f_lo = locate_entities_boundary(msh, fdim, lambda x, d=direction:      np.isclose(x[d], 0.))
    f_hi = locate_entities_boundary(msh, fdim, lambda x, d=direction, L=L: np.isclose(x[d], L))

    bcs = [
        fem.dirichletbc(PETSc.ScalarType(0.),
                        locate_dofs_topological(V.sub(direction), fdim, f_lo), V.sub(direction)),
        fem.dirichletbc(PETSc.ScalarType(delta),
                        locate_dofs_topological(V.sub(direction), fdim, f_hi), V.sub(direction)),
    ]

    # Resolve Lx/Lz placeholders and locate corner-point DOFs
    def resolve(pt):
        return (pt[0] if pt[0] != 1 or direction == 2 else Lx,
                pt[1],
                pt[2] * Lz)

    Vc = [V.sub(d).collapse()[0] for d in range(3)]
    for pt_raw, comp in _RBC[direction]:
        px, py, pz = resolve(pt_raw)
        def marker(x, px=px, py=py, pz=pz):
            return np.isclose(x[0], px) & np.isclose(x[1], py) & np.isclose(x[2], pz)
        dofs = locate_dofs_geometrical((V.sub(comp), Vc[comp]), marker)[0]
        if len(dofs):
            bcs.append(fem.dirichletbc(PETSc.ScalarType(0.), dofs, V.sub(comp)))

    return bcs


# ---------------------------------------------------------------------------
# Assemble, solve, post-process
# ---------------------------------------------------------------------------

def eps_fn(w):
    return ufl.sym(ufl.grad(w))


def sig_fn(w, lam, mu):
    return lam * ufl.tr(eps_fn(w)) * ufl.Identity(3) + 2 * mu * eps_fn(w)


def build_cte_bcs(msh, V, Lx: float, Ly: float):
    """
    3-2-1 fixture: 6 point constraints to remove all rigid body modes
    without restraining free thermal expansion.

    Point A (0, 0, 0)   : ux=uy=uz=0   (3 translations)
    Point B (Lx, 0, 0)  : uy=0, uz=0   (Rz, Ry rotations)
    Point C (0, Ly, 0)  : uz=0         (Rx rotation)
    """
    tol  = 1e-10 * max(Lx, Ly)
    bcs  = []
    pins = [
        (0, lambda x, Lx=Lx, Ly=Ly: np.isclose(x[0], 0,  atol=tol) & np.isclose(x[1], 0,  atol=tol) & np.isclose(x[2], 0, atol=tol)),
        (1, lambda x, Lx=Lx, Ly=Ly: np.isclose(x[0], 0,  atol=tol) & np.isclose(x[1], 0,  atol=tol) & np.isclose(x[2], 0, atol=tol)),
        (2, lambda x, Lx=Lx, Ly=Ly: np.isclose(x[0], 0,  atol=tol) & np.isclose(x[1], 0,  atol=tol) & np.isclose(x[2], 0, atol=tol)),
        (1, lambda x, Lx=Lx, Ly=Ly: np.isclose(x[0], Lx, atol=tol) & np.isclose(x[1], 0,  atol=tol) & np.isclose(x[2], 0, atol=tol)),
        (2, lambda x, Lx=Lx, Ly=Ly: np.isclose(x[0], Lx, atol=tol) & np.isclose(x[1], 0,  atol=tol) & np.isclose(x[2], 0, atol=tol)),
        (2, lambda x, Lx=Lx, Ly=Ly: np.isclose(x[0], 0,  atol=tol) & np.isclose(x[1], Ly, atol=tol) & np.isclose(x[2], 0, atol=tol)),
    ]
    for sub, marker in pins:
        Vc   = V.sub(sub).collapse()[0]
        dofs = locate_dofs_geometrical((V.sub(sub), Vc), marker)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs[0], V.sub(sub)))
    return bcs


def run_cte(msh, V, lam, mu, alpha_ip, alpha_op, bcs,
            delta_T: float, Lx: float, Ly: float, Lz: float) -> dict:
    """
    Thermoelastic free-expansion solve.

    Applies a uniform temperature change delta_T with a 3-2-1 fixture,
    then returns volume-averaged effective CTEs in x, y, z.

    Weak form:
        a(u,v) = ∫ σ_mech(u) : ε(v) dV
        L(v)   = ∫ σ_thermal  : ε(v) dV

    σ_thermal = λ·tr(α·ΔT)·I + 2μ·α·ΔT
    with α = diag(alpha_ip, alpha_ip, alpha_op) per cell.
    """
    dx   = ufl.Measure("dx", domain=msh)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = fem.form(ufl.inner(sig_fn(u, lam, mu), eps_fn(v)) * dx)

    # Thermal eigenstress per cell (anisotropic alpha tensor)
    alpha_t = ufl.as_tensor([[alpha_ip, 0, 0],
                              [0, alpha_ip, 0],
                              [0, 0,        alpha_op]])
    tr_alpha = 2 * alpha_ip + alpha_op
    sig_th   = (lam * tr_alpha * delta_T * ufl.Identity(3)
                + 2 * mu * alpha_t * delta_T)
    L = fem.form(ufl.inner(sig_th, eps_fn(v)) * dx)

    A = assemble_matrix(a, bcs=bcs)
    A.assemble()
    from dolfinx.fem.petsc import assemble_vector
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    uh = fem.Function(V)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc  = ksp.getPC()
    pc.setType("lu")
    try:
        pc.setFactorSolverType("mumps")
    except Exception:
        pass
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    vol  = Lx * Ly * Lz
    eps  = eps_fn(uh)
    a_x  = float(fem.assemble_scalar(fem.form(eps[0, 0] * dx))) / vol / delta_T
    a_y  = float(fem.assemble_scalar(fem.form(eps[1, 1] * dx))) / vol / delta_T
    a_z  = float(fem.assemble_scalar(fem.form(eps[2, 2] * dx))) / vol / delta_T

    return {"alpha_x_per_K": a_x, "alpha_y_per_K": a_y, "alpha_z_per_K": a_z, "uh": uh}


def run_tensile(msh, V, lam, mu, bcs,
                direction: int, delta: float,
                Lx: float, Ly: float, Lz: float):
    """Solve and return (uh, E_eff_GPa, nu_dict)."""
    dx   = ufl.Measure("dx", domain=msh)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = fem.form(ufl.inner(sig_fn(u, lam, mu), eps_fn(v)) * dx)
    A = assemble_matrix(a, bcs=bcs)
    A.assemble()

    b = la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    uh = fem.Function(V)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    try:
        pc.setFactorSolverType("mumps")
    except Exception:
        pass
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Homogenised moduli
    vol         = Lx * Ly * Lz
    eps_applied = delta / (Lx, Ly, Lz)[direction]
    names       = ["x", "y", "z"]

    sigma_avg = {i: float(fem.assemble_scalar(fem.form(sig_fn(uh, lam, mu)[i, i] * dx))) / vol
                 for i in range(3)}
    eps_avg   = {i: float(fem.assemble_scalar(fem.form(eps_fn(uh)[i, i] * dx))) / vol
                 for i in range(3)}

    E_eff  = sigma_avg[direction] / eps_applied
    nu_eff = {names[j]: -eps_avg[j] / eps_applied
              for j in range(3) if j != direction}

    return uh, E_eff, nu_eff


# ---------------------------------------------------------------------------
# XDMF output
# ---------------------------------------------------------------------------

def write_results(msh, uh, lam, mu, V0, results_dir: Path, tag: str) -> None:
    """Write displacement and von Mises stress to XDMF for ParaView."""
    from dolfinx.fem import Expression

    s    = sig_fn(uh, lam, mu)
    sdev = s - (ufl.tr(s) / 3) * ufl.Identity(3)
    svm  = ufl.sqrt((3 / 2) * ufl.inner(sdev, sdev))

    svm_expr = Expression(svm, V0.element.interpolation_points())
    svm_h    = fem.Function(V0, name="von_mises_Pa")
    svm_h.interpolate(svm_expr)
    uh.name = "displacement_m"

    with XDMFFile(msh.comm, str(results_dir / f"tensile_{tag}_disp.xdmf"), "w") as f:
        f.write_mesh(msh)
        f.write_function(uh)
    with XDMFFile(msh.comm, str(results_dir / f"tensile_{tag}_stress.xdmf"), "w") as f:
        f.write_mesh(msh)
        f.write_function(svm_h)


def write_cte_results(msh, uh, alpha_ip: fem.Function, results_dir: Path) -> None:
    """Write CTE displacement and in-plane alpha field to XDMF for ParaView."""
    uh.name = "cte_displacement_m"
    with XDMFFile(msh.comm, str(results_dir / "cte_disp.xdmf"), "w") as f:
        f.write_mesh(msh)
        f.write_function(uh)

    alpha_fn = fem.Function(alpha_ip.function_space, name="alpha_ip_per_K")
    alpha_fn.x.array[:] = alpha_ip.x.array[:]
    with XDMFFile(msh.comm, str(results_dir / "cte_alpha_ip.xdmf"), "w") as f:
        f.write_mesh(msh)
        f.write_function(alpha_fn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-direction tensile homogenization via FEniCSx linear elasticity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--nx", type=int, default=50,
                        help="Mesh elements in x (independent of material_map px count)")
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--nz-per-layer", type=int, default=2,
                        help="Hex elements per z-layer (bot Cu / FR4 core / top Cu)")
    parser.add_argument("--strain", type=float, default=1e-3,
                        help="Applied nominal strain")
    parser.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    parser.add_argument("--save-xdmf", action="store_true",
                        help="Write XDMF fields for ParaView (material, tensile, CTE). "
                             "Skipped by default to save disk space in dataset runs.")
    args = parser.parse_args()

    if not args.sample_dir.exists():
        sys.exit(f"Not found: {args.sample_dir}")

    comm = MPI.COMM_WORLD

    # ── Load inputs ───────────────────────────────────────────────────────────
    with open(args.sample_dir / "metadata.json") as f:
        meta = json.load(f)

    disc_path = args.sample_dir / "mesh" / "discretization.json"
    if not disc_path.exists():
        sys.exit(f"Run discretize.py first — not found: {disc_path}")

    with open(disc_path) as f:
        disc = json.load(f)

    # Load per-layer maps keyed by position ('top' / 'bottom').
    # Falls back to the combined OR map for both layers when per-layer files
    # are absent (e.g. samples discretized before this fix was applied).
    mesh_dir   = args.sample_dir / "mesh"
    layer_maps: dict[str, np.ndarray] = {}
    for _lname, linfo in disc.get("layer_maps", {}).items():
        pos  = linfo.get("position")
        fpath = mesh_dir / linfo["file"]
        if pos in ("top", "bottom") and fpath.exists():
            layer_maps[pos] = np.load(fpath)

    if not layer_maps:
        print("[warn] Per-layer maps not found — using combined map for both Cu layers.",
              file=sys.stderr)
        combined = np.load(mesh_dir / "material_map.npy")
        layer_maps = {"top": combined, "bottom": combined}

    mat = load_materials(args.materials)

    # ── Geometry [m] ──────────────────────────────────────────────────────────
    Lx    = meta["board"]["width_mm"]              * 1e-3
    Ly    = meta["board"]["height_mm"]             * 1e-3
    cu_t  = meta["stackup"]["copper_thickness_mm"] * 1e-3
    fr4_t = meta["stackup"]["dielectric_thickness_mm"] * 1e-3
    Lz    = meta["stackup"]["total_thickness_mm"]  * 1e-3

    z_cu_bot, z_cu_top = cu_t, cu_t + fr4_t
    nz_total = args.nz_per_layer * 3

    print(f"Board    : {Lx*1e3:.1f} × {Ly*1e3:.1f} × {Lz*1e3:.2f} mm")
    print(f"Stackup  : Cu {cu_t*1e6:.0f} µm | FR4 {fr4_t*1e3:.2f} mm | Cu {cu_t*1e6:.0f} µm")
    print(f"FE mesh  : {args.nx} × {args.ny} × {nz_total}  ({args.nx*args.ny*nz_total:,} cells)")

    # ── Build mesh ────────────────────────────────────────────────────────────
    msh, z_pts = create_layered_mesh(
        comm, Lx, Ly, [0., z_cu_bot, z_cu_top, Lz],
        args.nx, args.ny, args.nz_per_layer
    )

    # ── FE spaces ─────────────────────────────────────────────────────────────
    cname = msh.ufl_cell().cellname()
    V  = fem.functionspace(msh, basix.ufl.element("Lagrange", cname, 1, shape=(3,)))
    V0 = fem.functionspace(msh, basix.ufl.element("DG", cname, 0))

    # ── Materials ─────────────────────────────────────────────────────────────
    print("Assigning materials …")
    lam, mu, label = assign_materials(msh, V0, layer_maps, Lx, Ly, z_cu_bot, z_cu_top, mat)
    print(f"  Cu cell fraction: {label.mean():.4f}")

    results_dir = args.sample_dir / "results"
    results_dir.mkdir(exist_ok=True)

    if args.save_xdmf:
        mat_fn = fem.Function(V0, name="material_1Cu_0FR4")
        mat_fn.x.array[:] = label
        with XDMFFile(comm, str(results_dir / "material.xdmf"), "w") as f:
            f.write_mesh(msh)
            f.write_function(mat_fn)
        print("  Saved  results/material.xdmf")

    # ── Voigt / Reuss bounds ─────────────────────────────────────────────────
    phi_top = float(layer_maps["top"].mean())    if "top"    in layer_maps else 0.0
    phi_bot = float(layer_maps["bottom"].mean()) if "bottom" in layer_maps else 0.0
    bounds  = voigt_reuss_bounds(
        phi_top, phi_bot,
        E_cu=mat["copper"]["E_GPa"] * 1e9,
        E_fr4=mat["FR4"]["E_GPa"]   * 1e9,
        t_cu=cu_t, t_fr4=fr4_t,
        alpha_cu=mat["copper"].get("CTE_per_K", 17e-6),
        alpha_fr4_ip=mat["FR4"].get("CTE_inplane_per_K",    mat["FR4"].get("CTE_per_K", 14e-6)),
        alpha_fr4_op=mat["FR4"].get("CTE_outofplane_per_K", mat["FR4"].get("CTE_per_K", 60e-6)),
    )
    print(f"\nVoigt-Reuss bounds:")
    print(f"  phi_top={phi_top:.4f}  phi_bot={phi_bot:.4f}  phi_3D={bounds['phi_3D']:.4f}")
    print(f"  E_xy_voigt = {bounds['E_xy_voigt_GPa']:.4f} GPa  (structural upper bound for E_x, E_y)")
    print(f"  E_z_reuss  = {bounds['E_z_reuss_GPa']:.4f} GPa  (structural lower bound for E_z)")
    print(f"  E_voigt_global = {bounds['E_voigt_global_GPa']:.4f} GPa")
    print(f"  E_reuss_global = {bounds['E_reuss_global_GPa']:.4f} GPa")
    print(f"  alpha_xy_voigt = {bounds['alpha_xy_voigt_per_K']*1e6:.2f} ppm/K  (in-plane CTE bound)")
    print(f"  alpha_z_reuss  = {bounds['alpha_z_reuss_per_K']*1e6:.2f} ppm/K  (out-of-plane CTE bound)")

    # ── Tensile simulations ───────────────────────────────────────────────────
    names = ["x", "y", "z"]
    out: dict = {
        "sample_id":      meta["sample_id"],
        "mesh":           {"nx": args.nx, "ny": args.ny, "nz_per_layer": args.nz_per_layer,
                           "total_cells": args.nx * args.ny * nz_total},
        "applied_strain": args.strain,
        "materials":      {k: {p: v for p, v in mat[k].items() if p in ("E_GPa", "nu")}
                           for k in ("copper", "FR4")},
        "timestamp":      datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        **bounds,
    }

    for d, name in enumerate(names):
        print(f"\nTensile {name}  (strain = {args.strain}) …")
        bcs = build_bcs(msh, V, d, args.strain, Lx, Ly, Lz)
        uh, E_eff, nu_eff = run_tensile(msh, V, lam, mu, bcs, d, args.strain, Lx, Ly, Lz)

        out[f"E_{name}_GPa"] = round(E_eff / 1e9, 6)
        for j_name, nu in nu_eff.items():
            out[f"nu_{name}{j_name}"] = round(nu, 6)

        print(f"  E_{name}  = {E_eff/1e9:.4f} GPa")
        for j_name, nu in nu_eff.items():
            print(f"  nu_{name}{j_name} = {nu:.5f}")

        if args.save_xdmf:
            write_results(msh, uh, lam, mu, V0, results_dir, name)
            print(f"  Saved  results/tensile_{name}_{{disp,stress}}.xdmf")

    # ── CTE simulation ────────────────────────────────────────────────────────
    print("\nCTE simulation  (ΔT = 100 K) …")
    alpha_ip, alpha_op = assign_alpha(msh, V0, layer_maps, Lx, Ly, z_cu_bot, z_cu_top, mat)
    bcs_cte = build_cte_bcs(msh, V, Lx, Ly)
    cte = run_cte(msh, V, lam, mu, alpha_ip, alpha_op, bcs_cte,
                  delta_T=100.0, Lx=Lx, Ly=Ly, Lz=Lz)
    out["alpha_x_per_K"] = round(cte["alpha_x_per_K"], 9)
    out["alpha_y_per_K"] = round(cte["alpha_y_per_K"], 9)
    out["alpha_z_per_K"] = round(cte["alpha_z_per_K"], 9)
    print(f"  α_x = {cte['alpha_x_per_K']*1e6:.3f} ppm/K")
    print(f"  α_y = {cte['alpha_y_per_K']*1e6:.3f} ppm/K")
    print(f"  α_z = {cte['alpha_z_per_K']*1e6:.3f} ppm/K")
    if args.save_xdmf:
        write_cte_results(msh, cte["uh"], alpha_ip, results_dir)
        print("  Saved  results/cte_{disp,alpha_ip}.xdmf")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    (results_dir / "homogenized.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nDone. Effective properties → {results_dir / 'homogenized.json'}")


if __name__ == "__main__":
    main()
