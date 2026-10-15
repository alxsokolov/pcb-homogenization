# Restructuring plan: one n-layer pipeline, uniform grid, minimal surface

**Audience: the implementing agent (Sonnet) working in this repository.** Execute the phases in
order. Every phase ends with a working repo and a stated acceptance check — do not start the next
phase until the current one's check passes. Do not "improve" physics, formulas or defaults beyond
what is written here; this is a restructuring, not a modelling change. Where a behaviour change is
unavoidable, it is listed in §9 and must be recorded in `CHANGELOG.md`.

---

## 1. Requirements and how they map to work

| # | Requirement | Where it is handled |
|---|---|---|
| 0 | Copy to a new GitHub repo before changing anything | Phase 0 |
| 1 | Unify 2-layer and n-layer into one n-layer-capable workflow; a 2-layer board is just top + bottom with zero inner layers | Phases 1–5; the n-layer stackup model (`stackup.layers[]`, 3 slabs for a 2-layer board) becomes the only model |
| 2 | Drop quadtree; keep uniform quad discretization; homogenization grid independent of FE mesh | Phase 3 (`homogenize.py --gx/--gy`) + Phase 4/5 (`--nx/--ny`) |
| 3 | ODB++ and Gerber import both possible | Phase 2 (`import_board.py gerber|odb`) |
| 4 | Creep postprocessing, component placement, creep simulation stay as they are | Phases 5–6: `pcbh/component.py` and `postprocess_creep.py` keep their logic; only the board side changes |
| 5 | Maximally minimalistic, easy to follow | 34 Python files → 21; ~10.3 kLOC → ~6.5 kLOC; one pipeline, one artifact per step |

### Target pipeline (the whole project, after restructuring)

```
Gerber / ODB++ ──► import_board.py     ──► <sample>/metadata.json      (board + full stackup)
               ──► discretize.py       ──► <sample>/mesh/raster_*.npy  (one per copper layer)
               ──► homogenize.py       ──► <sample>/mesh/homogenization.json
                                            (uniform gx × gy grid × Voigt-Reuss materials)
               ──► simulate_board.py   ──► <sample>/results/board_*.json      (CTE / warpage / k_eff)
               ──► simulate_assembly.py──► <sample>/results/assembly.rst      (+ component, solder, creep)
               ──► postprocess_creep.py──► <sample>/results/zone_creep_strain.json
```

Two independent discretizations, which is the point of the project and must stay visible in the CLI:

* **homogenization grid** `--gx/--gy` (`homogenize.py`) — over how large a region the copper is averaged.
* **FE mesh** `--nx/--ny/--nz-per-layer` (`simulate_*.py`) — the element size the solver sees.

They are unrelated: `--gx 10 --gy 10` with `--nx 50 --ny 50` is legal and is the configuration the
existing `rv3032_homogz_50x50_nz10_g10/g20/g25` samples were produced with.

---

## 2. Decisions already taken

These are my calls. If the user disagrees with any of them, that line is the only thing that
changes — flag it before starting the affected phase rather than inventing a compromise.

| D | Decision | Rationale |
|---|---|---|
| D1 | Library code moves into a flat package `pcbh/`; only runnable steps stay at the repo root | Today `simulate_pyansys_assembly.py` imports internals of `simulate_pyansys_benchmark.py`, and three plot scripts import `quadtree._load_layer_maps`. "Which file do I run?" must be answerable from `ls` |
| D2 | No `pyproject.toml`, no installable package, no `src/` layout. Scripts are run from the repo root, `pcbh` imports work because the root is on `sys.path` | Minimalism; matches how the project is actually used |
| D3 | The n-layer schema is the **only** metadata schema. The legacy flat `copper_thickness_mm`/`dielectric_thickness_mm` keys are gone, including the `_slabs_from_legacy2()` reader | Two schemas is the main reason the current code has two of everything |
| D4 | Existing samples under `data/` are migrated once by a throwaway script (Phase 2), not supported forever by a compatibility branch in the library | Keeps the library single-path. `data/` is not in git; it lives on the MikroVAL share |
| D5 | `--board-model raster` disappears. The full-resolution reference becomes `homogenize.py --model projection --gx <nx> --gy <ny>` | One code path instead of two. Small, documented numerical change: an element is copper if its copper **area fraction** ≥ 0.5, not if the raster pixel at its centre is copper |
| D6 | `--cu-penetration-elements` and the whole `dilution` mechanism are deleted | The unified mesh bands z by physical slab, so every copper layer is resolved exactly at its true thickness. Dilution existed only to patch the 2-layer mesh's inability to do that |
| D7 | `solder_reliability.py`'s analytic Coffin-Manson QoI is deleted. Its pad coordinates survive as `pcbh/pads.py` | It reads quadtree leaves (gone), and its own docstring calls it illustrative and uncalibrated. The real QoI is the creep postprocessing, which stays untouched. **This is the one deletion a user might want vetoed** — if kept, it needs ~20 lines rewriting against the grid model |
| D8 | `simulate_pyansys_nlayer.py` (structural) and `simulate_pyansys_thermal_nlayer.py` (conductivity) merge into `simulate_board.py --analysis cte|k` | They already share mesh, materials and MAPDL lifecycle via imports; only element type, BCs and QoI differ |
| D9 | `QUICKSTART.md` is deleted; its content folds into `README.md` | It duplicates the README and the two will drift |
| D10 | `remap_board_materials()` / the board-MAT-remap dance is dropped: board materials are allocated from MAT 9001 upward at element creation, so they never collide with the component archive's MAT 1–10. `reassert_board_materials()` is **kept** and still called after `CDREAD` | The remap was only needed because the board used MAT 1/2. The reassert fixes a confirmed CDREAD bug and is cheap — keeping it is the low-risk choice |
| D11 | `odb_parser.parse_odb_attrlist()`'s whitespace bug is fixed (see CHANGELOG "Known issue in the 2-layer ODB++ path") | The reason not to fix it was "don't change 2-layer results"; there is no separate 2-layer path any more. Changes `hella`'s copper thickness 0.035 → 0.036614 mm — record it |
| D12 | Python 3.14, numpy, matplotlib, PyYAML, PyMAPDL pinned as today. `pytest` is added for the new `tests/` | No new runtime dependencies |

---

## 3. Target repository layout

```
pcb-homogenization/
  README.md            one page: pipeline, install, quickstart, limitations
  CHANGELOG.md
  LICENSE
  requirements.txt
  RESTRUCTURING_PLAN.md    (delete at the end of Phase 7)

  import_board.py          gerber | odb  -> sample
  discretize.py            rasterize every copper layer
  homogenize.py            uniform grid x region model -> materials
  simulate_board.py        board-only MAPDL solve (--analysis cte|k)
  simulate_assembly.py     board + CR0805 + SAC305 + thermal profile (+creep)
  postprocess_creep.py     zone-averaged creep strain from the .rst

  plot_board.py            before the solve: layout, per-layer phi grid, stackup profile
  plot_solved_board.py     after the solve: board as MAPDL saw it, read from the .rst
  plot_creep.py            creep history + per-cycle increment, one or several samples
  plot_convergence.py      QoI vs homogenization-grid and vs mesh resolution
  plot_component.py        3-D material render of the assembly

  pcbh/
    __init__.py            empty
    materials.py           load_materials(), PhaseProps
    homogenization.py      n-slab Voigt-Reuss (the only homogenization module)
    stackup.py             Slab, BoardStackup, raster lookup + block means
    raster.py              format-agnostic geometry model + rasterizer
    gerber.py              RS-274X parser -> LayerGeometry, coordinate shift
    odb.py                 ODB++ matrix/attrlist/features parser -> LayerGeometry
    odb_stackup.py         ODB++ -> physical stackup
    grid.py                uniform homogenization grid + material table + JSON I/O
    mesh.py                MAPDL: z-banded hex mesh, materials, fixture, lifecycle
    component.py           component archive import, placement, bonded contact
    thermal.py             thermal profile load / plot / apply
    pads.py                COMPONENT_NAME, PAD_1_XY_MM, PAD_2_XY_MM

  configs/
    materials.yaml
    thermal_profile.yaml
    thermal_profile_example.yaml
    stackup_example.yaml     NEW: documents the Gerber stackup YAML format

  tests/
    test_homogenization.py   analytic invariants + closed-form 3-slab values
    test_grid.py             conservation, dedup, element->region mapping
    test_io.py               gerber + odb parse of bundled samples, golden phi
    golden/                  small JSONs captured in Phase 0
    data/synth4/             synthetic 4-copper-layer sample (Phase 0)

  docs/
    overview/main.tex, make_figures.py, figures/
    workflow_overview.pptx
```

### File disposition (every current file is accounted for)

| Current file | Action |
|---|---|
| `homogenization.py` | **delete**. `load_materials()` → `pcbh/materials.py`; 2-layer bounds superseded |
| `homogenization_nlayer.py` | → `pcbh/homogenization.py`; `PhaseProps` → `pcbh/materials.py`; drop `dilution=`; `selftest()` → `tests/test_homogenization.py` |
| `stackup_nlayer.py` | → `pcbh/stackup.py`; drop `_slabs_from_legacy2()`; add `layer_phi_grid()` |
| `quadtree.py` | **delete** (requirement 2). `_load_layer_maps()` → `pcbh/stackup.attach_rasters()` |
| `coarsegrain_nlayer.py` | → `homogenize.py` + `pcbh/grid.py`; quadtree modes removed |
| `discretize.py` | stays as the CLI; its Gerber parser → `pcbh/gerber.py`, its duplicated polygon/raster helpers deleted in favour of `pcbh/raster.py` |
| `raster_geometry.py` | → `pcbh/raster.py` (already format-agnostic; now used by both inputs) |
| `odb_parser.py` | → `pcbh/odb.py` (+ D11 fix) |
| `odb_stackup.py` | → `pcbh/odb_stackup.py` (drop its private duplicate attrlist parser once D11 lands) |
| `import_real_board.py` | **delete** → `import_board.py gerber`; `_shift_gerber()` → `pcbh/gerber.shift_coordinates()` |
| `import_gerber_board_nlayer.py` | **delete** → `import_board.py gerber` |
| `import_odb_board.py` | **delete** (2-layer-only) |
| `import_odb_board_nlayer.py` | **delete** → `import_board.py odb` |
| `simulate_pyansys_benchmark.py` | **delete**. Mesh builder superseded by `pcbh/mesh.py`; its pad-displacement QoI superseded |
| `simulate_pyansys_nlayer.py` | → `simulate_board.py` + `pcbh/mesh.py` |
| `simulate_pyansys_thermal_nlayer.py` | → `simulate_board.py --analysis k` (D8) |
| `simulate_pyansys_assembly.py` | → `simulate_assembly.py`, board side rebuilt on `pcbh/mesh.py` |
| `component_import.py` | → `pcbh/component.py`, **logic unchanged** except D10 and the contact-selection change (§6, Phase 5) |
| `thermal_profile.py` | → `pcbh/thermal.py`, unchanged; its standalone plot CLI replaced by `simulate_assembly.py --dry-run` |
| `zone_creep_strain.py` | → `postprocess_creep.py`, **unchanged** (rename only, plus the import of nothing project-local — it has no project imports today) |
| `solder_reliability.py` | **delete** (D7); constants → `pcbh/pads.py` |
| `plot_board_preview.py` + `plot_coarsegrain_nlayer.py` | → `plot_board.py` |
| `plot_board_model.py` | → `plot_solved_board.py` (adapted to n slabs and the 9001+ MAT band) |
| `plot_creep_strain_history.py` + `plot_cycle_creep_increment.py` + `plot_comparison_creep.py` + `plot_delta_summary.py` | → `plot_creep.py` |
| `plot_leaf_resolution_sweep.py` + `plot_discretization_sensitivity.py` + `compare_nlayer_models.py` | → `plot_convergence.py` |
| `plot_cu_penetration_sweep.py` | **delete** (D6) |
| `plot_component_materials.py` | → `plot_component.py`, MAT legend updated for the 9001+ band |
| `docs/overview/make_figures.py` | update to the new plot scripts and sample layout |
| `QUICKSTART.md` | **delete** (D9) |
| `~$workflow_overview.pptx` | **delete** (Office lock turd, already gitignored by `~$*`) |

---

## 4. Data contracts

Freeze these first; everything else follows. All lengths in mm, all MAPDL material values in the
mm-N-MPa system (E, G in MPa; CTE in 1/K; conductivity in W/(mm·K)).

**Raster convention (unchanged, state it in `pcbh/raster.py` and `pcbh/stackup.py`):** a raster is
`(ny, nx)`, row index increases with y, column index with x, values in {0,1}, threshold 0.5. Only
PNG previews flip y.

**Mesh index convention (unchanged, state it in `pcbh/mesh.py`):** node id `(i*nyn + j)*nzn + k + 1`
with `i→x`, `j→y`, `k→z`; cells are iterated `cell = i*ny + j`.

### 4.1 `<sample>/metadata.json` (schema_version 2)

```json
{
  "schema_version": 2,
  "sample_id": "rv3032_example",
  "board": {"width_mm": 20.0, "height_mm": 15.0},
  "layers": [
    {"name": "F_Cu", "file": "gerber/F_Cu.gbr", "format": "gerber",
     "position": "top", "stackup_index": 2}
  ],
  "stackup": {
    "dielectric_material": "FR4",
    "total_thickness_mm": 1.6,
    "layers": [
      {"index": 0, "kind": "copper",     "name": "B_Cu", "layer_name": "B_Cu",
       "thickness_mm": 0.035, "z0_mm": 0.0,   "z1_mm": 0.035},
      {"index": 1, "kind": "dielectric", "name": "core",
       "thickness_mm": 1.53,  "z0_mm": 0.035, "z1_mm": 1.565, "material": "FR4"},
      {"index": 2, "kind": "copper",     "name": "F_Cu", "layer_name": "F_Cu",
       "thickness_mm": 0.035, "z0_mm": 1.565, "z1_mm": 1.6}
    ]
  },
  "source": {"format": "gerber", "bbox_mm": [122.0, -105.0, 142.0, -90.0],
             "files": ["..."], "stackup_file": null},
  "generation": {"script": "import_board.py", "timestamp": "..."}
}
```

Rules:
* `layers[]` is copper only, **top first**, each pointing into `stackup.layers[]` by `stackup_index`.
* `stackup.layers[]` is **every** slab, ascending z, `z0_mm == 0` at the bottom face, contiguous.
* Thickness lives in `stackup.layers[]` only — remove the duplicate `thickness_mm` from `layers[]`.
* A 2-layer board has exactly 3 slabs. An n-copper-layer board has `2n-1` slabs.
* No `type: "copper"` key (`layers[]` is copper by definition), no `n_copper_layers` (it is `len`),
  no flat thickness keys (D3).
* ODB++ samples additionally keep `source.odb = {job_step, outline_source, origin_offset_mm,
  drill_layers_not_imported}` — `origin_offset_mm` is read by `discretize.py`.

### 4.2 `<sample>/mesh/discretization.json`

```json
{
  "schema_version": 2,
  "sample_id": "...",
  "resolution_px_per_mm": 10.0,
  "grid": {"nx": 200, "ny": 150, "dx_mm": 0.1, "dy_mm": 0.1},
  "board": {"width_mm": 20.0, "height_mm": 15.0},
  "layers": {
    "F_Cu": {"file": "raster_F_Cu.npy", "stackup_index": 2, "copper_fraction": 0.4123}
  },
  "timestamp": "..."
}
```

`material_map.npy` (combined OR of all layers) is **not** written any more — it has no meaning for
an n-layer board and nothing in the new code reads it.

### 4.3 `<sample>/mesh/homogenization.json`

```json
{
  "schema_version": 1,
  "sample_id": "...",
  "generated_by": "homogenize.py",
  "board": {"width_mm": 20.0, "height_mm": 15.0, "total_thickness_mm": 1.6},
  "grid": {"gx": 20, "gy": 20, "dx_mm": 1.0, "dy_mm": 0.75},
  "region_model": "zlayers",
  "projection_threshold": null,
  "material_units": "MAPDL mm-N-MPa",
  "stackup": [ ... the slab records, copied from metadata ... ],
  "materials": {"m0": {"EX": ..., "EY": ..., "EZ": ..., "PRXY": ..., "PRYZ": ...,
                       "PRXZ": ..., "GXY": ..., "GYZ": ..., "GXZ": ...,
                       "ALPX": ..., "ALPY": ..., "ALPZ": ...,
                       "KXX": ..., "KYY": ..., "KZZ": ...}},
  "layers": [
    {"slab_index": 2, "layer_name": "F_Cu",
     "phi":      [[...gx values...], ...gy rows...],
     "material": [["m3", "m7", ...], ...gy rows...]}
  ],
  "column_material": null,
  "phi_3D": [[...]],
  "n_materials": 137
}
```

* Row index = y, column index = x (same as the rasters). Say so in the file's `generated_by` comment
  and in the loader docstring.
* `region_model`:
  * `zlayers` (default) — `layers[]` populated, one material per (region, copper slab) from that
    slab's own φ; dielectric slabs get the pure dielectric material.
  * `smeared` — `layers[]` null, `column_material` is a `gy × gx` array; every element in a region's
    column gets it, dielectrics included.
  * `projection` — same shape as `zlayers`, but each (region, slab) φ was first snapped to 0 or 1 at
    `projection_threshold`.
* Materials are deduplicated by rounded φ (6 decimals), as `coarsegrain_nlayer.py` does today.

---

## 5. Library API

Only the signatures that matter. Keep the existing docstrings wherever the code moves unchanged —
they carry empirically-won knowledge (CDREAD quirks, MAPDL process cleanup, aspect ratios, the
structural-vs-global bounds proof). Do not rewrite them to be shorter.

### `pcbh/materials.py`
```python
def load_materials(path: Path | None) -> dict                 # from homogenization.py, unchanged
@dataclass class PhaseProps:                                  # from homogenization_nlayer.py
    E_cu_MPa; nu_cu; alpha_cu; E_d_MPa; nu_d; alpha_d_ip; alpha_d_op; k_cu; k_d
    @classmethod from_materials(cls, materials: dict) -> PhaseProps
```

### `pcbh/homogenization.py`
```python
@dataclass class SlabSpec: thickness_mm; phi=0.0; kind="copper"; name=""
def slab_properties(phi, p) -> dict
def laminate_bounds(slabs: list[SlabSpec], p) -> dict
def orthotropic_slab(phi, p) -> dict          # drop the dilution argument (D6)
def orthotropic_smeared(bounds) -> dict
def pure_phase_material(is_copper: bool, p) -> dict
```
Unchanged formulas. Keep the module docstring's proof that the structural bounds equal the global
ones; drop the paragraphs that compare against the deleted 2-layer module.

### `pcbh/stackup.py`
```python
@dataclass class Slab: index; kind; name; thickness_mm; z0_mm; z1_mm; layer_name=None
                       material=None; raster=None
@dataclass class BoardStackup: sample_id; width_mm; height_mm; slabs; dielectric_material
    copper_slabs; n_copper; total_thickness_mm; z_interfaces_mm(); slab_at_z(z)
    is_copper_at(x, y, z); copper_fraction_3d(); describe()

def load_board(sample_dir, rasters=True) -> BoardStackup      # schema 2 only
def attach_rasters(sample_dir, board) -> None                 # replaces quadtree._load_layer_maps
def raster_lookup(raster, x, y, w, h) -> bool
def layer_phi_grid(raster, gx, gy) -> np.ndarray              # (gy, gx) area fractions   NEW
```
`layer_phi_grid` is the one new numeric routine. When `nx % gx == 0 and ny % gy == 0` it is an exact
reshape-and-mean (`raster.reshape(gy, ny//gy, gx, nx//gx).mean(axis=(1, 3))`) and conserves copper
area exactly. Otherwise fall back to the current per-cell `floor/ceil` pixel slice (identical to
`stackup_nlayer.raster_area_mean`) and print one warning that boundary pixels are counted twice.
`load_board` must raise a clear error on a schema-1 sample, naming the migration script.

### `pcbh/grid.py`
```python
@dataclass class GridModel:
    gx; gy; region_model; projection_threshold; board; stackup
    materials: dict[str, dict]
    layer_material: dict[int, np.ndarray]    # slab index -> (gy, gx) of material names
    layer_phi: dict[int, np.ndarray]
    column_material: np.ndarray | None
    phi_3D: np.ndarray

def build(board, gx, gy, region_model, props, threshold=0.5) -> GridModel
def save(model, path) -> None
def load(path) -> GridModel
def region_of_element(ix, iy, nx, ny, gx, gy) -> tuple[int, int]
    # ((ix + 0.5) * gx) // nx  clamped — pure arithmetic, no bbox search
def mat_resolver(model, mat_ids: dict[str, int], nx, ny) -> Callable[[int, int, int], int]
    # (ix, iy, slab_index) -> MAPDL MAT id, closed over the grid; this is what pcbh.mesh calls
```
This replaces `coarsegrain_nlayer.region_of_cells()` and
`simulate_pyansys_nlayer.region_of_cells()`'s chunked `(cells × regions)` boolean entirely — with a
uniform grid the lookup is integer arithmetic.

### `pcbh/mesh.py`
```python
@dataclass class ZGrid: z_pts; slab_of_row; nz
def build_z_grid(board, nz_per_layer) -> ZGrid

@dataclass class MeshInfo: corner_000; corner_Lx00; corner_0Ly0; corner_LxLy0
    top_nodes; x0_nodes; xL_nodes; y0_nodes; yL_nodes; z0_nodes; zL_nodes
    n_nodes; n_elements; n_materials_used; board_cm = "BOARD"

BOARD_MAT_BASE = 9001          # board materials live here and above (D10)

def register_materials(mapdl, model: GridModel | None, props) -> dict[str, int]
def build_mesh(mapdl, board, zgrid, nx, ny,
               mat_of_cell: Callable[[int, int, int], int],
               elem_type: str = "SOLID186") -> MeshInfo
def apply_321_fixture(mapdl, mesh: MeshInfo) -> None
def launch(run_location: Path, nproc: int | None) -> "Mapdl"
def shutdown(mapdl, run_location) -> None      # mapdl.exit() + _force_kill_mapdl_processes()
```
`build_mesh` must, right after creating the elements, `mapdl.esel("S","TYPE","",1)` →
`mapdl.cm("BOARD","ELEM")` → `mapdl.allsel()`. That named component is how the assembly script finds
the board afterwards, independent of material ids (see Phase 5). Keep the `_ELEMENT_WARN_LIMIT`
warning, the aspect-ratio report and the group-by-MAT element emission (one `mapdl.mat()` per group).
Keep `_force_kill_mapdl_processes()` verbatim, including its docstring — it is the only defence
against stale `file.lock`.

### `pcbh/component.py`, `pcbh/thermal.py`, `pcbh/pads.py`
Moved verbatim, except:
* `component.py`: delete `remap_board_materials()` and `_board_mat_cmname()`; change
  `reassert_board_materials(mapdl, new_mat_ids)` to `reassert_board_materials(mapdl, board_cm="BOARD")`
  which does `CMSEL,S,BOARD` and re-`EMODIF`s each element back to the MAT it had before `CDREAD`
  (capture `{elem_id: mat}` for the board elements before the import, or re-derive from the grid —
  capturing the map is simpler and exact). Change `generate_bonded_contact(...)` to take
  `board_cm: str` instead of `board_mat_ids`/`board_pad_mat_id`, selecting the board side with
  `CMSEL,S,<board_cm>` and stamping the TARGE170 elements with the copper pure-phase MAT id.
* `thermal.py`: unchanged, `main()` removed (the CLI moves to `simulate_assembly.py --dry-run`).

---

## 6. CLI specifications

Keep `argparse` with `ArgumentDefaultsHelpFormatter`, `--help` epilogs where the current scripts
have them, and the "set `PYTHONUTF8=1` on Windows" note in the README.

### `import_board.py`
```
python import_board.py gerber --sample-id ID --copper F_Cu.gbr [In1_Cu.gbr ...] B_Cu.gbr \
       --bbox XMIN YMIN XMAX YMAX [--stackup configs/stackup_x.yaml] [--output-root data/samples]
python import_board.py odb    --sample-id ID --odb-dir DIR [--step NAME] \
       [--stackup configs/stackup_x.yaml] [--default-copper-thickness-mm 0.035] [--source NOTE]
```
* Two subparsers, one shared `_write_sample(...)` that builds `metadata.json` (§4.1) and prints the
  stackup table + `Next: python discretize.py <sample>`.
* `gerber`: copper files **top first**; names `F_Cu`, `In1_Cu`…`In<n-2>_Cu`, `B_Cu`; coordinates
  shifted so the bbox starts at (0,0); `--stackup` required for > 2 copper layers; for exactly 2 it
  defaults to 0.035 mm copper / 1.53 mm dielectric (keeps `rv3032_example` reproducible).
* `odb`: all conductive layers (SIGNAL/POWER_GROUND/MIXED), stackup from `pcbh/odb_stackup.py`,
  bbox from `profile` or a name-matched outline layer, drill layers reported as not imported, the
  trimmed `odb/` subtree copied as today.
* Both: fail loudly on a zero/negative slab thickness, naming `--stackup`.

### `discretize.py`
```
python discretize.py <sample> [--resolution 10.0]
```
Per copper layer in `metadata.json`: parse with `pcbh.gerber` or `pcbh.odb` depending on
`layers[].format`, rasterize with `pcbh.raster.rasterize_layer`, save `raster_<name>.npy`, report
dropped unsupported shapes, write `discretization.json` (§4.2). Print each layer's copper fraction.

### `homogenize.py`
```
python homogenize.py <sample> --gx 20 --gy 20 [--model zlayers|smeared|projection]
       [--threshold 0.5] [--materials configs/materials.yaml]
```
Writes `mesh/homogenization.json`. Keep `coarsegrain_nlayer.py`'s closing **conservation check**
(area-weighted mean of region φ vs whole-layer φ per copper slab, with the `5e-3` warning) — with an
exact grid division it should now be ~1e-16, which makes it a real test. Warn if the raster is not an
integer multiple of the grid, and warn if `gx`/`gy` exceeds a mesh resolution the user may later use
(cannot be known here — just note in `--help` that regions finer than elements are not resolved).

### `simulate_board.py`
```
python simulate_board.py <sample> --nx 40 --ny 40 [--nz-per-layer 1]
       [--analysis cte|k] [--delta-t 100] [--ref-temp-c 20]        # cte
       [--heat-w 1.0] [--directions xyz]                            # k
       [--tag NAME] [--mapdl-dir DIR] [--nproc N] [--dry-run] [--materials ...]
```
* Requires `mesh/homogenization.json`.
* `--analysis cte` (default): SOLID186, 3-2-1 fixture, one static step to `ref+delta_t`; reports
  `alpha_eff_x/y`, warpage (peak-to-valley and RMS, best-fit plane removed), peak von Mises, each
  compared against `laminate_bounds()`; keeps the asymmetric-stackup note.
* `--analysis k`: SOLID70, per direction fix one face / lump `--heat-w` on the opposite face,
  `k_eff = Q·L/(A·ΔT)` against `k_xy_voigt`/`k_z_reuss`.
* Output `results/board_<analysis>_<tag>.json` (+ `von_mises_top_<tag>.png` for `cte`). `--dry-run`
  reports mesh, material count and analytical reference without launching MAPDL.
* Default tag = region model (`cte`) or `k`.

### `simulate_assembly.py`
```
python simulate_assembly.py <sample> --nx 20 --ny 20 [--nz-per-layer 1]
       [--creep] [--creep-ratio-limit X] [--thermal-profile configs/thermal_profile.yaml]
       [--cdb data/components/cr0805/...cdb] [--solder-mac data/materials/sac305/...mac]
       [--place-x-mm X] [--place-y-mm Y] [--rotation-deg 0|90] [--inspect]
       [--mapdl-dir DIR] [--nproc N] [--dry-run] [--materials ...]
```
Order of operations (this is the risky part — follow it exactly):
1. `pcbh.stackup.load_board`, `pcbh.grid.load`, `pcbh.thermal.load_thermal_profile`.
2. Copy the profile YAML + write its PNG into `results/`. `--dry-run` stops here (this replaces
   `python thermal_profile.py <yaml>`).
3. `mesh.launch()` → `register_materials()` (board materials at 9001+) → `build_mesh()` (creates the
   `BOARD` element component) → `apply_321_fixture()`.
4. Snapshot `{board elem id: MAT}` for the reassert step.
5. `compute_safe_offsets()` → `import_component()` → `reassert_board_materials(mapdl, "BOARD")` →
   `reapply_component_materials(mapdl, cdb, (2, 6, 9))`.
6. `apply_solder_material()` → `place_component(..., board_top_z_mm=board.total_thickness_mm, ...)`.
   `--inspect` returns after `eplot()`.
7. `generate_bonded_contact(..., board_cm="BOARD")`; abort before the solve if either side is empty.
8. `/SOLU`: `RATE,ON|OFF`, optional `CUTCONTROL,CRPLIMIT,x,1`, `ANTYPE,STATIC`, `OUTRES,ALL,ALL`,
   `apply_thermal_profile()`.
9. Copy `.db`/`.rst` to `results/assembly.db` / `results/assembly.rst`; `finally:` `mesh.shutdown()`.

No `--board-model` flag: the board model is whatever `homogenize.py` wrote. For the full-resolution
reference run, build the grid with `--model projection --gx <nx> --gy <ny>` first (D5).

### `postprocess_creep.py`
Identical CLI to today's `zone_creep_strain.py` (`<sample> [--target-ratio] [--time-start]
[--time-end] [--solder-mat]`), reading `results/assembly.rst`. Only the default `.rst` filename
changes (`pyansys_assembly.rst` → `assembly.rst`); accept `--rst PATH` so old samples still work.

### Plot scripts
* `plot_board.py <sample> [--out DIR]` — panel A top copper raster; panel B small multiples of every
  copper layer's φ on the homogenization grid, one shared 0–1 colourbar; panel C stackup through
  thickness with φ per copper slab at true z. Reads metadata + rasters + `homogenization.json`.
* `plot_solved_board.py <sample> [--rst PATH]` — today's `plot_board_model.py`, generalised from 3
  z-bands to `len(stackup.layers)` bands and from MAT 9001/9002 to "any MAT ≥ 9001 is board".
* `plot_creep.py <sample> [<sample> ...] [--labels ...] [--out PATH]` — zone creep history and
  per-cycle Δε; one sample → the two single-sample figures, several → the overlay/bar comparison.
* `plot_convergence.py [--grid N DIR]... [--mesh N DIR]... [--reference DIR] [--out PATH]` —
  Δε (or a board QoI from `board_cte_*.json`) vs homogenization-grid resolution at fixed mesh, and
  vs mesh resolution at fixed model. This is the convergence study the project already ran
  (`plot_leaf_resolution_sweep.py`'s two series), with "leaves" renamed to "grid".
* `plot_component.py <sample>` — unchanged render, MAT legend updated.

---

## 7. Phased work plan

Commit at the end of every phase (and freely inside it). Conventional, plain commit subjects; end
each message with the attribution line this repo uses.

### Phase 0 — copy the repo, freeze a baseline  *(no code changes)*

1. **The user must create the empty GitHub repo** `alxsokolov/pcb-homogenization` (no README, no
   licence, no .gitignore) — `gh` is not installed on this machine, so this cannot be scripted here.
   Stop and ask if it does not exist yet.
2. From the repo root:
   ```
   git branch -a                       # note every branch, incl. wip/2026-10 if present
   git tag v0.1.2-pre-restructure      # if not already tagged
   git remote add newrepo git@github.com:alxsokolov/pcb-homogenization.git
   git push newrepo 'refs/heads/*:refs/heads/*' --tags
   git ls-remote newrepo               # verify every branch and tag arrived
   ```
   Then make the new repo the working remote, keeping the old one reachable:
   ```
   git remote rename origin old-origin
   git remote rename newrepo origin
   ```
   `data/` and `mapdl_run*/` are gitignored, so the copy is code only — the sample data stays on the
   MikroVAL share. Say so in the final report.
3. **Capture the golden baseline** with the *current* code, into `tests/golden/`:
   ```
   python discretize.py data/samples/rv3032_example --resolution 10
   python quadtree.py   data/samples/rv3032_example --per-element-grid 20 20
   cp data/samples/rv3032_example/mesh/quadtree.json tests/golden/rv3032_qt_20x20.json
   python discretize.py data/samples/odb_kitchen_sink --resolution 10
   python discretize.py data/samples/hella --resolution 10
   python coarsegrain_nlayer.py data/samples/rv3032_example --mode grid --nx 20 --ny 20
   cp data/samples/rv3032_example/mesh/coarsegrain_nlayer.json tests/golden/rv3032_cg_20x20.json
   python simulate_pyansys_nlayer.py data/samples/rv3032_example --nx 20 --ny 20 --dry-run > tests/golden/rv3032_dryrun.txt
   python homogenization_nlayer.py   > tests/golden/homogenization_selftest.txt
   ```
   Also record, from each sample's `discretization.json`, every layer's `copper_fraction` into
   `tests/golden/copper_fractions.json`. These numbers are the contract: the restructured code must
   reproduce the per-layer φ, the per-region φ and the laminate bounds **exactly**.
4. **Build a synthetic 4-copper-layer fixture** so n > 2 is testable without partner data:
   ```
   python import_gerber_board_nlayer.py --sample-id synth4 --output-root tests/data \
       --copper data/samples/rv3032_example/gerber/F_Cu.gbr \
                data/samples/rv3032_example/gerber/B_Cu.gbr \
                data/samples/rv3032_example/gerber/F_Cu.gbr \
                data/samples/rv3032_example/gerber/B_Cu.gbr \
       --bbox 0 0 20 15 --stackup configs/stackup_synth4.yaml
   ```
   with `copper_thickness_mm: [0.035, 0.018, 0.018, 0.035]`,
   `dielectric_thickness_mm: [0.2, 1.1, 0.2]`. Commit the sample (two small Gerbers + metadata) —
   it is the only n-layer board the repo can test against.
5. **Optional but strongly recommended MAPDL baseline** (needs a licence, ~10 min): with the current
   code, run the assembly case that the new code can reproduce exactly —
   ```
   python quadtree.py data/samples/rv3032_example --per-element-grid 20 20
   python simulate_pyansys_assembly.py data/samples/rv3032_example --nx 20 --ny 20 \
       --nz-per-layer 2 --board-model homogenized_zlayers --creep \
       --thermal-profile configs/thermal_profile_example.yaml --mapdl-dir C:\temp\mapdl_base
   python zone_creep_strain.py data/samples/rv3032_example --target-ratio 0.25
   ```
   Copy `results/zone_creep_strain.json` and `results/cycle_creep_increment.json` into
   `tests/golden/`. This is the acceptance gate for Phase 5. If no licence is available, say so
   explicitly in the final report and fall back to the element/material-count comparison described
   there.

**Acceptance:** new remote holds every branch and tag; `tests/golden/` committed; `git status` clean.

### Phase 1 — library skeleton, pure-Python moves

Create `pcbh/` and move, with no behaviour change: `materials.py`, `homogenization.py` (from
`homogenization_nlayer.py`, minus `dilution` and the 2-layer reduction selftest), `stackup.py`
(minus `_slabs_from_legacy2`, plus `layer_phi_grid`), `raster.py`, `gerber.py` (the parser from
`discretize.py`, emitting `LayerGeometry`), `odb.py`, `odb_stackup.py`, `thermal.py`, `pads.py`.
Delete `homogenization.py`'s 2-layer bounds, `raster_geometry.py`, and the ~120 duplicated polygon/
rasterize lines in `discretize.py`. Apply the D11 attrlist fix and delete `odb_stackup.py`'s private
duplicate parser.

Write `tests/test_homogenization.py` and `tests/test_io.py`:
* pure phases: `slab_properties(0)` == dielectric, `slab_properties(1)` == copper, exactly;
* slab-count invariance at uniform φ (from the old selftest);
* structural == global bounds on the 8-layer reference stackup (from the old selftest);
* closed-form check of the 3-slab laminate against values computed by hand in the test for
  `phi_bot=0.3, phi_top=0.7, t_cu=0.035, t_fr4=1.53` (this replaces the comparison against the
  deleted 2-layer module — hardcode the expected numbers from the Phase 0 golden run);
* Gerber and ODB++ parse of the bundled samples reproduce `tests/golden/copper_fractions.json`.

**Acceptance:** `pytest -q` green; `python -c "import pcbh.homogenization, pcbh.stackup, pcbh.odb"`
clean; old scripts may be temporarily broken (they are deleted in Phases 2–5).

### Phase 2 — unified import + discretize

Write `import_board.py` (§6) and rewrite `discretize.py` on `pcbh.gerber`/`pcbh.odb`/`pcbh.raster`.
Delete the four old importers. Add `configs/stackup_example.yaml`.

Write `tools/migrate_metadata.py` (throwaway, ~60 lines, deleted in Phase 7): rewrites a
schema-1 `metadata.json` in place to §4.1 — reconstructing `[bottom Cu, dielectric, top Cu]` from the
flat keys exactly as `_slabs_from_legacy2()` did, writing `schema_version: 2`, keeping a `.bak`.
Run it over every `data/samples/*`, then re-run `discretize.py` for each (seconds; the old
`*_map.npy`/`material_map.npy` become stale and must be deleted).

**Acceptance:** for `rv3032_example`, `odb_kitchen_sink`, `hella` and `tests/data/synth4`, re-import
or migrate + `discretize.py` reproduces `tests/golden/copper_fractions.json` bit-for-bit (except
`hella`'s copper thickness, which D11 changes — assert the new value and note it).
`python import_board.py gerber --help` / `odb --help` readable on one screen.

### Phase 3 — the uniform homogenization grid

Write `pcbh/grid.py` and `homogenize.py`; delete `quadtree.py` and `coarsegrain_nlayer.py`.
Write `tests/test_grid.py`:
* conservation: for every copper slab, `(phi * area).sum() / total_area == raster.mean()` to 1e-12
  when the raster divides evenly;
* `region_of_element` maps element centres to the right region for `nx == gx`, `nx == 5*gx`,
  `nx == 3*gx + 1`, and never goes out of range;
* material dedup: a uniform board yields exactly one material per copper slab;
* **equivalence with the old code:** `build(board, 20, 20, "zlayers")`'s per-region φ equals the
  `phi_top`/`phi_bot` in `tests/golden/rv3032_qt_20x20.json` and the per-slab φ in
  `tests/golden/rv3032_cg_20x20.json`, region for region (both used the same pixel rule, so this must
  match exactly — if it does not, the grid geometry or the row/column order is wrong).

Then port `plot_board.py` (merging `plot_board_preview.py` + `plot_coarsegrain_nlayer.py`).

**Acceptance:** `pytest -q` green; `python homogenize.py data/samples/rv3032_example --gx 20 --gy 20`
prints a conservation table with deltas ≤ 1e-12; `plot_board.py` renders for both a 2-layer and the
synthetic 4-layer sample.

### Phase 4 — board-only solve

Write `pcbh/mesh.py` and `simulate_board.py` (D8); delete `simulate_pyansys_nlayer.py`,
`simulate_pyansys_thermal_nlayer.py`, `simulate_pyansys_benchmark.py`, `solder_reliability.py`,
`compare_nlayer_models.py`. Port `plot_convergence.py`.

Verification without a licence: `--dry-run` for `rv3032_example --nx 20 --ny 20 --nz-per-layer 2`
must report 3 slabs × 2 = 6 z rows, 2400 elements, 3087 nodes and the same `phi_3D`/`alpha_xy`/
`E_xy`/`E_z` as `tests/golden/rv3032_dryrun.txt`.
With a licence: `--analysis cte` on `rv3032_example` reproduces the old n-layer run's `alpha_eff_x/y`
to ≤ 0.1 %, and `--analysis k`'s `k_eff` still brackets `k_xy_voigt`/`k_z_reuss` as before.

**Acceptance:** the dry-run numbers match golden; one real `cte` solve completes and its
`alpha_eff` sits within a few percent of the analytical laminate value (the existing criterion).

### Phase 5 — assembly: component, solder, creep  *(highest risk)*

Rewrite `simulate_pyansys_assembly.py` as `simulate_assembly.py` on `pcbh.mesh` + `pcbh.component`
(§6). Apply D10 and the `BOARD`-component contact selection. Delete nothing from
`pcbh/component.py`'s docstrings.

Watch for, in this order:
1. **MAT collision.** Board materials from 9001; component archive keeps 1–10; solder stays MAT 9
   (`postprocess_creep.py` defaults to it). After `CDREAD`, `MPLIST` a board MAT and confirm the
   properties survived; `reassert_board_materials(mapdl, "BOARD")` then restores any element whose
   MAT field `CDREAD` reverted (this is a confirmed, not hypothetical, bug — see the docstring).
2. **Contact pair.** `generate_bonded_contact` must produce non-zero counts on both sides. The
   pre-restructure run at `--nx 20 --ny 20` gave 10 TARGE170 + 260 CONTA174; at `--nx 50 --ny 50`,
   55 + 260. Reproduce those counts — a change here means the board-side selection is wrong.
3. **Board top z** is `board.total_thickness_mm`, i.e. the top copper slab's `z1_mm`; `place_component`
   and the contact both key off it.
4. **nz-per-layer.** With 3 slabs and `--nz-per-layer 2` the new mesh is *identical* to the old
   2-layer mesh at `--nz-per-layer 2` (bands `[0, t_cu, t_cu+t_core, Lz]`, 2 rows each). Use that for
   the regression run. The new default is 1 (one row per slab) — README must say that reproducing a
   pre-restructure run needs `--nz-per-layer 2`.

**Acceptance (with a licence):** rerun the Phase 0.5 case as
```
python homogenize.py data/samples/<copy> --gx 20 --gy 20 --model zlayers
python simulate_assembly.py data/samples/<copy> --nx 20 --ny 20 --nz-per-layer 2 --creep \
    --thermal-profile configs/thermal_profile_example.yaml --mapdl-dir C:\temp\mapdl_new
python postprocess_creep.py data/samples/<copy> --target-ratio 0.25
```
and compare `cycle_creep_increment.json`'s cycle-1 Δε against `tests/golden/`. Expect agreement to
within ~1 % (identical mesh, identical materials, identical profile); investigate anything larger
before continuing. **Without a licence:** compare the printed element counts, material counts,
contact-pair counts and the grid's material table against the Phase 0 log, state clearly in the final
report that the solve-level check could not be run, and leave the commands above in the README as the
acceptance procedure.

### Phase 6 — postprocessing and plots

Rename `zone_creep_strain.py` → `postprocess_creep.py` (content unchanged, `--rst` added). Merge the
creep plots into `plot_creep.py`; rename `plot_board_model.py` → `plot_solved_board.py` and
generalise its z-band handling; rename `plot_component_materials.py` → `plot_component.py`. Delete
`plot_cu_penetration_sweep.py`, `plot_delta_summary.py`, `plot_comparison_creep.py`,
`plot_creep_strain_history.py`, `plot_cycle_creep_increment.py`, `plot_leaf_resolution_sweep.py`,
`plot_discretization_sensitivity.py`.

**Acceptance:** every plot script runs on an existing solved sample (e.g.
`data/samples/rv3032_homogz_50x50_nz10_g20`) and writes a PNG; `plot_creep.py` with three sample dirs
reproduces the old comparison figure's content.

### Phase 7 — docs, cleanup, release

1. `README.md`: one page — what the tool does, the pipeline diagram from §1, install, a single
   quickstart for `rv3032_example` (2-layer) and one for an n-layer ODB++ board, the two-discretization
   explanation, data layout, limitations (vias not modelled, one component placed manually, TREF 180 °C,
   Gerber arcs approximated, `matrix/stackup.xml` not parsed), credits. Delete `QUICKSTART.md`.
2. `CHANGELOG.md`: a `v0.2.0` section listing §9 verbatim, plus the migration note ("`data/` samples
   need `tools/migrate_metadata.py` and a re-run of `discretize.py`/`homogenize.py`; solved `.rst`
   files stay valid").
3. `docs/overview/make_figures.py`: point at the new plot scripts and sample paths; rebuild
   `main.pdf` if `latexmk` is available, otherwise note it.
4. Delete `tools/migrate_metadata.py`, `RESTRUCTURING_PLAN.md`, `~$workflow_overview.pptx`; add
   `tests/golden/` and `tests/data/` to git (they are small) and make sure `.gitignore` does not
   exclude them (`data/` is ignored — `tests/data/` is a different path, verify).
5. `requirements.txt`: add `pytest`.
6. `git tag v0.2.0` and push.

**Acceptance:** `pytest -q` green; every script's `--help` works; `README.md`'s quickstart commands
exist exactly as written; `rg -n "quadtree|nlayer|cu_penetration|dilution|solder_reliability" --glob '!CHANGELOG.md'`
returns nothing outside `docs/` history text.

---

## 8. Verification summary

| Level | What | When |
|---|---|---|
| Unit | `tests/test_homogenization.py`, `test_grid.py`, `test_io.py` | Phases 1, 2, 3; green from then on |
| Golden | per-layer φ, per-region φ, laminate bounds, dry-run mesh report vs `tests/golden/` | Phases 2, 3, 4 |
| Integration, no MAPDL | `--dry-run` on both a 2-layer and the synthetic 4-layer sample; every plot script on an existing sample | Phases 4, 6 |
| Integration, MAPDL | one `simulate_board.py --analysis cte` solve; one `simulate_assembly.py --creep` run compared to the Phase 0 creep baseline | Phases 4, 5 |

Report honestly which of these actually ran. A skipped MAPDL check is a result, not a detail.

---

## 9. Behaviour changes to record in CHANGELOG

1. **Adaptive quadtree coarse-graining removed** (boundary-straddle, leaf budget, component-focused,
   2:1 balancing). The in-plane discretization is a uniform `gx × gy` grid, independent of the FE mesh.
2. **Single pipeline.** 2-layer and n-layer boards go through the same scripts; a 2-layer board is a
   3-slab stackup. The separate 2-layer homogenization module, mesh builder, importers and board
   benchmark are gone.
3. **`metadata.json` schema 2.** Flat `copper_thickness_mm`/`dielectric_thickness_mm` removed; the
   stackup slab list is the single source of truth. Old samples need `tools/migrate_metadata.py`.
4. **Copper-penetration / dilution removed.** The mesh bands z by physical slab, so each copper layer
   is represented at its true thickness; nothing is smeared over a thicker region.
5. **`--board-model raster` replaced** by `homogenize.py --model projection --gx <nx> --gy <ny>`. An
   element is copper if its copper area fraction ≥ 0.5 rather than if its centre pixel is copper —
   results differ slightly on trace edges.
6. **Default `--nz-per-layer` is 1** (one element row per physical slab). Pre-restructure 2-layer runs
   used 2; pass `--nz-per-layer 2` to reproduce them.
7. **Board materials live at MAT 9001+** from element creation; the board-material remap before the
   component import is gone (the reassert-after-`CDREAD` fix stays). Board elements are selected by
   the `BOARD` element component, not by material-id range.
8. **ODB++ `attrlist` whitespace bug fixed.** `hella`'s resolved copper thickness changes
   0.035 → 0.036614 mm.
9. **Analytic CTE-mismatch QoI (`solder_reliability.py`) removed**; the pad coordinates live in
   `pcbh/pads.py`. The creep postprocessing is the QoI.
10. **`material_map.npy` no longer written**; rasters are per layer (`raster_<layer>.npy`).
11. **Result file names:** `results/assembly.{db,rst}`, `results/board_<analysis>_<tag>.json`,
    `mesh/homogenization.json`.
12. **Unchanged on purpose:** component archive import and placement, bonded-contact generation,
    solder material and creep law, thermal-profile application, zone-averaged creep postprocessing,
    all Voigt-Reuss formulas.

---

## 10. Risks

| R | Risk | Mitigation |
|---|---|---|
| R1 | The assembly solve behaves differently on the new mesh | The 3-slab mesh at `--nz-per-layer 2` is identical to the old 2-layer mesh; regression against the Phase 0 creep baseline (Phase 5) |
| R2 | `CDREAD` clobbers board materials in the 9001+ band | Keep `reassert_board_materials()`; verify with `MPLIST` right after import |
| R3 | Contact pair comes out empty after switching the board-side selection to the `BOARD` component | Compare TARGE170/CONTA174 counts against the known 10+260 / 55+260; abort before the solve if either side is zero (already in the code) |
| R4 | Grid φ differs from the old per-element quadtree φ because of row/column order or pixel rounding | `tests/test_grid.py`'s exact equivalence check against `tests/golden/rv3032_qt_20x20.json` |
| R5 | No MAPDL licence available during the work | Phases 1–3 and 6 need none; 4 and 5 fall back to dry-run/count comparisons with the gap reported |
| R6 | Existing samples on the share break | `tools/migrate_metadata.py` with `.bak`; solved `.rst` files are untouched and still postprocess |
| R7 | Only 2-layer sample data is in the repo, so n-layer paths stay untested | `tests/data/synth4` built in Phase 0; run every no-MAPDL acceptance check on it as well |
| R8 | Scope creep into physics | §9 item 12 is the contract: the formulas and the component/creep side do not change |
