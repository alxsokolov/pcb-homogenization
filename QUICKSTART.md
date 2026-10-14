# Quick Start: 2-Layer vs n-Layer Workflows

## 2-Layer Board (Gerber) — with component + solder + thermal cycling

For boards with exactly two copper layers (top/bottom + dielectric core).

```bash
# 1. Import from Gerber
python import_real_board.py \
    --f-cu F_Cu.gbr --b-cu B_Cu.gbr \
    --bbox XMIN YMIN XMAX YMAX \
    --sample-id my_board

# 2. Rasterize copper
python discretize.py data/samples/my_board

# 3. Coarse-grain with Voigt-Reuss homogenization (quadtree)
python quadtree.py data/samples/my_board --max-level 6

# 4. Solve board + CR0805 + SAC305 solder with thermal cycling
python simulate_pyansys_assembly.py data/samples/my_board \
    --nx 20 --ny 20 --creep \
    --thermal-profile configs/thermal_profile.yaml \
    --mapdl-dir C:\temp\mapdl_board

# 5. Post-process creep strain per cycle
python zone_creep_strain.py data/samples/my_board --target-ratio 0.25
python plot_cycle_creep_increment.py data/samples/my_board
python plot_board_model.py data/samples/my_board
```

**Demo run (fast):** use `configs/thermal_profile_example.yaml` instead (1 cycle, ~8 min on 8 cores).

---

## n-Layer Board (ODB++ or Gerber) — board only, no component yet

For boards with 3+ copper layers. No component or solder; validates homogenization method.

### Option A: From ODB++ (recovers stackup automatically)

```bash
# 1. Extract ODB++ job and import all copper layers
python import_odb_board_nlayer.py \
    --odb-dir /path/to/extracted/job \
    --sample-id my_8layer

# 2. Rasterize all copper layers
python discretize.py data/samples/my_8layer --resolution 20

# 3. Coarse-grain plane and assign materials
python coarsegrain_nlayer.py data/samples/my_8layer \
    --mode grid --nx 40 --ny 40 \
    --region-model homogenized_zlayers

# 4. Inspect before solving
python plot_coarsegrain_nlayer.py data/samples/my_8layer

# 5. Solve board (board-only, z-banded mesh)
python simulate_pyansys_nlayer.py data/samples/my_8layer \
    --nx 40 --ny 40 \
    --mapdl-dir C:\temp\mapdl_nlayer

# 6. Compare against full-resolution reference
python simulate_pyansys_nlayer.py data/samples/my_8layer \
    --nx 40 --ny 40 --board-model raster \
    --tag raster --mapdl-dir C:\temp\mapdl_nlayer
python compare_nlayer_models.py data/samples/my_8layer
```

### Option B: From Gerber (requires stackup YAML)

```bash
# 1. Create stackup YAML (example: configs/stackup_example.yaml)
cat > stackup.yaml <<EOF
copper_thickness_mm: [0.035, 0.018, 0.018, 0.035]
dielectric_thickness_mm: [0.2, 1.1, 0.2]
dielectric_material: FR4
EOF

# 2. Import n copper Gerbers, top first
python import_gerber_board_nlayer.py \
    --copper F_Cu.gbr In1_Cu.gbr In2_Cu.gbr B_Cu.gbr \
    --bbox XMIN YMIN XMAX YMAX \
    --stackup stackup.yaml \
    --sample-id my_4layer

# 3–6. Continue as above (discretize → coarsegrain → solve → compare)
```

---

## Key Options

### Discretization modes (`coarsegrain_nlayer.py --mode`)
- `grid`: uniform regions, one per element (default) or coarser
- `quadtree`: adaptive refinement where layout is mixed (slower but detail where it matters)

### Region models (`coarsegrain_nlayer.py --region-model`)
- `homogenized_zlayers` (default): each copper layer gets its own Voigt-Reuss material → best warpage accuracy
- `homogenized`: one smeared material through thickness → faster, less accurate warpage
- `projection`: black/white per layer, no homogenization → shows model without homogenization

### MAPDL tips
- Put `--mapdl-dir` on a **local SSD**, not network. MAPDL writes multi-GB scratch files.
- Use `--nproc N` for parallel solve (e.g., `--nproc 8` for 8 cores).
- Add `--dry-run` to check the mesh without solving.
- Set `PYTHONUTF8=1` on Windows if output is piped/redirected.

---

## Example: Full n-layer run (ublox 8-layer board)

```bash
python import_odb_board_nlayer.py --odb-dir Y:\...\JODY-W6\odb --sample-id ublox_nlayer
python discretize.py data/samples/ublox_nlayer --resolution 20
python coarsegrain_nlayer.py data/samples/ublox_nlayer --mode grid --nx 40 --ny 40
python plot_coarsegrain_nlayer.py data/samples/ublox_nlayer
python simulate_pyansys_nlayer.py data/samples/ublox_nlayer --nx 40 --ny 40 --mapdl-dir C:\temp\mapdl_nlayer --nproc 8

# Compare homogenized_zlayers against full-resolution raster
python simulate_pyansys_nlayer.py data/samples/ublox_nlayer --nx 40 --ny 40 --board-model raster --tag raster --mapdl-dir C:\temp\mapdl_nlayer --nproc 8
python compare_nlayer_models.py data/samples/ublox_nlayer --reference raster
```

---

## What Each Script Does

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| `import_real_board.py` | Gerber F_Cu, B_Cu | metadata.json, gerber/ | 2-layer only |
| `import_odb_board_nlayer.py` | ODB++ job | metadata.json, all layers | Recovers stackup |
| `import_gerber_board_nlayer.py` | n Gerbers + YAML | metadata.json, all layers | Manual stackup |
| `discretize.py` | metadata.json | mesh/rasters_*.npy | Shared; rasterizes all layers |
| `quadtree.py` | rasters | mesh/quadtree.json | 2-layer; per-leaf Voigt-Reuss |
| `coarsegrain_nlayer.py` | rasters, stackup | mesh/coarsegrain_nlayer.json | n-layer; grid or quadtree |
| `simulate_pyansys_assembly.py` | quadtree.json, component CDB | results/pyansys_assembly.rst | 2-layer; board + component + solder |
| `simulate_pyansys_nlayer.py` | coarsegrain.json | results/pyansys_nlayer_\*.json | n-layer; board only |
| `plot_coarsegrain_nlayer.py` | coarsegrain.json | results/coarsegrain_layers_\*.png | Inspect regions, per-layer φ |
| `compare_nlayer_models.py` | multiple pyansys_nlayer_\*.json | results/nlayer_model_comparison.png | Table + deviation bar plot |

---

## Validation & Self-Tests

```bash
# Homogenization formula agreement (n-layer vs 2-layer)
python homogenization_nlayer.py

# Stackup_nlayer reads legacy 2-layer schemas (cross-check)
python -c "from stackup_nlayer import load_board_stackup; print(load_board_stackup('data/samples/rv3032_example'))"
```
