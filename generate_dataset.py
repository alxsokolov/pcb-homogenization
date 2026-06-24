#!/usr/bin/env python3
"""
Dataset generation master script for PCB homogenization.

Sweeps random layout parameters, runs the full pipeline per sample
(generate Gerber → discretize → FEniCSx tensile simulation), and
collects results into a flat index.csv for ML training.

Directory layout produced:
    data/datasets/<dataset_id>/
        dataset.json          sweep config (self-documenting)
        index.csv             one row per sample: params + E_x/y/z + nu + status
        samples/
            000001/           gerber/, mesh/, results/, metadata.json
            000002/
            ...

Resumable: samples whose results/homogenized.json already exists are
skipped, so interrupted runs can be restarted with the same command.

Usage:
    python generate_dataset.py --n-samples 100
    python generate_dataset.py --n-samples 500 --dataset-id sweep_v2 --workers 4
    python generate_dataset.py --dataset-id sweep_v1          # resume existing
    python generate_dataset.py --n-samples 200 --nx 30 --ny 30 --px-per-mm 5
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from discretize import discretize_sample
from generate_gerber import generate_sample

_SIMULATE = _HERE / "simulate_tensile.py"

# ---------------------------------------------------------------------------
# CSV schema — order defines column order in index.csv
# ---------------------------------------------------------------------------

_FIELDS = [
    "sample_id",
    "seed",
    "n_traces_nominal",
    "n_pours_nominal",
    "trace_width_min_mm",
    "trace_width_max_mm",
    "pad_size_min_mm",
    "pad_size_max_mm",
    "h_bias",
    "cu_fraction",
    # FEM elasticity results
    "E_x_GPa",
    "E_y_GPa",
    "E_z_GPa",
    "nu_xy",
    "nu_xz",
    "nu_yz",
    # FEM CTE results
    "alpha_x_per_K",
    "alpha_y_per_K",
    "alpha_z_per_K",
    # Analytical Voigt/Reuss bounds (ML baselines)
    "phi_top",
    "phi_bot",
    "phi_3D",
    "E_voigt_global_GPa",
    "E_reuss_global_GPa",
    "E_xy_voigt_GPa",
    "E_z_reuss_GPa",
    "alpha_xy_voigt_per_K",
    "alpha_z_reuss_per_K",
    "status",
]

# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def _draw_params(rng: np.random.Generator, cfg: dict) -> dict:
    """Sample one set of layout parameters from config ranges."""
    return dict(
        n_traces=int(rng.integers(cfg["n_traces_min"], cfg["n_traces_max"] + 1)),
        n_pours=int(rng.integers(cfg["n_pours_min"],  cfg["n_pours_max"]  + 1)),
        trace_width_range=(cfg["trace_width_min_mm"], cfg["trace_width_max_mm"]),
        pad_size_range=(cfg["pad_size_min_mm"], cfg["pad_size_max_mm"]),
        h_bias=float(rng.uniform(cfg.get("h_bias_min", 0.0), cfg.get("h_bias_max", 1.0))),
        seed=int(rng.integers(0, 2**31)),
    )

# ---------------------------------------------------------------------------
# Single-sample pipeline
# ---------------------------------------------------------------------------

def _run_sample(dataset_dir: Path, sample_id: str, params: dict, cfg: dict) -> dict:
    """
    Run the full pipeline for one sample and return a CSV row dict.
    Safe to call from a worker process.
    """
    sample_dir  = dataset_dir / "samples" / sample_id
    homog_path  = sample_dir / "results" / "homogenized.json"

    # ── Resume: already finished ──────────────────────────────────────────
    if homog_path.exists():
        return _load_completed_row(sample_id, sample_dir, homog_path)

    board = cfg["board"]

    # ── Stage 1: generate Gerber ──────────────────────────────────────────
    generate_sample(
        output_root=dataset_dir / "samples",
        sample_id=sample_id,
        width_mm=board["width_mm"],
        height_mm=board["height_mm"],
        cu_thickness_mm=board.get("cu_thickness_mm", 0.035),
        dielectric_thickness_mm=board.get("dielectric_thickness_mm", 1.53),
        layout_type="random",
        seed=params["seed"],
        n_traces=params["n_traces"],
        n_pours=params["n_pours"],
        trace_width_range=params["trace_width_range"],
        pad_size_range=params["pad_size_range"],
        h_bias=params["h_bias"],
    )

    # ── Stage 2: discretize ───────────────────────────────────────────────
    mat = discretize_sample(
        sample_dir,
        resolution_px_per_mm=cfg.get("px_per_mm", 10.0),
        save_png=False,
    )
    cu_fraction = float(mat.mean())

    # ── Stage 3: simulate ─────────────────────────────────────────────────
    cmd = [
        sys.executable, str(_SIMULATE), str(sample_dir),
        "--nx", str(cfg.get("nx", 50)),
        "--ny", str(cfg.get("ny", 50)),
    ]
    if cfg.get("save_xdmf"):
        cmd.append("--save-xdmf")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_HERE))
    if proc.returncode != 0:
        raise RuntimeError(
            f"simulate_tensile.py failed for {sample_id}:\n{proc.stderr[-3000:]}"
        )

    # ── Collect result ────────────────────────────────────────────────────
    with open(homog_path) as f:
        homog = json.load(f)

    with open(sample_dir / "metadata.json") as f:
        gen = json.load(f)["generation"]

    return _build_row(sample_id, cu_fraction, gen, homog)


def _load_completed_row(sample_id: str, sample_dir: Path, homog_path: Path) -> dict:
    """Reconstruct a CSV row from an already-finished sample."""
    with open(homog_path) as f:
        homog = json.load(f)
    with open(sample_dir / "metadata.json") as f:
        gen = json.load(f)["generation"]
    disc_path = sample_dir / "mesh" / "discretization.json"
    cu_fraction = 0.0
    if disc_path.exists():
        with open(disc_path) as f:
            cu_fraction = json.load(f).get("copper_fraction", 0.0)
    return _build_row(sample_id, cu_fraction, gen, homog)


def _build_row(sample_id: str, cu_fraction: float, gen: dict, homog: dict) -> dict:
    tw = gen.get("trace_width_range_mm", ["", ""])
    ps = gen.get("pad_size_range_mm",    ["", ""])
    return {
        "sample_id":           sample_id,
        "seed":                gen.get("seed", ""),
        "n_traces_nominal":    gen.get("n_traces_nominal", ""),
        "n_pours_nominal":     gen.get("n_pours_nominal", ""),
        "trace_width_min_mm":  tw[0],
        "trace_width_max_mm":  tw[1],
        "pad_size_min_mm":     ps[0],
        "pad_size_max_mm":     ps[1],
        "h_bias":              gen.get("h_bias", ""),
        "cu_fraction":         round(cu_fraction, 6),
        "E_x_GPa":             homog.get("E_x_GPa", ""),
        "E_y_GPa":             homog.get("E_y_GPa", ""),
        "E_z_GPa":             homog.get("E_z_GPa", ""),
        "nu_xy":               homog.get("nu_xy",   ""),
        "nu_xz":               homog.get("nu_xz",   ""),
        "nu_yz":               homog.get("nu_yz",   ""),
        "alpha_x_per_K":       homog.get("alpha_x_per_K", ""),
        "alpha_y_per_K":       homog.get("alpha_y_per_K", ""),
        "alpha_z_per_K":       homog.get("alpha_z_per_K", ""),
        "phi_top":             homog.get("phi_top",             ""),
        "phi_bot":             homog.get("phi_bot",             ""),
        "phi_3D":              homog.get("phi_3D",              ""),
        "E_voigt_global_GPa":  homog.get("E_voigt_global_GPa", ""),
        "E_reuss_global_GPa":  homog.get("E_reuss_global_GPa", ""),
        "E_xy_voigt_GPa":      homog.get("E_xy_voigt_GPa",     ""),
        "E_z_reuss_GPa":       homog.get("E_z_reuss_GPa",      ""),
        "alpha_xy_voigt_per_K": homog.get("alpha_xy_voigt_per_K", ""),
        "alpha_z_reuss_per_K":  homog.get("alpha_z_reuss_per_K",  ""),
        "status":              "complete",
    }

# ---------------------------------------------------------------------------
# Dataset initialisation / index management
# ---------------------------------------------------------------------------

def _init_dataset(dataset_dir: Path, cfg: dict) -> None:
    """Create directory structure and write dataset.json (idempotent)."""
    (dataset_dir / "samples").mkdir(parents=True, exist_ok=True)
    ds_json = dataset_dir / "dataset.json"
    if not ds_json.exists():
        ds_json.write_text(
            json.dumps({**cfg, "created": datetime.now(tz=timezone.utc).isoformat(timespec="seconds")},
                       indent=2) + "\n"
        )


def _load_completed_ids(index_csv: Path) -> set[str]:
    """Return the set of sample_ids already marked complete in index.csv."""
    if not index_csv.exists():
        return set()
    with open(index_csv, newline="") as f:
        return {row["sample_id"] for row in csv.DictReader(f)
                if row.get("status") == "complete"}


def _append_rows(index_csv: Path, rows: list[dict]) -> None:
    """Append rows to index.csv, writing the header if the file is new."""
    write_header = not index_csv.exists()
    with open(index_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# Worker wrapper (needed for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> dict:
    dataset_dir, sample_id, params, cfg = args
    return _run_sample(dataset_dir, sample_id, params, cfg)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PCB homogenization dataset (Gerber → FEM → index.csv).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-id", default=None,
                        help="Dataset name under data/datasets/ (default: auto timestamp)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--master-seed", type=int, default=0,
                        help="RNG seed for parameter sampling (reproducible datasets)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (1 = sequential)")
    # Board
    parser.add_argument("--width",  type=float, default=50.0, help="Board width [mm]")
    parser.add_argument("--height", type=float, default=50.0, help="Board height [mm]")
    # Layout sweep ranges
    parser.add_argument("--n-traces-min", type=int, default=5)
    parser.add_argument("--n-traces-max", type=int, default=40)
    parser.add_argument("--n-pours-min",  type=int, default=0)
    parser.add_argument("--n-pours-max",  type=int, default=6)
    parser.add_argument("--trace-width-min", type=float, default=0.1,  help="[mm]")
    parser.add_argument("--trace-width-max", type=float, default=3.0,  help="[mm]")
    parser.add_argument("--pad-size-min",    type=float, default=0.5,  help="[mm]")
    parser.add_argument("--pad-size-max",    type=float, default=4.0,  help="[mm]")
    parser.add_argument("--h-bias-min", type=float, default=0.0,
                        help="Min horizontal trace bias (0=all vertical, 1=all horizontal)")
    parser.add_argument("--h-bias-max", type=float, default=1.0,
                        help="Max horizontal trace bias")
    # FEM
    parser.add_argument("--px-per-mm", type=float, default=10.0,
                        help="Rasterisation resolution [px/mm]")
    parser.add_argument("--nx", type=int, default=50, help="FEM elements in x")
    parser.add_argument("--ny", type=int, default=50, help="FEM elements in y")
    parser.add_argument("--save-xdmf", action="store_true",
                        help="Pass --save-xdmf to each simulation (ParaView output). "
                             "Avoid for large dataset runs — writes many files.")

    args = parser.parse_args()

    dataset_id  = args.dataset_id or f"dataset_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    dataset_dir = _HERE / "data" / "datasets" / dataset_id
    index_csv   = dataset_dir / "index.csv"

    cfg = {
        "dataset_id":          dataset_id,
        "n_samples":           args.n_samples,
        "master_seed":         args.master_seed,
        "board":               {"width_mm": args.width, "height_mm": args.height},
        "n_traces_min":        args.n_traces_min,
        "n_traces_max":        args.n_traces_max,
        "n_pours_min":         args.n_pours_min,
        "n_pours_max":         args.n_pours_max,
        "trace_width_min_mm":  args.trace_width_min,
        "trace_width_max_mm":  args.trace_width_max,
        "pad_size_min_mm":     args.pad_size_min,
        "pad_size_max_mm":     args.pad_size_max,
        "h_bias_min":          args.h_bias_min,
        "h_bias_max":          args.h_bias_max,
        "px_per_mm":           args.px_per_mm,
        "save_xdmf":           args.save_xdmf,
        "nx":                  args.nx,
        "ny":                  args.ny,
    }

    _init_dataset(dataset_dir, cfg)

    # ── Build the work list, skipping already-completed samples ──────────
    completed = _load_completed_ids(index_csv)
    rng = np.random.default_rng(args.master_seed)

    work: list[tuple] = []
    for i in range(1, args.n_samples + 1):
        sample_id = f"{i:06d}"
        params    = _draw_params(rng, cfg)
        if sample_id in completed:
            print(f"  skip {sample_id}  (already complete)")
            continue
        work.append((dataset_dir, sample_id, params, cfg))

    total    = len(work)
    skipped  = args.n_samples - total
    print(f"\nDataset : {dataset_id}")
    print(f"Samples : {total} to run, {skipped} already complete")
    print(f"Workers : {args.workers}\n")

    if total == 0:
        print("Nothing to do.")
        return

    # ── Run pipeline ──────────────────────────────────────────────────────
    done = 0
    failed = 0

    if args.workers == 1:
        for job in work:
            ds_dir, sid, params, cfg_ = job
            try:
                row = _run_sample(ds_dir, sid, params, cfg_)
                _append_rows(index_csv, [row])
                done += 1
                ax = row['alpha_x_per_K']
                az = row['alpha_z_per_K']
                ax_s = f"{float(ax)*1e6:.2f}" if ax != "" else "?"
                az_s = f"{float(az)*1e6:.2f}" if az != "" else "?"
                print(f"  [{done}/{total}]  {sid}  "
                      f"E_x={row['E_x_GPa']:.3f}  E_z={row['E_z_GPa']:.3f}  "
                      f"α_x={ax_s} α_z={az_s} ppm/K  "
                      f"cu={row['cu_fraction']:.3f}")
            except Exception as exc:
                failed += 1
                print(f"  [{done+failed}/{total}]  {sid}  FAILED: {exc}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, job): job[1] for job in work}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    row = fut.result()
                    _append_rows(index_csv, [row])
                    done += 1
                    print(f"  [{done}/{total}]  {sid}  "
                          f"E_x={row['E_x_GPa']:.3f}  "
                          f"E_z={row['E_z_GPa']:.3f}  "
                          f"cu={row['cu_fraction']:.3f}")
                except Exception as exc:
                    failed += 1
                    print(f"  FAILED {sid}: {exc}", file=sys.stderr)

    print(f"\nDone. {done} complete, {failed} failed.")
    print(f"Index  → {index_csv}")
    print(f"Config → {dataset_dir / 'dataset.json'}")


if __name__ == "__main__":
    main()
