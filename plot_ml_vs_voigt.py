#!/usr/bin/env python3
"""
Parity-plot comparison of the CNN leaf-homogenization surrogate
(ml_homogenization.py) against the analytical Voigt-Reuss bounds
(homogenization.voigt_reuss_bounds()), evaluated on a held-out dataset.

Every sample already carries ground-truth FEniCSx-simulated properties and
the VR baseline in index.csv (both written by generate_dataset.py); this
script additionally runs the trained CNN on each sample's raster to get ML
predictions, then plots predicted vs actual for both methods side by side
-- the six targets Voigt-Reuss has an analytical value for (E_x/y/z,
alpha_x/y/z; nu has no VR baseline to compare against).

Usage:
    python plot_ml_vs_voigt.py data/datasets/test_dataset_ml
    python plot_ml_vs_voigt.py data/datasets/test_dataset_ml --model models/leaf_cnn_v1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from homogenization import load_materials
import ml_homogenization as mlh

_ML_COLOR = "#2a78d6"   # dataviz skill categorical slot 1 (blue)
_VR_COLOR = "#e34948"   # dataviz skill categorical slot 6 (red) -- far in the
                        # fixed hue order from slot 1, for CVD-safe separation
_REF_COLOR = "#9a9990"

# (target field, VR-baseline field it's compared against, axis label)
_PANELS = [
    ("E_x_GPa", "E_xy_voigt_GPa", "E_x [GPa]"),
    ("E_y_GPa", "E_xy_voigt_GPa", "E_y [GPa]"),
    ("E_z_GPa", "E_z_reuss_GPa", "E_z [GPa]"),
    ("alpha_x_per_K", "alpha_xy_voigt_per_K", "alpha_x [1/K]"),
    ("alpha_y_per_K", "alpha_xy_voigt_per_K", "alpha_y [1/K]"),
    ("alpha_z_per_K", "alpha_z_reuss_per_K", "alpha_z [1/K]"),
]


def _load_rows(dataset_dir: Path) -> list[dict]:
    with open(dataset_dir / "index.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "complete"]
    if not rows:
        raise SystemExit(f"No complete samples in {dataset_dir}/index.csv")
    return rows


def _stackup_mm(dataset_dir: Path, sample_id: str) -> tuple[float, float]:
    meta_path = dataset_dir / "samples" / sample_id / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["stackup"]["copper_thickness_mm"], meta["stackup"]["dielectric_thickness_mm"]


def _ml_predict_all(dataset_dir: Path, rows: list[dict], model, config,
                    materials: dict, t_cu: float, t_fr4: float) -> list[dict]:
    preds = []
    for row in rows:
        sample_dir = dataset_dir / "samples" / row["sample_id"]
        top = np.load(sample_dir / "mesh" / "F_Cu_map.npy")
        bot = np.load(sample_dir / "mesh" / "B_Cu_map.npy")
        preds.append(mlh.predict_leaf_properties(
            top, bot, float(row["phi_top"]), float(row["phi_bot"]),
            materials, t_cu, t_fr4, model, config))
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/leaf_cnn_v1"))
    parser.add_argument("--materials", type=Path, default=Path("configs/materials.yaml"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: <dataset_dir>/plots/ml_vs_voigt.png)")
    args = parser.parse_args()

    rows = _load_rows(args.dataset_dir)
    t_cu, t_fr4 = _stackup_mm(args.dataset_dir, rows[0]["sample_id"])
    materials = load_materials(args.materials)
    model, config = mlh.load_model(args.model)

    print(f"Running CNN surrogate on {len(rows)} samples from {args.dataset_dir} …")
    preds = _ml_predict_all(args.dataset_dir, rows, model, config, materials, t_cu, t_fr4)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (target, vr_field, label) in zip(axes.flat, _PANELS):
        actual = np.array([float(r[target]) for r in rows])
        vr = np.array([float(r[vr_field]) for r in rows])
        ml = np.array([p[target] for p in preds])

        lo = min(actual.min(), vr.min(), ml.min())
        hi = max(actual.max(), vr.max(), ml.max())
        pad = (hi - lo) * 0.05 or abs(hi) * 0.05 or 1.0
        lo, hi = lo - pad, hi + pad

        ax.plot([lo, hi], [lo, hi], "--", color=_REF_COLOR, linewidth=1, zorder=0)
        ax.scatter(actual, vr, s=28, marker="^", facecolors="none",
                  edgecolors=_VR_COLOR, linewidths=1.1, zorder=1)
        ax.scatter(actual, ml, s=24, marker="o", facecolors=_ML_COLOR,
                  edgecolors="none", alpha=0.75, zorder=2)

        mae_ml = float(np.mean(np.abs(ml - actual)))
        mae_vr = float(np.mean(np.abs(vr - actual)))
        ax.text(0.03, 0.97, f"MAE  ml={mae_ml:.3g}\n         vr={mae_vr:.3g}",
               transform=ax.transAxes, va="top", ha="left", fontsize=8, color="#52514e")

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"actual {label}"); ax.set_ylabel(f"predicted {label}")
        ax.set_aspect("equal", adjustable="box")

    ml_handle = plt.Line2D([], [], marker="o", linestyle="", color=_ML_COLOR, label="ML (CNN)")
    vr_handle = plt.Line2D([], [], marker="^", linestyle="", markerfacecolor="none",
                           markeredgecolor=_VR_COLOR, label="Voigt-Reuss")
    ref_handle = plt.Line2D([], [], linestyle="--", color=_REF_COLOR, label="y = x (ideal)")
    fig.legend(handles=[ml_handle, vr_handle, ref_handle], loc="upper center",
              ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"ML vs Voigt-Reuss homogenization -- {args.dataset_dir.name} "
                f"({len(rows)} held-out samples)", y=1.07)
    fig.tight_layout()

    out_path = args.out or (args.dataset_dir / "plots" / "ml_vs_voigt.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    print("\nPer-target MAE on this held-out set, ML vs Voigt-Reuss:")
    for target, vr_field, label in _PANELS:
        actual = np.array([float(r[target]) for r in rows])
        vr = np.array([float(r[vr_field]) for r in rows])
        ml = np.array([p[target] for p in preds])
        print(f"  {label:14s}  ml={np.mean(np.abs(ml - actual)):.5g}   "
             f"voigt_reuss={np.mean(np.abs(vr - actual)):.5g}")


if __name__ == "__main__":
    main()
