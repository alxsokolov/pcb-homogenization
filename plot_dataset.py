#!/usr/bin/env python3
"""
Plot simulation results against Voigt/Reuss bounds for a PCB dataset.

Generates one figure per output parameter, each showing simulation points
and analytical bound curves versus phi_3D (3D copper volume fraction).

Note: E_xy_voigt == E_voigt_global and E_z_reuss == E_reuss_global
algebraically — both pairs reduce to the same formula given phi_3D.
The plots will confirm this visually (curves overlap).

Output: <dataset_dir>/plots/*.png

Usage:
    python plot_dataset.py data/datasets/random_v1
    python plot_dataset.py data/datasets/random_v1 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi":       150,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# Colours
C_SIM   = "#2563EB"   # blue   — FEM simulation
C_VOIGT = "#F97316"   # orange — Voigt upper bound
C_REUSS = "#16A34A"   # green  — Reuss lower bound


def _sort(df: pd.DataFrame, col: str):
    """Return phi_3D and col sorted by phi_3D for clean line plots."""
    s = df[["phi_3D", col]].dropna().sort_values("phi_3D")
    return s["phi_3D"].values, s[col].values


def plot_E(ax, df: pd.DataFrame, col: str) -> None:
    """Young's modulus: simulation + Voigt upper bound + Reuss lower bound."""
    ax.scatter(df["phi_3D"], df[col],
               c=C_SIM, s=18, alpha=0.7, zorder=3, label="FEM simulation")

    x, y = _sort(df, "E_voigt_global_GPa")
    ax.plot(x, y, color=C_VOIGT, lw=2, label="Voigt bound (upper)")

    x, y = _sort(df, "E_reuss_global_GPa")
    ax.plot(x, y, color=C_REUSS, lw=2, label="Reuss bound (lower)")

    ax.set_xlabel("φ₃D  (3D copper volume fraction)")
    ax.set_ylabel("E  [GPa]")
    ax.legend(fontsize=8)


def plot_nu(ax, df: pd.DataFrame, col: str, label: str) -> None:
    """Poisson's ratio: simulation only (no standard analytical bound)."""
    ax.scatter(df["phi_3D"], df[col],
               c=C_SIM, s=18, alpha=0.7, zorder=3, label="FEM simulation")
    ax.set_xlabel("φ₃D  (3D copper volume fraction)")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)


def plot_alpha(ax, df: pd.DataFrame, col: str, voigt_col: str | None, reuss_col: str | None) -> None:
    """CTE: simulation + optional Voigt upper / Reuss lower bounds."""
    sub = df.dropna(subset=[col])
    ax.scatter(sub["phi_3D"], sub[col] * 1e6,
               c=C_SIM, s=18, alpha=0.7, zorder=3, label="FEM simulation")

    if voigt_col and voigt_col in df.columns:
        x, y = _sort(df.dropna(subset=[voigt_col]), voigt_col)
        ax.plot(x, y * 1e6, color=C_VOIGT, lw=2, label="Voigt bound (upper)")

    if reuss_col and reuss_col in df.columns:
        x, y = _sort(df.dropna(subset=[reuss_col]), reuss_col)
        ax.plot(x, y * 1e6, color=C_REUSS, lw=2, label="Reuss bound (lower)")

    ax.set_xlabel("φ₃D  (3D copper volume fraction)")
    ax.set_ylabel("CTE  [ppm/K]")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Figure definitions
# ---------------------------------------------------------------------------

_FIGURES = [
    # (stem,      title,                              kind,    col/args)
    ("E_x",     "In-plane Young's modulus  E_x",    "E",     "E_x_GPa"),
    ("E_y",     "In-plane Young's modulus  E_y",    "E",     "E_y_GPa"),
    ("E_z",     "Out-of-plane Young's modulus  E_z","E",     "E_z_GPa"),
    ("nu_xy",   "Poisson's ratio  ν_xy",            "nu",    ("nu_xy", "ν_xy")),
    ("nu_xz",   "Poisson's ratio  ν_xz",            "nu",    ("nu_xz", "ν_xz")),
    ("nu_yz",   "Poisson's ratio  ν_yz",            "nu",    ("nu_yz", "ν_yz")),
    # CTE: (col, voigt_bound_col, reuss_bound_col)
    ("alpha_x", "In-plane CTE  α_x",               "alpha", ("alpha_x_per_K", "alpha_xy_voigt_per_K", None)),
    ("alpha_y", "In-plane CTE  α_y",               "alpha", ("alpha_y_per_K", "alpha_xy_voigt_per_K", None)),
    ("alpha_z", "Out-of-plane CTE  α_z",           "alpha", ("alpha_z_per_K", None, "alpha_z_reuss_per_K")),
]


def make_figures(df: pd.DataFrame, out_dir: Path, show: bool) -> None:
    out_dir.mkdir(exist_ok=True)

    for stem, title, kind, args in _FIGURES:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title, fontsize=11)

        if kind == "E":
            plot_E(ax, df, args)
        elif kind == "nu":
            col, ylabel = args
            plot_nu(ax, df, col, ylabel)
        elif kind == "alpha":
            col, voigt_col, reuss_col = args
            plot_alpha(ax, df, col, voigt_col, reuss_col)

        fig.tight_layout()
        path = out_dir / f"{stem}.png"
        fig.savefig(path)
        print(f"  Saved {path.relative_to(out_dir.parent.parent.parent)}")
        if not show:
            plt.close(fig)

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot dataset simulation results vs Voigt/Reuss bounds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset_dir", type=Path,
                        help="Dataset directory containing index.csv")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively in addition to saving")
    args = parser.parse_args()

    index_csv = args.dataset_dir / "index.csv"
    if not index_csv.exists():
        raise SystemExit(f"Not found: {index_csv}")

    df = pd.read_csv(index_csv)
    df = df[df["status"] == "complete"].reset_index(drop=True)

    if df.empty:
        raise SystemExit("No complete samples in index.csv — run generate_dataset.py first.")

    print(f"Loaded {len(df)} complete samples from {index_csv}")

    out_dir = args.dataset_dir / "plots"
    make_figures(df, out_dir, show=args.show)
    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
