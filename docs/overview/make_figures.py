#!/usr/bin/env python3
"""
Regenerate the figures for docs/overview/main.tex from a solved sample.

Draws the overview-only input figure (both copper layers) and copies the
pipeline's own plots into figures/, cropping the white margins of the 3-D
renders. Run plot_board_model.py (and the other plot scripts) on the sample
first.

Usage (from the repository root):
    python docs/overview/make_figures.py data/samples/rv3032_example
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from solder_reliability import PAD_1_XY_MM, PAD_2_XY_MM  # noqa: E402

FR4, CU, PAD = "#cfd8c4", "#c8702a", "#1f4fd1"
CMAP = ListedColormap([FR4, CU])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})


def plot_input_layers(top, bot, Lx, Ly, pads, out: Path) -> None:
    ext = [0, Lx, 0, Ly]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, m, name in zip(axs, [top, bot], ["Top copper (F_Cu)", "Bottom copper (B_Cu)"]):
        ax.imshow(m > 0.5, origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"{name}  -  Cu fraction {np.mean(m > 0.5):.0%}")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    axs[0].plot(pads[:, 0], pads[:, 1], "+", color=PAD, ms=9, mew=1.8)
    axs[0].text(pads[:, 0].min() - 0.7, pads[:, 1].mean(), "R1", color=PAD, fontsize=9,
                fontweight="bold", ha="right", va="center")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, color=CU), Rectangle((0, 0), 1, 1, color=FR4)],
               labels=["copper", "FR4 (no copper)"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.savefig(out)
    plt.close(fig)


def copy_trimmed(src: Path, out: Path, cut_top: float = 0.0, pad: int = 12) -> None:
    """Copy src to out, dropping a top band (legend/title far from a 3-D model)
    and cropping uniform white margins."""
    if not src.exists():
        print(f"[skip] missing {src}")
        return
    im = Image.open(src).convert("RGB")
    if cut_top:
        im = im.crop((0, int(im.height * cut_top), im.width, im.height))
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bbox:
        im = im.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                      min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height)))
    im.save(out)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = parser.parse_args()

    sample, out = args.sample_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((sample / "metadata.json").read_text())
    Lx, Ly = meta["board"]["width_mm"], meta["board"]["height_mm"]
    top = np.load(sample / "mesh/F_Cu_map.npy")
    bot = np.load(sample / "mesh/B_Cu_map.npy")
    pads = np.array([PAD_1_XY_MM, PAD_2_XY_MM])

    plot_input_layers(top, bot, Lx, Ly, pads, out / "input_layers.png")
    print(f"Wrote {out / 'input_layers.png'}")

    res = sample / "results"
    copy_trimmed(res / "board_model.png", out / "board_model.png")
    copy_trimmed(res / "component_materials.png", out / "component_materials.png", cut_top=0.13)
    copy_trimmed(res / "thermal_profile.png", out / "thermal_profile.png")
    copy_trimmed(res / "zone_elements_highlight.png", out / "zone_elements.png", cut_top=0.09)
    copy_trimmed(res / "creep_strain_field.png", out / "creep_strain_field.png", cut_top=0.03)
    copy_trimmed(res / "zone_creep_strain.png", out / "zone_creep_strain.png")
    copy_trimmed(res / "cycle_creep_increment.png", out / "cycle_creep_increment.png")


if __name__ == "__main__":
    main()

