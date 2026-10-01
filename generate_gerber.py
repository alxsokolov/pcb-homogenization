#!/usr/bin/env python3
"""
Generate PCB Gerber layouts for the FEniCSx homogenization pipeline.

Layout types (--type):
  full_copper  Solid copper pour on both layers (baseline).
  example      Central pour + corner pads + traces (top), crossing buses (bottom).
  random       Randomised traces, pads, and pours; seed + counts are configurable.

Directory layout produced for each sample:
    <output_dir>/<sample_id>/
        gerber/
            F_Cu.gbr          RS-274X, top copper layer
            B_Cu.gbr          RS-274X, bottom copper layer
            Edge_Cuts.gbr     RS-274X, board outline
        mesh/                 populated by discretize.py
        results/              populated by the simulation scripts
        metadata.json         board & stackup parameters

Usage:
    python generate_gerber.py
    python generate_gerber.py --width 40 --height 40 --sample-id board_001
    python generate_gerber.py --type random --n-traces 30 --n-pads 15 --n-pours 4
    python generate_gerber.py --type random --seed 42
    python generate_gerber.py --config configs/default.yaml
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Gerber helpers
# ---------------------------------------------------------------------------

# RS-274X X46Y46 format: 4 integer digits + 6 decimal digits → 1 mm = 1 000 000 units
_SCALE = 1_000_000


def _mm(value: float) -> int:
    return int(round(value * _SCALE))


def _gerber_header(layer_name: str, file_function: str, width_mm: float, height_mm: float) -> list[str]:
    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    return [
        f"G04 {layer_name}*",
        f"G04 PCB Homogenization — {width_mm:.4f} mm x {height_mm:.4f} mm*",
        f"G04 Generated: {ts}*",
        "%FSLAX46Y46*%",
        "%MOMM*%",
        f"%TF.FileFunction,{file_function}*%",
        "%TF.FilePolarity,Positive*%",
        "%LPD*%",
    ]


def write_copper_layer(path: Path, width_mm: float, height_mm: float,
                       layer_name: str, file_function: str) -> None:
    """RS-274X file with a solid copper region fill over the full rectangle."""
    x0, y0 = _mm(0.0), _mm(0.0)
    x1, y1 = _mm(width_mm), _mm(height_mm)

    lines = _gerber_header(layer_name, file_function, width_mm, height_mm) + [
        # A tiny circle aperture is required before G36 by most parsers
        "%ADD10C,0.010*%",
        "D10*",
        "G01*",  # linear interpolation mode
        "G36*",  # begin region
        f"X{x0}Y{y0}D02*",   # move to origin (no draw)
        f"X{x1}Y{y0}D01*",   # bottom edge →
        f"X{x1}Y{y1}D01*",   # right edge ↑
        f"X{x0}Y{y1}D01*",   # top edge ←
        f"X{x0}Y{y0}D01*",   # left edge ↓ (close contour)
        "G37*",  # end region
        "M02*",
    ]
    path.write_text("\n".join(lines) + "\n")


def _region(x0: float, y0: float, x1: float, y1: float) -> list[str]:
    """G36/G37 block for a closed rectangular copper pour."""
    return [
        "G36*",
        f"X{_mm(x0)}Y{_mm(y0)}D02*",
        f"X{_mm(x1)}Y{_mm(y0)}D01*",
        f"X{_mm(x1)}Y{_mm(y1)}D01*",
        f"X{_mm(x0)}Y{_mm(y1)}D01*",
        f"X{_mm(x0)}Y{_mm(y0)}D01*",
        "G37*",
    ]


def write_example_f_cu(path: Path, width_mm: float, height_mm: float) -> None:
    """
    F.Cu example layer with partial coverage:
      - Central copper pour  (20 %–80 % of each board dimension, 36 % area)
      - Four corner pads     (3 mm diameter circles)
      - Four diagonal traces (0.3 mm wide) from each pad to the nearest pour corner
    Exercises all three geometry types: region, flash, stroke.
    """
    W, H = width_mm, height_mm
    px0, py0 = 0.2 * W, 0.2 * H   # pour lower-left
    px1, py1 = 0.8 * W, 0.8 * H   # pour upper-right
    pads         = [(0.1*W, 0.1*H), (0.9*W, 0.1*H), (0.9*W, 0.9*H), (0.1*W, 0.9*H)]
    pour_corners = [(px0,   py0),   (px1,   py0),   (px1,   py1),   (px0,   py1)]

    lines = _gerber_header("F.Cu — Example partial copper", "Copper,L1,Top", W, H) + [
        "%ADD10C,3.000*%",   # pad aperture  (3 mm circle)
        "%ADD11C,0.300*%",   # trace aperture (0.3 mm circle)
        "%ADD12C,0.010*%",   # region helper aperture (required before G36)
        "G01*",
    ]

    # Central pour
    lines += ["D12*"] + _region(px0, py0, px1, py1)

    # Corner pad flashes
    lines.append("D10*")
    for x, y in pads:
        lines.append(f"X{_mm(x)}Y{_mm(y)}D03*")

    # Diagonal traces: pad → pour corner
    lines.append("D11*")
    for (px, py), (cx, cy) in zip(pads, pour_corners):
        lines += [f"X{_mm(px)}Y{_mm(py)}D02*",
                  f"X{_mm(cx)}Y{_mm(cy)}D01*"]

    lines.append("M02*")
    path.write_text("\n".join(lines) + "\n")


def write_example_b_cu(path: Path, width_mm: float, height_mm: float) -> None:
    """
    B.Cu example layer: two crossing 1 mm bus traces (horizontal + vertical).
    """
    W, H = width_mm, height_mm
    mx, my = W / 2, H / 2

    lines = _gerber_header("B.Cu — Example bus traces", "Copper,L2,Bot", W, H) + [
        "%ADD10C,1.000*%",   # 1 mm trace aperture
        "G01*",
        "D10*",
        f"X{_mm(0)}Y{_mm(my)}D02*", f"X{_mm(W)}Y{_mm(my)}D01*",   # horizontal bus
        f"X{_mm(mx)}Y{_mm(0)}D02*", f"X{_mm(mx)}Y{_mm(H)}D01*",   # vertical bus
        "M02*",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_random_layer(
    path: Path,
    width_mm: float,
    height_mm: float,
    layer_name: str,
    file_function: str,
    rng: np.random.Generator,
    *,
    n_pads: int,
    n_traces: int,
    n_pours: int,
    trace_width_range: tuple,
    pad_size_range: tuple,
    h_bias: float = 0.5,
) -> None:
    """
    Random RS-274X layer: rectangular pours, axis-aligned traces, ring pads at endpoints.

    All traces on a layer share one width (realistic).  Traces run only along X
    or Y (Manhattan routing).  h_bias controls the probability of a horizontal
    trace (0 = all vertical, 1 = all horizontal, 0.5 = isotropic).  A solid
    circular pad is flashed at both endpoints of every trace.
    """
    W, H = width_mm, height_mm
    lines = _gerber_header(layer_name, file_function, W, H)

    ap_id = 10

    # One trace width for the whole layer
    trace_w = float(rng.uniform(*trace_width_range))

    # Ring (pad) diameter — larger than trace width to be visible as a ring
    pad_d = max(trace_w * 1.8, float(rng.uniform(*pad_size_range)))

    trace_ap = ap_id
    lines.append(f"%ADD{ap_id}C,{trace_w:.4f}*%")
    ap_id += 1

    pad_ap = ap_id
    lines.append(f"%ADD{ap_id}C,{pad_d:.4f}*%")
    ap_id += 1

    region_ap = ap_id
    if n_pours > 0:
        lines.append(f"%ADD{ap_id}C,0.010*%")
        ap_id += 1

    lines.append("G01*")

    # Rectangular copper pours
    if n_pours > 0:
        lines.append(f"D{region_ap}*")
        for _ in range(n_pours):
            pw = float(rng.uniform(W * 0.05, W * 0.6))
            ph = float(rng.uniform(H * 0.05, H * 0.6))
            x0 = float(rng.uniform(0.0, W - pw))
            y0 = float(rng.uniform(0.0, H - ph))
            lines += _region(x0, y0, x0 + pw, y0 + ph)

    # Axis-aligned traces; collect endpoints for ring pads
    endpoints: list[tuple[float, float]] = []
    if n_traces > 0:
        lines.append(f"D{trace_ap}*")
        for _ in range(n_traces):
            if rng.random() < h_bias:   # horizontal if h_bias→1, vertical if h_bias→0
                x0 = float(rng.uniform(0.0, W))
                x1 = float(rng.uniform(0.0, W))
                y0 = y1 = float(rng.uniform(0.0, H))
            else:                      # vertical
                x0 = x1 = float(rng.uniform(0.0, W))
                y0 = float(rng.uniform(0.0, H))
                y1 = float(rng.uniform(0.0, H))
            lines += [
                f"X{_mm(x0)}Y{_mm(y0)}D02*",
                f"X{_mm(x1)}Y{_mm(y1)}D01*",
            ]
            endpoints.extend([(x0, y0), (x1, y1)])

    # Flash a ring pad at every trace endpoint
    if endpoints:
        lines.append(f"D{pad_ap}*")
        for x, y in endpoints:
            lines.append(f"X{_mm(x)}Y{_mm(y)}D03*")

    lines.append("M02*")
    path.write_text("\n".join(lines) + "\n")


def write_edge_cuts(path: Path, width_mm: float, height_mm: float) -> None:
    """RS-274X board outline layer (Edge.Cuts)."""
    x0, y0 = _mm(0.0), _mm(0.0)
    x1, y1 = _mm(width_mm), _mm(height_mm)

    lines = _gerber_header("Edge.Cuts — board outline", "Profile,NP", width_mm, height_mm) + [
        "%ADD10C,0.050*%",
        "D10*",
        "G01*",
        f"X{x0}Y{y0}D02*",
        f"X{x1}Y{y0}D01*",
        f"X{x1}Y{y1}D01*",
        f"X{x0}Y{y1}D01*",
        f"X{x0}Y{y0}D01*",
        "M02*",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_metadata(path: Path, *, sample_id: str, width_mm: float, height_mm: float,
                   cu_thickness_mm: float, dielectric_thickness_mm: float,
                   generation_type: str = "full_copper",
                   generation_params: dict | None = None) -> None:
    generation = {
        "script": "generate_gerber.py",
        "type": generation_type,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    if generation_params:
        generation.update(generation_params)

    meta = {
        "sample_id": sample_id,
        "board": {
            "width_mm": width_mm,
            "height_mm": height_mm,
        },
        "layers": [
            {
                "name": "F_Cu",
                "file": "gerber/F_Cu.gbr",
                "type": "copper",
                "position": "top",
                "thickness_mm": cu_thickness_mm,
            },
            {
                "name": "B_Cu",
                "file": "gerber/B_Cu.gbr",
                "type": "copper",
                "position": "bottom",
                "thickness_mm": cu_thickness_mm,
            },
        ],
        "stackup": {
            "n_copper_layers": 2,
            "copper_thickness_mm": cu_thickness_mm,
            "dielectric_material": "FR4",
            "dielectric_thickness_mm": dielectric_thickness_mm,
            "total_thickness_mm": round(2 * cu_thickness_mm + dielectric_thickness_mm, 6),
        },
        "generation": generation,
    }
    path.write_text(json.dumps(meta, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Sample assembly
# ---------------------------------------------------------------------------

_LAYOUT_TYPES = ("full_copper", "example", "random")


def generate_sample(
    output_root: Path,
    sample_id: str,
    width_mm: float,
    height_mm: float,
    cu_thickness_mm: float = 0.035,
    dielectric_thickness_mm: float = 1.53,
    layout_type: str = "full_copper",
    # random-layout parameters (ignored for other types)
    seed: int | None = None,
    n_pads: int = 10,
    n_traces: int = 20,
    n_pours: int = 3,
    trace_width_range: tuple = (0.1, 2.0),
    pad_size_range: tuple = (0.5, 3.0),
    h_bias: float = 0.5,
) -> Path:
    """Build one sample directory and return its path."""
    if layout_type not in _LAYOUT_TYPES:
        raise ValueError(f"Unknown layout_type {layout_type!r}; choose from {_LAYOUT_TYPES}")

    sample_dir = output_root / sample_id
    gerber_dir = sample_dir / "gerber"

    for d in [gerber_dir, sample_dir / "mesh", sample_dir / "results"]:
        d.mkdir(parents=True, exist_ok=True)

    generation_params: dict | None = None

    if layout_type == "full_copper":
        write_copper_layer(
            gerber_dir / "F_Cu.gbr", width_mm, height_mm,
            layer_name="F.Cu — Top copper full pour",
            file_function="Copper,L1,Top",
        )
        write_copper_layer(
            gerber_dir / "B_Cu.gbr", width_mm, height_mm,
            layer_name="B.Cu — Bottom copper full pour",
            file_function="Copper,L2,Bot",
        )

    elif layout_type == "example":
        write_example_f_cu(gerber_dir / "F_Cu.gbr", width_mm, height_mm)
        write_example_b_cu(gerber_dir / "B_Cu.gbr", width_mm, height_mm)

    elif layout_type == "random":
        # Resolve seed now so it ends up in metadata for reproducibility.
        resolved_seed = seed if seed is not None else int(np.random.default_rng().integers(0, 2**31))
        rng = np.random.default_rng(resolved_seed)

        def _layer_count(n: int) -> int:
            """Draw a count in [n//2, n*3//2] so the two layers differ."""
            if n == 0:
                return 0
            return int(rng.integers(max(0, n // 2), max(1, n * 3 // 2) + 1))

        write_random_layer(
            gerber_dir / "F_Cu.gbr", width_mm, height_mm,
            layer_name="F.Cu — Random layout",
            file_function="Copper,L1,Top",
            rng=rng,
            n_pads=_layer_count(n_pads),
            n_traces=_layer_count(n_traces),
            n_pours=_layer_count(n_pours),
            trace_width_range=trace_width_range,
            pad_size_range=pad_size_range,
            h_bias=h_bias,
        )
        write_random_layer(
            gerber_dir / "B_Cu.gbr", width_mm, height_mm,
            layer_name="B.Cu — Random layout",
            file_function="Copper,L2,Bot",
            rng=rng,
            n_pads=_layer_count(n_pads),
            n_traces=_layer_count(n_traces),
            n_pours=_layer_count(n_pours),
            trace_width_range=trace_width_range,
            pad_size_range=pad_size_range,
            h_bias=h_bias,
        )
        generation_params = {
            "seed": resolved_seed,
            "n_pads_nominal": n_pads,
            "n_traces_nominal": n_traces,
            "n_pours_nominal": n_pours,
            "trace_width_range_mm": list(trace_width_range),
            "pad_size_range_mm": list(pad_size_range),
            "h_bias": h_bias,
        }

    write_edge_cuts(gerber_dir / "Edge_Cuts.gbr", width_mm, height_mm)
    write_metadata(
        sample_dir / "metadata.json",
        sample_id=sample_id,
        width_mm=width_mm,
        height_mm=height_mm,
        cu_thickness_mm=cu_thickness_mm,
        dielectric_thickness_mm=dielectric_thickness_mm,
        generation_type=layout_type,
        generation_params=generation_params,
    )
    return sample_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    if not _YAML_AVAILABLE:
        sys.exit("PyYAML is required to load a config file: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dummy full-copper Gerber layouts for FEniCSx homogenization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, metavar="FILE",
                        help="YAML config file (values override CLI defaults)")
    parser.add_argument("--width", type=float, default=50.0, help="Board width [mm]")
    parser.add_argument("--height", type=float, default=50.0, help="Board height [mm]")
    parser.add_argument("--cu-thickness", type=float, default=0.035,
                        help="Copper layer thickness [mm] (1 oz = 0.035 mm)")
    parser.add_argument("--dielectric-thickness", type=float, default=1.53,
                        help="FR4 core thickness [mm]")
    parser.add_argument("--sample-id", type=str, default=None,
                        help="Sample identifier (default: auto timestamp)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples"),
                        help="Root directory for generated samples")
    parser.add_argument("--type", dest="layout_type", default="full_copper",
                        choices=_LAYOUT_TYPES,
                        help=(
                            "full_copper: solid copper pour on both layers; "
                            "example: central pour + corner pads + traces (top), "
                            "crossing bus traces (bottom); "
                            "random: randomised pours, traces, and pads"
                        ))

    rnd = parser.add_argument_group("random layout options (only used with --type random)")
    rnd.add_argument("--seed", type=int, default=None,
                     help="RNG seed for reproducibility (default: random)")
    rnd.add_argument("--n-pads", type=int, default=10,
                     help="Nominal pad count per layer (actual count varies ±50%%)")
    rnd.add_argument("--n-traces", type=int, default=20,
                     help="Nominal trace count per layer (actual count varies ±50%%)")
    rnd.add_argument("--n-pours", type=int, default=3,
                     help="Nominal rectangular-pour count per layer (actual count varies ±50%%)")
    rnd.add_argument("--trace-width-min", type=float, default=0.1,
                     help="Minimum trace width [mm]")
    rnd.add_argument("--trace-width-max", type=float, default=2.0,
                     help="Maximum trace width [mm]")
    rnd.add_argument("--pad-size-min", type=float, default=0.5,
                     help="Minimum pad diameter / side [mm]")
    rnd.add_argument("--pad-size-max", type=float, default=3.0,
                     help="Maximum pad diameter / side [mm]")
    rnd.add_argument("--h-bias", type=float, default=0.5,
                     help="Probability of horizontal trace (0=all vertical, 1=all horizontal, 0.5=isotropic)")

    args = parser.parse_args()

    # Merge YAML config (if given) with CLI args; CLI wins
    cfg: dict = {}
    if args.config:
        cfg = _load_config(args.config)

    width = args.width if "--width" in sys.argv else cfg.get("board", {}).get("width_mm", args.width)
    height = args.height if "--height" in sys.argv else cfg.get("board", {}).get("height_mm", args.height)
    cu = args.cu_thickness if "--cu-thickness" in sys.argv else cfg.get("stackup", {}).get("copper_thickness_mm", args.cu_thickness)
    diel = args.dielectric_thickness if "--dielectric-thickness" in sys.argv else cfg.get("stackup", {}).get("dielectric_thickness_mm", args.dielectric_thickness)
    out_dir = args.output_dir if "--output-dir" in sys.argv else Path(cfg.get("output", {}).get("dir", args.output_dir))

    sample_id = args.sample_id or f"sample_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    sample_path = generate_sample(
        output_root=out_dir,
        sample_id=sample_id,
        width_mm=width,
        height_mm=height,
        cu_thickness_mm=cu,
        dielectric_thickness_mm=diel,
        layout_type=args.layout_type,
        seed=args.seed,
        n_pads=args.n_pads,
        n_traces=args.n_traces,
        n_pours=args.n_pours,
        trace_width_range=(args.trace_width_min, args.trace_width_max),
        pad_size_range=(args.pad_size_min, args.pad_size_max),
        h_bias=args.h_bias,
    )

    print(f"Generated sample → {sample_path}")
    for f in sorted(sample_path.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(sample_path)}")


if __name__ == "__main__":
    main()
