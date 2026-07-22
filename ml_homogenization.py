#!/usr/bin/env python3
"""
CNN surrogate for per-leaf homogenization: rasterized copper layout
(F_Cu, B_Cu) -> effective anisotropic elastic + CTE properties.

Trained on data/datasets/<id>/ samples produced by generate_dataset.py:
each sample is a random 50x50mm board with a full-resolution copper
raster and a FEniCSx-simulated ground-truth property set, aggregated in
the dataset's index.csv (which also already carries the analytical
Voigt-Reuss bounds computed by homogenization.voigt_reuss_bounds()). Those
bounds are fed to the network as side-features so it can learn a
"baseline + correction" rather than the raw raster-to-property mapping
from scratch.

Kept import-light at module load (numpy/scipy/torch only, no dolfinx) so
it can run in a plain CPU env — see quadtree.py for how it's wired in as
an alternative to voigt_reuss_bounds() for per-leaf homogenization.

Usage:
    python ml_homogenization.py train --datasets data/datasets/sweep_v3 data/datasets/training_data_1 \
        --out models/leaf_cnn_v1
    python ml_homogenization.py train --epochs 100 --img-size 128 --datasets data/datasets/training_data_1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from homogenization import load_materials, voigt_reuss_bounds

SIDE_FEATURES = [
    "phi_top", "phi_bot", "phi_3D",
    "E_xy_voigt_GPa", "E_z_reuss_GPa",
    "alpha_xy_voigt_per_K", "alpha_z_reuss_per_K",
]
TARGETS = [
    "E_x_GPa", "E_y_GPa", "E_z_GPa",
    "nu_xy", "nu_xz", "nu_yz",
    "alpha_x_per_K", "alpha_y_per_K", "alpha_z_per_K",
]
# 90/270-degree rotation swaps the x/y axes: these target pairs must swap too.
_SWAP_PAIRS = [("E_x_GPa", "E_y_GPa"), ("alpha_x_per_K", "alpha_y_per_K"), ("nu_xz", "nu_yz")]

# VR baseline field each target is measured against, where one exists (nu has none).
_VR_BASELINE_FIELD = {
    "E_x_GPa": "E_xy_voigt_GPa", "E_y_GPa": "E_xy_voigt_GPa", "E_z_GPa": "E_z_reuss_GPa",
    "alpha_x_per_K": "alpha_xy_voigt_per_K", "alpha_y_per_K": "alpha_xy_voigt_per_K",
    "alpha_z_per_K": "alpha_z_reuss_per_K",
}


# ---------------------------------------------------------------------------
# Shared preprocessing -- used identically at train time (full 500x500 board
# raster -> img_size) and inference time (arbitrary leaf-crop -> img_size)
# so preprocessing can never drift between training and serving.
# ---------------------------------------------------------------------------

def _resize_to(arr: np.ndarray, size: int) -> np.ndarray:
    zoomed = zoom(arr, (size / arr.shape[0], size / arr.shape[1]), order=1)
    if zoomed.shape == (size, size):
        return zoomed.astype(np.float32)
    out = np.zeros((size, size), dtype=np.float32)
    ys, xs = min(size, zoomed.shape[0]), min(size, zoomed.shape[1])
    out[:ys, :xs] = zoomed[:ys, :xs]
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_index_rows(dataset_dir: Path) -> list[dict]:
    idx_path = dataset_dir / "index.csv"
    if not idx_path.exists():
        sys.exit(f"Not found: {idx_path}")
    with open(idx_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "complete"]
    for r in rows:
        r["_dataset_dir"] = str(dataset_dir)
    return rows


def _gather_rows(dataset_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for d in dataset_dirs:
        rows.extend(_load_index_rows(d))
    return rows


def _compute_stats(rows: list[dict], fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array([[float(r[f]) for f in fields] for r in rows], dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _augment_d4(img: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a uniformly random element of the board's D4 in-plane symmetry
    group (4 rotations x mirror-or-not), transforming labels consistently.
    Free ~8x effective dataset size -- physically motivated, not generic
    image augmentation, since the board has no preferred in-plane
    orientation."""
    k = random.randint(0, 3)
    img = np.rot90(img, k=k, axes=(1, 2))
    if random.random() < 0.5:
        img = np.flip(img, axis=2)
    out_target = target.copy()
    if k % 2 == 1:
        idx = {name: i for i, name in enumerate(TARGETS)}
        for a, b in _SWAP_PAIRS:
            out_target[idx[a]], out_target[idx[b]] = target[idx[b]], target[idx[a]]
    return np.ascontiguousarray(img), out_target


class LeafRasterDataset(Dataset):
    def __init__(self, rows: list[dict], img_size: int,
                 side_mean: np.ndarray, side_std: np.ndarray,
                 target_mean: np.ndarray, target_std: np.ndarray,
                 augment: bool):
        self.rows = rows
        self.img_size = img_size
        self.side_mean, self.side_std = side_mean, side_std
        self.target_mean, self.target_std = target_mean, target_std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        sample_dir = Path(row["_dataset_dir"]) / "samples" / row["sample_id"]
        top = _resize_to(np.load(sample_dir / "mesh" / "F_Cu_map.npy"), self.img_size)
        bot = _resize_to(np.load(sample_dir / "mesh" / "B_Cu_map.npy"), self.img_size)
        img = np.stack([top, bot], axis=0)

        side = np.array([float(row[f]) for f in SIDE_FEATURES], dtype=np.float32)
        target = np.array([float(row[f]) for f in TARGETS], dtype=np.float32)

        if self.augment:
            img, target = _augment_d4(img, target)

        side = (side - self.side_mean) / self.side_std
        target = (target - self.target_mean) / self.target_std
        return torch.from_numpy(img), torch.from_numpy(side), torch.from_numpy(target)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LeafCNN(nn.Module):
    def __init__(self, n_side: int, n_target: int):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.conv = nn.Sequential(block(2, 16), block(16, 32), block(32, 64), block(64, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(64 + n_side, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, n_target),
        )

    def forward(self, img: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv(img)).flatten(1)
        return self.head(torch.cat([x, side], dim=1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, loss_fn, opt=None) -> float:
    train_mode = opt is not None
    model.train(train_mode)
    tot, n = 0.0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for img, side, target in loader:
            if train_mode:
                opt.zero_grad()
            pred = model(img, side)
            loss = loss_fn(pred, target)
            if train_mode:
                loss.backward()
                opt.step()
            tot += loss.item() * img.shape[0]
            n += img.shape[0]
    return tot / n


def _evaluate_physical(model, val_rows, img_size, side_mean, side_std,
                       target_mean, target_std) -> dict:
    ds = LeafRasterDataset(val_rows, img_size, side_mean, side_std, target_mean, target_std, augment=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for img, side, _ in loader:
            preds.append(model(img, side).numpy())
    preds = np.concatenate(preds) * target_std + target_mean
    targets = np.array([[float(r[t]) for t in TARGETS] for r in val_rows], dtype=np.float64)

    metrics = {}
    for i, t in enumerate(TARGETS):
        entry = {"mae_ml": float(np.mean(np.abs(preds[:, i] - targets[:, i])))}
        vr_field = _VR_BASELINE_FIELD.get(t)
        if vr_field is not None:
            vr_vals = np.array([float(r[vr_field]) for r in val_rows])
            entry["mae_voigt_reuss"] = float(np.mean(np.abs(vr_vals - targets[:, i])))
        metrics[t] = entry
    return metrics


def train(args: argparse.Namespace) -> None:
    dataset_dirs = [Path(d) for d in args.datasets]
    rows = _gather_rows(dataset_dirs)
    if len(rows) < 10:
        sys.exit(f"Only {len(rows)} complete samples found across {dataset_dirs} -- need more to train.")

    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_frac))
    val_rows, train_rows = shuffled[:n_val], shuffled[n_val:]
    print(f"{len(rows)} complete samples -> {len(train_rows)} train / {len(val_rows)} val")

    side_mean, side_std = _compute_stats(train_rows, SIDE_FEATURES)
    target_mean, target_std = _compute_stats(train_rows, TARGETS)

    train_ds = LeafRasterDataset(train_rows, args.img_size, side_mean, side_std,
                                 target_mean, target_std, augment=not args.no_augment)
    val_ds = LeafRasterDataset(val_rows, args.img_size, side_mean, side_std,
                               target_mean, target_std, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)
    model = LeafCNN(n_side=len(SIDE_FEATURES), n_target=len(TARGETS))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state = float("inf"), None
    for epoch in range(args.epochs):
        train_loss = _run_epoch(model, train_loader, loss_fn, opt)
        val_loss = _run_epoch(model, val_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics = _evaluate_physical(model, val_rows, args.img_size, side_mean, side_std,
                                 target_mean, target_std)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "checkpoint.pt")

    config = {
        "img_size": args.img_size,
        "side_features": SIDE_FEATURES,
        "targets": TARGETS,
        "side_mean": side_mean.tolist(), "side_std": side_std.tolist(),
        "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
        "dataset_dirs": [str(d) for d in dataset_dirs],
        "n_train": len(train_rows), "n_val": len(val_rows),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "metrics.json").write_text(json.dumps(
        {"best_val_loss": best_val, "per_target": metrics}, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("standardized MSE loss"); ax.set_yscale("log")
    ax.legend(); ax.set_title("LeafCNN training curve")
    fig.tight_layout()
    fig.savefig(args.out / "training_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote {args.out}/ (checkpoint.pt, config.json, metrics.json, training_curve.png)")
    print("\nPer-target val MAE, ML vs Voigt-Reuss baseline:")
    for t in TARGETS:
        m = metrics[t]
        if "mae_voigt_reuss" in m:
            print(f"  {t:16s}  ml={m['mae_ml']:.5g}   voigt_reuss={m['mae_voigt_reuss']:.5g}")
        else:
            print(f"  {t:16s}  ml={m['mae_ml']:.5g}   (no analytical baseline)")


# ---------------------------------------------------------------------------
# Inference -- used by quadtree.py as a drop-in for voigt_reuss_bounds()
# ---------------------------------------------------------------------------

def load_model(model_dir: Path) -> tuple[LeafCNN, dict]:
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    model = LeafCNN(n_side=len(config["side_features"]), n_target=len(config["targets"]))
    model.load_state_dict(torch.load(model_dir / "checkpoint.pt", map_location="cpu"))
    model.eval()
    return model, config


def predict_leaf_properties(top_crop: np.ndarray, bot_crop: np.ndarray,
                            phi_top: float, phi_bot: float,
                            materials: dict, t_cu_mm: float, t_fr4_mm: float,
                            model: LeafCNN, config: dict) -> dict:
    E_cu = materials["copper"]["E_GPa"] * 1e9
    E_fr4 = materials["FR4"]["E_GPa"] * 1e9
    alpha_cu = materials["copper"].get("CTE_per_K", 17e-6)
    alpha_fr4_ip = materials["FR4"].get("CTE_inplane_per_K", materials["FR4"].get("CTE_per_K", 14e-6))
    alpha_fr4_op = materials["FR4"].get("CTE_outofplane_per_K", materials["FR4"].get("CTE_per_K", 60e-6))
    baseline = voigt_reuss_bounds(phi_top, phi_bot, E_cu=E_cu, E_fr4=E_fr4,
                                  t_cu=t_cu_mm, t_fr4=t_fr4_mm, alpha_cu=alpha_cu,
                                  alpha_fr4_ip=alpha_fr4_ip, alpha_fr4_op=alpha_fr4_op)

    img_size = config["img_size"]
    top = _resize_to(np.asarray(top_crop, dtype=np.float32), img_size)
    bot = _resize_to(np.asarray(bot_crop, dtype=np.float32), img_size)
    img = np.stack([top, bot], axis=0)[None, ...]

    side_mean = np.array(config["side_mean"], dtype=np.float32)
    side_std = np.array(config["side_std"], dtype=np.float32)
    target_mean = np.array(config["target_mean"], dtype=np.float32)
    target_std = np.array(config["target_std"], dtype=np.float32)
    side_raw = np.array([baseline[f] for f in config["side_features"]], dtype=np.float32)
    side = ((side_raw - side_mean) / side_std)[None, ...]

    with torch.no_grad():
        pred = model(torch.from_numpy(img), torch.from_numpy(side)).numpy()[0]
    pred = pred * target_std + target_mean
    out = dict(zip(config["targets"], (float(v) for v in pred)))

    return {
        "phi_top": round(baseline["phi_top"], 6),
        "phi_bot": round(baseline["phi_bot"], 6),
        "phi_3D": round(baseline["phi_3D"], 6),
        "E_xy_voigt_GPa": round((out["E_x_GPa"] + out["E_y_GPa"]) / 2, 6),
        "E_z_reuss_GPa": round(out["E_z_GPa"], 6),
        "alpha_xy_voigt_per_K": round((out["alpha_x_per_K"] + out["alpha_y_per_K"]) / 2, 9),
        "alpha_z_reuss_per_K": round(out["alpha_z_per_K"], 9),
        "E_x_GPa": round(out["E_x_GPa"], 6),
        "E_y_GPa": round(out["E_y_GPa"], 6),
        "E_z_GPa": round(out["E_z_GPa"], 6),
        "nu_xy": round(out["nu_xy"], 6),
        "nu_xz": round(out["nu_xz"], 6),
        "nu_yz": round(out["nu_yz"], 6),
        "alpha_x_per_K": round(out["alpha_x_per_K"], 9),
        "alpha_y_per_K": round(out["alpha_y_per_K"], 9),
        "alpha_z_per_K": round(out["alpha_z_per_K"], 9),
        "source": "ml",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train / use the CNN leaf-homogenization surrogate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train on one or more datasets under data/datasets/.")
    p_train.add_argument("--datasets", nargs="+", required=True, type=Path)
    p_train.add_argument("--img-size", type=int, default=96)
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--val-frac", type=float, default=0.15)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--out", type=Path, default=Path("models/leaf_cnn_v1"))
    p_train.add_argument("--no-augment", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
