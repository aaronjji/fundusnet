"""
train.py — FundusNet training script.

Five ablation variants:
  baseline          : no conditioning, no dark aug, pos_weight=2.0
  aug-only          : no conditioning, dark aug + 3× dark oversample, pos_weight=2.0
  tone-only         : ODCR conditioning, no dark aug, pos_weight=2.0
  full              : ODCR conditioning, dark aug + 3× dark oversample, pos_weight=2.0
  balanced-baseline : no conditioning, equalised group sampling, pos_weight=1.0
                      (proves disparity is feature-learning not data-counting)

Usage:
  python src/models/train.py --model baseline
  python src/models/train.py --model full
  python src/models/train.py --model all        # all five sequentially

Outputs (per variant):
  checkpoints/<variant>/best.pt
  results/training/<variant>/history.json
  results/training/<variant>/summary.json
"""

import argparse
import json
import os
import site
import sys
import time
from pathlib import Path

# Windows: ensure torch CUDA DLLs are findable before first torch import
if sys.platform == "win32":
    for _sp in site.getsitepackages():
        _lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_lib):
            os.add_dll_directory(_lib)
            break

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
_cache      = ROOT / "data" / "train_512"
IMAGE_DIR   = _cache if _cache.exists() else ROOT / "data" / "train"
SPLITS_DIR  = ROOT / "data" / "splits"
ODCR_DIR    = ROOT / "data" / "odcr"
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results" / "training"

# ODCR thresholds recalibrated to population tertiles (25th/75th pct)
ODCR_DARK_THRESH  = -29.0
ODCR_LIGHT_THRESH =  20.0

# Variant configs: use_odcr, use_dark_aug, oversample_dark, balanced
VARIANTS = {
    "baseline":          {"use_odcr": False, "use_dark_aug": False, "oversample_dark": False, "balanced": False},
    "aug-only":          {"use_odcr": False, "use_dark_aug": True,  "oversample_dark": True,  "balanced": False},
    "tone-only":         {"use_odcr": True,  "use_dark_aug": False, "oversample_dark": False, "balanced": False},
    "full":              {"use_odcr": True,  "use_dark_aug": True,  "oversample_dark": True,  "balanced": False},
    "balanced-baseline": {"use_odcr": False, "use_dark_aug": False, "oversample_dark": False, "balanced": True},
}


# ── Dark fundus augmentation ──────────────────────────────────────

class DarkFundusAug:
    """
    Simulate darker choroidal pigmentation by darkening the L* channel.
    Applied with p=0.5; darkening strength uniform in [0.15, 0.25].
    """
    def __init__(self, p: float = 0.5, strength: tuple = (0.15, 0.25)):
        self.p = p
        self.strength = strength

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return img_np
        factor = 1.0 - np.random.uniform(*self.strength)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * factor, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


# ── Dataset ───────────────────────────────────────────────────────

def _odcr_group(odcr_deg):
    """Assign recalibrated tertile group from a scalar ODCR degree value."""
    if pd.isna(odcr_deg):
        return "medium"
    if odcr_deg >= ODCR_LIGHT_THRESH:
        return "light"
    if odcr_deg <= ODCR_DARK_THRESH:
        return "dark"
    return "medium"


class FundusDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        odcr_csv: Path,
        image_dir: Path,
        augment: bool = False,
        use_dark_aug: bool = False,
    ):
        df_split = pd.read_csv(split_csv)
        df_odcr  = pd.read_csv(odcr_csv)[
            ["image", "odcr", "odcr_sin", "odcr_cos", "odcr_norm"]
        ]
        self.df = df_split.merge(df_odcr, on="image", how="left")

        for col in ["odcr_sin", "odcr_cos", "odcr_norm"]:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # Recalibrated group — used for sampling weights
        self.df["group"] = self.df["odcr"].apply(_odcr_group)

        self.image_dir = image_dir
        self.dark_aug  = DarkFundusAug() if use_dark_aug else None
        size = 260  # EfficientNet-B2 canonical native resolution

        spatial_aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ] if augment else []

        self.tensor_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.pil_aug = transforms.Compose(spatial_aug) if spatial_aug else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = None
        for ext in [".jpeg", ".jpg", ".png"]:
            p = self.image_dir / f"{row['image']}{ext}"
            if p.exists():
                img_path = p
                break

        img_bgr = cv2.imread(str(img_path)) if img_path else None
        if img_bgr is None:
            img_bgr = np.zeros((456, 456, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.dark_aug is not None:
            img_rgb = self.dark_aug(img_rgb)

        img_pil = transforms.ToPILImage()(img_rgb)
        if self.pil_aug is not None:
            img_pil = self.pil_aug(img_pil)
        img = self.tensor_transform(img_pil)

        label    = torch.tensor(float(row["label"]),   dtype=torch.float32)
        odcr_enc = torch.tensor(
            [row["odcr_sin"], row["odcr_cos"], row["odcr_norm"]], dtype=torch.float32
        )
        return img, label, odcr_enc, row["image"]

    def make_sampler(self, oversample_dark: bool = False, balanced: bool = False):
        """Return a WeightedRandomSampler or None for the given sampling strategy."""
        if oversample_dark:
            w = self.df["group"].map({"dark": 3.0, "medium": 1.0, "light": 1.0}).fillna(1.0).values
        elif balanced:
            counts = self.df["group"].value_counts()
            w = self.df["group"].map(lambda g: 1.0 / counts.get(g, 1)).values
        else:
            return None
        return WeightedRandomSampler(torch.from_numpy(w.astype(np.float32)), len(w))


# ── Model ─────────────────────────────────────────────────────────

class FundusNet(nn.Module):
    def __init__(self, use_odcr: bool = True):
        super().__init__()
        self.use_odcr = use_odcr
        self.backbone  = timm.create_model("efficientnet_b2", pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features  # 1408 for B2

        if use_odcr:
            self.odcr_branch = nn.Sequential(
                nn.Linear(3, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
            )
            self.head = nn.Linear(feat_dim + 32, 1)
        else:
            self.odcr_branch = None
            self.head = nn.Linear(feat_dim, 1)

    def forward(self, img, odcr_enc=None):
        feats = self.backbone(img)
        if self.use_odcr and odcr_enc is not None:
            feats = torch.cat([feats, self.odcr_branch(odcr_enc)], dim=1)
        return self.head(feats).squeeze(1)


# ── Training loop ─────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels, odcr_encs, _ in tqdm(loader, leave=False):
            imgs      = imgs.to(device)
            labels    = labels.to(device)
            odcr_encs = odcr_encs.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits = model(imgs, odcr_encs if model.use_odcr else None)
                loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.sigmoid(logits).detach().float().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    valid = np.isfinite(preds_arr)
    if valid.sum() < len(preds_arr):
        print(f"  [warn] {(~valid).sum()} NaN/Inf predictions filtered before AUC")
    auc = roc_auc_score(labels_arr[valid], preds_arr[valid])
    return avg_loss, auc


def compute_group_auc(model, loader, device):
    """Val AUC broken down by recalibrated ODCR tertile group."""
    model.eval()
    records = []
    with torch.no_grad():
        for imgs, labels, odcr_encs, _ in loader:
            imgs      = imgs.to(device)
            odcr_encs = odcr_encs.to(device)
            logits    = model(imgs, odcr_encs if model.use_odcr else None)
            probs     = torch.sigmoid(logits).float().cpu().numpy()
            for label, prob, enc in zip(labels.numpy(), probs, odcr_encs.cpu().numpy()):
                odcr_deg = np.degrees(np.arctan2(enc[0], enc[1]))
                records.append({"label": label, "prob": prob,
                                "group": _odcr_group(odcr_deg)})

    df = pd.DataFrame(records)
    group_aucs = {}
    for g in ["light", "medium", "dark"]:
        sub = df[df["group"] == g]
        if len(sub) > 10 and sub["label"].nunique() == 2:
            group_aucs[g] = float(roc_auc_score(sub["label"], sub["prob"]))
    return group_aucs


# ── Per-variant training ──────────────────────────────────────────

def train_model(variant: str, args):
    cfg = VARIANTS[variant]
    use_odcr        = cfg["use_odcr"]
    use_dark_aug    = cfg["use_dark_aug"]
    oversample_dark = cfg["oversample_dark"]
    balanced        = cfg["balanced"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Variant : {variant}")
    print(f"Config  : odcr={use_odcr}  dark_aug={use_dark_aug}  "
          f"oversample_dark={oversample_dark}  balanced={balanced}")
    print(f"Device  : {device}")
    print(f"{'='*60}")

    train_ds = FundusDataset(
        SPLITS_DIR / "train.csv", ODCR_DIR / "train_odcr.csv",
        IMAGE_DIR, augment=True, use_dark_aug=use_dark_aug,
    )
    val_ds = FundusDataset(
        SPLITS_DIR / "val.csv", ODCR_DIR / "val_odcr.csv",
        IMAGE_DIR, augment=False, use_dark_aug=False,
    )

    pin     = device.type == "cuda"
    persist = args.workers > 0
    sampler = train_ds.make_sampler(oversample_dark=oversample_dark, balanced=balanced)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=persist, prefetch_factor=(2 if persist else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=persist, prefetch_factor=(2 if persist else None),
    )

    pos_weight_val = 1.0 if balanced else 2.0
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    )

    model = FundusNet(use_odcr=use_odcr).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr / 10},
        {"params": (p for n, p in model.named_parameters() if "backbone" not in n),
         "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    ckpt_dir    = CKPT_DIR / variant
    results_dir = RESULTS_DIR / variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_auc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_auc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"  Epoch {epoch:02d}/{args.epochs}  "
            f"train {train_loss:.4f}/{train_auc:.4f}  "
            f"val {val_loss:.4f}/{val_auc:.4f}  "
            f"({elapsed:.0f}s)"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"    ✓ best val AUC: {val_auc:.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_auc": train_auc,
            "val_loss": val_loss, "val_auc": val_auc,
        })
        with open(results_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    group_aucs   = compute_group_auc(model, val_loader, device)
    fairness_gap = (max(group_aucs.values()) - min(group_aucs.values())
                    if len(group_aucs) >= 2 else None)

    summary = {
        "variant": variant, **cfg,
        "best_val_auc": best_val_auc,
        "group_aucs": group_aucs,
        "fairness_gap": fairness_gap,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  {variant}: AUC={best_val_auc:.4f}  groups={group_aucs}  gap={fairness_gap}")
    return summary


# ── Main ──────────────────────────────────────────────────────────

def main(args):
    targets = list(VARIANTS.keys()) if args.model == "all" else [args.model]
    summaries = {}
    for v in targets:
        summaries[v] = train_model(v, args)

    if len(summaries) > 1:
        w = max(len(v) for v in summaries)
        print("\n" + "=" * 65)
        print("Ablation summary")
        print("=" * 65)
        print(f"  {'Variant':<{w}}  {'AUC':>6}  {'Light':>6}  {'Med':>6}  {'Dark':>6}  {'Gap':>6}")
        for v, s in summaries.items():
            g   = s.get("group_aucs", {})
            gap = s.get("fairness_gap")
            print(
                f"  {v:<{w}}  {s['best_val_auc']:>6.4f}  "
                f"{g.get('light',  float('nan')):>6.4f}  "
                f"{g.get('medium', float('nan')):>6.4f}  "
                f"{g.get('dark',   float('nan')):>6.4f}  "
                f"{gap if gap is not None else float('nan'):>6.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--workers",    type=int,   default=4)
    main(parser.parse_args())
