"""
evaluate_aptos.py — FundusNet zero-shot external validation on APTOS 2019.

Loads checkpoints/<variant>/best.pt for all 5 trained variants (no fine-tuning),
runs inference on data/aptos/train_images/ using APTOS 2019 train.csv labels,
and produces per-variant per-tone-group statistics mirroring the EyePACS evaluation.

APTOS grades binarised: 0-1 → non-referable (0), 2-4 → referable (1).
ODCR computed on-the-fly using compute_odcr() from src/features/odcr.py,
cached to data/aptos/aptos_odcr.csv for subsequent runs.
Groups assigned using ODCR_DARK_THRESH=-29.0 / ODCR_LIGHT_THRESH=20.0 (training tertiles).

Statistical methodology (mirrors src/evaluation/evaluate.py exactly):
  - Bootstrap CIs: 2,000 iterations, percentile method
  - Permutation tests: 5,000 iterations, 3 pre-specified tests
  - Cohen's h for effect sizes on proportions
  - Score distribution by group × label (SNR mechanism check)

Outputs (all to results/evaluation/):
  aptos_results.json                — full numerical results
  aptos_roc_curves.png              — ROC curves by tone group
  aptos_specificity_sensitivity.png — spec/sens asymmetry bar chart
  aptos_ablation_auc.png            — ablation AUC bars with 95% CI
  aptos_threshold_sweep.png         — sens/spec vs. threshold for 'full' variant

Usage:
  cd fundusnet
  python src/evaluation/evaluate_aptos.py
  python src/evaluation/evaluate_aptos.py --threshold 0.45 --n_bootstrap 2000 --n_perm 5000
"""

import argparse
import json
import os
import site
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ── Shared imports from sibling modules ───────────────────────────
_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "models"))
sys.path.insert(0, str(_SRC / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import (
    FundusNet, VARIANTS, _odcr_group,
    CKPT_DIR, ROOT, ODCR_DARK_THRESH, ODCR_LIGHT_THRESH,
)
from odcr import process_single_image
from evaluate import (
    bootstrap_auc, sensitivity_specificity, cohen_h,
    _permutation_gap, _sens_fn, _spec_fn, _json_default,
    analyse_variant, run_permutation_tests,
    GROUPS, VARIANT_ORDER, ODCR_DARK_HC, ODCR_LIGHT_HC,
    GROUP_COLORS, GROUP_LABELS, VARIANT_COLORS,
)

# ── APTOS-specific paths ───────────────────────────────────────────
APTOS_DIR        = ROOT / "data" / "aptos"
APTOS_IMAGES     = APTOS_DIR / "train_images"
APTOS_CSV        = APTOS_DIR / "train.csv"
APTOS_ODCR_CACHE = APTOS_DIR / "aptos_odcr.csv"
EVAL_DIR         = ROOT / "results" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading and ODCR computation ────────────────────────────

def load_or_compute_aptos_odcr(n_workers: int = 8) -> pd.DataFrame:
    """
    Load APTOS 2019 train.csv, binarise grades (0-1 → 0, 2-4 → 1),
    and compute (or load cached) ODCR for each image.

    Caches ODCR to data/aptos/aptos_odcr.csv so repeated runs skip recomputation.
    Groups are assigned using the same recalibrated training tertile thresholds
    (ODCR_DARK_THRESH=-29.0°, ODCR_LIGHT_THRESH=20.0°) as the EyePACS evaluation.

    Returns a merged DataFrame with columns:
      image, label, grade, odcr, odcr_sin, odcr_cos, odcr_norm, group
    """
    df_labels = pd.read_csv(APTOS_CSV)
    df_labels = df_labels.rename(columns={"id_code": "image", "diagnosis": "grade"})
    df_labels["label"] = (df_labels["grade"] >= 2).astype(int)

    if APTOS_ODCR_CACHE.exists():
        print(f"  Loading cached ODCR from {APTOS_ODCR_CACHE.name}")
        df_odcr = pd.read_csv(APTOS_ODCR_CACHE)
    else:
        print(f"  Computing ODCR for {len(df_labels)} APTOS images "
              f"(workers={n_workers}) …")
        image_paths = [APTOS_IMAGES / f"{iid}.png" for iid in df_labels["image"]]
        raw = []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(process_single_image, p): p for p in image_paths}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="  ODCR", unit="img"):
                raw.append(fut.result())
        df_odcr = pd.DataFrame(raw)
        df_odcr.to_csv(APTOS_ODCR_CACHE, index=False)
        print(f"  Cached to {APTOS_ODCR_CACHE.name}")

    df = df_labels.merge(
        df_odcr[["image", "odcr", "odcr_sin", "odcr_cos", "odcr_norm"]],
        on="image", how="left",
    )
    # Fill missing encodings with population medians (mirrors FundusDataset)
    for col in ["odcr_sin", "odcr_cos", "odcr_norm"]:
        df[col] = df[col].fillna(df[col].median())

    # Assign recalibrated tone group using training tertile thresholds
    df["group"] = df["odcr"].apply(
        lambda x: _odcr_group(x) if pd.notna(x) else "medium"
    )

    n_failed = df["odcr"].isna().sum()
    n_ok = len(df) - n_failed
    print(f"  ODCR: {n_ok} succeeded, {n_failed} failed "
          f"(→ assigned to 'medium' group)")
    print(f"  Group distribution: {df['group'].value_counts().to_dict()}")
    print(f"  Prevalence: {df['label'].mean():.4f} "
          f"({df['label'].sum()} / {len(df)} referable)")
    return df


# ── Dataset ───────────────────────────────────────────────────────

class AptosDataset(Dataset):
    """APTOS 2019 PNG fundus images for zero-shot FundusNet inference."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        size = 260  # EfficientNet-B2 canonical native resolution (matches training)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = APTOS_IMAGES / f"{row['image']}.png"
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            img_bgr = np.zeros((260, 260, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = self.transform(transforms.ToPILImage()(img_rgb))
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        odcr_enc = torch.tensor(
            [float(row["odcr_sin"]),
             float(row["odcr_cos"]),
             float(row["odcr_norm"])],
            dtype=torch.float32,
        )
        return img, label, odcr_enc, str(row["image"])


# ── Zero-shot inference ───────────────────────────────────────────

def run_inference(variant: str, df_aptos: pd.DataFrame,
                  device, batch_size: int = 64, workers: int = 4) -> pd.DataFrame:
    """
    Load best.pt for one variant and run zero-shot inference on APTOS.
    Mirrors evaluate.py:run_inference exactly — no fine-tuning.

    Returns a DataFrame with columns: image, label, prob, odcr_deg, group.
    """
    cfg = VARIANTS[variant]
    ckpt_path = CKPT_DIR / variant / "best.pt"

    model = FundusNet(use_odcr=cfg["use_odcr"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    ds = AptosDataset(df_aptos)
    persist = workers > 0
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persist,
        prefetch_factor=(2 if persist else None),
    )

    records = []
    with torch.no_grad():
        for imgs, labels, odcr_encs, image_ids in tqdm(
            loader, desc=f"  {variant:<20}", leave=False
        ):
            imgs = imgs.to(device)
            odcr_encs = odcr_encs.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits = model(imgs, odcr_encs if model.use_odcr else None)
            probs = torch.sigmoid(logits).float().cpu().numpy()

            for image_id, label, prob, enc in zip(
                image_ids, labels.numpy(), probs, odcr_encs.cpu().numpy()
            ):
                odcr_deg = float(np.degrees(np.arctan2(enc[0], enc[1])))
                records.append({
                    "image":    image_id,
                    "label":    int(label),
                    "prob":     float(prob),
                    "odcr_deg": odcr_deg,
                    "group":    _odcr_group(odcr_deg),
                })

    df = pd.DataFrame(records)
    n_nan = (~np.isfinite(df["prob"])).sum()
    if n_nan > 0:
        print(f"  [warn] {n_nan} NaN/Inf probabilities in {variant} — filtering")
        df = df[np.isfinite(df["prob"])]
    return df


# ── Plotting (APTOS-specific titles, same structure as evaluate.py) ──

def plot_roc_curves(preds_by_variant, results, out_path):
    n = len(VARIANT_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.5), sharey=True)
    for ax, variant in zip(axes, VARIANT_ORDER):
        df = preds_by_variant[variant]
        for g in GROUPS:
            sub = df[df["group"] == g]
            if len(sub) < 10 or sub["label"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(sub["label"].values, sub["prob"].values)
            g_auc = results[variant]["groups"][g]["auc"]
            ax.plot(fpr, tpr, color=GROUP_COLORS[g], lw=1.8,
                    label=f"{GROUP_LABELS[g]}\n(AUC={g_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.45)
        ax.set_title(variant, fontsize=10, fontweight="bold")
        ax.set_xlabel("1 – Specificity", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Sensitivity", fontsize=9)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)
    fig.suptitle(
        "ROC Curves by Fundus Tone Group — APTOS 2019 External Validation",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_specificity_sensitivity(results, threshold, out_path):
    x = np.arange(len(VARIANT_ORDER))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (metric_key, ylabel) in zip(
        axes, [("sensitivity", "Sensitivity"), ("specificity", "Specificity")]
    ):
        for gi, g in enumerate(GROUPS):
            vals = []
            for v in VARIANT_ORDER:
                grp = results[v]["groups"].get(g)
                vals.append(grp[metric_key] if grp else float("nan"))
            offset = (gi - 1) * width
            ax.bar(x + offset, vals, width, color=GROUP_COLORS[g],
                   label=GROUP_LABELS[g], alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANT_ORDER, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} at t={threshold:.2f}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower right")
        ax.axhline(0.9, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    fig.suptitle(
        "Sensitivity & Specificity by Fundus Tone Group — APTOS 2019 External Validation",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_ablation_auc(results, out_path):
    n_panels = 1 + len(GROUPS)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5), sharey=False)

    ax = axes[0]
    aucs    = [results[v]["auc"] for v in VARIANT_ORDER]
    yerr_lo = [results[v]["auc"] - results[v]["auc_ci_lo"] for v in VARIANT_ORDER]
    yerr_hi = [results[v]["auc_ci_hi"] - results[v]["auc"] for v in VARIANT_ORDER]
    colors  = [VARIANT_COLORS[v] for v in VARIANT_ORDER]
    ax.bar(range(len(VARIANT_ORDER)), aucs, color=colors, alpha=0.85, edgecolor="white")
    ax.errorbar(range(len(VARIANT_ORDER)), aucs,
                yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=5, lw=1.5)
    ax.set_title("Overall AUC", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(VARIANT_ORDER)))
    ax.set_xticklabels(VARIANT_ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_ylim(0.5, 1.0)

    for ax, g in zip(axes[1:], GROUPS):
        g_aucs, g_lo, g_hi = [], [], []
        for v in VARIANT_ORDER:
            grp = results[v]["groups"].get(g)
            if grp:
                g_aucs.append(grp["auc"])
                g_lo.append(grp["auc"] - grp["auc_ci_lo"])
                g_hi.append(grp["auc_ci_hi"] - grp["auc"])
            else:
                g_aucs.append(float("nan"))
                g_lo.append(0.0)
                g_hi.append(0.0)
        ax.bar(range(len(VARIANT_ORDER)), g_aucs,
               color=GROUP_COLORS[g], alpha=0.85, edgecolor="white")
        ax.errorbar(range(len(VARIANT_ORDER)), g_aucs,
                    yerr=[g_lo, g_hi], fmt="none", color="black", capsize=5, lw=1.5)
        ax.set_title(f"{GROUP_LABELS[g]} AUC", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(VARIANT_ORDER)))
        ax.set_xticklabels(VARIANT_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.5, 1.0)

    fig.suptitle(
        "Ablation AUC — APTOS 2019 External Validation (95% Bootstrap CI)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_threshold_sweep(preds_by_variant, out_path):
    df = preds_by_variant["full"]
    thresholds = np.arange(0.30, 0.66, 0.05)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, g in zip(axes, GROUPS):
        sub = df[df["group"] == g]
        if len(sub) < 10 or sub["label"].nunique() < 2:
            ax.set_title(f"{GROUP_LABELS[g]}\n(insufficient data)", fontsize=10)
            continue
        gl = sub["label"].values.astype(float)
        gp = sub["prob"].values
        sens_vals, spec_vals = [], []
        for t in thresholds:
            s, sp = sensitivity_specificity(gl, gp, t)
            sens_vals.append(s)
            spec_vals.append(sp)
        ax.plot(thresholds, sens_vals, "o-", color="#D62728", label="Sensitivity", lw=2)
        ax.plot(thresholds, spec_vals, "s-", color="#1F77B4", label="Specificity", lw=2)
        ax.axvline(0.5, color="gray", linestyle="--", lw=0.8, alpha=0.6)
        ax.set_title(GROUP_LABELS[g], fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Metric value", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05); ax.set_xlim(0.27, 0.68)
        ax.tick_params(labelsize=9)
    fig.suptitle(
        "'Full' Variant: Sensitivity & Specificity vs. Threshold — APTOS 2019",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"APTOS   : {APTOS_CSV}")
    print(f"Images  : {APTOS_IMAGES}")
    print(f"Output  : {EVAL_DIR / 'aptos_results.json'}")
    print(f"Thresholds used for grouping: dark≤{ODCR_DARK_THRESH}°, light≥{ODCR_LIGHT_THRESH}°")

    # ── Step 1: ODCR computation ─────────────────────────────────
    print(f"\n[1/5] Loading APTOS data and computing ODCR …")
    df_aptos = load_or_compute_aptos_odcr(n_workers=args.workers)

    # ── Step 2: zero-shot inference ──────────────────────────────
    print(f"\n[2/5] Running zero-shot inference on APTOS ({len(df_aptos)} images) …")
    preds_by_variant = {}
    for variant in VARIANT_ORDER:
        print(f"  {variant}")
        preds_by_variant[variant] = run_inference(
            variant, df_aptos, device,
            batch_size=args.batch_size, workers=args.workers,
        )
        df = preds_by_variant[variant]
        group_counts = df["group"].value_counts().to_dict()
        print(f"         n={len(df)}  groups={group_counts}  "
              f"pos={df['label'].sum()}  "
              f"overall_auc={roc_auc_score(df['label'].values, df['prob'].values):.4f}")

    # ── Step 3: per-variant statistics ───────────────────────────
    print(f"\n[3/5] Computing per-variant statistics "
          f"(bootstrap n={args.n_bootstrap}) …")
    results = {}
    for variant in VARIANT_ORDER:
        print(f"  {variant}")
        results[variant] = analyse_variant(
            preds_by_variant[variant], variant,
            threshold=args.threshold,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )

    # ── Step 4: permutation tests ─────────────────────────────────
    print(f"\n[4/5] Running pre-specified permutation tests "
          f"(n={args.n_perm}) …")
    perm_results = run_permutation_tests(
        preds_by_variant, results,
        threshold=args.threshold,
        n_perm=args.n_perm,
        rng=rng,
    )
    results["_permutation_tests"] = perm_results
    results["_metadata"] = {
        "dataset":       "APTOS 2019",
        "n_images":      int(len(df_aptos)),
        "n_referable":   int(df_aptos["label"].sum()),
        "prevalence":    float(df_aptos["label"].mean()),
        "binarisation":  "grades 0-1 → 0 (non-referable), 2-4 → 1 (referable)",
        "evaluation":    "zero-shot (no fine-tuning)",
        "odcr_dark_thresh":  ODCR_DARK_THRESH,
        "odcr_light_thresh": ODCR_LIGHT_THRESH,
        "odcr_dark_hc":      ODCR_DARK_HC,
        "odcr_light_hc":     ODCR_LIGHT_HC,
        "threshold":     args.threshold,
        "n_bootstrap":   args.n_bootstrap,
        "n_perm":        args.n_perm,
        "seed":          args.seed,
    }

    # ── Step 5: save outputs ──────────────────────────────────────
    print(f"\n[5/5] Saving results and plots …")
    json_path = EVAL_DIR / "aptos_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"  Saved: {json_path.name}")

    plot_roc_curves(preds_by_variant, results,
                    EVAL_DIR / "aptos_roc_curves.png")
    plot_specificity_sensitivity(results, args.threshold,
                                 EVAL_DIR / "aptos_specificity_sensitivity.png")
    plot_ablation_auc(results,
                      EVAL_DIR / "aptos_ablation_auc.png")
    plot_threshold_sweep(preds_by_variant,
                         EVAL_DIR / "aptos_threshold_sweep.png")

    # ── Console summary ───────────────────────────────────────────
    w = max(len(v) for v in VARIANT_ORDER)
    print("\n" + "=" * 76)
    print("APTOS 2019 External Validation — AUC Summary (zero-shot transfer)")
    print("=" * 76)
    print(f"  {'Variant':<{w}}  {'AUC':>6}  {'95% CI':^13}  "
          f"{'Light':>6}  {'Med':>6}  {'Dark':>6}  {'Gap':>6}")
    for v in VARIANT_ORDER:
        r = results[v]
        g = r["groups"]
        lo, hi = r["auc_ci_lo"], r["auc_ci_hi"]
        light_auc = g["light"]["auc"]  if g.get("light")  else float("nan")
        med_auc   = g["medium"]["auc"] if g.get("medium") else float("nan")
        dark_auc  = g["dark"]["auc"]   if g.get("dark")   else float("nan")
        defined   = [x for x in [light_auc, med_auc, dark_auc] if np.isfinite(x)]
        gap = max(defined) - min(defined) if len(defined) >= 2 else float("nan")
        print(
            f"  {v:<{w}}  {r['auc']:>6.4f}  [{lo:.4f},{hi:.4f}]"
            f"  {light_auc:>6.4f}  {med_auc:>6.4f}  {dark_auc:>6.4f}  {gap:>6.4f}"
        )

    print(f"\nSensitivity / Specificity at t={args.threshold:.2f}")
    print(f"  {'Variant':<{w}}  {'Sens(D)':>7}  {'Spec(D)':>7}  "
          f"{'Sens(L)':>7}  {'Spec(L)':>7}")
    for v in VARIANT_ORDER:
        g   = results[v]["groups"]
        ds  = g["dark"]["sensitivity"]  if g.get("dark")  else float("nan")
        dsp = g["dark"]["specificity"]  if g.get("dark")  else float("nan")
        ls  = g["light"]["sensitivity"] if g.get("light") else float("nan")
        lsp = g["light"]["specificity"] if g.get("light") else float("nan")
        print(f"  {v:<{w}}  {ds:>7.4f}  {dsp:>7.4f}  {ls:>7.4f}  {lsp:>7.4f}")

    print("\nScore distribution — SNR mechanism check "
          "(mean P̂(referable) by group × true label):")
    print(f"  {'Variant':<{w}}  {'Light–':>7}  {'Light+':>7}  "
          f"{'Dark–':>7}  {'Dark+':>7}")
    for v in VARIANT_ORDER:
        sd  = results[v]["score_distribution"]
        ln  = sd.get("light", {}).get("neg_mean", float("nan"))
        lp  = sd.get("light", {}).get("pos_mean", float("nan"))
        dn  = sd.get("dark",  {}).get("neg_mean", float("nan"))
        dp  = sd.get("dark",  {}).get("pos_mean", float("nan"))
        print(f"  {v:<{w}}  {ln:>7.4f}  {lp:>7.4f}  {dn:>7.4f}  {dp:>7.4f}")

    print("\nPre-specified permutation tests (dark vs. light):")
    for v in VARIANT_ORDER:
        pt = perm_results.get(v)
        if pt is None:
            print(f"  {v:<{w}}  —")
            continue
        print(
            f"  {v:<{w}}"
            f"  spec_gap={pt['spec_gap']:.4f} p={pt['spec_gap_p']:.4f}"
            f" h={pt['spec_cohen_h']:+.3f}"
            f"  sens_gap={pt['sens_gap']:.4f} p={pt['sens_gap_p']:.4f}"
            f" h={pt['sens_cohen_h']:+.3f}"
        )

    auc_gap_test = perm_results.get("_auc_gap_full_vs_baseline_dark")
    if auc_gap_test:
        print(
            f"\nAUC gap test (full vs. baseline, dark subgroup):\n"
            f"  full={auc_gap_test['full_dark_auc']:.4f}  "
            f"baseline={auc_gap_test['baseline_dark_auc']:.4f}  "
            f"gap={auc_gap_test['auc_gap']:.4f}  p={auc_gap_test['auc_gap_p']:.4f}"
        )

    print(f"\nAll outputs saved to: {EVAL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FundusNet zero-shot external validation on APTOS 2019"
    )
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--n_bootstrap", type=int,   default=2000,
                        help="Bootstrap iterations for AUC CIs (default: 2000)")
    parser.add_argument("--n_perm",      type=int,   default=5000,
                        help="Permutation test iterations (default: 5000)")
    parser.add_argument("--batch_size",  type=int,   default=64,
                        help="Inference batch size (default: 64)")
    parser.add_argument("--workers",     type=int,   default=4,
                        help="DataLoader workers; use 0 if multiprocessing "
                             "issues on Windows (default: 4)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="RNG seed for reproducibility (default: 42)")
    main(parser.parse_args())
