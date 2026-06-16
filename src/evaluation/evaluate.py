"""
evaluate.py — FundusNet ablation evaluation on the locked test set.

Loads checkpoints/<variant>/best.pt for all 5 trained variants, runs
inference on data/splits/test.csv, and produces per-variant per-tone-group
statistics mirroring the SkinToneNet NeurIPS 2026 methodology.

Statistical methodology:
  - Bootstrap CIs: 2,000 iterations, percentile method
  - Permutation tests: 5,000 iterations, 3 pre-specified tests
  - Cohen's h for effect sizes on proportions
  - Threshold sweep t=0.30–0.65

Outputs (all to results/evaluation/):
  results.json                 — full numerical results
  roc_curves.png               — ROC curves by tone group (Fig. 1 analog)
  specificity_sensitivity.png  — spec/sens asymmetry bar chart (Fig. 2 analog)
  ablation_auc.png             — ablation AUC bars with 95% CI (Fig. 3 analog)
  threshold_sweep.png          — sens/spec vs. threshold for 'full' variant

Usage:
  python src/evaluation/evaluate.py
  python src/evaluation/evaluate.py --threshold 0.45 --n_bootstrap 2000 --n_perm 5000
"""

import argparse
import json
import os
import site
import sys
from pathlib import Path

# Windows: ensure torch CUDA DLLs are findable before first torch import
if sys.platform == "win32":
    for _sp in site.getsitepackages():
        _lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_lib):
            os.add_dll_directory(_lib)
            break

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import shared definitions from train.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from train import (
    FundusDataset, FundusNet, VARIANTS, _odcr_group,
    IMAGE_DIR, SPLITS_DIR, ODCR_DIR, CKPT_DIR, ROOT,
)

EVAL_DIR = ROOT / "results" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["light", "medium", "dark"]
VARIANT_ORDER = ["baseline", "aug-only", "tone-only", "full", "balanced-baseline"]

# High-confidence ODCR thresholds (stricter than training -29°/+20°)
ODCR_DARK_HC  = -40.0
ODCR_LIGHT_HC =  30.0

GROUP_COLORS = {"light": "#E6A817", "medium": "#5C8A5E", "dark": "#3A3A7A"}
GROUP_LABELS = {"light": "Light fundus", "medium": "Medium fundus", "dark": "Dark fundus"}
VARIANT_COLORS = {
    "baseline":          "#636EFA",
    "aug-only":          "#EF553B",
    "tone-only":         "#00CC96",
    "full":              "#AB63FA",
    "balanced-baseline": "#FFA15A",
}


# ── Statistical functions ─────────────────────────────────────────

def bootstrap_auc(labels, probs, n=2000, rng=None):
    """Percentile bootstrap 95% CI for AUC (SkinToneNet §2.3 methodology)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_samples = len(labels)
    boot_aucs = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        bl, bp = labels[idx], probs[idx]
        if bl.sum() == 0 or bl.sum() == n_samples:
            continue
        boot_aucs.append(roc_auc_score(bl, bp))
    boot_aucs = np.array(boot_aucs)
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


def sensitivity_specificity(labels, probs, threshold=0.5):
    """Sensitivity and specificity at a given threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sens, spec


def cohen_h(p1, p2):
    """Cohen's h for comparing two proportions."""
    return float(2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1))) -
                 2 * np.arcsin(np.sqrt(np.clip(p2, 0, 1))))


def _permutation_gap(labels_a, probs_a, labels_b, probs_b, metric_fn, n, rng):
    """
    Two-sided permutation test for a metric gap between two groups.
    Returns (observed_gap, p_value).
    """
    obs_a = metric_fn(labels_a, probs_a)
    obs_b = metric_fn(labels_b, probs_b)
    obs_gap = abs(obs_a - obs_b)

    all_labels = np.concatenate([labels_a, labels_b])
    all_probs  = np.concatenate([probs_a,  probs_b])
    na = len(labels_a)

    null_gaps = []
    for _ in range(n):
        perm = rng.permutation(len(all_labels))
        la, pa = all_labels[perm[:na]], all_probs[perm[:na]]
        lb, pb = all_labels[perm[na:]], all_probs[perm[na:]]
        if la.sum() == 0 or la.sum() == len(la): continue
        if lb.sum() == 0 or lb.sum() == len(lb): continue
        ga = metric_fn(la, pa)
        gb = metric_fn(lb, pb)
        if np.isfinite(ga) and np.isfinite(gb):
            null_gaps.append(abs(ga - gb))

    null_gaps = np.array(null_gaps)
    p_val = float((null_gaps >= obs_gap).mean()) if len(null_gaps) > 0 else float("nan")
    return float(obs_gap), p_val


def _sens_fn(labels, probs, threshold):
    return sensitivity_specificity(labels, probs, threshold)[0]


def _spec_fn(labels, probs, threshold):
    return sensitivity_specificity(labels, probs, threshold)[1]


# ── Inference ─────────────────────────────────────────────────────

def run_inference(variant, device, batch_size=64, workers=4):
    """Load best.pt and run inference on the locked test set. Returns a DataFrame."""
    cfg = VARIANTS[variant]
    ckpt_path = CKPT_DIR / variant / "best.pt"

    model = FundusNet(use_odcr=cfg["use_odcr"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    test_ds = FundusDataset(
        SPLITS_DIR / "test.csv",
        ODCR_DIR   / "test_odcr.csv",
        IMAGE_DIR,
        augment=False,
        use_dark_aug=False,
    )

    persist = workers > 0
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persist,
        prefetch_factor=(2 if persist else None),
    )

    records = []
    with torch.no_grad():
        for imgs, labels, odcr_encs, image_ids in tqdm(loader, desc=f"  {variant:<20}", leave=False):
            imgs      = imgs.to(device)
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


# ── Per-variant analysis ──────────────────────────────────────────

def analyse_variant(df, variant, threshold, n_bootstrap, rng):
    labels_all = df["label"].values.astype(float)
    probs_all  = df["prob"].values

    auc_all = float(roc_auc_score(labels_all, probs_all))
    ci_lo, ci_hi = bootstrap_auc(labels_all, probs_all, n=n_bootstrap, rng=rng)
    sens_all, spec_all = sensitivity_specificity(labels_all, probs_all, threshold)

    result = {
        "variant": variant,
        "n": int(len(df)),
        "n_pos": int(labels_all.sum()),
        "prevalence": float(labels_all.mean()),
        "auc": auc_all,
        "auc_ci_lo": ci_lo,
        "auc_ci_hi": ci_hi,
        "sensitivity": float(sens_all),
        "specificity": float(spec_all),
        "groups": {},
        "hc_dark": None,
        "hc_light": None,
        "score_distribution": {},
        "threshold_sweep": {},
    }

    # Per-group stats
    for g in GROUPS:
        sub = df[df["group"] == g]
        if len(sub) < 10 or sub["label"].nunique() < 2:
            result["groups"][g] = None
            continue
        gl = sub["label"].values.astype(float)
        gp = sub["prob"].values
        g_auc = float(roc_auc_score(gl, gp))
        g_ci_lo, g_ci_hi = bootstrap_auc(gl, gp, n=n_bootstrap, rng=rng)
        sens, spec = sensitivity_specificity(gl, gp, threshold)
        p_benign = 1.0 - gl.mean()
        referral_burden = (1.0 - spec) * p_benign * 1000

        result["groups"][g] = {
            "n": int(len(sub)),
            "n_pos": int(gl.sum()),
            "prevalence": float(gl.mean()),
            "auc": g_auc,
            "auc_ci_lo": g_ci_lo,
            "auc_ci_hi": g_ci_hi,
            "sensitivity": float(sens),
            "specificity": float(spec),
            "referral_burden_per1000": float(referral_burden),
        }

    # High-confidence ODCR subset (stricter thresholds)
    for label_hc, mask in [
        ("hc_dark",  df["odcr_deg"] <= ODCR_DARK_HC),
        ("hc_light", df["odcr_deg"] >= ODCR_LIGHT_HC),
    ]:
        sub_hc = df[mask]
        if len(sub_hc) >= 10 and sub_hc["label"].nunique() == 2:
            gl = sub_hc["label"].values.astype(float)
            gp = sub_hc["prob"].values
            hc_auc = float(roc_auc_score(gl, gp))
            hc_lo, hc_hi = bootstrap_auc(gl, gp, n=n_bootstrap, rng=rng)
            sens_hc, spec_hc = sensitivity_specificity(gl, gp, threshold)
            result[label_hc] = {
                "n": int(len(sub_hc)),
                "n_pos": int(gl.sum()),
                "auc": hc_auc,
                "auc_ci_lo": hc_lo,
                "auc_ci_hi": hc_hi,
                "sensitivity": float(sens_hc),
                "specificity": float(spec_hc),
            }

    # Score distribution: mean predicted probability by (group × label)
    for g in GROUPS:
        sub = df[df["group"] == g]
        neg_probs = sub[sub["label"] == 0]["prob"]
        pos_probs = sub[sub["label"] == 1]["prob"]
        result["score_distribution"][g] = {
            "neg_mean": float(neg_probs.mean()) if len(neg_probs) > 0 else float("nan"),
            "neg_std":  float(neg_probs.std())  if len(neg_probs) > 0 else float("nan"),
            "pos_mean": float(pos_probs.mean()) if len(pos_probs) > 0 else float("nan"),
            "pos_std":  float(pos_probs.std())  if len(pos_probs) > 0 else float("nan"),
        }

    # Threshold sweep t=0.30–0.65
    for g in GROUPS:
        sub = df[df["group"] == g]
        if len(sub) < 10 or sub["label"].nunique() < 2:
            continue
        gl = sub["label"].values.astype(float)
        gp = sub["prob"].values
        sweep = []
        for t in np.arange(0.30, 0.651, 0.05):
            s, sp = sensitivity_specificity(gl, gp, t)
            sweep.append({
                "threshold":   round(float(t), 2),
                "sensitivity": float(s),
                "specificity": float(sp),
            })
        result["threshold_sweep"][g] = sweep

    return result


def run_permutation_tests(preds_by_variant, results, threshold, n_perm, rng):
    """
    Three pre-specified permutation tests (SkinToneNet §2.4 methodology):
      (i)   Dark specificity gap at t (dark vs. medium), per variant
      (ii)  Dark sensitivity gap (dark vs. medium), per variant
      (iii) AUC gap: full vs. baseline in dark subgroup only
    """
    perm_results = {}

    for variant in VARIANT_ORDER:
        df = preds_by_variant[variant]
        dark_df  = df[df["group"] == "dark"]
        light_df = df[df["group"] == "light"]

        if (len(dark_df) < 10 or dark_df["label"].nunique() < 2 or
                len(light_df) < 10 or light_df["label"].nunique() < 2):
            perm_results[variant] = None
            continue

        dl = dark_df["label"].values.astype(float)
        dp = dark_df["prob"].values
        ll = light_df["label"].values.astype(float)
        lp = light_df["prob"].values

        # (i) specificity gap
        spec_gap, spec_p = _permutation_gap(
            dl, dp, ll, lp,
            lambda la, pr: _spec_fn(la, pr, threshold), n_perm, rng,
        )
        # (ii) sensitivity gap
        sens_gap, sens_p = _permutation_gap(
            dl, dp, ll, lp,
            lambda la, pr: _sens_fn(la, pr, threshold), n_perm, rng,
        )

        dark_spec  = _spec_fn(dl, dp, threshold)
        light_spec = _spec_fn(ll, lp, threshold)
        dark_sens  = _sens_fn(dl, dp, threshold)
        light_sens = _sens_fn(ll, lp, threshold)

        perm_results[variant] = {
            "dark_spec":    float(dark_spec),
            "light_spec":   float(light_spec),
            "spec_gap":     float(spec_gap),
            "spec_gap_p":   float(spec_p),
            "spec_cohen_h": cohen_h(dark_spec, light_spec),
            "dark_sens":    float(dark_sens),
            "light_sens":   float(light_sens),
            "sens_gap":     float(sens_gap),
            "sens_gap_p":   float(sens_p),
            "sens_cohen_h": cohen_h(dark_sens, light_sens),
        }

    # (iii) AUC gap: full vs. baseline, dark subgroup
    full_dark     = preds_by_variant["full"][preds_by_variant["full"]["group"] == "dark"]
    baseline_dark = preds_by_variant["baseline"][preds_by_variant["baseline"]["group"] == "dark"]

    if (len(full_dark) >= 10 and full_dark["label"].nunique() == 2 and
            len(baseline_dark) >= 10 and baseline_dark["label"].nunique() == 2):
        fl = full_dark["label"].values.astype(float)
        fp = full_dark["prob"].values
        bl = baseline_dark["label"].values.astype(float)
        bpv = baseline_dark["prob"].values

        auc_gap, auc_p = _permutation_gap(
            fl, fp, bl, bpv, roc_auc_score, n_perm, rng,
        )
        perm_results["_auc_gap_full_vs_baseline_dark"] = {
            "full_dark_auc":     float(roc_auc_score(fl, fp)),
            "baseline_dark_auc": float(roc_auc_score(bl, bpv)),
            "auc_gap":           float(auc_gap),
            "auc_gap_p":         float(auc_p),
        }

    return perm_results


# ── Plotting ──────────────────────────────────────────────────────

def plot_roc_curves(preds_by_variant, results, out_path):
    """Figure 1 analog: ROC curves by tone group, one panel per variant."""
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
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=8)

    fig.suptitle("ROC Curves by Fundus Tone Group — Locked Test Set",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_specificity_sensitivity(results, threshold, out_path):
    """Figure 2 analog: sensitivity and specificity by group and variant."""
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

    fig.suptitle("Sensitivity & Specificity by Fundus Tone Group — Locked Test Set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _safe_val(v):
    return v if (v is not None and np.isfinite(v)) else float("nan")


def plot_ablation_auc(results, out_path):
    """Figure 3 analog: ablation AUC bars with 95% bootstrap CI, overall + per group."""
    n_panels = 1 + len(GROUPS)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5), sharey=False)

    # Overall AUC panel
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

    # Per-group panels
    for ax, g in zip(axes[1:], GROUPS):
        g_aucs  = []
        g_lo    = []
        g_hi    = []
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

    fig.suptitle("Ablation AUC — Locked Test Set (95% Bootstrap CI)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_threshold_sweep(preds_by_variant, out_path):
    """Threshold sweep: sensitivity & specificity vs. threshold for the 'full' variant."""
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
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0.27, 0.68)
        ax.tick_params(labelsize=9)

    fig.suptitle("'Full' Variant: Sensitivity & Specificity vs. Threshold by Tone Group",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── JSON serialiser ───────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    raise TypeError(f"Not serialisable: {type(obj)}")


# ── Main ──────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Test set: {SPLITS_DIR / 'test.csv'}")
    print(f"Images  : {IMAGE_DIR}")
    rng = np.random.default_rng(args.seed)

    # ── Step 1: inference ────────────────────────────────────────
    print(f"\n[1/4] Running inference on test set …")
    preds_by_variant = {}
    for variant in VARIANT_ORDER:
        print(f"  {variant}")
        preds_by_variant[variant] = run_inference(
            variant, device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        df = preds_by_variant[variant]
        group_counts = df["group"].value_counts().to_dict()
        print(f"         n={len(df)}  groups={group_counts}  "
              f"pos={df['label'].sum()}  "
              f"overall_auc={roc_auc_score(df['label'].values, df['prob'].values):.4f}")

    # ── Step 2: per-variant statistics ───────────────────────────
    print(f"\n[2/4] Computing per-variant statistics (bootstrap n={args.n_bootstrap}) …")
    results = {}
    for variant in VARIANT_ORDER:
        print(f"  {variant}")
        results[variant] = analyse_variant(
            preds_by_variant[variant], variant,
            threshold=args.threshold,
            n_bootstrap=args.n_bootstrap,
            rng=rng,
        )

    # ── Step 3: permutation tests ─────────────────────────────────
    print(f"\n[3/4] Running pre-specified permutation tests (n={args.n_perm}) …")
    perm_results = run_permutation_tests(
        preds_by_variant, results,
        threshold=args.threshold,
        n_perm=args.n_perm,
        rng=rng,
    )
    results["_permutation_tests"] = perm_results

    # ── Step 4: save outputs ──────────────────────────────────────
    print(f"\n[4/4] Saving results and plots …")
    json_path = EVAL_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"  Saved: {json_path.name}")

    plot_roc_curves(preds_by_variant, results,    EVAL_DIR / "roc_curves.png")
    plot_specificity_sensitivity(results, args.threshold, EVAL_DIR / "specificity_sensitivity.png")
    plot_ablation_auc(results,                            EVAL_DIR / "ablation_auc.png")
    plot_threshold_sweep(preds_by_variant,                EVAL_DIR / "threshold_sweep.png")

    # ── Console summary ───────────────────────────────────────────
    w = max(len(v) for v in VARIANT_ORDER)
    print("\n" + "=" * 76)
    print("Test-set AUC summary")
    print("=" * 76)
    print(f"  {'Variant':<{w}}  {'AUC':>6}  {'95% CI':^13}  "
          f"{'Light':>6}  {'Med':>6}  {'Dark':>6}  {'Gap':>6}")
    for v in VARIANT_ORDER:
        r  = results[v]
        g  = r["groups"]
        lo = r["auc_ci_lo"]
        hi = r["auc_ci_hi"]
        light_auc = g["light"]["auc"] if g.get("light") else float("nan")
        med_auc   = g["medium"]["auc"] if g.get("medium") else float("nan")
        dark_auc  = g["dark"]["auc"] if g.get("dark") else float("nan")
        defined   = [x for x in [light_auc, med_auc, dark_auc] if np.isfinite(x)]
        gap = max(defined) - min(defined) if len(defined) >= 2 else float("nan")
        print(
            f"  {v:<{w}}  {r['auc']:>6.4f}  [{lo:.4f},{hi:.4f}]"
            f"  {light_auc:>6.4f}  {med_auc:>6.4f}  {dark_auc:>6.4f}  {gap:>6.4f}"
        )

    print(f"\nSensitivity / Specificity at t={args.threshold:.2f}")
    print(f"  {'Variant':<{w}}  {'Sens(D)':>7}  {'Spec(D)':>7}  {'Sens(L)':>7}  {'Spec(L)':>7}")
    for v in VARIANT_ORDER:
        g = results[v]["groups"]
        ds = g["dark"]["sensitivity"]  if g.get("dark")  else float("nan")
        dsp = g["dark"]["specificity"] if g.get("dark")  else float("nan")
        ls = g["light"]["sensitivity"] if g.get("light") else float("nan")
        lsp = g["light"]["specificity"] if g.get("light") else float("nan")
        print(f"  {v:<{w}}  {ds:>7.4f}  {dsp:>7.4f}  {ls:>7.4f}  {lsp:>7.4f}")

    print("\nPre-specified permutation tests (dark vs. light):")
    for v in VARIANT_ORDER:
        pt = perm_results.get(v)
        if pt is None:
            print(f"  {v:<{w}}  —")
            continue
        print(
            f"  {v:<{w}}"
            f"  spec_gap={pt['spec_gap']:.4f} p={pt['spec_gap_p']:.4f} h={pt['spec_cohen_h']:+.3f}"
            f"  sens_gap={pt['sens_gap']:.4f} p={pt['sens_gap_p']:.4f} h={pt['sens_cohen_h']:+.3f}"
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
    parser = argparse.ArgumentParser(description="FundusNet ablation evaluation")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--n_bootstrap", type=int,   default=2000,
                        help="Bootstrap iterations for AUC CIs (default: 2000)")
    parser.add_argument("--n_perm",      type=int,   default=5000,
                        help="Permutation test iterations (default: 5000)")
    parser.add_argument("--batch_size",  type=int,   default=64,
                        help="Inference batch size (default: 64)")
    parser.add_argument("--workers",     type=int,   default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="RNG seed for reproducibility (default: 42)")
    main(parser.parse_args())
