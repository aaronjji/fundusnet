"""
snr.py — Vessel-background SNR analysis by ODCR tertile.
This is the most important pre-training experiment.

Mirrors SkinToneNet Section 3.3 SNR Framework:
    SNR(ODCR) = ||μ_lesion - μ_background|| / σ_background

In fundus images:
    - "lesion"     = microaneurysm / haemorrhage regions (dark spots in green channel)
    - "background" = peripapillary tissue (same region as ODCR computation)

The SNR framework predicts a monotonic decrease in SNR with decreasing ODCR
(increasing choroidal pigmentation). Document the SNR ratio between light
and dark tertiles — your 5.2× equivalent from SkinToneNet.

If SNR ratio < 1.5×: weaker effect than dermoscopy — report honestly
If SNR ratio > 3×:   strong support for mechanistic framework
Either way — this is a finding.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from scipy import stats


# ── SNR computation ───────────────────────────────────────────────

def detect_lesion_regions(img_rgb: np.ndarray) -> np.ndarray:
    """
    Detect candidate microaneurysm / haemorrhage regions.
    These appear as dark red spots in the green channel.

    Returns a binary mask of candidate lesion pixels.
    """
    # Green channel: vessels and lesions are dark, background is bright
    green = img_rgb[:, :, 1].astype(np.float32)

    # CLAHE equalisation for consistent contrast across pigmentation levels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_eq = clahe.apply(img_rgb[:, :, 1])

    # Adaptive thresholding: dark regions relative to local background
    # Lesions are smaller than vessels — use small block size
    lesion_binary = cv2.adaptiveThreshold(
        green_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=8
    )

    # Morphological cleanup — remove vessels (elongated) keep spots (round)
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lesion_binary = cv2.morphologyEx(lesion_binary, cv2.MORPH_OPEN,  kernel_open)
    lesion_binary = cv2.morphologyEx(lesion_binary, cv2.MORPH_CLOSE, kernel_close)

    return lesion_binary.astype(bool)


def extract_background_region(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract background pixels (non-vessel, non-lesion retinal tissue).
    Uses the green channel — background is the mid-brightness tissue.
    """
    green = img_rgb[:, :, 1].astype(np.float32)
    h, w = green.shape

    # Exclude image edges (vignetting artifact common in fundus cameras)
    mask = np.zeros((h, w), dtype=bool)
    margin = int(min(h, w) * 0.08)
    mask[margin:h-margin, margin:w-margin] = True

    # Exclude very dark pixels (vessels, lesions) and very bright (disc, artifacts)
    p10 = np.percentile(green[mask], 10)
    p90 = np.percentile(green[mask], 90)
    background_mask = mask & (green > p10 * 1.3) & (green < p90 * 0.95)

    return green[background_mask]


def compute_snr(img_rgb: np.ndarray) -> dict:
    """
    Compute vessel-background SNR for a single fundus image.

    SNR = |μ_lesion - μ_background| / σ_background

    Returns dict with snr, mu_lesion, mu_background, sigma_background.
    Returns None values if computation fails.
    """
    try:
        img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
        green   = img_rgb[:, :, 1].astype(np.float32)

        lesion_mask = detect_lesion_regions(img_rgb)
        bg_pixels   = extract_background_region(img_rgb)

        # Need enough lesion pixels to be meaningful
        n_lesion = lesion_mask.sum()
        n_bg     = len(bg_pixels)

        if n_lesion < 30 or n_bg < 100:
            return {"snr": None, "mu_lesion": None, "mu_background": None,
                    "sigma_background": None, "n_lesion": n_lesion}

        mu_lesion = float(np.mean(green[lesion_mask]))
        mu_bg     = float(np.mean(bg_pixels))
        sigma_bg  = float(np.std(bg_pixels))

        if sigma_bg < 1e-6:
            return {"snr": None, "mu_lesion": mu_lesion, "mu_background": mu_bg,
                    "sigma_background": sigma_bg, "n_lesion": n_lesion}

        snr = float(abs(mu_lesion - mu_bg) / sigma_bg)

        return {
            "snr": snr,
            "mu_lesion": mu_lesion,
            "mu_background": mu_bg,
            "sigma_background": sigma_bg,
            "n_lesion": int(n_lesion)
        }

    except Exception as e:
        return {"snr": None, "mu_lesion": None, "mu_background": None,
                "sigma_background": None, "error": str(e)}


# ── Batch SNR computation ─────────────────────────────────────────

def compute_snr_for_split(
    image_dir: Path,
    odcr_csv: Path,
    output_path: Path,
    sample_n: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute SNR for a stratified sample of images across ODCR groups.
    Sampling is stratified by tone_group to ensure enough images per group.
    """
    df = pd.read_csv(odcr_csv)
    df = df[df["odcr"].notna() & (df["tone_group"] != "unknown")]

    # Stratified sample — equal N per tone group
    n_per_group = sample_n // 3
    df_sample = pd.concat([
        grp.sample(min(len(grp), n_per_group), random_state=seed)
        for _, grp in df.groupby("tone_group", group_keys=False)
    ], ignore_index=True)
    print(f"\nSNR analysis sample: {len(df_sample):,} images")
    print(df_sample["tone_group"].value_counts())

    results = []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Computing SNR"):
        image_name = row["image"]
        img_path = None
        for ext in [".jpeg", ".jpg", ".png"]:
            p = image_dir / f"{image_name}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is None:
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        snr_result = compute_snr(img_rgb)
        results.append({
            "image": image_name,
            "odcr": row["odcr"],
            "tone_group": row["tone_group"],
            "label": row.get("label"),
            **snr_result
        })

    df_snr = pd.DataFrame(results)
    df_snr = df_snr[df_snr["snr"].notna()]

    df_snr.to_csv(output_path, index=False)
    return df_snr


# ── Analysis & reporting ──────────────────────────────────────────

def analyse_snr(df_snr: pd.DataFrame, output_dir: Path) -> dict:
    """
    The core analysis — mirrors SkinToneNet Section 3.3.
    Produces:
      1. SNR by tone group table
      2. SNR ratio (light/dark) — your 5.2× equivalent
      3. Statistical test (Kruskal-Wallis + pairwise Mann-Whitney)
      4. SNR vs ODCR scatter plot
      5. Box plot by tone group
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = ["light", "medium", "dark"]
    colours = {"light": "#4C9BE8", "medium": "#F0A500", "dark": "#C0392B"}

    # ── Per-group statistics ──────────────────────────────────────
    stats_rows = []
    group_snrs = {}
    for g in groups:
        snrs = df_snr[df_snr["tone_group"] == g]["snr"].dropna()
        if len(snrs) == 0:
            continue
        group_snrs[g] = snrs
        stats_rows.append({
            "tone_group": g,
            "n": len(snrs),
            "mean_snr": snrs.mean(),
            "median_snr": snrs.median(),
            "std_snr": snrs.std(),
            "p25": snrs.quantile(0.25),
            "p75": snrs.quantile(0.75),
        })

    df_stats = pd.DataFrame(stats_rows)
    print("\n── SNR by Tone Group ──────────────────────────────────────")
    print(df_stats.round(3).to_string(index=False))

    # ── SNR ratio ────────────────────────────────────────────────
    snr_ratio = None
    if "light" in group_snrs and "dark" in group_snrs:
        snr_ratio = group_snrs["light"].mean() / group_snrs["dark"].mean()
        print(f"\n✦ SNR ratio (light/dark): {snr_ratio:.2f}×")
        print(f"  SkinToneNet benchmark:  5.20×")

        if snr_ratio >= 3.0:
            print("  → Strong support for mechanistic SNR framework")
        elif snr_ratio >= 1.5:
            print("  → Moderate support — fundus SNR effect smaller than dermoscopy (expected)")
        else:
            print("  → Weak SNR effect — check ODCR proxy calibration and disc localisation")

    # ── Statistical tests ────────────────────────────────────────
    if len(group_snrs) >= 2:
        kw_stat, kw_p = stats.kruskal(*group_snrs.values())
        print(f"\nKruskal-Wallis test (H={kw_stat:.2f}, p={kw_p:.4f})")

        if "light" in group_snrs and "dark" in group_snrs:
            mw_stat, mw_p = stats.mannwhitneyu(
                group_snrs["light"], group_snrs["dark"], alternative="greater"
            )
            print(f"Mann-Whitney U (light > dark): U={mw_stat:.0f}, p={mw_p:.4f}")

    # ── Plot 1: Box plot by tone group ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    plot_data  = [group_snrs[g].values for g in groups if g in group_snrs]
    plot_labels = [f"{g}\n(n={len(group_snrs[g]):,})"
                   for g in groups if g in group_snrs]
    plot_colors = [colours[g] for g in groups if g in group_snrs]

    bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Vessel-Background SNR", fontsize=12)
    ax.set_title("SNR by Fundus Pigmentation Group\n(vessel-background contrast by ODCR tertile)", fontsize=12)
    ax.set_xlabel("ODCR Tone Group", fontsize=12)
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3)

    n_clipped = int((df_snr["snr"] > 10).sum())
    ax.text(0.02, 0.98, f"{n_clipped} outliers > 10 not shown",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color="gray", style="italic")

    if snr_ratio:
        ax.text(0.98, 0.98, f"SNR ratio\n(light/dark)\n{snr_ratio:.2f}×",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # ── Plot 2: SNR vs ODCR scatter ───────────────────────────────
    ax2 = axes[1]
    for g in groups:
        if g not in group_snrs:
            continue
        subset = df_snr[df_snr["tone_group"] == g]
        ax2.scatter(subset["odcr"], subset["snr"], alpha=0.3, s=10,
                    color=colours[g], label=g.capitalize())

    # Regression line
    valid = df_snr[["odcr", "snr"]].dropna()
    if len(valid) > 10:
        slope, intercept, r, p, _ = stats.linregress(valid["odcr"], valid["snr"])
        x_line = np.linspace(valid["odcr"].min(), valid["odcr"].max(), 100)
        ax2.plot(x_line, slope * x_line + intercept, "k--", linewidth=2,
                 label=f"r={r:.2f}, p={p:.3f}")

    ax2.set_xlabel("ODCR (degrees)", fontsize=12)
    ax2.set_ylabel("Vessel-Background SNR", fontsize=12)
    ax2.set_title("SNR vs ODCR — Mechanistic Relationship\n(Predicts: positive correlation)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "snr_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {plot_path}")

    # ── Plot 3: ODCR distribution ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for g in groups:
        if g not in group_snrs:
            continue
        subset = df_snr[df_snr["tone_group"] == g]["odcr"].dropna()
        ax.hist(subset, bins=40, alpha=0.6, color=colours[g], label=g.capitalize(), density=True)

    ax.axvline(35, color="gray", linestyle="--", alpha=0.7, label="Light/Medium boundary (35°)")
    ax.axvline(10, color="gray", linestyle=":",  alpha=0.7, label="Medium/Dark boundary (10°)")
    ax.set_xlabel("ODCR (degrees)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("ODCR Distribution by Tone Group", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    dist_path = output_dir / "odcr_distribution.png"
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution plot saved to {dist_path}")

    # ── Save results ──────────────────────────────────────────────
    summary = {
        "snr_by_group": df_stats.to_dict(orient="records"),
        "snr_ratio_light_over_dark": snr_ratio,
        "skintone_benchmark": 5.2,
        "kruskal_wallis_p": float(kw_p) if "kw_p" in dir() else None,
        "n_images_analysed": len(df_snr),
    }
    with open(output_dir / "snr_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def main(args):
    image_dir  = Path(args.image_dir)
    odcr_csv   = Path(args.odcr_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snr_cache = output_dir / "snr_raw.csv"

    if snr_cache.exists() and not args.recompute:
        print(f"Loading cached SNR from {snr_cache}")
        df_snr = pd.read_csv(snr_cache)
    else:
        df_snr = compute_snr_for_split(
            image_dir=image_dir,
            odcr_csv=odcr_csv,
            output_path=snr_cache,
            sample_n=args.sample_n,
            seed=42,
        )

    summary = analyse_snr(df_snr, output_dir)
    print("\n── Final Summary ──────────────────────────────────────────")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  default="data/train")
    parser.add_argument("--odcr_csv",   default="data/odcr/train_odcr.csv")
    parser.add_argument("--output_dir", default="results/snr_analysis")
    parser.add_argument("--sample_n",   type=int, default=2000,
                        help="Images to sample per group (stratified)")
    parser.add_argument("--recompute",  action="store_true",
                        help="Recompute even if cache exists")
    main(parser.parse_args())
