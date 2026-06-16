"""
odcr.py — Optic Disc Colour Ratio (ODCR) computation
Your ITA equivalent for fundus images.

Formula (mirrors SkinToneNet ITA, Section 3.2):
    ODCR = arctan((L* - 50) / b*) × (180/π)

Region: peripapillary annulus (ring around optic disc) in CIE L*a*b* space.
Disc localisation: brightest region in red channel (standard heuristic for
non-pathological discs — sufficient for a population-level fairness proxy).

Encoding for model input (angular continuity, mirrors SkinToneNet):
    [sin(θ), cos(θ), θ/90]  where θ = ODCR in degrees
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Disc localisation ─────────────────────────────────────────────

def localise_disc(img_rgb: np.ndarray) -> tuple[int, int, int]:
    """
    Estimate optic disc centre and radius from the red channel.
    The disc is the brightest compact region in the red channel.

    Returns: (cx, cy, radius) in pixels.
    Falls back to image centre if localisation fails.
    """
    h, w = img_rgb.shape[:2]

    # Work on red channel — disc is bright, vessels are dark
    red = img_rgb[:, :, 0].astype(np.float32)

    # Remove bright artifacts at image edges (common in RetCam / fundus cameras)
    mask = np.zeros_like(red, dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2 - 10, 255, -1)
    red = cv2.bitwise_and(red, red, mask=mask)

    # Gaussian blur to suppress vessels and noise
    blurred = cv2.GaussianBlur(red, (0, 0), sigmaX=w // 30)

    # Brightest point = disc centre estimate
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx, cy = max_loc

    # Estimate disc radius as ~8% of image width (typical for fundus images)
    radius = max(int(w * 0.08), 20)

    # Sanity check — if centre is within 5% of edge, fall back to image centre
    margin = int(min(h, w) * 0.05)
    if cx < margin or cx > w - margin or cy < margin or cy > h - margin:
        cx, cy = w // 2, h // 2

    return cx, cy, radius


# ── Peripapillary annulus extraction ─────────────────────────────

def extract_peripapillary_region(
    img_rgb: np.ndarray,
    cx: int, cy: int, radius: int,
    inner_scale: float = 1.1,   # inner boundary: just outside disc edge
    outer_scale: float = 2.5    # outer boundary: 2.5x disc radius
) -> np.ndarray | None:
    """
    Extract pixels from the annular region surrounding the optic disc.
    Mirrors SkinToneNet's perilesional background extraction (Section 3.2).

    Inner radius = disc_radius × inner_scale  (avoid disc itself)
    Outer radius = disc_radius × outer_scale  (capture choroidal pigmentation)

    Returns: flat array of pixel values in L*a*b* space, or None if region invalid.
    """
    h, w = img_rgb.shape[:2]
    inner_r = int(radius * inner_scale)
    outer_r = int(radius * outer_scale)

    # Build annular mask
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    annulus_mask = (dist >= inner_r) & (dist <= outer_r)

    # Exclude image boundary and very dark pixels (retinal edge artifacts)
    boundary_mask = np.zeros((h, w), dtype=bool)
    boundary_margin = 10
    boundary_mask[boundary_margin:h-boundary_margin,
                  boundary_margin:w-boundary_margin] = True
    annulus_mask &= boundary_mask

    n_pixels = annulus_mask.sum()
    if n_pixels < 50:
        return None  # degenerate image — skip

    # Convert to L*a*b* and extract annulus pixels
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV L*a*b* ranges: L* [0,255], a* [0,255], b* [0,255]
    # Rescale to standard ranges: L* [0,100], a* [-128,127], b* [-128,127]
    img_lab[:, :, 0] = img_lab[:, :, 0] * (100.0 / 255.0)
    img_lab[:, :, 1] = img_lab[:, :, 1] - 128.0
    img_lab[:, :, 2] = img_lab[:, :, 2] - 128.0

    return img_lab[annulus_mask]  # shape: (n_pixels, 3)


# ── ODCR computation ──────────────────────────────────────────────

def compute_odcr(img_rgb: np.ndarray) -> float | None:
    """
    Compute the Optic Disc Colour Ratio for a single fundus image.

    ODCR = arctan((L* - 50) / b*) × (180/π)

    Uses median L* and b* of the peripapillary annulus for robustness
    against vessel pixels and local artifacts.

    Returns ODCR in degrees, or None if computation fails.
    """
    try:
        cx, cy, radius = localise_disc(img_rgb)
        pixels = extract_peripapillary_region(img_rgb, cx, cy, radius)

        if pixels is None or len(pixels) < 50:
            return None

        L_star = float(np.median(pixels[:, 0]))
        b_star  = float(np.median(pixels[:, 2]))

        # Guard against b* ≈ 0 (achromatic region — rare but possible)
        if abs(b_star) < 1e-3:
            return None

        odcr = np.degrees(np.arctan((L_star - 50.0) / b_star))
        return float(odcr)

    except Exception:
        return None


def odcr_to_encoding(odcr_deg: float) -> list[float]:
    """
    Encode ODCR for model input — angular continuity (mirrors SkinToneNet ITA encoding).
    Returns [sin(θ), cos(θ), θ/90] where θ is in radians.
    """
    theta = np.radians(odcr_deg)
    return [float(np.sin(theta)), float(np.cos(theta)), float(odcr_deg / 90.0)]


# ── Tone group assignment ─────────────────────────────────────────

def assign_tone_group(odcr: float) -> str:
    """
    Assign fundus pigmentation group by ODCR.
    Thresholds chosen to mirror ITA light/medium/dark boundaries
    in SkinToneNet (ITA > 41° light, 10°–41° medium, ≤ 10° dark).

    These thresholds are a starting hypothesis — validate against
    APTOS population distribution before fixing them.
    """
    if odcr > 35.0:
        return "light"
    elif odcr > 10.0:
        return "medium"
    else:
        return "dark"


# ── Batch processing ──────────────────────────────────────────────

def process_single_image(image_path: Path) -> dict:
    """Process one image — used in parallel executor."""
    try:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {"image": image_path.stem, "odcr": None, "tone_group": "unknown", "error": "load_failed"}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to standard size for consistent disc localisation
        img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)

        odcr = compute_odcr(img_rgb)
        tone_group = assign_tone_group(odcr) if odcr is not None else "unknown"
        encoding = odcr_to_encoding(odcr) if odcr is not None else [0.0, 1.0, 0.0]

        return {
            "image": image_path.stem,
            "odcr": odcr,
            "tone_group": tone_group,
            "odcr_sin": encoding[0],
            "odcr_cos": encoding[1],
            "odcr_norm": encoding[2],
            "error": None
        }
    except Exception as e:
        return {"image": image_path.stem, "odcr": None, "tone_group": "unknown", "error": str(e)}


def compute_odcr_batch(
    image_dir: Path,
    split_csv: Path,
    output_path: Path,
    n_workers: int = 8,
    max_images: int | None = None,
) -> pd.DataFrame:
    """
    Compute ODCR for all images in a split CSV.
    Saves results to output_path as CSV.
    """
    df_split = pd.read_csv(split_csv)
    image_names = df_split["image"].tolist()

    if max_images:
        image_names = image_names[:max_images]
        print(f"  [Debug mode] Processing first {max_images} images only")

    # Build full paths — EyePACS images are .jpeg
    image_paths = []
    for name in image_names:
        for ext in [".jpeg", ".jpg", ".png"]:
            p = image_dir / f"{name}{ext}"
            if p.exists():
                image_paths.append(p)
                break

    print(f"Found {len(image_paths):,} images (of {len(image_names):,} in split)")

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_image, p): p for p in image_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing ODCR"):
            results.append(future.result())

    df_odcr = pd.DataFrame(results)

    # Merge with split to keep labels
    df_merged = df_split.merge(df_odcr, on="image", how="left")

    # Summary
    n_failed = df_merged["odcr"].isna().sum()
    print(f"\nODCR computation complete:")
    print(f"  Succeeded: {(~df_merged['odcr'].isna()).sum():,}")
    print(f"  Failed:    {n_failed:,} ({n_failed/len(df_merged)*100:.1f}%) — excluded from analysis")
    print(f"\nTone group distribution:")
    print(df_merged["tone_group"].value_counts())
    print(f"\nODCR statistics:")
    print(df_merged["odcr"].describe().round(2))

    df_merged.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    return df_merged


def main(args):
    image_dir  = Path(args.image_dir)
    split_csv  = Path(args.split_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_name = Path(args.split_csv).stem  # e.g. "train", "test"
    output_path = output_dir / f"{split_name}_odcr.csv"

    compute_odcr_batch(
        image_dir=image_dir,
        split_csv=split_csv,
        output_path=output_path,
        n_workers=args.workers,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  default="data/train")
    parser.add_argument("--split_csv",  default="data/splits/train.csv")
    parser.add_argument("--output_dir", default="data/odcr")
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit for quick debugging (e.g. 500)")
    main(parser.parse_args())
