"""
run_preprocessing.py — Master runner for FundusNet preprocessing pipeline.

Run this in order after downloading EyePACS:
    python run_preprocessing.py --step all

Or step by step:
    python run_preprocessing.py --step split
    python run_preprocessing.py --step odcr --split train
    python run_preprocessing.py --step odcr --split val
    python run_preprocessing.py --step odcr --split test
    python run_preprocessing.py --step snr

For quick debugging on 500 images before running full dataset:
    python run_preprocessing.py --step all --debug
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ── Default paths — adjust to your setup ─────────────────────────
DEFAULTS = {
    "labels_csv":  "data/trainLabels.csv",
    "image_dir":   "data/train",
    "splits_dir":  "data/splits",
    "odcr_dir":    "data/odcr",
    "results_dir": "results/snr_analysis",
}


def run(cmd: list[str]):
    """Run a subprocess command, exit on failure."""
    print(f"\n$ {' '.join(cmd)}\n{'─' * 60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n✗ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✓ Done")


def step_split(args):
    print("\n═══ STEP 1: Patient-Level Split ═══")
    run([
        sys.executable, "src/data/split.py",
        "--labels_csv", args.labels_csv,
        "--output_dir", args.splits_dir,
        "--seed", "42",
    ])


def step_odcr(args, split: str = "train"):
    print(f"\n═══ STEP 2: ODCR Computation ({split}) ═══")
    cmd = [
        sys.executable, "src/features/odcr.py",
        "--image_dir",  args.image_dir,
        "--split_csv",  str(Path(args.splits_dir) / f"{split}.csv"),
        "--output_dir", args.odcr_dir,
        "--workers",    str(args.workers),
    ]
    if args.debug:
        cmd += ["--max_images", "500"]
    run(cmd)


def step_snr(args):
    print("\n═══ STEP 3: SNR Analysis ═══")
    cmd = [
        sys.executable, "src/analysis/snr.py",
        "--image_dir",  args.image_dir,
        "--odcr_csv",   str(Path(args.odcr_dir) / "train_odcr.csv"),
        "--output_dir", args.results_dir,
        "--sample_n",   "500" if args.debug else "2000",
    ]
    run(cmd)


def main(args):
    # Create directory structure
    for d in [args.splits_dir, args.odcr_dir, args.results_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if args.step in ("split", "all"):
        step_split(args)

    if args.step in ("odcr", "all"):
        for split in ["train", "val", "test"]:
            step_odcr(args, split)

    if args.step in ("snr", "all"):
        step_snr(args)

    if args.step == "all":
        print("\n" + "═" * 60)
        print("✓ Full preprocessing pipeline complete.")
        print("\nNext steps:")
        print("  1. Check results/snr_analysis/snr_summary.json")
        print("     → If SNR ratio > 1.5×: mechanistic premise confirmed, start training")
        print("     → If SNR ratio < 1.5×: review ODCR proxy, check disc localisation")
        print("  2. Open results/snr_analysis/snr_analysis.png")
        print("  3. Check data/odcr/train_odcr.csv tone_group distribution")
        print("     → Dark group should be 15-30% of images (if less, check ODCR thresholds)")
        print("═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FundusNet preprocessing pipeline")
    parser.add_argument("--step", choices=["split", "odcr", "snr", "all"], default="all")
    parser.add_argument("--labels_csv",  default=DEFAULTS["labels_csv"])
    parser.add_argument("--image_dir",   default=DEFAULTS["image_dir"])
    parser.add_argument("--splits_dir",  default=DEFAULTS["splits_dir"])
    parser.add_argument("--odcr_dir",    default=DEFAULTS["odcr_dir"])
    parser.add_argument("--results_dir", default=DEFAULTS["results_dir"])
    parser.add_argument("--workers",     type=int, default=8)
    parser.add_argument("--debug",       action="store_true",
                        help="Run on 500 images only — quick sanity check before full run")
    main(parser.parse_args())
