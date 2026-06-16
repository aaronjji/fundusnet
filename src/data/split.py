"""
split.py — Patient-level train/val/test split for EyePACS
Mirrors SkinToneNet Section 3.1 methodology exactly.

EyePACS filenames: {patient_id}_left.jpeg / {patient_id}_right.jpeg
Patient ID is the integer prefix before the underscore.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import json
import argparse


# ── Label binarisation ────────────────────────────────────────────
# DR grade 0-1 = non-referable (negative class)
# DR grade 2-4 = referable    (positive class)
# Standard clinical threshold — equivalent to SkinToneNet's benign/malignant split
GRADE_TO_BINARY = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1}


def extract_patient_id(image_name: str) -> str:
    """
    Extract patient ID from EyePACS filename.
    '10_left'  -> '10'
    '10_right' -> '10'
    Both eyes belong to the same patient — must be kept in the same split.
    """
    return image_name.split("_")[0]


def load_and_prepare(labels_csv: Path) -> pd.DataFrame:
    """Load trainLabels.csv, add patient_id and binary label columns."""
    df = pd.read_csv(labels_csv)

    # Normalise column names (Kaggle CSV uses 'image' and 'level')
    df.columns = df.columns.str.strip().str.lower()
    assert "image" in df.columns and "level" in df.columns, \
        "Expected columns: 'image', 'level'. Check your CSV."

    df["patient_id"] = df["image"].apply(extract_patient_id)
    df["label"] = df["level"].map(GRADE_TO_BINARY)
    df["dr_grade"] = df["level"]  # keep original for reference

    print(f"\nLoaded {len(df):,} images from {df['patient_id'].nunique():,} patients")
    print(f"DR grade distribution:\n{df['dr_grade'].value_counts().sort_index()}")
    print(f"\nBinary label distribution:")
    print(f"  Non-referable (0): {(df['label']==0).sum():,} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"  Referable     (1): {(df['label']==1).sum():,} ({(df['label']==1).mean()*100:.1f}%)")

    return df


def patient_level_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    GroupShuffleSplit on patient_id — zero patient overlap between splits.
    70 / 15 / 15 mirrors SkinToneNet exactly.
    """
    patients = df["patient_id"].values
    indices = np.arange(len(df))

    # First split: train vs. (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=seed)
    train_idx, temp_idx = next(gss1.split(indices, groups=patients))

    df_train = df.iloc[train_idx].copy()
    df_temp  = df.iloc[temp_idx].copy()

    # Second split: val vs. test (50/50 of the remaining ~30%)
    patients_temp = df_temp["patient_id"].values
    indices_temp  = np.arange(len(df_temp))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(gss2.split(indices_temp, groups=patients_temp))

    df_val  = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()

    # ── Verify zero patient overlap (critical) ──────────────────────
    train_patients = set(df_train["patient_id"])
    val_patients   = set(df_val["patient_id"])
    test_patients  = set(df_test["patient_id"])

    assert len(train_patients & val_patients)  == 0, "LEAKAGE: train/val patient overlap!"
    assert len(train_patients & test_patients) == 0, "LEAKAGE: train/test patient overlap!"
    assert len(val_patients   & test_patients) == 0, "LEAKAGE: val/test patient overlap!"
    print("\n✓ Zero patient overlap verified across all splits")

    return df_train, df_val, df_test


def print_split_stats(df_train, df_val, df_test):
    total = len(df_train) + len(df_val) + len(df_test)
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        pos_rate = df["label"].mean() * 100
        n_patients = df["patient_id"].nunique()
        print(f"  {name:5s}: {len(df):6,} images | {n_patients:5,} patients | "
              f"{pos_rate:.1f}% referable ({df['label'].sum():,} positives)")
    print(f"  Total: {total:,} images")


def main(args):
    labels_csv = Path(args.labels_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(labels_csv)
    df_train, df_val, df_test = patient_level_split(df, seed=args.seed)

    print("\nSplit summary:")
    print_split_stats(df_train, df_val, df_test)

    # Save splits
    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv",     index=False)
    df_test.to_csv(output_dir / "test.csv",   index=False)

    # Save metadata for reproducibility (mirrors SkinToneNet locked test set)
    meta = {
        "seed": args.seed,
        "train_images": len(df_train),
        "val_images":   len(df_val),
        "test_images":  len(df_test),
        "train_patients": df_train["patient_id"].nunique(),
        "val_patients":   df_val["patient_id"].nunique(),
        "test_patients":  df_test["patient_id"].nunique(),
        "test_positive_rate": float(df_test["label"].mean()),
        "overlap_verified": True
    }
    with open(output_dir / "split_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSplits saved to {output_dir}/")
    print("Locked test set: test.csv — do not peek at metrics until ablation is complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv",  default="data/trainLabels.csv")
    parser.add_argument("--output_dir",  default="data/splits")
    parser.add_argument("--seed",        type=int, default=42)
    main(parser.parse_args())
