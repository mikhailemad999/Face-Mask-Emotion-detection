"""
Step 1 & 2: Load, Explore, and Check Duplicates
================================================
Face Mask & Emotion Detection — Graduation Project
Run this script from: project/ml/
Usage:  python notebooks/01_explore_duplicates.py
"""

import sys
import os
import io
from pathlib import Path

# Fix Windows console encoding for emoji/unicode output
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "project" / "ml"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

from utils.dataset_utils import (
    build_emotion_df, build_mask_df,
    load_image_pil, EMOTION_CLASSES, MASK_CLASSES
)

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "project" / "ml" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)

print("=" * 70)
print("STEP 1: LOAD AND EXPLORE DATASET")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1.1  Build DataFrames
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1.1] Building emotion DataFrame …")
df_em = build_emotion_df()

print("\n[1.1] Building mask DataFrame …")
df_mask = build_mask_df()

# Combine for joint analysis
df_all = pd.concat([df_em, df_mask], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.2  Basic Shape / Types / Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1.2] SHAPE:")
print(f"  Emotion dataset : {len(df_em):>6} images")
print(f"  Mask dataset    : {len(df_mask):>6} images")
print(f"  Combined        : {len(df_all):>6} images")

print("\n[1.2] DTYPES:")
print(df_all.dtypes)

print("\n[1.2] SUMMARY STATISTICS (numeric columns):")
numeric_cols = ["width", "height", "channels", "file_size_bytes"]
print(df_all[numeric_cols + ["dataset"]].groupby("dataset").describe().round(2))

print("\n[1.2] SAMPLE ROWS (emotion):")
print(df_em.head(3).to_string())

print("\n[1.2] SAMPLE ROWS (mask):")
print(df_mask.head(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 1.3  Identify data type mismatches
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1.3] COLOR MODE DISTRIBUTION:")
print(df_all.groupby(["dataset", "color_mode"]).size().reset_index(name="count"))

# Grayscale vs RGB inconsistency check
em_rgb   = df_em[df_em["color_mode"] == "RGB"]
em_gray  = df_em[df_em["color_mode"] == "L"]
mask_gray = df_mask[df_mask["color_mode"] == "L"]
mask_rgb  = df_mask[df_mask["color_mode"] == "RGB"]

print(f"\n  Emotion — RGB: {len(em_rgb)} | Grayscale: {len(em_gray)}")
print(f"  Mask    — RGB: {len(mask_rgb)} | Grayscale: {len(mask_gray)}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.4  Valid / Invalid breakdown
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1.4] VALIDITY CHECK:")
validity = df_all.groupby(["dataset", "valid"]).size().reset_index(name="count")
print(validity)

invalid_imgs = df_all[~df_all["valid"]]
if len(invalid_imgs) > 0:
    print(f"\n  ⚠️  {len(invalid_imgs)} invalid/corrupt images found:")
    print(invalid_imgs[["filepath", "label", "error"]].to_string())
else:
    print("  ✅ No corrupt images detected.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: CHECK FOR AND HANDLE DUPLICATES")
print("=" * 70)

valid_df = df_all[df_all["valid"] & df_all["file_hash"].notna()].copy()

# 2.1  Exact duplicates by file hash
dup_mask_exact = valid_df.duplicated(subset=["file_hash"], keep=False)
exact_dups = valid_df[dup_mask_exact]

print(f"\n[2.1] Exact duplicates (same file hash): {len(exact_dups)}")
print(f"      = {len(exact_dups)/len(valid_df)*100:.2f}% of valid images")

if len(exact_dups) > 0:
    print("\n  Sample duplicate groups:")
    grp = exact_dups.groupby("file_hash").agg(
        count=("filepath", "count"),
        labels=("label", lambda x: list(x.unique())),
        datasets=("dataset", lambda x: list(x.unique())),
    ).reset_index()
    print(grp.head(10).to_string())

# 2.2  Duplicates within same dataset and class
print("\n[2.2] Intra-class duplicates per dataset:")
for ds_name, sub in valid_df.groupby("dataset"):
    dup_intra = sub.duplicated(subset=["file_hash"], keep=False)
    n = dup_intra.sum()
    print(f"  {ds_name}: {n} ({n/len(sub)*100:.2f}%)")

# 2.3  Cross-class duplicates (same image, different label — problematic!)
cross_class = (
    exact_dups.groupby("file_hash")["label"]
    .nunique()
    .reset_index(name="n_labels")
)
cross_class_problematic = cross_class[cross_class["n_labels"] > 1]
print(f"\n[2.3] Cross-class duplicates (same image, different labels): {len(cross_class_problematic)}")
if len(cross_class_problematic) > 0:
    print("  ⚠️  These are LABEL CONFLICTS and must be investigated!")
    print(cross_class_problematic.to_string())
else:
    print("  ✅ No cross-class label conflicts found.")

# 2.4  Drop duplicates (keep='first' strategy)
print("\n[2.4] Dropping duplicates (keep first occurrence)…")
df_em_clean   = df_em[df_em["valid"]].drop_duplicates(subset=["file_hash"], keep="first")
df_mask_clean = df_mask[df_mask["valid"]].drop_duplicates(subset=["file_hash"], keep="first")

print(f"  Emotion: {len(df_em)} → {len(df_em_clean)} (dropped {len(df_em)-len(df_em_clean)})")
print(f"  Mask   : {len(df_mask)} → {len(df_mask_clean)} (dropped {len(df_mask)-len(df_mask_clean)})")

# ─────────────────────────────────────────────────────────────────────────────
# Save clean DataFrames
# ─────────────────────────────────────────────────────────────────────────────
df_em_clean.to_csv(OUT_DIR / "df_emotion_clean.csv", index=False)
df_mask_clean.to_csv(OUT_DIR / "df_mask_clean.csv", index=False)
print(f"\n[SAVED] Clean DataFrames → {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION — Sample image grids (Step 5 contribution)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[VIZ] Generating sample image grids…")

def show_class_samples(df: pd.DataFrame, classes: list, n_per_class: int = 5,
                        title: str = "Samples", out_path: Path = None):
    """Plot a grid of sample images per class."""
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, n_per_class,
                              figsize=(n_per_class * 2.5, n_classes * 2.5))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    for row, cls in enumerate(classes):
        class_df = df[(df["label"] == cls) & df["valid"]]
        samples  = class_df.sample(min(n_per_class, len(class_df)), random_state=42)
        for col in range(n_per_class):
            ax = axes[row][col] if n_classes > 1 else axes[col]
            ax.axis("off")
            if col < len(samples):
                fp = samples.iloc[col]["filepath"]
                try:
                    img = Image.open(fp).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "ERR", ha="center")
            if col == 0:
                ax.set_ylabel(cls, fontsize=9, rotation=90, labelpad=40,
                              va="center", fontweight="bold")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close()


show_class_samples(df_em_clean,   EMOTION_CLASSES, n_per_class=5,
                   title="Emotion Dataset — Sample Images",
                   out_path=OUT_DIR / "viz_emotion_samples.png")

show_class_samples(df_mask_clean, MASK_CLASSES, n_per_class=5,
                   title="Mask Dataset — Sample Images",
                   out_path=OUT_DIR / "viz_mask_samples.png")

print("\n✅ Steps 1 & 2 complete. Outputs saved to:", OUT_DIR)
print("   Next: Run 02_missing_outliers.py")
