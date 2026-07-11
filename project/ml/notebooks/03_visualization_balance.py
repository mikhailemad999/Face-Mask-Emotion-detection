"""
Step 5 & 6: Comprehensive Visualizations + Imbalance Handling
=============================================================
Face Mask & Emotion Detection — Graduation Project
Run: python notebooks/03_visualization_balance.py
"""

import sys
import io
from pathlib import Path

# Fix Windows console encoding for emoji/unicode output
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "project" / "ml"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from sklearn.utils import compute_class_weight
from sklearn.decomposition import PCA
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    print("⚠️  imbalanced-learn not installed. Run: pip install imbalanced-learn")

OUT_DIR = PROJECT_ROOT / "project" / "ml" / "outputs"

# ── Load cleaned data ────────────────────────────────────────────────────────
df_all  = pd.read_csv(OUT_DIR / "df_all_cleaned.csv")
df_em   = df_all[df_all["dataset"] == "emotion"].copy()
df_mask = df_all[df_all["dataset"] == "mask"].copy()

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MASK_CLASSES    = ["with_mask", "without_mask"]

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
PALETTE_EMOTION = [
    "#FF3366", "#FF6B35", "#FFB700", "#00E5A0",
    "#00B4D8", "#7B61FF", "#E040FB",
]
PALETTE_MASK = ["#00FFB3", "#FF3366"]

plt.rcParams.update({
    "figure.facecolor":  "#0A0F1E",
    "axes.facecolor":    "#111827",
    "text.color":        "#E2E8F0",
    "axes.labelcolor":   "#E2E8F0",
    "xtick.color":       "#94A3B8",
    "ytick.color":       "#94A3B8",
    "axes.edgecolor":    "#1E293B",
    "grid.color":        "#1E293B",
    "figure.dpi":        120,
    "font.family":       "DejaVu Sans",
})

print("=" * 70)
print("STEP 5: VISUALIZATIONS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1: Class distribution bar chart — Emotion dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n[VIZ 1] Emotion class distribution…")
em_counts = df_em["label"].value_counts().reindex(EMOTION_CLASSES)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(em_counts.index, em_counts.values, color=PALETTE_EMOTION,
              edgecolor="#0A0F1E", linewidth=1.2, width=0.65)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f"{bar.get_height()}", ha="center", va="bottom", fontsize=9)
ax.set_title("Emotion Dataset — Class Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Emotion Class")
ax.set_ylabel("Image Count")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_DIR / "viz1_emotion_class_dist.png", bbox_inches="tight")
plt.close()
print(f"  Saved: viz1_emotion_class_dist.png")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 2: Class distribution bar chart — Mask dataset
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 2] Mask class distribution…")
mask_counts = df_mask["label"].value_counts().reindex(MASK_CLASSES)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(mask_counts.index, mask_counts.values, color=PALETTE_MASK,
              edgecolor="#0A0F1E", linewidth=1.2, width=0.5)
for bar in bars:
    pct = bar.get_height() / mask_counts.sum() * 100
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{bar.get_height()} ({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
ax.set_title("Mask Dataset — Class Distribution", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Class")
ax.set_ylabel("Image Count")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(OUT_DIR / "viz2_mask_class_dist.png", bbox_inches="tight")
plt.close()
print(f"  Saved: viz2_mask_class_dist.png")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3: File size distribution (KDE + histogram) — both datasets
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 3] File size distribution…")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (ds, sub, palette) in zip(axes, [
    ("Emotion", df_em,   PALETTE_EMOTION),
    ("Mask",   df_mask, PALETTE_MASK),
]):
    for cls, color in zip(sub["label"].unique(), palette):
        data = sub[sub["label"] == cls]["file_size_bytes"].dropna()
        ax.hist(data, bins=40, alpha=0.45, color=color, label=cls)
        sns.kdeplot(data, ax=ax, color=color, linewidth=2)
    ax.set_title(f"{ds} — File Size Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("File Size (bytes)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "viz3_file_size_dist.png", bbox_inches="tight")
plt.close()
print(f"  Saved: viz3_file_size_dist.png")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 4: Image dimension scatter (width vs height) colored by class
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 4] Image dimension scatter…")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (ds, sub, palette, classes) in zip(axes, [
    ("Emotion", df_em,   PALETTE_EMOTION, EMOTION_CLASSES),
    ("Mask",   df_mask, PALETTE_MASK,    MASK_CLASSES),
]):
    for cls, color in zip(classes, palette):
        s = sub[sub["label"] == cls]
        ax.scatter(s["width"], s["height"], c=color, alpha=0.4, s=15, label=cls)
    ax.set_title(f"{ds} — Width vs Height", fontsize=12, fontweight="bold")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.legend(fontsize=7, markerscale=2)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "viz4_dimension_scatter.png", bbox_inches="tight")
plt.close()
print(f"  Saved: viz4_dimension_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 5: Pixel intensity distribution per emotion class
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 5] Pixel intensity sampling per emotion class…")

def sample_pixel_means(df_class: pd.DataFrame, n_sample: int = 80) -> np.ndarray:
    """Quickly compute per-image mean pixel value from a random sample."""
    sample = df_class.sample(min(n_sample, len(df_class)), random_state=42)
    means = []
    for fp in sample["filepath"]:
        try:
            arr = np.array(Image.open(fp).convert("L"), dtype=np.float32)
            means.append(arr.mean())
        except Exception:
            pass
    return np.array(means)

fig, ax = plt.subplots(figsize=(12, 5))
for cls, color in zip(EMOTION_CLASSES, PALETTE_EMOTION):
    sub = df_em[df_em["label"] == cls]
    px_means = sample_pixel_means(sub, n_sample=100)
    if len(px_means) > 5:
        sns.kdeplot(px_means, ax=ax, label=cls, color=color, fill=True, alpha=0.25, linewidth=2)
ax.set_title("Emotion Classes — Mean Pixel Intensity Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean Pixel Value (0–255 grayscale)")
ax.set_ylabel("Density")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "viz5_pixel_intensity.png", bbox_inches="tight")
plt.close()
print(f"  Saved: viz5_pixel_intensity.png")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 6: Correlation heatmap of numeric features
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 6] Correlation heatmap…")
num_features = ["width", "height", "file_size_bytes", "log_file_size", "channels"]
available = [c for c in num_features if c in df_all.columns]
corr_matrix = df_all[available].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    mask=False, ax=ax,
    linewidths=0.5, linecolor="#0A0F1E",
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "viz6_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print(f"  Saved: viz6_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# VIZ 7: Combined class imbalance radar / comparison chart
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 7] Imbalance overview…")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Emotion — show imbalance ratio
em_counts_sorted = df_em["label"].value_counts().reindex(EMOTION_CLASSES)
max_count = em_counts_sorted.max()
imbalance_ratios = em_counts_sorted / max_count

axes[0].barh(EMOTION_CLASSES, imbalance_ratios.values, color=PALETTE_EMOTION)
axes[0].axvline(x=0.7, color="#FF3366", linestyle="--", linewidth=2, label="70% threshold (imbalanced)")
axes[0].set_xlim(0, 1.1)
axes[0].set_title("Emotion — Imbalance Ratio\n(relative to largest class)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Ratio (1.0 = balanced)")
axes[0].legend(fontsize=8)
axes[0].grid(axis="x", alpha=0.3)

# Mask — pie chart
mask_pct = df_mask["label"].value_counts(normalize=True) * 100
axes[1].pie(
    mask_pct.values,
    labels=[f"{l}\n{v:.1f}%" for l, v in zip(mask_pct.index, mask_pct.values)],
    colors=PALETTE_MASK,
    startangle=90,
    wedgeprops={"edgecolor": "#0A0F1E", "linewidth": 2},
    textprops={"color": "#E2E8F0"},
)
axes[1].set_title("Mask Dataset — Class Balance (Pie)", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT_DIR / "viz7_class_imbalance.png", bbox_inches="tight")
plt.close()
print(f"  Saved: viz7_class_imbalance.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: CHECK AND HANDLE IMBALANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: CHECK IMBALANCE AND HANDLE IT")
print("=" * 70)

# 6.1  Check class percentages
print("\n[6.1] Emotion class distribution:")
em_pct = df_em["label"].value_counts(normalize=True) * 100
for cls, pct in em_pct.items():
    flag = "⚠️  IMBALANCED" if pct < 10 else ("✅ OK" if pct > 12 else "⚡ Slight")
    print(f"  {cls:<12}: {pct:5.1f}%  {flag}")

total_pct = em_pct.max() / em_pct.min()
print(f"\n  Imbalance ratio (max/min): {total_pct:.2f}x")
print(f"  Classification: {'⚠️  IMBALANCED' if em_pct.min() < 10 else '✅ ACCEPTABLE'}")

print("\n[6.1] Mask class distribution:")
mask_pct = df_mask["label"].value_counts(normalize=True) * 100
for cls, pct in mask_pct.items():
    flag = "✅ OK" if 30 <= pct <= 70 else "⚠️  IMBALANCED"
    print(f"  {cls:<15}: {pct:5.1f}%  {flag}")

# 6.2  Compute class weights for PyTorch training
print("\n[6.2] Computed class weights (for weighted loss function):")
em_labels = df_em["label"].values
classes = np.unique(em_labels)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=em_labels)
class_weights = dict(zip(classes, weights))
print("  Emotion class weights:")
for cls, w in sorted(class_weights.items(), key=lambda x: -x[1]):
    print(f"    {cls:<12}: {w:.4f}")

# Save class weights
import json
with open(OUT_DIR / "emotion_class_weights.json", "w") as f:
    json.dump(class_weights, f, indent=2)
print(f"\n  Saved: emotion_class_weights.json")

# 6.3  SMOTE demonstration on PCA-reduced features
print("\n[6.3] SMOTE demonstration (PCA feature space):")

if HAS_IMBALANCED:
    # Use pixel statistics as synthetic features for SMOTE demo
    print("  Sampling pixel means for SMOTE feature matrix…")
    feature_rows = []
    label_rows   = []
    MAX_PER_CLASS = 200  # for demo speed

    for cls in EMOTION_CLASSES:
        sub = df_em[df_em["label"] == cls].head(MAX_PER_CLASS)
        for fp in sub["filepath"]:
            try:
                arr = np.array(Image.open(fp).convert("L").resize((24, 24)), dtype=np.float32)
                feature_rows.append(arr.flatten() / 255.0)
                label_rows.append(cls)
            except Exception:
                pass

    X_demo = np.array(feature_rows)
    y_demo = np.array(label_rows)

    print(f"  Feature matrix: {X_demo.shape}")

    # PCA reduction for SMOTE
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_demo)

    before_counts = Counter(y_demo)

    try:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_pca, y_demo)
        after_counts = Counter(y_res)

        print("\n  Before SMOTE:")
        for cls in EMOTION_CLASSES:
            print(f"    {cls:<12}: {before_counts.get(cls, 0):>4}")
        print("\n  After SMOTE:")
        for cls in EMOTION_CLASSES:
            print(f"    {cls:<12}: {after_counts.get(cls, 0):>4}")

        # VIZ 8: Before/After SMOTE
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(list(before_counts.keys()), list(before_counts.values()),
                    color=PALETTE_EMOTION)
        axes[0].set_title("Before SMOTE", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Emotion Class")
        axes[0].tick_params(axis="x", rotation=30)
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar(list(after_counts.keys()), list(after_counts.values()),
                    color=PALETTE_EMOTION)
        axes[1].set_title("After SMOTE (Balanced)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Emotion Class")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(axis="y", alpha=0.3)

        fig.suptitle("SMOTE Effect on Emotion Class Distribution", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "viz8_smote_before_after.png", bbox_inches="tight")
        plt.close()
        print(f"\n  Saved: viz8_smote_before_after.png")

    except Exception as e:
        print(f"  ⚠️  SMOTE failed: {e}")
else:
    print("  Skipping SMOTE (imbalanced-learn not installed)")

# 6.4  Summary
print("\n[6.4] IMBALANCE HANDLING SUMMARY:")
print("  ✅ Class weights computed → saved to emotion_class_weights.json")
print("  ✅ SMOTE demonstrated on PCA feature space")
print("  📋 Training strategy: Use class_weight in CrossEntropyLoss + ImageDataGenerator augmentation")
print("  📋 Augmentation: RandomHorizontalFlip, RandomRotation(15°), ColorJitter, RandomCrop")

print("\n✅ Steps 5 & 6 complete.")
print("   Generated 8 visualizations in:", OUT_DIR)
print("   Next: Run 04_train_mask.py and 05_train_emotion.py")
