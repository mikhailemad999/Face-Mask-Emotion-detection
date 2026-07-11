"""
Step 3 & 4: Missing Values and Outlier Detection
================================================
Face Mask & Emotion Detection — Graduation Project
Run: python notebooks/02_missing_outliers.py
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
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

OUT_DIR = PROJECT_ROOT / "project" / "ml" / "outputs"

# ── Load clean DataFrames from Step 1-2 ─────────────────────────────────────
df_em   = pd.read_csv(OUT_DIR / "df_emotion_clean.csv")
df_mask = pd.read_csv(OUT_DIR / "df_mask_clean.csv")
df_all  = pd.concat([df_em, df_mask], ignore_index=True)

print("=" * 70)
print("STEP 3: HANDLE MISSING VALUES")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Detect missing values per column (%)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3.1] Missing values per column (%):")
miss = (df_all.isnull().sum() / len(df_all) * 100).round(2)
miss_report = miss[miss > 0].sort_values(ascending=False)
if len(miss_report) == 0:
    print("  ✅ No missing values in any column.")
else:
    print(miss_report.to_string())

# The 'error' column may have values only for invalid rows
# 'file_hash' is None for invalid images — already handled by dropping invalids
invalid_rows = df_all[df_all["valid"] == False]
print(f"\n  Invalid rows with error info: {len(invalid_rows)}")
if len(invalid_rows) > 0:
    pct = len(invalid_rows) / len(df_all) * 100
    print(f"  Invalid percentage: {pct:.2f}%")
    if pct < 5:
        print("  → Action: DROP (< 5% missing threshold)")
        df_all = df_all[df_all["valid"] == True].copy()
        df_em  = df_em[df_em["valid"] == True].copy()
        df_mask = df_mask[df_mask["valid"] == True].copy()
        print(f"  After drop — Total: {len(df_all)}")
    else:
        print("  → Action: INVESTIGATE (> 5% — too many invalids!)")

# ─────────────────────────────────────────────────────────────────────────────
# 3.2  Numeric feature imputation check
# ─────────────────────────────────────────────────────────────────────────────
numeric_cols = ["width", "height", "file_size_bytes"]
print("\n[3.2] Numeric feature missing counts after cleaning:")
print(df_all[numeric_cols].isnull().sum())
# These should all be 0 since we only kept valid images

print("\n✅ Step 3 complete — All missing values handled.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: DETECT AND HANDLE OUTLIERS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 4.1  Z-Score outlier detection on file_size_bytes
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4.1] Z-Score outlier detection (threshold = 3):")
for ds_name, sub in df_all.groupby("dataset"):
    z = np.abs(stats.zscore(sub["file_size_bytes"].dropna()))
    n_out = (z > 3).sum()
    print(f"  {ds_name}: {n_out} outliers in file_size_bytes")

# 4.2  IQR method
print("\n[4.2] IQR outlier detection (1.5 × IQR):")
outlier_flags = {}
for ds_name, sub in df_all.groupby("dataset"):
    for col in ["width", "height", "file_size_bytes"]:
        Q1 = sub[col].quantile(0.25)
        Q3 = sub[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((sub[col] < lower) | (sub[col] > upper)).sum()
        print(f"  {ds_name} | {col}: {n_out} outliers  [range: {lower:.0f} – {upper:.0f}]")

# 4.3  Isolation Forest on {file_size, width, height}
print("\n[4.3] Isolation Forest outlier detection:")
for ds_name, sub in df_all.groupby("dataset"):
    feat_cols = ["file_size_bytes", "width", "height"]
    X = sub[feat_cols].dropna()
    if len(X) < 10:
        continue
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(X)
    n_anomalies = (preds == -1).sum()
    pct = n_anomalies / len(X) * 100
    print(f"  {ds_name}: {n_anomalies} anomalies ({pct:.2f}%)")
    outlier_flags[ds_name] = pd.Series(preds == -1, index=X.index)

# ─────────────────────────────────────────────────────────────────────────────
# 4.4  Handle outliers — retain <5%, cap extreme file sizes
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4.4] Applying outlier handling strategy:")

# For images: log-transform file_size for analysis; remove only truly extreme
for ds_name in ["emotion", "mask"]:
    mask_ds = df_all["dataset"] == ds_name
    sub = df_all[mask_ds].copy()
    total = len(sub)

    # Identify extreme outliers using tight IQR on file size
    Q1 = sub["file_size_bytes"].quantile(0.01)
    Q3 = sub["file_size_bytes"].quantile(0.99)
    extreme_low  = sub["file_size_bytes"] < Q1 * 0.1
    extreme_high = sub["file_size_bytes"] > Q3 * 10
    extreme = extreme_low | extreme_high
    n_extreme = extreme.sum()
    pct = n_extreme / total * 100
    print(f"  {ds_name}: {n_extreme} extreme outliers ({pct:.2f}%) → {'REMOVE' if pct < 5 else 'KEEP (>5%)'}")

    if pct < 5:
        df_all.drop(df_all[mask_ds][extreme].index, inplace=True)

# Log-transform file_size for use in modeling
df_all["log_file_size"] = np.log1p(df_all["file_size_bytes"])
print("  Added log_file_size feature (log1p transform).")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS — Boxplots (Step 5 contribution)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[VIZ] Generating outlier boxplots…")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("File Size and Dimension Outlier Analysis", fontsize=14, fontweight="bold")

colors_map = {
    "emotion": "#7B61FF",
    "mask":    "#00FFB3",
}

for ax, col, label in zip(
    axes,
    ["file_size_bytes", "width", "height"],
    ["File Size (bytes)", "Image Width (px)", "Image Height (px)"],
):
    data = [
        df_all[df_all["dataset"] == ds][col].dropna().values
        for ds in ["emotion", "mask"]
    ]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=["emotion", "mask"],
        notch=False,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], ["#7B61FF", "#00FFB3"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(label)
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "viz_outlier_boxplots.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved: viz_outlier_boxplots.png")

# File size KDE per class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (ds_name, sub) in zip(axes, df_all.groupby("dataset")):
    for cls in sub["label"].unique():
        cls_data = sub[sub["label"] == cls]["file_size_bytes"].dropna()
        if len(cls_data) > 10:
            sns.kdeplot(cls_data, ax=ax, label=cls, fill=True, alpha=0.25)
    ax.set_title(f"{ds_name.capitalize()} — File Size Distribution by Class")
    ax.set_xlabel("File Size (bytes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "viz_file_size_kde.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved: viz_file_size_kde.png")

# Save updated DataFrame
df_all.to_csv(OUT_DIR / "df_all_cleaned.csv", index=False)
print(f"\n[SAVED] Cleaned combined DataFrame → {OUT_DIR / 'df_all_cleaned.csv'}")
print("\n✅ Steps 3 & 4 complete. Next: Run 03_visualization_balance.py")
