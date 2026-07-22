"""
Step 7: Mask Detection Model — MobileNetV2 Fine-tuning
=======================================================
GPU: NVIDIA RTX 2060 (CUDA)
Run: python notebooks/04_train_mask.py
"""

import sys
import io
import json
import time
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score
)

OUT_DIR   = PROJECT_ROOT / "project" / "ml" / "outputs"
MODEL_DIR = PROJECT_ROOT / "project" / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── GPU check ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*60}")
print(f"MASK DETECTION MODEL TRAINING")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "img_size":      224,
    "batch_size":    32,
    "epochs":        30,
    "lr":            1e-4,
    "weight_decay":  1e-4,
    "patience":      7,       # early stopping
    "val_split":     0.15,
    "test_split":    0.15,
    "seed":          42,
    "num_workers":   0,
    "label_map":     {"with_mask": 1, "without_mask": 0},
}

with open(MODEL_DIR / "mask_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class MaskDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        fp    = self.filepaths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            img = Image.new("RGB", (CFG["img_size"], CFG["img_size"]))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms (augmentation for training)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT_DIR / "df_all_cleaned.csv")
df_mask = df[df["dataset"] == "mask"].copy()
df_mask["label_int"] = df_mask["label"].map(CFG["label_map"])
df_mask = df_mask.dropna(subset=["label_int"])
print(f"Mask dataset: {len(df_mask)} images")
print(df_mask["label"].value_counts())

raw_filepaths = df_mask["filepath"].values
labels        = df_mask["label_int"].values.astype(int)
filenames     = df_mask["filename"].values
label_names   = df_mask["label"].values

resolved_filepaths = []
missing_count = 0
for i, fp_str in enumerate(raw_filepaths):
    fp = Path(fp_str)
    if not fp.exists():
        alternative_path = PROJECT_ROOT / label_names[i] / filenames[i]
        if alternative_path.exists():
            fp = alternative_path
        else:
            missing_count += 1
    resolved_filepaths.append(str(fp))

if missing_count > 0:
    print(f"[WARNING] {missing_count}/{len(raw_filepaths)} mask images were not found on disk.")

filepaths = np.array(resolved_filepaths)

# Train/Val/Test split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    filepaths, labels,
    test_size=CFG["test_split"],
    random_state=CFG["seed"],
    stratify=labels,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=CFG["val_split"] / (1 - CFG["test_split"]),
    random_state=CFG["seed"],
    stratify=y_trainval,
)

print(f"\nSplit - Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Weighted sampler for imbalanced training set
class_counts = np.bincount(y_train)
class_weights_sampler = 1.0 / class_counts
sample_weights = class_weights_sampler[y_train]
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).float(),
    num_samples=len(y_train),
    replacement=True,
)

train_ds = MaskDataset(X_train, y_train, transform=train_transform)
val_ds   = MaskDataset(X_val,   y_val,   transform=val_transform)
test_ds  = MaskDataset(X_test,  y_test,  transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                          num_workers=CFG["num_workers"], pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model — MobileNetV2 fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
print("\n[MODEL] Loading MobileNetV2 (pretrained ImageNet)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze all backbone layers first (feature extraction phase)
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier head for binary classification
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(256, 1),  # binary: with_mask vs without_mask
)
model = model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Loss, Optimizer, Scheduler
# ─────────────────────────────────────────────────────────────────────────────
pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG["lr"], weight_decay=CFG["weight_decay"]
)
scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss = float("inf")
patience_counter = 0

print(f"\n[TRAINING] Starting {CFG['epochs']} epochs on {device}...\n")

for epoch in range(1, CFG["epochs"] + 1):
    # Phase 1: unfreeze all layers after epoch 5
    if epoch == 6:
        print("[INFO] Unfreezing all backbone layers (fine-tuning mode)...")
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=CFG["lr"] * 0.1,
                               weight_decay=CFG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"] - 5)

    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    t0 = time.time()

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss   = criterion(logits, lbls)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * imgs.size(0)
        preds          = (torch.sigmoid(logits) > 0.5).long()
        train_correct += (preds == lbls.long()).sum().item()
        train_total   += imgs.size(0)

    train_loss /= train_total
    train_acc   = train_correct / train_total

    # ── Validate ───────────────────────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, lbls)
            val_loss    += loss.item() * imgs.size(0)
            preds        = (torch.sigmoid(logits) > 0.5).long()
            val_correct += (preds == lbls.long()).sum().item()
            val_total   += imgs.size(0)

    val_loss /= val_total
    val_acc   = val_correct / val_total
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - t0
    lr_now  = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch {epoch:02d}/{CFG['epochs']} | "
        f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
        f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
        f"LR: {lr_now:.2e} | {elapsed:.1f}s"
    )

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_DIR / "mask_model_best.pt")
        print(f"  [OK] Best model saved (val_loss={val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= CFG["patience"]:
            print(f"\n[INFO] Early stopping at epoch {epoch} (patience={CFG['patience']})")
            break

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: MODEL EVALUATION — MASK DETECTION")
print("=" * 60)

# Load best model
model.load_state_dict(torch.load(MODEL_DIR / "mask_model_best.pt", map_location=device))
model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy().astype(int))

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
acc     = accuracy_score(all_labels, all_preds)
f1      = f1_score(all_labels, all_preds, average="weighted")
roc_auc = roc_auc_score(all_labels, all_probs)
cm      = confusion_matrix(all_labels, all_preds)
report  = classification_report(all_labels, all_preds,
                                 target_names=["without_mask", "with_mask"])

print(f"\nTest Accuracy : {acc:.4f}")
print(f"F1 (weighted) : {f1:.4f}")
print(f"ROC-AUC       : {roc_auc:.4f}")
print(f"\nClassification Report:\n{report}")
print(f"\nConfusion Matrix:\n{cm}")

# Save metrics
metrics = {
    "accuracy": acc, "f1_weighted": f1, "roc_auc": roc_auc,
    "confusion_matrix": cm.tolist(),
}
with open(MODEL_DIR / "mask_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── Plots ──────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0A0F1E", "axes.facecolor": "#111827",
                     "text.color": "#E2E8F0", "axes.labelcolor": "#E2E8F0",
                     "xtick.color": "#94A3B8", "ytick.color": "#94A3B8",
                     "axes.edgecolor": "#1E293B", "grid.color": "#1E293B"})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Mask Detection — Training Results & Evaluation", fontsize=14, fontweight="bold")

# Learning curves
epochs_range = range(1, len(history["train_loss"]) + 1)
axes[0].plot(epochs_range, history["train_loss"], color="#00FFB3", label="Train Loss", linewidth=2)
axes[0].plot(epochs_range, history["val_loss"],   color="#FF3366", label="Val Loss",   linewidth=2)
axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], color="#00FFB3", label="Train Acc", linewidth=2)
axes[1].plot(epochs_range, history["val_acc"],   color="#FF3366", label="Val Acc",   linewidth=2)
axes[1].set_title("Accuracy Curves")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)

# Confusion matrix
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["without_mask", "with_mask"],
            yticklabels=["without_mask", "with_mask"],
            linewidths=1, linecolor="#0A0F1E")
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(OUT_DIR / "viz_mask_training_results.png", bbox_inches="tight", dpi=120)
plt.close()
print(f"\nSaved: viz_mask_training_results.png")

# ONNX export
print("\n[EXPORT] Exporting model to ONNX...")
try:
    import torch.onnx
    dummy_input = torch.randn(1, 3, CFG["img_size"], CFG["img_size"]).to(device)
    torch.onnx.export(
        model, dummy_input,
        str(MODEL_DIR / "mask_model.onnx"),
        export_params=True, opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"  Saved: mask_model.onnx")
except Exception as e:
    print(f"  [WARNING] ONNX export failed: {e}")

print("\n[SUCCESS] Mask model training complete!")
print(f"   Best model: {MODEL_DIR / 'mask_model_best.pt'}")
