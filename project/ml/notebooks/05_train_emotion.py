"""
Step 7 & 8: Emotion Recognition Model — EfficientNet-B0
========================================================
GPU: NVIDIA RTX 2060 (CUDA)
Run: python notebooks/05_train_emotion.py
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
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    roc_auc_score, roc_curve, auc
)

OUT_DIR   = PROJECT_ROOT / "project" / "ml" / "outputs"
MODEL_DIR = PROJECT_ROOT / "project" / "ml" / "models"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_MAP       = {cls: i for i, cls in enumerate(EMOTION_CLASSES)}
IDX_TO_LABEL    = {i: cls for cls, i in LABEL_MAP.items()}

# ── GPU check ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*60}")
print(f"EMOTION RECOGNITION MODEL TRAINING")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "img_size":       224,
    "batch_size":     64,
    "epochs":         50,
    "lr":             1e-4,
    "weight_decay":   1e-2,
    "label_smoothing": 0.1,
    "patience":       10,
    "val_split":      0.15,
    "test_split":     0.15,
    "seed":           42,
    "num_workers":    0,
    "num_classes":    7,
    "dropout":        0.4,
}

with open(MODEL_DIR / "emotion_config.json", "w") as f:
    json.dump(CFG, f, indent=2)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
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
            # Emotion images are grayscale 48×48; convert to RGB for EfficientNet
            img = Image.open(fp).convert("RGB")
        except Exception:
            img = Image.new("RGB", (CFG["img_size"], CFG["img_size"]))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────────────────────
# Load data + class weights
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(OUT_DIR / "df_all_cleaned.csv")
df_em = df[df["dataset"] == "emotion"].copy()
df_em["label_int"] = df_em["label"].map(LABEL_MAP)
df_em = df_em.dropna(subset=["label_int"])
print(f"Emotion dataset: {len(df_em)} images")
print(df_em["label"].value_counts())

with open(MODEL_DIR.parent / "outputs" / "emotion_class_weights.json") as f:
    raw_weights = json.load(f)

class_weights_tensor = torch.zeros(CFG["num_classes"], dtype=torch.float32)
for cls, w in raw_weights.items():
    idx = LABEL_MAP[cls]
    class_weights_tensor[idx] = w
class_weights_tensor = class_weights_tensor.to(device)
print(f"\nClass weights: {class_weights_tensor.cpu().numpy()}")

filepaths = df_em["filepath"].values
labels    = df_em["label_int"].values.astype(int)

# Train / Val / Test split (stratified)
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

# Weighted sampler
class_counts     = np.bincount(y_train)
sample_weights   = (1.0 / class_counts)[y_train]
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).float(),
    num_samples=len(y_train),
    replacement=True,
)

train_ds = EmotionDataset(X_train, y_train, transform=train_transform)
val_ds   = EmotionDataset(X_val,   y_val,   transform=val_transform)
test_ds  = EmotionDataset(X_test,  y_test,  transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler,
                          num_workers=CFG["num_workers"], pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model — EfficientNet-B0
# ─────────────────────────────────────────────────────────────────────────────
print("\n[MODEL] Loading EfficientNet-B0 (pretrained ImageNet)...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

# Custom classification head
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(CFG["dropout"]),
    nn.Linear(num_features, 512),
    nn.SiLU(),                      # Swish activation (EfficientNet style)
    nn.BatchNorm1d(512),
    nn.Dropout(CFG["dropout"] * 0.5),
    nn.Linear(512, CFG["num_classes"]),
)
model = model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Loss, Optimizer, Scheduler
# ─────────────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(
    label_smoothing=CFG["label_smoothing"],
)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG["lr"], weight_decay=CFG["weight_decay"]
)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=CFG["lr"],
    steps_per_epoch=len(train_loader),
    epochs=CFG["epochs"],
    pct_start=0.1,
)

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss = float("inf")
patience_counter = 0

print(f"\n[TRAINING] Starting {CFG['epochs']} epochs...\n")

for epoch in range(1, CFG["epochs"] + 1):
    # Unfreeze all layers at epoch 10
    if epoch == 11:
        print("[INFO] Unfreezing all layers (full fine-tuning)...")
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"] * 0.2,
                                weight_decay=CFG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG["epochs"] - 10
        )

    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    t0 = time.time()

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch <= 10:
            scheduler.step()
        train_loss    += loss.item() * imgs.size(0)
        preds          = logits.argmax(dim=1)
        train_correct += (preds == lbls).sum().item()
        train_total   += imgs.size(0)

    train_loss /= train_total
    train_acc   = train_correct / train_total

    # ── Validate ───────────────────────────────────────────────────────────
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            val_loss    += loss.item() * imgs.size(0)
            preds        = logits.argmax(dim=1)
            val_correct += (preds == lbls).sum().item()
            val_total   += imgs.size(0)

    val_loss /= val_total
    val_acc   = val_correct / val_total
    if epoch > 10:
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

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_DIR / "emotion_model_best.pt")
        print(f"  [OK] Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= CFG["patience"]:
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: MODEL EVALUATION - EMOTION RECOGNITION")
print("=" * 60)

model.load_state_dict(torch.load(MODEL_DIR / "emotion_model_best.pt", map_location=device))
model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
acc    = accuracy_score(all_labels, all_preds)
f1_w   = f1_score(all_labels, all_preds, average="weighted")
f1_m   = f1_score(all_labels, all_preds, average="macro")
cm     = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds,
                                target_names=EMOTION_CLASSES)
try:
    roc_auc_ovr = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
except Exception:
    roc_auc_ovr = 0.0

print(f"\nTest Accuracy   : {acc:.4f}")
print(f"F1 (weighted)   : {f1_w:.4f}")
print(f"F1 (macro)      : {f1_m:.4f}")
print(f"ROC-AUC (OvR)   : {roc_auc_ovr:.4f}")
print(f"\nClassification Report:\n{report}")

metrics = {
    "accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m,
    "roc_auc_ovr": roc_auc_ovr, "confusion_matrix": cm.tolist(),
}
with open(MODEL_DIR / "emotion_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── Plots ──────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.facecolor": "#0A0F1E", "axes.facecolor": "#111827",
                     "text.color": "#E2E8F0", "axes.labelcolor": "#E2E8F0",
                     "xtick.color": "#94A3B8", "ytick.color": "#94A3B8"})

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Emotion Recognition — Training & Evaluation", fontsize=14, fontweight="bold")

epochs_r = range(1, len(history["train_loss"]) + 1)
axes[0].plot(epochs_r, history["train_loss"], "#00FFB3", label="Train", lw=2)
axes[0].plot(epochs_r, history["val_loss"],   "#FF3366", label="Val",   lw=2)
axes[0].set_title("Loss Curves"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_r, history["train_acc"], "#00FFB3", label="Train", lw=2)
axes[1].plot(epochs_r, history["val_acc"],   "#FF3366", label="Val",   lw=2)
axes[1].set_title("Accuracy Curves"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)

sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=axes[2],
            xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES,
            linewidths=0.5, linecolor="#0A0F1E")
axes[2].set_title("Confusion Matrix")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")
axes[2].tick_params(axis="x", rotation=30)
axes[2].tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig(OUT_DIR / "viz_emotion_training_results.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: viz_emotion_training_results.png")

# ONNX export
try:
    dummy_input = torch.randn(1, 3, CFG["img_size"], CFG["img_size"]).to(device)
    torch.onnx.export(
        model, dummy_input,
        str(MODEL_DIR / "emotion_model.onnx"),
        export_params=True, opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print("Saved: emotion_model.onnx")
except Exception as e:
    print(f"ONNX export failed: {e}")

print("\n[SUCCESS] Emotion model training complete!")
