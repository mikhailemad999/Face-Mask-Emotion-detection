import sys
import io
import json
from pathlib import Path

# Fix Windows console encoding for emoji/unicode output
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "project" / "ml"))

OUT_DIR   = PROJECT_ROOT / "project" / "ml" / "outputs"
MODEL_DIR = PROJECT_ROOT / "project" / "ml" / "models"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_MAP       = {cls: i for i, cls in enumerate(EMOTION_CLASSES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation running on: {device}")

CFG = {
    "img_size":       224,
    "batch_size":     64,
    "num_classes":    7,
    "dropout":        0.4,
    "test_split":     0.15,
    "seed":           42,
}

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
            img = Image.open(fp).convert("RGB")
        except Exception:
            img = Image.new("RGB", (CFG["img_size"], CFG["img_size"]))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((CFG["img_size"], CFG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Load data
def main():
    csv_path = OUT_DIR / "df_all_cleaned.csv"
    if not csv_path.exists():
        print(f"[ERROR] Cleaned dataset CSV not found at: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df_em = df[df["dataset"] == "emotion"].copy()
    df_em["label_int"] = df_em["label"].map(LABEL_MAP)
    df_em = df_em.dropna(subset=["label_int"])
    print(f"Emotion dataset: {len(df_em)} images")

    raw_filepaths = df_em["filepath"].values
    labels        = df_em["label_int"].values.astype(int)
    filenames     = df_em["filename"].values
    label_names   = df_em["label"].values

    # Resolve filepaths dynamically relative to PROJECT_ROOT if the absolute path doesn't exist
    resolved_filepaths = []
    missing_count = 0
    for i, fp_str in enumerate(raw_filepaths):
        fp = Path(fp_str)
        if not fp.exists():
            # Try to resolve relative to project root
            alternative_path = PROJECT_ROOT / label_names[i] / filenames[i]
            if alternative_path.exists():
                fp = alternative_path
            else:
                missing_count += 1
        resolved_filepaths.append(str(fp))
    
    if missing_count > 0:
        print(f"[WARNING] {missing_count}/{len(raw_filepaths)} images were not found on disk.")

    resolved_filepaths = np.array(resolved_filepaths)

    # Split
    _, X_test, _, y_test = train_test_split(
        resolved_filepaths, labels,
        test_size=CFG["test_split"],
        random_state=CFG["seed"],
        stratify=labels,
    )
    print(f"Test split: {len(X_test)}")

    test_ds  = EmotionDataset(X_test,  y_test,  transform=val_transform)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # Build model
    model_path = MODEL_DIR / "emotion_model_best.pt"
    if not model_path.exists():
        print(f"[ERROR] Best model weights file not found at: {model_path}")
        print("Please train the model first by running: python project/ml/notebooks/05_train_emotion.py")
        return

    print("Loading model...")
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(CFG["dropout"]),
        nn.Linear(num_features, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(CFG["dropout"] * 0.5),
        nn.Linear(512, CFG["num_classes"]),
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
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
    report = classification_report(all_labels, all_preds, target_names=EMOTION_CLASSES)
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

    # Plots
    plt.rcParams.update({"figure.facecolor": "#0A0F1E", "axes.facecolor": "#111827",
                         "text.color": "#E2E8F0", "axes.labelcolor": "#E2E8F0",
                         "xtick.color": "#94A3B8", "ytick.color": "#94A3B8"})

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Emotion Recognition — Confusion Matrix", fontsize=14, fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax,
                xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES,
                linewidths=0.5, linecolor="#0A0F1E")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_emotion_training_results.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: viz_emotion_training_results.png")

    # Export to ONNX
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

    print("\n[SUCCESS] Emotion model evaluation & export complete!")

if __name__ == "__main__":
    main()
