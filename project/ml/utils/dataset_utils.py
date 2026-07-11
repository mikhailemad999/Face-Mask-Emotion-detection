"""
dataset_utils.py — Shared utilities for Face Mask & Emotion Detection project
Handles image loading, validation, hashing and DataFrame construction
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Dataset root discovery
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parents[3]   # project root → workspace root

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MASK_CLASSES    = ["with_mask", "without_mask"]


def get_emotion_dataset_root() -> Path:
    return BASE_DIR  # each emotion class folder lives at top level


def get_mask_dataset_root() -> Path:
    return BASE_DIR


# ─────────────────────────────────────────────
# 2. Image hash (for duplicate detection)
# ─────────────────────────────────────────────

def compute_file_hash(filepath: str, algorithm: str = "md5") -> str:
    """Compute cryptographic hash of raw file bytes."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────
# 3. Image validation
# ─────────────────────────────────────────────

def validate_image(filepath: str) -> Dict:
    """
    Try to open an image and return metadata.
    Returns dict with 'valid' flag and image properties.
    """
    result = {
        "filepath": filepath,
        "valid": False,
        "width": None,
        "height": None,
        "channels": None,
        "file_size_bytes": None,
        "color_mode": None,
        "error": None,
    }

    result["file_size_bytes"] = os.path.getsize(filepath)

    try:
        with Image.open(filepath) as img:
            img.verify()   # detects corrupt files without full decode

        # Re-open after verify (verify closes the file)
        with Image.open(filepath) as img:
            img.load()
            result["width"], result["height"] = img.size
            result["color_mode"] = img.mode
            result["channels"] = len(img.getbands())
            result["valid"] = True

    except (UnidentifiedImageError, OSError, Exception) as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────
# 4. Build dataset DataFrames
# ─────────────────────────────────────────────

def build_dataset_df(
    class_folders: Dict[str, Path],
    dataset_name: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> pd.DataFrame:
    """
    Walk through class folders, validate each image,
    compute hash, and return a clean DataFrame.

    Args:
        class_folders: {label: folder_path}
        dataset_name:  'emotion' | 'mask'
        extensions:    allowed file extensions

    Returns:
        pd.DataFrame with columns:
            filepath, label, dataset, valid, width, height,
            channels, file_size_bytes, color_mode, file_hash, error
    """
    records: List[Dict] = []

    for label, folder in class_folders.items():
        if not folder.exists():
            logger.warning(f"Folder not found: {folder}")
            continue

        files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        logger.info(f"[{dataset_name}] Class '{label}': {len(files)} files found")

        for fpath in files:
            info = validate_image(str(fpath))
            info["label"]        = label
            info["dataset"]      = dataset_name
            info["filename"]     = fpath.name

            if info["valid"]:
                info["file_hash"] = compute_file_hash(str(fpath))
            else:
                info["file_hash"] = None

            records.append(info)

    df = pd.DataFrame(records)
    logger.info(f"[{dataset_name}] Total records: {len(df)}, Valid: {df['valid'].sum()}, Invalid: {(~df['valid']).sum()}")
    return df


def build_emotion_df() -> pd.DataFrame:
    """Build DataFrame for the 7-class emotion dataset."""
    root = get_emotion_dataset_root()
    class_folders = {cls: root / cls for cls in EMOTION_CLASSES}
    return build_dataset_df(class_folders, dataset_name="emotion")


def build_mask_df() -> pd.DataFrame:
    """Build DataFrame for the binary mask dataset."""
    root = get_mask_dataset_root()
    class_folders = {cls: root / cls for cls in MASK_CLASSES}
    return build_dataset_df(class_folders, dataset_name="mask")


# ─────────────────────────────────────────────
# 5. Image loading for ML (OpenCV / numpy)
# ─────────────────────────────────────────────

def load_image_cv2(
    filepath: str,
    target_size: Optional[Tuple[int, int]] = None,
    grayscale: bool = False,
) -> Optional[np.ndarray]:
    """
    Load image using OpenCV. Returns numpy array or None on failure.
    Args:
        target_size: (width, height) for resize. None = no resize.
        grayscale:   convert to single-channel grayscale.
    """
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(filepath, flag)
    if img is None:
        return None
    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def load_image_pil(
    filepath: str,
    target_size: Optional[Tuple[int, int]] = None,
    mode: str = "RGB",
) -> Optional[Image.Image]:
    """Load image using PIL and optionally resize + convert mode."""
    try:
        img = Image.open(filepath).convert(mode)
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
        return img
    except Exception:
        return None


# ─────────────────────────────────────────────
# 6. Pixel statistics per image (for outlier detection)
# ─────────────────────────────────────────────

def compute_pixel_stats(filepath: str, grayscale: bool = True) -> Dict:
    """Compute mean, std, min, max pixel values of an image."""
    img = load_image_cv2(filepath, grayscale=grayscale)
    if img is None:
        return {}
    arr = img.astype(np.float32)
    return {
        "pixel_mean": float(arr.mean()),
        "pixel_std":  float(arr.std()),
        "pixel_min":  float(arr.min()),
        "pixel_max":  float(arr.max()),
    }


if __name__ == "__main__":
    # Quick smoke test
    print("Building emotion dataset DataFrame...")
    df_emotion = build_emotion_df()
    print(df_emotion[["label", "valid", "width", "height", "file_size_bytes"]].groupby("label").describe())

    print("\nBuilding mask dataset DataFrame...")
    df_mask = build_mask_df()
    print(df_mask[["label", "valid", "width", "height", "file_size_bytes"]].groupby("label").describe())
