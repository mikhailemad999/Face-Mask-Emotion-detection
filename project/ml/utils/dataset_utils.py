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
    """
    Get the filesystem path pointing to the top-level directory containing emotion class folders.

    Returns:
        Path: Absolute pathlib.Path to the workspace root directory.
    """
    return BASE_DIR  # each emotion class folder lives at top level


def get_mask_dataset_root() -> Path:
    """
    Get the filesystem path pointing to the top-level directory containing mask class folders.

    Returns:
        Path: Absolute pathlib.Path to the workspace root directory.
    """
    return BASE_DIR


# ─────────────────────────────────────────────
# 2. Image hash (for duplicate detection)
# ─────────────────────────────────────────────

def compute_file_hash(filepath: str, algorithm: str = "md5") -> str:
    """
    Compute cryptographic hash of raw file bytes to identify duplicate dataset images.

    Args:
        filepath (str): Absolute or relative filesystem path to the target file.
        algorithm (str): Hash algorithm identifier (default: "md5").

    Returns:
        str: Hexadecimal digest of the file hash.
    """
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
    Validate image file integrity, color mode, dimensions, and file size.

    Args:
        filepath (str): Path to image file.

    Returns:
        Dict: Metadata dictionary containing validity flag, width, height, channel count, file size, color mode, and error string.
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
    Walk through specified class directories, validate images, compute hashes, and build a pandas DataFrame.

    Args:
        class_folders (Dict[str, Path]): Map of class labels to folder paths.
        dataset_name (str): Identifier name for the dataset ('emotion' or 'mask').
        extensions (Tuple[str, ...]): Permitted image file extensions.

    Returns:
        pd.DataFrame: Structured DataFrame containing image metadata, validity flags, and hashes.
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
    """
    Build structured pandas DataFrame for the 7-class emotion recognition dataset.

    Returns:
        pd.DataFrame: Emotion dataset metadata DataFrame.
    """
    root = get_emotion_dataset_root()
    class_folders = {cls: root / cls for cls in EMOTION_CLASSES}
    return build_dataset_df(class_folders, dataset_name="emotion")


def build_mask_df() -> pd.DataFrame:
    """
    Build structured pandas DataFrame for the binary face mask dataset.

    Returns:
        pd.DataFrame: Mask dataset metadata DataFrame.
    """
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
    Load image using OpenCV, with optional spatial resizing and grayscale conversion.

    Args:
        filepath (str): Path to image file.
        target_size (Optional[Tuple[int, int]]): Target (width, height) tuple for resizing.
        grayscale (bool): Whether to load image as single-channel grayscale.

    Returns:
        Optional[np.ndarray]: Loaded OpenCV image numpy array, or None if unreadable.
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
    """
    Load image using PIL Image, with optional spatial resizing and mode conversion.

    Args:
        filepath (str): Path to image file.
        target_size (Optional[Tuple[int, int]]): Target (width, height) tuple for resizing.
        mode (str): PIL color mode (default: "RGB").

    Returns:
        Optional[Image.Image]: PIL Image object, or None if unreadable.
    """
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
    """
    Compute mean, standard deviation, minimum, and maximum pixel intensity values for an image.

    Args:
        filepath (str): Path to image file.
        grayscale (bool): Convert image to grayscale prior to statistic computation.

    Returns:
        Dict: Dictionary containing 'pixel_mean', 'pixel_std', 'pixel_min', and 'pixel_max'.
    """
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
