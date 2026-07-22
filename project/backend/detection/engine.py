"""
Detection Engine — Face Mask + Emotion Inference Pipeline
Uses PyTorch models loaded at startup (singleton pattern)
Supports: image file, numpy array, base64 encoded frame
"""

import io
import base64
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from django.conf import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton model loader
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"[DetectionEngine] Running on: {device}")


def _build_mask_model() -> nn.Module:
    """
    Rebuild MobileNetV2 architecture for binary face mask classification.
    
    Returns:
        nn.Module: PyTorch MobileNetV2 model initialized with custom classifier head.
    """
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    )
    return model


def _build_emotion_model() -> nn.Module:
    """
    Rebuild EfficientNet-B0 architecture for 7-class facial emotion recognition.
    
    Returns:
        nn.Module: PyTorch EfficientNet-B0 model initialized with custom 7-class linear classifier.
    """
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, 7),
    )
    return model


class DetectionEngine:
    """
    Singleton detection pipeline executing end-to-end inference:
    1. Face detection via multi-cascade OpenCV Haar Cascades
    2. Mask status classification via MobileNetV2
    3. Facial emotion recognition via EfficientNet-B0
    """

    _instance = None

    def __new__(cls):
        """
        Enforce single instance creation across application threads.
        
        Returns:
            DetectionEngine: Singleton instance of the engine.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the detection engine singleton, loading PyTorch models and OpenCV detectors."""
        if self._initialized:
            return
        self._initialized = True
        self.device = device
        self._load_models()
        self._load_face_detector()

    def _load_models(self):
        """
        Load trained PyTorch weights for mask and emotion models from configured disk paths.
        Falls back gracefully with warnings if model files are missing.
        """
        logger.info("[DetectionEngine] Loading ML models…")

        # Mask model
        self.mask_model = _build_mask_model().to(device)
        mask_path = settings.MASK_MODEL_PATH
        if Path(mask_path).exists():
            self.mask_model.load_state_dict(
                torch.load(mask_path, map_location=device, weights_only=True)
            )
            logger.info(f"  ✅ Mask model loaded from {mask_path}")
        else:
            logger.warning(f"  ⚠️  Mask model not found at {mask_path} — running in demo mode")
        self.mask_model.eval()

        # Emotion model
        self.emotion_model = _build_emotion_model().to(device)
        emotion_path = settings.EMOTION_MODEL_PATH
        if Path(emotion_path).exists():
            self.emotion_model.load_state_dict(
                torch.load(emotion_path, map_location=device, weights_only=True)
            )
            logger.info(f"  ✅ Emotion model loaded from {emotion_path}")
        else:
            logger.warning(f"  ⚠️  Emotion model not found at {emotion_path} — running in demo mode")
        self.emotion_model.eval()

    def _load_face_detector(self):
        """
        Load OpenCV Haar Cascade face detectors including default, alt2, and profile cascades
        to ensure robust face localization across varying lighting and angles.
        """
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        alt_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self.face_cascade_alt = cv2.CascadeClassifier(alt_path)
        
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
        self.face_cascade_profile = cv2.CascadeClassifier(profile_path)
        
        logger.info(f"  ✅ Face detectors loaded (Default, Alt2, Profile)")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect_from_pil(self, pil_image: Image.Image) -> Dict:
        """
        Run face detection, mask classification, and emotion recognition on a PIL Image instance.

        Args:
            pil_image (Image.Image): Input PIL Image object.

        Returns:
            Dict: Inference results containing detected face count, bounding boxes, predictions, processing time, and execution device.
        """
        start = time.perf_counter()
        img_rgb = np.array(pil_image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        faces = self._detect_faces(img_bgr)
        results = []

        for (x, y, w, h) in faces:
            face_img = pil_image.crop((x, y, x + w, y + h))
            mask_result    = self._classify_mask(face_img)
            emotion_result = self._classify_emotion(face_img)
            results.append({
                "bbox":    {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "mask":    mask_result,
                "emotion": emotion_result,
            })

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "faces_detected":   len(faces),
            "predictions":       results,
            "processing_time_ms": round(elapsed_ms, 2),
            "device":            str(device),
        }

    def detect_from_bytes(self, image_bytes: bytes) -> Dict:
        """
        Run inference pipeline on raw image byte data (e.g. from HTTP file uploads).

        Args:
            image_bytes (bytes): Binary contents of an uploaded image file.

        Returns:
            Dict: Inference result dictionary containing predictions and performance metrics.
        """
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.detect_from_pil(pil_image)

    def detect_from_base64(self, b64_string: str) -> Dict:
        """
        Run inference pipeline on base64-encoded image string (e.g. from webcam canvas data URLs).

        Args:
            b64_string (str): Base64-encoded image string (with optional data header).

        Returns:
            Dict: Inference result dictionary.
        """
        if b64_string.startswith("data:image"):
            b64_string = b64_string.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_string)
        return self.detect_from_bytes(raw_bytes)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_faces(self, img_bgr: np.ndarray) -> list:
        """
        Detect face bounding boxes using layered multi-cascade classifiers with adaptive parameter fallbacks.

        Args:
            img_bgr (np.ndarray): Input BGR OpenCV image array.

        Returns:
            list: List of bounding box tuples `(x, y, w, h)`.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Try 1: Default cascade (strict, high precision)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            return list(faces)
            
        # Try 2: Alternative cascade (better for tilted/partially occluded faces)
        faces = self.face_cascade_alt.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            return list(faces)
            
        # Try 3: Side/profile face detector
        faces = self.face_cascade_profile.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            return list(faces)

        # Try 4: Adaptive relaxation (highly sensitive, allows smaller/dimmer faces)
        faces = self.face_cascade_alt.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            return list(faces)

        return []

    @torch.no_grad()
    def _classify_mask(self, face_pil: Image.Image) -> Dict:
        """
        Perform binary mask classification on a cropped face image.

        Args:
            face_pil (Image.Image): Cropped PIL image of a detected face.

        Returns:
            Dict: Classification result with predicted label ('with_mask' / 'without_mask') and confidence probabilities.
        """
        tensor = _transform(face_pil.convert("RGB")).unsqueeze(0).to(device)
        logit  = self.mask_model(tensor).squeeze()
        prob   = torch.sigmoid(logit).item()
        label  = settings.MASK_CLASSES[1 if prob > 0.5 else 0]
        return {
            "label":      label,
            "confidence": round(prob if prob > 0.5 else 1 - prob, 4),
            "with_mask_prob":    round(prob, 4),
            "without_mask_prob": round(1 - prob, 4),
        }

    @torch.no_grad()
    def _classify_emotion(self, face_pil: Image.Image) -> Dict:
        """
        Perform 7-class emotion recognition on a cropped face image.

        Args:
            face_pil (Image.Image): Cropped PIL image of a detected face.

        Returns:
            Dict: Classification result with top predicted emotion label, confidence score, and probability distribution across all 7 emotions.
        """
        # Convert to grayscale first, then RGB to match the training data distribution (FER2013-like)
        tensor = _transform(face_pil.convert("L").convert("RGB")).unsqueeze(0).to(device)
        logits = self.emotion_model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        top_idx   = int(probs.argmax())
        top_label = settings.EMOTION_CLASSES[top_idx]
        all_probs = {cls: round(float(p), 4)
                     for cls, p in zip(settings.EMOTION_CLASSES, probs)}
        return {
            "label":      top_label,
            "confidence": round(float(probs[top_idx]), 4),
            "all_probs":  all_probs,
        }


# Module-level singleton
engine = DetectionEngine()
