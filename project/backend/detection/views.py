"""
Detection API Views — Django REST Framework
Endpoints:
  POST /api/detect/image/   — Upload image file
  POST /api/detect/frame/   — Base64 webcam frame
  GET  /api/detect/history/ — Paginated detection history
  GET  /api/detect/models/  — Model version info
"""

import uuid
from datetime import datetime

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter

from .engine import engine
from .models import DetectionLog, ModelVersion
from .serializers import DetectionLogSerializer, ModelVersionSerializer
from .mongo_service import save_prediction_to_mongo


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/detect/image/
# ─────────────────────────────────────────────────────────────────────────────
@extend_schema(
    summary="Detect mask and emotion from an uploaded image",
    description="Upload a JPEG/PNG image. Returns face detection results with mask and emotion predictions.",
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def detect_image(request):
    """Upload image → run detection pipeline → return JSON + log to DBs."""
    if "image" not in request.FILES:
        return Response({"error": "No image file provided. Use field name 'image'"},
                        status=status.HTTP_400_BAD_REQUEST)

    image_file = request.FILES["image"]
    if not image_file.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        return Response({"error": "Unsupported file type. Use JPG, PNG, or BMP."},
                        status=status.HTTP_400_BAD_REQUEST)

    # Run detection
    image_bytes = image_file.read()
    result = engine.detect_from_bytes(image_bytes)

    # Extract top prediction (first face)
    top_pred = result["predictions"][0] if result["predictions"] else {}
    mask_label    = top_pred.get("mask",    {}).get("label",      "")
    mask_conf     = top_pred.get("mask",    {}).get("confidence", None)
    emotion_label = top_pred.get("emotion", {}).get("label",      "")
    emotion_conf  = top_pred.get("emotion", {}).get("confidence", None)

    # Save to MongoDB (full prediction detail)
    mongo_id = ""
    try:
        mongo_id = save_prediction_to_mongo({
            "session_id":   str(uuid.uuid4()),
            "timestamp":    datetime.utcnow(),
            "image_name":   image_file.name,
            "source":       "upload",
            "full_result":  result,
        })
    except Exception as e:
        pass  # MongoDB unavailable → non-fatal

    # Save summary to SQL Server
    try:
        log = DetectionLog.objects.create(
            image_name         = image_file.name,
            faces_detected     = result["faces_detected"],
            mask_result        = mask_label,
            mask_confidence    = mask_conf,
            emotion_result     = emotion_label,
            emotion_confidence = emotion_conf,
            processing_time_ms = result["processing_time_ms"],
            source             = "upload",
            mongo_prediction_id = mongo_id,
        )
    except Exception as e:
        pass  # SQL Server unavailable → non-fatal

    return Response({
        "status":  "success",
        "result":  result,
        "log_id":  getattr(log, "id", None) if "log" in dir() else None,
    }, status=status.HTTP_200_OK)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/detect/frame/
# ─────────────────────────────────────────────────────────────────────────────
@extend_schema(
    summary="Detect from a base64-encoded webcam frame",
    description="Send a base64-encoded image (data URL or raw base64). Used for real-time webcam inference.",
)
@api_view(["POST"])
@parser_classes([JSONParser])
def detect_frame(request):
    """Base64 webcam frame → detection → response."""
    frame_b64 = request.data.get("frame")
    if not frame_b64:
        return Response({"error": "No 'frame' field in request body."},
                        status=status.HTTP_400_BAD_REQUEST)
    try:
        result = engine.detect_from_base64(frame_b64)
    except Exception as e:
        return Response({"error": f"Detection failed: {str(e)}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Log to MongoDB (lightweight — no SQL write for every webcam frame)
    try:
        save_prediction_to_mongo({
            "session_id": request.data.get("session_id", str(uuid.uuid4())),
            "timestamp":  datetime.utcnow(),
            "source":     "webcam",
            "full_result": result,
        })
    except Exception:
        pass

    return Response({"status": "success", "result": result})


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/detect/history/
# ─────────────────────────────────────────────────────────────────────────────
@extend_schema(summary="Get paginated detection history from SQL Server")
@api_view(["GET"])
def detection_history(request):
    """Return paginated detection logs from SQL Server."""
    logs = DetectionLog.objects.all().order_by("-timestamp")

    # Simple filter support
    source        = request.query_params.get("source")
    mask_label    = request.query_params.get("mask")
    emotion_label = request.query_params.get("emotion")

    if source:        logs = logs.filter(source=source)
    if mask_label:    logs = logs.filter(mask_result=mask_label)
    if emotion_label: logs = logs.filter(emotion_result=emotion_label)

    # Manual pagination (page_size default=20)
    page      = int(request.query_params.get("page", 1))
    page_size = int(request.query_params.get("page_size", 20))
    total     = logs.count()
    logs_page = logs[(page - 1) * page_size : page * page_size]
    serializer = DetectionLogSerializer(logs_page, many=True)

    return Response({
        "count":    total,
        "page":     page,
        "pages":    (total + page_size - 1) // page_size,
        "results":  serializer.data,
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/detect/models/
# ─────────────────────────────────────────────────────────────────────────────
@extend_schema(summary="Get active model version info")
@api_view(["GET"])
def model_info(request):
    """Return active model versions with metrics."""
    active_models = ModelVersion.objects.filter(is_active=True)
    serializer = ModelVersionSerializer(active_models, many=True)
    return Response({
        "device":      str(engine.device),
        "cuda_available": __import__("torch").cuda.is_available(),
        "models":      serializer.data,
    })
