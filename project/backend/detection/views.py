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
    """
    Handle multipart image upload, run face/mask/emotion detection pipeline,
    and persist results asynchronously to SQL Server and MongoDB databases.

    Args:
        request (Request): Django REST framework HTTP request containing 'image' file in request.FILES.

    Returns:
        Response: DRF Response with status success/error, detection payload, and persisted log ID.
    """
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
    """
    Handle real-time base64 webcam frame inference requests.

    Args:
        request (Request): DRF HTTP JSON request containing 'frame' base64 string and optional 'session_id'.

    Returns:
        Response: DRF Response with status and detection predictions array.
    """
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
    """
    Query paginated detection log entries stored in SQL Server with optional filtering.

    Args:
        request (Request): DRF GET request with query params for 'page', 'page_size', 'source', 'mask', 'emotion'.

    Returns:
        Response: JSON payload with total record count, page numbers, and serialized detection logs.
    """
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
    """
    Retrieve active model version metadata, evaluation metrics, and GPU execution system status.

    Args:
        request (Request): DRF GET request.

    Returns:
        Response: JSON payload detailing active model versions, GPU device name, and PyTorch CUDA availability.
    """
    active_models = ModelVersion.objects.filter(is_active=True)
    serializer = ModelVersionSerializer(active_models, many=True)
    return Response({
        "device":      str(engine.device),
        "cuda_available": __import__("torch").cuda.is_available(),
        "models":      serializer.data,
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/detect/batch/
# ─────────────────────────────────────────────────────────────────────────────
@extend_schema(
    summary="Detect mask and emotion across a batch/folder of uploaded images",
    description="Upload multiple image files from a folder. Returns aggregated emotion breakdowns and per-file predictions.",
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def detect_batch(request):
    """
    Process a batch or folder of image files, aggregating emotion statistics and mask compliance counts.

    Args:
        request (Request): DRF POST request containing list of image files in request.FILES.getlist('images').

    Returns:
        Response: JSON payload with batch execution metrics, emotion distribution breakdown, mask compliance, and file predictions.
    """
    try:
        files = request.FILES.getlist("images") or request.FILES.getlist("files")
    except Exception as exc:
        return Response(
            {"error": f"File payload processing error: {str(exc)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not files:
        return Response({"error": "No image files provided. Use field name 'images'"},
                        status=status.HTTP_400_BAD_REQUEST)

    allowed_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    valid_files = [f for f in files if f.name.lower().endswith(allowed_exts)]

    if not valid_files:
        return Response({"error": "No valid image files (JPG, PNG, WebP) found in batch upload."},
                        status=status.HTTP_400_BAD_REQUEST)

    batch_session_id = str(uuid.uuid4())
    start_total = datetime.utcnow()

    emotion_counts = {"happy": 0, "sad": 0, "angry": 0, "disgust": 0, "fear": 0, "neutral": 0, "surprise": 0}
    mask_counts    = {"with_mask": 0, "without_mask": 0}
    total_faces    = 0
    file_results   = []

    for img_file in valid_files:
        try:
            image_bytes = img_file.read()
            result = engine.detect_from_bytes(image_bytes)

            faces_in_img = result.get("faces_detected", 0)
            total_faces += faces_in_img

            top_pred = result["predictions"][0] if result["predictions"] else {}
            top_mask    = top_pred.get("mask",    {}).get("label", "")
            top_emotion = top_pred.get("emotion", {}).get("label", "")

            # Count all face predictions in this file
            for pred in result.get("predictions", []):
                m_label = pred.get("mask", {}).get("label")
                e_label = pred.get("emotion", {}).get("label")
                if m_label in mask_counts:
                    mask_counts[m_label] += 1
                if e_label in emotion_counts:
                    emotion_counts[e_label] += 1

            file_results.append({
                "filename":       img_file.name,
                "faces_detected": faces_in_img,
                "top_mask":       top_mask,
                "top_emotion":    top_emotion,
                "predictions":    result.get("predictions", []),
                "processing_ms":  result.get("processing_time_ms", 0),
            })

            # Save summary to SQL Server
            if top_pred:
                try:
                    DetectionLog.objects.create(
                        image_name=img_file.name,
                        faces_detected=faces_in_img,
                        mask_result=top_mask,
                        mask_confidence=top_pred.get("mask", {}).get("confidence"),
                        emotion_result=top_emotion,
                        emotion_confidence=top_pred.get("emotion", {}).get("confidence"),
                        processing_time_ms=result.get("processing_time_ms"),
                        source="batch",
                    )
                except Exception:
                    pass

        except Exception as e:
            file_results.append({
                "filename": img_file.name,
                "error": str(e),
                "faces_detected": 0,
            })

    # Determine top overall dominant emotion
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if total_faces > 0 else "N/A"

    return Response({
        "status":            "success",
        "batch_session_id":  batch_session_id,
        "total_files":       len(valid_files),
        "total_faces":       total_faces,
        "dominant_emotion":  dominant_emotion,
        "emotion_counts":    emotion_counts,
        "mask_counts":       mask_counts,
        "file_results":      file_results,
    }, status=status.HTTP_200_OK)
