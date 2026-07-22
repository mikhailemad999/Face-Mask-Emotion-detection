from django.urls import path
from django.db.models import Avg, Count
from rest_framework.decorators import api_view
from rest_framework.response import Response
from detection.models import DetectionLog, ModelVersion
from detection.mongo_service import get_analytics_summary


@api_view(["GET"])
def analytics_summary(request):
    """
    Return comprehensive aggregated detection statistics, performance metrics, and class distributions.

    Args:
        request (Request): DRF GET HTTP request.

    Returns:
        Response: DRF Response with statistical analytics dashboard data payload.
    """
    # SQL Server aggregation
    total_logs = DetectionLog.objects.count()
    avg_latency = DetectionLog.objects.aggregate(avg=Avg("processing_time_ms"))["avg"] or 0.0

    # Mask distribution from SQL
    mask_qs = DetectionLog.objects.values("mask_result").annotate(count=Count("id")).order_by()
    mask_counts = {item["mask_result"]: item["count"] for item in mask_qs if item["mask_result"]}

    # Emotion distribution from SQL
    emotion_qs = DetectionLog.objects.values("emotion_result").annotate(count=Count("id")).order_by()
    emotion_counts = {item["emotion_result"]: item["count"] for item in emotion_qs if item["emotion_result"]}

    # Source breakdown from SQL
    source_qs = DetectionLog.objects.values("source").annotate(count=Count("id")).order_by()
    source_counts = {item["source"]: item["count"] for item in source_qs if item["source"]}

    # Dominant emotion & mask compliance
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "N/A"
    with_mask_cnt    = mask_counts.get("with_mask", 0)
    without_mask_cnt = mask_counts.get("without_mask", 0)
    total_mask_faces = with_mask_cnt + without_mask_cnt
    compliance_rate  = round((with_mask_cnt / total_mask_faces * 100), 1) if total_mask_faces > 0 else 0.0

    # MongoDB fallback / extended analytics
    mongo_data = get_analytics_summary()

    # Active model versions
    active_models = list(ModelVersion.objects.filter(is_active=True).values("model_type", "version", "accuracy", "f1_score", "roc_auc"))

    return Response({
        "status":               "success",
        "total_predictions":     total_logs or mongo_data.get("total_predictions", 0),
        "last_24h":              mongo_data.get("last_24h", 0),
        "avg_processing_ms":     round(avg_latency, 1),
        "mask_compliance_rate":  compliance_rate,
        "dominant_emotion":      dominant_emotion,
        "mask_counts":           mask_counts,
        "emotion_counts":        emotion_counts,
        "source_counts":         source_counts,
        "active_models":         active_models,
        "dataset_statistics": {
            "total_images":   18681,
            "emotion_images": 11388,
            "mask_images":    7293,
            "classes_count":  9, # 7 emotions + 2 mask status
        },
    })


urlpatterns = [
    path("summary/", analytics_summary, name="analytics-summary"),
]
