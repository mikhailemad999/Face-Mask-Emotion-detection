"""
Detection App — Django Models (SQL Server)
"""

from django.db import models


class DetectionLog(models.Model):
    """Stores every detection event in SQL Server."""
    timestamp           = models.DateTimeField(auto_now_add=True, db_index=True)
    image_name          = models.CharField(max_length=255, blank=True, default="")
    faces_detected      = models.IntegerField(default=0)
    mask_result         = models.CharField(max_length=30, blank=True, default="")
    mask_confidence     = models.FloatField(null=True, blank=True)
    emotion_result      = models.CharField(max_length=30, blank=True, default="")
    emotion_confidence  = models.FloatField(null=True, blank=True)
    processing_time_ms  = models.FloatField(null=True, blank=True)
    source              = models.CharField(
        max_length=20,
        choices=[("upload", "Upload"), ("webcam", "Webcam"), ("api", "API")],
        default="upload",
    )
    mongo_prediction_id = models.CharField(max_length=50, blank=True, default="")

    class Meta:
        db_table  = "DetectionLog"
        ordering  = ["-timestamp"]
        indexes   = [
            models.Index(fields=["timestamp"]),
            models.Index(fields=["mask_result"]),
            models.Index(fields=["emotion_result"]),
        ]

    def __str__(self):
        return f"[{self.timestamp}] mask={self.mask_result} emotion={self.emotion_result}"


class ModelVersion(models.Model):
    """Tracks trained model versions."""
    model_type  = models.CharField(
        max_length=20,
        choices=[("mask", "Mask"), ("emotion", "Emotion")],
    )
    version     = models.CharField(max_length=50)
    accuracy    = models.FloatField(null=True, blank=True)
    f1_score    = models.FloatField(null=True, blank=True)
    roc_auc     = models.FloatField(null=True, blank=True)
    trained_at  = models.DateTimeField()
    is_active   = models.BooleanField(default=False)
    notes       = models.TextField(blank=True, default="")
    model_path  = models.CharField(max_length=500, blank=True, default="")

    class Meta:
        db_table = "ModelVersion"
        ordering = ["-trained_at"]

    def __str__(self):
        return f"{self.model_type} v{self.version} (active={self.is_active})"
