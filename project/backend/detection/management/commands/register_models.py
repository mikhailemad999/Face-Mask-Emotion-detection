import json
from datetime import datetime
from django.core.management.base import BaseCommand
from django.conf import settings
from detection.models import ModelVersion

class Command(BaseCommand):
    help = "Register newly trained models and metrics into SQL Server registry"

    def handle(self, *args, **options):
        # 1. Mask Model
        mask_metrics_path = settings.ML_MODELS_DIR / "mask_metrics.json"
        if mask_metrics_path.exists():
            with open(mask_metrics_path, "r") as f:
                metrics = json.load(f)
            
            # Deactivate previous active mask models
            ModelVersion.objects.filter(model_type="mask", is_active=True).update(is_active=False)
            
            version_str = f"mobilenetv2_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ModelVersion.objects.create(
                model_type="mask",
                version=version_str,
                accuracy=metrics.get("accuracy"),
                f1_score=metrics.get("f1_weighted"),
                roc_auc=metrics.get("roc_auc"),
                trained_at=datetime.now(),
                is_active=True,
                notes="Trained via 04_train_mask.py script",
                model_path=str(settings.MASK_MODEL_PATH),
            )
            self.stdout.write(self.style.SUCCESS(f"Registered mask model version: {version_str}"))
        else:
            self.stdout.write(self.style.WARNING("mask_metrics.json not found in models directory"))

        # 2. Emotion Model
        emotion_metrics_path = settings.ML_MODELS_DIR / "emotion_metrics.json"
        if emotion_metrics_path.exists():
            with open(emotion_metrics_path, "r") as f:
                metrics = json.load(f)
            
            # Deactivate previous active emotion models
            ModelVersion.objects.filter(model_type="emotion", is_active=True).update(is_active=False)
            
            version_str = f"efficientnetb0_emotion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ModelVersion.objects.create(
                model_type="emotion",
                version=version_str,
                accuracy=metrics.get("accuracy"),
                f1_score=metrics.get("f1_weighted"),
                roc_auc=metrics.get("roc_auc_ovr"),
                trained_at=datetime.now(),
                is_active=True,
                notes="Trained via 05_train_emotion.py script",
                model_path=str(settings.EMOTION_MODEL_PATH),
            )
            self.stdout.write(self.style.SUCCESS(f"Registered emotion model version: {version_str}"))
        else:
            self.stdout.write(self.style.WARNING("emotion_metrics.json not found in models directory"))
