from rest_framework import serializers
from .models import DetectionLog, ModelVersion


class DetectionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model  = DetectionLog
        fields = "__all__"


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model  = ModelVersion
        fields = "__all__"
