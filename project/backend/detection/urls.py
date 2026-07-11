from django.urls import path
from . import views

urlpatterns = [
    path("image/",   views.detect_image,       name="detect-image"),
    path("frame/",   views.detect_frame,        name="detect-frame"),
    path("history/", views.detection_history,   name="detect-history"),
    path("models/",  views.model_info,          name="model-info"),
]
