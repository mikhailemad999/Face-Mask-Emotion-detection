"""
Django Settings — Face Mask & Emotion Detection API
Supports: SQL Server (primary DB) + MongoDB (analytics)
"""

import os
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config("DJANGO_SECRET_KEY", default="dev-secret-key-change-in-production")
DEBUG      = config("DEBUG", default=True, cast=bool)
ALLOWED_HOSTS = config("ALLOWED_HOSTS", default="*").split(",")

# ─────────────────────────────────────────────────────────────────────────────
# Applications
# ─────────────────────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    "corsheaders",
    "django_filters",
    "drf_spectacular",
    "channels",
    "django_celery_results",
    "django_celery_beat",
    # Local apps
    "api",
    "detection",
    "analytics",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION  = "config.asgi.application"

# ─────────────────────────────────────────────────────────────────────────────
# Database — SQL Server (primary)
# ─────────────────────────────────────────────────────────────────────────────
DATABASES = {
    "default": {
        "ENGINE":   "mssql",
        "NAME":     config("SQL_SERVER_DB",   default="FaceMaskEmotionDB"),
        "USER":     config("SQL_SERVER_USER", default="sa"),
        "PASSWORD": config("SQL_SERVER_PASS", default="YourStrong!Pass123"),
        "HOST":     config("SQL_SERVER_HOST", default="localhost"),
        "PORT":     config("SQL_SERVER_PORT", default="1433"),
        "OPTIONS":  {
            "driver": "ODBC Driver 17 for SQL Server",
            "extra_params": "TrustServerCertificate=yes",
        },
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# MongoDB — analytics & prediction logs
# ─────────────────────────────────────────────────────────────────────────────
MONGODB_URI  = config("MONGODB_URI", default="mongodb://localhost:27017")
MONGODB_NAME = config("MONGODB_DB",  default="face_mask_emotion")

# ─────────────────────────────────────────────────────────────────────────────
# Django REST Framework
# ─────────────────────────────────────────────────────────────────────────────
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES":  ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES":    [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
    "DEFAULT_FILTER_BACKENDS":   ["django_filters.rest_framework.DjangoFilterBackend"],
    "DEFAULT_PAGINATION_CLASS":  "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE":                  20,
    "DEFAULT_SCHEMA_CLASS":      "drf_spectacular.openapi.AutoSchema",
}

SPECTACULAR_SETTINGS = {
    "TITLE":       "Face Mask & Emotion Detection API",
    "DESCRIPTION": "AI-powered detection system — graduation project",
    "VERSION":     "1.0.0",
    "SERVE_INCLUDE_SCHEMA": False,
}

# Batch folder upload limits (allow up to 5,000 files in a single folder upload)
DATA_UPLOAD_MAX_NUMBER_FILES = 5000
DATA_UPLOAD_MAX_MEMORY_SIZE = 104857600   # 100 MB max request payload
FILE_UPLOAD_MAX_MEMORY_SIZE = 104857600

# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]
CORS_ALLOW_ALL_ORIGINS = DEBUG

# ─────────────────────────────────────────────────────────────────────────────
# Channels (WebSocket)
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG":  {"hosts": [config("REDIS_URL", default="redis://localhost:6379")]},
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Celery
# ─────────────────────────────────────────────────────────────────────────────
CELERY_BROKER_URL         = config("REDIS_URL", default="redis://localhost:6379")
CELERY_RESULT_BACKEND     = "django-db"
CELERY_ACCEPT_CONTENT     = ["json"]
CELERY_TASK_SERIALIZER    = "json"
CELERY_RESULT_SERIALIZER  = "json"
CELERY_TIMEZONE           = "UTC"

# ─────────────────────────────────────────────────────────────────────────────
# Static files & Media
# ─────────────────────────────────────────────────────────────────────────────
STATIC_URL  = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL  = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ─────────────────────────────────────────────────────────────────────────────
# ML Model Paths
# ─────────────────────────────────────────────────────────────────────────────
ml_dir_raw = config("ML_MODELS_DIR", default=None)
if ml_dir_raw:
    ml_dir_path = Path(ml_dir_raw)
    if not ml_dir_path.is_absolute():
        ML_MODELS_DIR = (BASE_DIR / ml_dir_path).resolve()
    else:
        ML_MODELS_DIR = ml_dir_path
else:
    ML_MODELS_DIR = (BASE_DIR.parent / "ml" / "models").resolve()
MASK_MODEL_PATH    = ML_MODELS_DIR / "mask_model_best.pt"
EMOTION_MODEL_PATH = ML_MODELS_DIR / "emotion_model_best.pt"
MASK_ONNX_PATH     = ML_MODELS_DIR / "mask_model.onnx"
EMOTION_ONNX_PATH  = ML_MODELS_DIR / "emotion_model.onnx"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MASK_CLASSES    = ["without_mask", "with_mask"]
IMG_SIZE        = 224

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
LANGUAGE_CODE      = "en-us"
TIME_ZONE          = "UTC"
USE_I18N           = True
USE_TZ             = True
