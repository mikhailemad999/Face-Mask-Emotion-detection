"""
MongoDB Service — Stores full prediction payloads
Uses pymongo (synchronous) for Django compatibility
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from django.conf import settings

logger = logging.getLogger(__name__)

try:
    from pymongo import MongoClient
    from bson import ObjectId

    _client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=3000)
    _db     = _client[settings.MONGODB_NAME]
    _predictions_col = _db["predictions"]
    _sessions_col    = _db["sessions"]
    _stats_col       = _db["stats"]
    _client.admin.command("ping")
    logger.info(f"[MongoDB] Connected to {settings.MONGODB_URI}/{settings.MONGODB_NAME}")
    MONGO_AVAILABLE = True
except Exception as e:
    logger.warning(f"[MongoDB] Not available: {e}. Mongo features disabled.")
    MONGO_AVAILABLE = False


def save_prediction_to_mongo(payload: Dict[str, Any]) -> str:
    """
    Save a full detection prediction to MongoDB.
    Returns the inserted document _id as string, or "" on failure.
    """
    if not MONGO_AVAILABLE:
        return ""
    try:
        payload["created_at"] = datetime.utcnow()
        result = _predictions_col.insert_one(payload)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"[MongoDB] insert failed: {e}")
        return ""


def get_analytics_summary() -> Dict:
    """
    Aggregate detection statistics from MongoDB.
    Returns: total_predictions, class_distributions, avg_confidence
    """
    if not MONGO_AVAILABLE:
        return {"error": "MongoDB unavailable"}
    try:
        total = _predictions_col.count_documents({})

        # Mask distribution
        mask_pipeline = [
            {"$unwind": "$full_result.predictions"},
            {"$group": {
                "_id":   "$full_result.predictions.mask.label",
                "count": {"$sum": 1},
                "avg_conf": {"$avg": "$full_result.predictions.mask.confidence"},
            }},
        ]
        mask_stats = list(_predictions_col.aggregate(mask_pipeline))

        # Emotion distribution
        emotion_pipeline = [
            {"$unwind": "$full_result.predictions"},
            {"$group": {
                "_id":   "$full_result.predictions.emotion.label",
                "count": {"$sum": 1},
                "avg_conf": {"$avg": "$full_result.predictions.emotion.confidence"},
            }},
        ]
        emotion_stats = list(_predictions_col.aggregate(emotion_pipeline))

        # Recent 24h
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent = _predictions_col.count_documents({"created_at": {"$gte": cutoff}})

        return {
            "total_predictions":    total,
            "last_24h":             recent,
            "mask_distribution":    mask_stats,
            "emotion_distribution": emotion_stats,
        }
    except Exception as e:
        logger.error(f"[MongoDB] aggregation failed: {e}")
        return {"error": str(e)}


def get_recent_predictions(limit: int = 50) -> list:
    """Return the most recent predictions from MongoDB."""
    if not MONGO_AVAILABLE:
        return []
    try:
        docs = list(
            _predictions_col
            .find({}, {"_id": 1, "timestamp": 1, "source": 1,
                       "full_result.faces_detected": 1,
                       "full_result.processing_time_ms": 1})
            .sort("created_at", -1)
            .limit(limit)
        )
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs
    except Exception as e:
        logger.error(f"[MongoDB] query failed: {e}")
        return []
