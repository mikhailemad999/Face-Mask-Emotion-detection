from django.urls import path
from rest_framework.decorators import api_view
from rest_framework.response import Response
from detection.mongo_service import get_analytics_summary


@api_view(["GET"])
def analytics_summary(request):
    """Aggregated detection statistics from MongoDB."""
    data = get_analytics_summary()
    return Response(data)


urlpatterns = [
    path("summary/", analytics_summary, name="analytics-summary"),
]
