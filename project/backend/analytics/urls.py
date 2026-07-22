from django.urls import path
from rest_framework.decorators import api_view
from rest_framework.response import Response
from detection.mongo_service import get_analytics_summary


@api_view(["GET"])
def analytics_summary(request):
    """
    Return aggregated detection statistics and class distributions retrieved from MongoDB.

    Args:
        request (Request): DRF GET HTTP request.

    Returns:
        Response: DRF Response with analytics summary data payload.
    """
    data = get_analytics_summary()
    return Response(data)


urlpatterns = [
    path("summary/", analytics_summary, name="analytics-summary"),
]
