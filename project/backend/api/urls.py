from django.urls import path
from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(["GET"])
def health_check(request):
    """
    Return basic service health check status.

    Args:
        request (Request): DRF GET HTTP request.

    Returns:
        Response: DRF Response with status 'ok' and service identifier.
    """
    return Response({"status": "ok", "service": "FaceGuard.AI"})


urlpatterns = [
    path("health/", health_check, name="health-check"),
]
