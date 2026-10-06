from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.services.cnn_model import model as cnn

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    available = cnn.available
    return HealthResponse(
        status="ok",
        mlAvailable=available,
        model=cnn.name if available else None,
    )
