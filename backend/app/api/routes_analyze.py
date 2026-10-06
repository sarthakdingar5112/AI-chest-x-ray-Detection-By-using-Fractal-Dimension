from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.models.schemas import AnalysisResponse
from app.services.pipeline import analyze
from app.services.validation import NotAnXrayError

router = APIRouter(tags=["analyze"])

_ALLOWED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}


@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_xray(file: UploadFile = File(...)) -> AnalysisResponse:
    name = (file.filename or "").lower()
    if "." in name and name[name.rfind("."):] not in _ALLOWED:
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Allowed: {sorted(_ALLOWED)}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB.")

    try:
        return analyze(data, file.filename or "upload.png")
    except NotAnXrayError as e:
        # 422 with a clean, user-facing message (no "Could not analyze" prefix).
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not analyze image: {e}")
