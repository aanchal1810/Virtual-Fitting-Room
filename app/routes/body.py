from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import shutil
import os
from app.models.body_model import BodyMeasurementExtractor

router = APIRouter()

@router.post("/extract-measurements")
async def extract_measurements(
    image: UploadFile = File(...),
    user_height_cm: float = Form(...)
):
    try:
        # Save uploaded image temporarily
        temp_path = f"temp_{image.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Run measurement extraction
        extractor = BodyMeasurementExtractor()
        measurements = extractor.extract_measurements(temp_path, user_height_cm)

        # Delete temp file
        os.remove(temp_path)

        return {"success": True, "measurements": measurements}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
