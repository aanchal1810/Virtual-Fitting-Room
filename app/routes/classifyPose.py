from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse
import os, uuid, cv2

from app.models.pose_estimator import PoseEstimator

# Initialize router
router = APIRouter(prefix="/pose", tags=["Pose Estimation"])

# Initialize pose estimator
pose_estimator = PoseEstimator(debug=True)
UPLOADS = "uploads"
os.makedirs(UPLOADS, exist_ok=True)


@router.post("/")
async def detect_pose(file: UploadFile = File(...), debug: bool = Query(True)):
    """
    Detect human pose, return keypoints and joint angles.
    """
    # Save uploaded image
    file_id = str(uuid.uuid4())
    in_path = os.path.join(UPLOADS, f"{file_id}_{file.filename}")
    with open(in_path, "wb") as f:
        f.write(await file.read())

    # Run pose estimation
    result = pose_estimator.process_image(in_path)

    # If no person detected
    if result.get("annotated_image") is None:
        return JSONResponse(
            {"error": "no person detected"},
            status_code=400,
        )

    # Save annotated output
    out_path = os.path.join(UPLOADS, f"annotated_{file_id}.jpg")
    cv2.imwrite(out_path, result["annotated_image"])

    return {
        "angles": result["angles"],
        "landmarks": result["landmarks"],
        "annotated_image_path": out_path,
    }
