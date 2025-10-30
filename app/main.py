from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import body as measurements
from app.routes import classifyPose as pose
from app.routes import tryon

app = FastAPI(
    title="Virtual Dressing Room API",
    description="Backend for body measurement, garment extraction, and try-on visualization",
    version="1.0.0"
)

# Allow CORS for UI (Gradio or React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(measurements.router, prefix="/api", tags=["Measurements"])
app.include_router(pose.router, prefix="/api", tags=["Pose Detection"])
app.include_router(tryon.router, prefix="/tryon", tags=["Virtual Try-On"])
@app.get("/")
def root():
    return {"message": "Virtual Dressing Room API is running"}
