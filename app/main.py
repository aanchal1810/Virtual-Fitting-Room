from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
def root():
    return {"message": "Virtual Dressing Room API is running"}
