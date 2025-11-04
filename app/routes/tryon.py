import os
import uuid
import torch
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

# Assuming your helper files are in these locations
from app.utils.image_tools import remove_background
from app.models.pose_estimator import PoseEstimator

router = APIRouter()

UPLOADS = "uploads"
OUTPUTS = "outputs"
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# Define a standard size for your model
MODEL_WIDTH = 512
MODEL_HEIGHT = 768

# --- 1. Load models ---
print("Loading models...")
pose_estimator = PoseEstimator(debug=False)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=torch.float16  # Use float16 for memory efficiency
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# --- Move pipeline to CUDA if available, else CPU ---
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
print(f"Models loaded on {device}.")

@router.post("/tryon")
async def try_on(user: UploadFile = File(...), outfit: UploadFile = File(...)):
    # Save user and outfit images
    uid = str(uuid.uuid4())
    user_path = os.path.join(UPLOADS, f"user_{uid}.jpg")
    outfit_path = os.path.join(UPLOADS, f"outfit_{uid}.jpg")
    with open(user_path, "wb") as f:
        f.write(await user.read())
    with open(outfit_path, "wb") as f:
        f.write(await outfit.read())

    # Resize images before processing
    target_size = (MODEL_WIDTH, MODEL_HEIGHT)

    try:
        # Resize user image
        user_img_pil = Image.open(user_path).convert("RGB")
        user_img_pil = user_img_pil.resize(target_size, Image.Resampling.LANCZOS)
        user_img_pil.save(user_path)

        # Resize outfit image
        outfit_img_pil = Image.open(outfit_path).convert("RGB")
        outfit_img_pil = outfit_img_pil.resize(target_size, Image.Resampling.LANCZOS)
        outfit_img_pil.save(outfit_path)
    except Exception as e:
        print(f"Error resizing images: {e}")
        return {"error": "Could not process uploaded images."}

    # 1️⃣ Get pose image (skeleton) from the resized user image
    pose_data = pose_estimator.process_image(user_path)
    pose_img = Image.fromarray(pose_data["annotated_image"])

    # Save pose image for debugging
    pose_img.save(os.path.join(OUTPUTS, f"debug_pose_{uid}.png"))

    # 2️⃣ Get outfit mask (remove background)
    outfit_img = remove_background(outfit_path)

    # --- Pipeline inference ---
    prompt = "a realistic photo of a person, full body, high quality clothing, studio lighting"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, gross, deformed, blurry"

    # Use torch autocast for GPU float16 efficiency
    with torch.autocast(device_type=device if device=="cuda" else "cpu"):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pose_img,  # Pass pose image as control
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]

    # Save output
    output_path = os.path.join(OUTPUTS, f"tryon_{uid}.png")
    result.save(output_path)

    return FileResponse(output_path)
