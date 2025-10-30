from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torch, os, uuid
from app.utils.image_tools import remove_background
from app.models.pose_estimator import PoseEstimator

router = APIRouter()

UPLOADS = "uploads"
OUTPUTS = "outputs"
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# Load pose detector
pose_estimator = PoseEstimator(debug=False)

# Load ControlNet model (for pose-guided generation)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
).to("cpu")

@router.post("/tryon")
async def try_on(user: UploadFile = File(...), outfit: UploadFile = File(...)):
    # Save user image
    uid = str(uuid.uuid4())
    user_path = os.path.join(UPLOADS, f"user_{uid}.jpg")
    outfit_path = os.path.join(UPLOADS, f"outfit_{uid}.jpg")
    with open(user_path, "wb") as f:
        f.write(await user.read())
    with open(outfit_path, "wb") as f:
        f.write(await outfit.read())

    # 1️⃣ Get pose image (skeleton)
    pose_data = pose_estimator.process_image(user_path)
    pose_img = Image.fromarray(pose_data["annotated_image"])

    # 2️⃣ Get outfit mask (remove background)
    outfit_img = remove_background(outfit_path)

    # 3️⃣ Generate try-on result
    prompt = "a realistic photo of the same person wearing this outfit"
    result = pipe(
        prompt=prompt,
        image=pose_img,
        control_image=pose_img,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    # 4️⃣ Save output
    output_path = os.path.join(OUTPUTS, f"tryon_{uid}.png")
    result.save(output_path)

    return FileResponse(output_path)
