import io
from PIL import Image
from rembg import remove

def remove_background(image_path: str) -> Image.Image:
    """Removes background from an outfit image."""
    with open(image_path, "rb") as f:
        input_image = f.read()
    output = remove(input_image)
    return Image.open(io.BytesIO(output)).convert("RGBA")