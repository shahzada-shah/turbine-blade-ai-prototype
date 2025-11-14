from PIL import Image

from io import BytesIO

def load_image_from_bytes(image_bytes: bytes):

    return Image.open(BytesIO(image_bytes)).convert("RGB")

def preprocess_for_model(pil_image):

    # Placeholder until real model

    return pil_image
