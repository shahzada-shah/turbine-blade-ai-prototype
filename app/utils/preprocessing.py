from PIL import Image

from io import BytesIO

# Optional torchvision import (for future ML model)
try:
    import torchvision.transforms as T
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    T = None



# Basic preprocessing to prepare input for a CNN

def load_image_from_bytes(image_bytes: bytes):

    return Image.open(BytesIO(image_bytes)).convert("RGB")



# Torch transforms (lazy initialization)

_transform = None

def _get_transform():
    """Get or create the transform, only when needed"""
    global _transform
    if _transform is None:
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for model preprocessing. Install with: pip install torch torchvision")
        _transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    return _transform



def preprocess_for_model(pil_image: Image.Image):

    transform = _get_transform()
    return transform(pil_image).unsqueeze(0)  # add batch dimension
