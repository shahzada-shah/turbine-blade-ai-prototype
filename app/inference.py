import uuid

from PIL import Image

from io import BytesIO

import random

from .utils.visualization import create_dummy_heatmap

from .utils.preprocessing import load_image_from_bytes



def analyze_blade_image(image_bytes: bytes):

    # Load the image

    pil_image = load_image_from_bytes(image_bytes)



    # Fake prediction (random for now)

    classes = ["healthy", "minor_damage", "severe_damage"]

    selected = random.choice(classes)

    confidence = round(random.uniform(0.6, 0.98), 2)

    

    damage_detected = selected != "healthy"



    # Generate "heatmap"

    heatmap_filename = f"results/heatmap_{uuid.uuid4().hex}.png"

    create_dummy_heatmap(pil_image, heatmap_filename)



    return {

        "damage_detected": damage_detected,

        "severity": selected,

        "confidence": confidence,

        "heatmap_path": heatmap_filename

    }
