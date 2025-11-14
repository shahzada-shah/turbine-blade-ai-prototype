from PIL import Image, ImageDraw

import os

def create_dummy_heatmap(pil_image, output_path):

    img = pil_image.copy()

    draw = ImageDraw.Draw(img)

    # Draw a transparent red circle (fake crack highlight)

    w, h = img.size

    radius = min(w, h) // 4

    bbox = (w//2 - radius, h//2 - radius, w//2 + radius, h//2 + radius)

    draw.ellipse(bbox, outline="red", width=10)

    os.makedirs("results", exist_ok=True)

    img.save(output_path)

    return output_path
