#!/usr/bin/env python3
"""
generate_sample_crops.py - Generates sample crop dataset using PIL for testing MobileNetV2 CNN training.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
LABELED_DIR = PROJECT_ROOT / "crops" / "labeled"
CLASSES = ["bottle_cap", "coin", "gem", "pick", "background"]

np.random.seed(42)

def create_synthetic_crop(cls_name):
    # Base background
    img_arr = np.full((128, 128, 3), fill_value=220, dtype=np.uint8)
    noise = np.random.normal(0, 12, (128, 128, 3)).astype(np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_arr, mode="RGB")
    draw = ImageDraw.Draw(img)

    if cls_name == "coin":
        # Draw metallic coin
        draw.ellipse([28, 28, 100, 100], fill=(180, 180, 180), outline=(100, 100, 100), width=3)
        draw.ellipse([38, 38, 90, 90], outline=(140, 140, 140), width=1)
    elif cls_name == "bottle_cap":
        # Draw red plastic bottle cap with ridges
        draw.ellipse([24, 24, 104, 104], fill=(200, 40, 40), outline=(255, 150, 150), width=4)
        for angle in range(0, 360, 30):
            rad = np.radians(angle)
            x1 = int(64 + 35 * np.cos(rad))
            y1 = int(64 + 35 * np.sin(rad))
            x2 = int(64 + 43 * np.cos(rad))
            y2 = int(64 + 43 * np.sin(rad))
            draw.line([(x1, y1), (x2, y2)], fill=(255, 200, 200), width=2)
    elif cls_name == "gem":
        # Draw blue crystal gemstone
        pts = [(64, 24), (94, 44), (94, 84), (64, 104), (34, 84), (34, 44)]
        draw.polygon(pts, fill=(50, 180, 240), outline=(150, 220, 255), width=2)
        draw.line([(64, 24), (64, 104)], fill=(200, 240, 255), width=2)
    elif cls_name == "pick":
        # Draw green triangular guitar pick
        pts = [(64, 100), (28, 35), (100, 35)]
        draw.polygon(pts, fill=(40, 180, 60), outline=(20, 120, 30), width=2)
    elif cls_name == "background":
        # Just desk texture/lines
        draw.line([(10, 20), (110, 110)], fill=(160, 160, 160), width=2)
        draw.line([(30, 10), (120, 90)], fill=(170, 170, 170), width=1)

    # Random slight rotation
    rot_angle = np.random.uniform(-15, 15)
    img = img.rotate(rot_angle, fillcolor=(220, 220, 220))
    return img

def main():
    print("Generating sample labeled crops dataset...")
    for cls in CLASSES:
        cls_dir = LABELED_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(25):
            crop = create_synthetic_crop(cls)
            file_path = cls_dir / f"crop_{cls}_{i:03d}.png"
            crop.save(file_path)
    print(f"Dataset created with {len(CLASSES) * 25} total images in {LABELED_DIR}")

if __name__ == "__main__":
    main()
