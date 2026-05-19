# src/phase2_artifact.py
# Hair artifact removal using OpenCV inpainting (DullRazor-inspired)

import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def remove_hair(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Black-hat transform isolates dark hair strands
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold → hair mask
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint fills hair regions with surrounding skin texture
    cleaned = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, cleaned)
    return True

def process_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    images = glob.glob(f"{input_dir}/*.jpg")
    images = images[:5000]  # 5000 images

    print(f"Total images to process: {len(images)}")

    # Only process images not already cleaned
    pending = []
    for img_path in images:
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        if not os.path.exists(out_path):
            pending.append(img_path)

    print(f"Already cleaned : {len(images) - len(pending)}")
    print(f"Remaining       : {len(pending)}")

    if len(pending) == 0:
        print("✅ All images already cleaned!")
        return

    success = 0
    for img_path in tqdm(pending, desc="Removing hair artifacts"):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        if remove_hair(img_path, out_path):
            success += 1

    print(f"\n✅ Artifact removal complete!")
    print(f"   Cleaned images saved to : {output_dir}")
    print(f"   Newly processed         : {success}/{len(pending)}")
    print(f"   Total cleaned now       : {len(images)}")
if __name__ == "__main__":
    process_all("data/ham10000", "data/ham10000_cleaned")