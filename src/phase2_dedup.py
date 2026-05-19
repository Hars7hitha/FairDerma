# src/phase2_dedup.py
# Duplicate removal using perceptual hashing (Cassidy et al. 2022)

import imagehash
from PIL import Image
import glob
import os
from tqdm import tqdm

def remove_duplicates(image_dir):
    all_images = glob.glob(f"{image_dir}/*.jpg")
    all_images = all_images[:100]  # 100 images for now

    print(f"Checking {len(all_images)} images for duplicates...")

    seen_hashes = {}
    duplicates = []

    for img_path in tqdm(all_images, desc="Hashing images"):
        try:
            h = imagehash.phash(Image.open(img_path))
            if h in seen_hashes:
                duplicates.append(img_path)
            else:
                seen_hashes[h] = img_path
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

    print(f"\nDuplicates found : {len(duplicates)}")
    print(f"Unique images    : {len(all_images) - len(duplicates)}")

    for d in duplicates:
        print(f"  Removing: {d}")
        os.remove(d)

    print("✅ Deduplication complete!")

if __name__ == "__main__":
    remove_duplicates("data/ham10000")