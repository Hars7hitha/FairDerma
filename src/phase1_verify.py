# src/phase1_verify.py
import glob

images = glob.glob("data/ham10000/**/*.jpg", recursive=True)
csvs   = glob.glob("data/ham10000/**/*.csv", recursive=True)

print(f"Images found : {len(images)}")
print(f"CSVs found   : {csvs}")
print(f"First 3 images: {images[:3]}")