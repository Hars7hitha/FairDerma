# src/phase2_dataset.py
# PyTorch Dataset with resize, normalize, augmentation

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob

# ImageNet normalization — standard for ResNet50 transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Training transform — augmentation applied uniformly across all skin tone groups
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation transform — no augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Binary label mapping — 7 HAM10000 classes → Malignant / Benign
LABEL_MAP = {
    "mel"  : 1,  # Melanoma           → Malignant
    "bcc"  : 1,  # Basal cell carcinoma → Malignant
    "akiec": 1,  # Actinic keratosis  → Malignant
    "bkl"  : 0,  # Benign keratosis   → Benign
    "df"   : 0,  # Dermatofibroma     → Benign
    "vasc" : 0,  # Vascular lesion    → Benign
    "nv"   : 0,  # Melanocytic nevi   → Benign
}

class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, limit=100):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        # 1. Map to binary labels first
        self.df["binary_label"] = self.df["dx"].map(LABEL_MAP)
        self.df = self.df.dropna(subset=["binary_label"])
        self.df["binary_label"] = self.df["binary_label"].astype(int)

        # 2. Check path existence before limiting
        self.df["img_path"] = self.df["image_id"].apply(
            lambda x: os.path.join(image_dir, f"{x}.jpg")
        )
        self.df = self.df[self.df["img_path"].apply(os.path.exists)]

        # --- THE CRITICAL CHANGE ---
        # 3. SHUFFLE the dataframe randomly
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 4. NOW apply the limit
        self.df = self.df.head(limit)
        # ---------------------------

        print(f"\n📦 HAM10000 Dataset")
        print(f"   Total samples : {len(self.df)}")
        print(f"   Benign (0)    : {(self.df['binary_label']==0).sum()}")
        print(f"   Malignant (1) : {(self.df['binary_label']==1).sum()}")
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        img_path = row["img_path"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, row["binary_label"]


if __name__ == "__main__":
    CSV_PATH  = "data/ham10000/HAM10000_metadata.csv"
    IMAGE_DIR = "data/ham10000_cleaned"  # cleaned images from phase2_artifact

    # Test with train transform
    dataset = HAM10000Dataset(
        csv_path=CSV_PATH,
        image_dir=IMAGE_DIR,
        transform=train_transform,
        limit=100
    )

    # Quick DataLoader test
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    imgs, labels = next(iter(loader))

    print(f"\n✅ DataLoader working!")
    print(f"   Batch image shape : {imgs.shape}")
    print(f"   Batch labels      : {labels.tolist()}")
   