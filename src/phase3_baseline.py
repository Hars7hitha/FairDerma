# src/phase3_baseline.py
# Baseline Model — Standard Cross-Entropy, NO Dynamic Fairness Loss
# Saved as outputs/resnet50_baseline.pth
# Used as control experiment to prove DFAL works

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Running on: {device}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

LABEL_MAP = {
    "mel": 1, "bcc": 1, "akiec": 1,
    "bkl": 0, "df" : 0, "vasc" : 0, "nv": 0,
}

class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, limit=2000):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        self.df["binary_label"] = self.df["dx"].map(LABEL_MAP)
        self.df = self.df.dropna(subset=["binary_label"])
        self.df["binary_label"] = self.df["binary_label"].astype(int)
        self.df["img_path"] = self.df["image_id"].apply(
            lambda x: os.path.join(image_dir, f"{x}.jpg")
        )
        self.df = self.df[self.df["img_path"].apply(os.path.exists)]
        self.df = self.df.head(limit).reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["binary_label"])


def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model.to(device)


def train_baseline(num_epochs=10):
    dataset = HAM10000Dataset(
        csv_path  = "data/ham10000/HAM10000_metadata.csv",
        image_dir = "data/ham10000_cleaned",
        transform = train_transform,
        limit     = 2000
    )

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)

    model     = build_model()

    # ── STANDARD static loss — NO dynamic fairness adjustment ──
    # Class weights only — this is the "control" experiment
    class_weights = torch.tensor([1.0, 5.0]).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    print(f"\nBaseline Training — {num_epochs} epochs (NO DFAL)")
    print(f"Train: {train_size} | Val: {val_size}")
    print("─" * 55)

    history = []

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct    += (torch.argmax(outputs, 1) == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs   = model(imgs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (torch.argmax(outputs, 1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch:2d} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {val_acc:.4f}")

        history.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss / len(train_loader), 4),
            "train_acc" : round(train_acc, 4),
            "val_loss"  : round(val_loss / len(val_loader), 4),
            "val_acc"   : round(val_acc, 4),
        })

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/resnet50_baseline.pth")
    print("\n✅ Baseline model saved: outputs/resnet50_baseline.pth")

    pd.DataFrame(history).to_csv(
        "outputs/baseline_history.csv", index=False
    )
    print("✅ Baseline history saved: outputs/baseline_history.csv")
    print("\n🎉 Baseline training complete!")
    print("   Now run phase4_evaluate.py for both models")


if __name__ == "__main__":
    train_baseline(num_epochs=10)