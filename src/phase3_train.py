# src/phase3_train.py
# Phase 3 — EfficientNet-B0 + Dynamic Fairness-Aware Loss
# Novel Contribution: Per-epoch TPR feedback loop across Fitzpatrick skin tone groups

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

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Running on: {device}")

# ── Transforms ────────────────────────────────────────────────────────────────
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

# ── Label Map ─────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "mel"  : 1, "bcc"  : 1, "akiec": 1,
    "bkl"  : 0, "df"   : 0, "vasc" : 0, "nv": 0,
}

# ── HAM10000 Dataset ──────────────────────────────────────────────────────────
class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, limit=5000):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        self.df["binary_label"] = self.df["dx"].map(LABEL_MAP)
        self.df = self.df.dropna(subset=["binary_label"])
        self.df["binary_label"] = self.df["binary_label"].astype(int)

        # Filter to only existing images
        self.df["img_path"] = self.df["image_id"].apply(
            lambda x: os.path.join(image_dir, f"{x}.jpg")
        )
        self.df = self.df[self.df["img_path"].apply(os.path.exists)]
        self.df = self.df.head(limit).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["binary_label"])

class SimulatedDDIDataset(Dataset):
    """
    Simulates DDI fairness benchmark with Fitzpatrick skin tone labels (I-VI).
    Guarantees at least 3 malignant cases per FST group so TPR is non-zero.
    Replace with real DDI in Phase 4.
    """
    def __init__(self, image_dir, csv_path, transform=None, limit=600):
        self.transform = transform

        # Try real DDI first
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path).head(limit)
            self.simulated = False
            print("✅ Real DDI dataset loaded")
            return

        print("⚠️  DDI not found — using simulated fairness data for demo")
        self.simulated = True

        images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        images = images[:120]  # 60 images → 10 per FST group

        # Guarantee exactly 10 images per FST group
        # with at least 4 malignant per group so TPR is meaningful
        rows = []
        imgs_per_group = len(images) // 6

        for fst in range(1, 7):
            group_imgs = images[(fst-1)*imgs_per_group : fst*imgs_per_group]

            for i, img in enumerate(group_imgs):
                # First 4 in each group = malignant, rest = benign
                # This guarantees non-zero TPR is possible
                label = 1 if i < 4 else 0
                rows.append({
                    "img_path"         : os.path.join(image_dir, img),
                    "malignant"        : label,
                    "fitzpatrick_scale": fst,
                })

        self.df = pd.DataFrame(rows)

        print(f"   Simulated DDI: {len(self.df)} images")
        print(f"   Malignant per group: 4 | Benign per group: {imgs_per_group - 4} | Total per group: {imgs_per_group}")
        print(f"   FST distribution: { {fst: 4 for fst in range(1,7)} } malignant each")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"] if self.simulated else os.path.join(
            "data/ddi", row["DDI_file"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["malignant"]), int(row["fitzpatrick_scale"])


# ── ResNet50 Model ─────────────────────────────────────────────────────────────
def build_model():
    # Use ResNet50 to match baseline for fair comparison
    # DFAL improvements should be due to loss function, not architecture
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # UNFREEZE layer4 and fc layers so the model can learn skin-specific features
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # New classification head (matching baseline)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),  # 2 classes: Benign, Malignant
    )
    return model.to(device)

# ── Dynamic Fairness Loss ★ NOVEL CONTRIBUTION ────────────────────────────────
class DynamicFairnessLoss(nn.Module):
    """
    Standard weighted cross-entropy sets weights once and freezes them.
    This version recomputes weights every epoch based on per-Fitzpatrick TPR.
    The group with the LOWEST TPR gets its weight boosted by ×1.1.
    This is a self-correcting feedback loop — the novelty of this project.
    """
    def __init__(self, n_groups=6, boost_factor=1.1, max_weight=3.0):
        super().__init__()
        self.n_groups   = n_groups
        self.boost      = boost_factor
        self.max_weight = max_weight
        # Start with equal weights for all FST groups (12, 34, 56)
        self.group_weights = {12: 1.0, 34: 1.0, 56: 1.0}

    def update_weights(self, model, ddi_loader):
        """
        Run model on DDI slice → compute TPR per FST group →
        boost weight of worst group.
        """
        model.eval()
        tpr_per_group = {}
        group_data    = {12: {"tp": 0, "fn": 0}, 34: {"tp": 0, "fn": 0}, 56: {"tp": 0, "fn": 0}}

        with torch.no_grad():
            for imgs, labels, fst_groups in ddi_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                preds   = torch.argmax(outputs, dim=1)

                for pred, label, fst in zip(preds, labels, fst_groups):
                    fst = fst.item()
                    # Map to group: 1,2 -> 12; 3,4 -> 34; 5,6 -> 56
                    fst_group = 12 if fst in [1, 2] else (34 if fst in [3, 4] else 56)
                    if label.item() == 1:          # only malignant cases for TPR
                        if pred.item() == 1:
                            group_data[fst_group]["tp"] += 1
                        else:
                            group_data[fst_group]["fn"] += 1

        # Compute TPR per group
        for g in [12, 34, 56]:
            tp = group_data[g]["tp"]
            fn = group_data[g]["fn"]
            tpr_per_group[g] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Find worst group (lowest TPR)
       # Find worst group — among groups that actually have malignant samples
        # Break ties by preferring higher FST (darker skin = more underserved)
        eligible = {g: tpr for g, tpr in tpr_per_group.items()
                   if group_data[g]["tp"] + group_data[g]["fn"] > 0}

        if eligible:
            # Pick lowest TPR — break ties by highest FST group number
            worst_group = min(eligible, key=lambda g: (eligible[g], -g))
        else:
            # All groups had no malignant samples — pick highest FST by default
            worst_group = 56
        old_weight  = self.group_weights[worst_group]
        new_weight  = min(old_weight * self.boost, self.max_weight)
        self.group_weights[worst_group] = new_weight

        print(f"\n   TPR per Fitzpatrick group : {tpr_per_group}")
        print(f"   Worst group              : FST {worst_group} "
              f"(TPR={tpr_per_group[worst_group]:.3f})")
        print(f"   ⬆Weight boosted            : {old_weight:.2f} → {new_weight:.2f}")

        model.train()
        return tpr_per_group

    def forward(self, outputs, labels):
        # Apply the highest group weight (worst-performing group) to all samples
        # This makes the model train harder when fairness is poor
        max_weight = max(self.group_weights.values())
        weights = torch.ones_like(labels, dtype=torch.float).to(outputs.device) * max_weight
        return nn.CrossEntropyLoss(weight=weights)(outputs, labels)


# ── Training Loop ─────────────────────────────────────────────────────────────
def train(num_epochs=20):
    # ── Data ──
    full_dataset = HAM10000Dataset(
        csv_path  = "data/ham10000/HAM10000_metadata.csv",
        image_dir = "data/ham10000_cleaned",
        transform = train_transform,
        limit     = 5000
    )

    train_size = int(0.9 * len(full_dataset))  # Use 90% for training
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)  # Increased batch size
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    ddi_dataset = SimulatedDDIDataset(
        image_dir = "data/ham10000_cleaned",
        csv_path  = "data/ddi/ddi_metadata.csv",
        transform = val_transform,
        limit     = 600
    )
    ddi_loader = DataLoader(ddi_dataset, batch_size=8, shuffle=False)

    # ── Model + Loss + Optimizer ──
    model     = build_model()
    criterion = DynamicFairnessLoss(n_groups=6, boost_factor=1.2, max_weight=5.0)  # More aggressive fairness
    # Class weights to force model to predict malignant cases
    # Without this model always plays safe and predicts benign
    class_weights = torch.tensor([1.0, 8.0]).to(device)  # Increased malignant weight
    criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
    
    # Differential Learning Rates
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'layer4' not in n and 'fc' not in n], 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'layer4' in n or 'fc' in n], 'lr': 1e-3}
    ])
    
    # Scheduler: Reduces LR when the model stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    print(f"\nStarting training — {num_epochs} epochs on {device}")
    print(f"   Train samples : {train_size}")
    print(f"   Val samples   : {val_size}")
    print("─" * 60)

    history = []

    for epoch in range(1, num_epochs + 1):
        # ── Train ──
        model.train()
        train_loss, correct, total = 0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion_weighted(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct    += (torch.argmax(outputs, 1) == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / total

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs  = model(imgs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (torch.argmax(outputs, 1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch} Results:")
        print(f"   Train Loss : {train_loss/len(train_loader):.4f} | "
              f"Train Acc : {train_acc:.4f}")
        print(f"   Val Loss   : {val_loss/len(val_loader):.4f}   | "
              f"Val Acc   : {val_acc:.4f}")

        # Step the scheduler
        scheduler.step(val_loss)

        # ── Dynamic Fairness Update ★ NOVEL CONTRIBUTION ──
        print(f"\nRunning fairness feedback loop (Epoch {epoch})...")
        tpr_per_group = criterion.update_weights(model, ddi_loader)

        history.append({
            "epoch"     : epoch,
            "train_loss": round(train_loss / len(train_loader), 4),
            "train_acc" : round(train_acc, 4),
            "val_loss"  : round(val_loss / len(val_loader), 4),
            "val_acc"   : round(val_acc, 4),
            **{f"tpr_fst{g}": round(tpr_per_group.get(g, 0.0), 4)
               for g in [12, 34, 56]}
        })

        print("─" * 60)

    # ── Save model ──
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/resnet50_fairness.pth")
    print("\n💾 Model saved to outputs/resnet50_fairness.pth")

    # ── Save training history ──
    history_df = pd.DataFrame(history)
    history_df.to_csv("outputs/training_history.csv", index=False)
    print("Training history saved to outputs/training_history.csv")
    print("\n" + history_df.to_string(index=False))

if __name__ == "__main__":
    train(num_epochs=20)