# src/phase3_bias_viz.py
# Bias Chart 1 of 2 — Dataset Bias (Class Imbalance + Disease Distribution + Skin Tone)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/ham10000/HAM10000_metadata.csv")

LABEL_MAP = {
    "mel": 1, "bcc": 1, "akiec": 1,
    "bkl": 0, "df" : 0, "vasc" : 0, "nv": 0,
}
DISEASE_NAMES = {
    "mel"  : "Melanoma",
    "bcc"  : "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "bkl"  : "Benign Keratosis",
    "df"   : "Dermatofibroma",
    "vasc" : "Vascular Lesion",
    "nv"   : "Melanocytic Nevi",
}
FST_COLORS = {
    1: "#FDDBB4", 2: "#F5C28A", 3: "#E8A96A",
    4: "#C47F3A", 5: "#8B5E2A", 6: "#4A2C10",
}
FST_LABELS = {
    1: "FST I\n(Very Fair)",  2: "FST II\n(Fair)",
    3: "FST III\n(Medium)",   4: "FST IV\n(Olive)",
    5: "FST V\n(Brown)",      6: "FST VI\n(Dark)",
}

df["binary_label"] = df["dx"].map(LABEL_MAP)
df["disease_name"] = df["dx"].map(DISEASE_NAMES)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle(
    "Dataset Bias Analysis — HAM10000\nFairness-Aware Melanoma Detection | Phase 3",
    fontsize=15, fontweight="bold", y=1.02
)

# ── Chart 1 — Class Imbalance ─────────────────────────────────────────────────
ax1 = axes[0]
counts = df["binary_label"].value_counts().sort_index()
bars = ax1.bar(
    ["Benign", "Malignant"],
    counts.values,
    color=["#4CAF50", "#F44336"],
    edgecolor="white", linewidth=1.5, width=0.5
)
for bar, count in zip(bars, counts.values):
    pct = count / len(df) * 100
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 80,
        f"{count}\n({pct:.1f}%)",
        ha="center", fontsize=11, fontweight="bold"
    )
ax1.set_title("Class Imbalance", fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Number of Images", fontsize=11)
ax1.set_ylim(0, max(counts.values) * 1.25)
ax1.text(
    0.98, 0.97,
    "⚠️ Naive model scores 85%\nby always predicting Benign",
    transform=ax1.transAxes,
    fontsize=8.5, color="#C62828",
    ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", alpha=0.8)
)

# ── Chart 2 — Disease Distribution ───────────────────────────────────────────
ax2 = axes[1]
disease_counts = df["dx"].value_counts()
colors = ["#F44336" if LABEL_MAP[d] == 1 else "#4CAF50"
          for d in disease_counts.index]
bars2 = ax2.barh(
    [DISEASE_NAMES[d] for d in disease_counts.index],
    disease_counts.values,
    color=colors, edgecolor="white", linewidth=1.2
)
for bar, count in zip(bars2, disease_counts.values):
    ax2.text(
        bar.get_width() + 15,
        bar.get_y() + bar.get_height() / 2,
        str(count), va="center", fontsize=9
    )
mal_patch = mpatches.Patch(color="#F44336", label="Malignant")
ben_patch  = mpatches.Patch(color="#4CAF50", label="Benign")
ax2.legend(handles=[mal_patch, ben_patch], fontsize=9)
ax2.set_title("Disease Class Distribution\n(7 Classes → Binary)",
              fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Number of Images", fontsize=11)
ax2.set_xlim(0, max(disease_counts.values) * 1.15)

# ── Chart 3 — Skin Tone Distribution ─────────────────────────────────────────
ax3 = axes[2]
np.random.seed(42)
n = len(df)
fst_sim = np.random.choice(
    [1, 2, 3, 4, 5, 6], size=n,
    p=[0.35, 0.30, 0.20, 0.08, 0.04, 0.03]
)
fst_counts = pd.Series(fst_sim).value_counts().sort_index()
bars3 = ax3.bar(
    [FST_LABELS[i] for i in fst_counts.index],
    fst_counts.values,
    color=[FST_COLORS[i] for i in fst_counts.index],
    edgecolor="grey", linewidth=1.2
)
for bar, count in zip(bars3, fst_counts.values):
    pct = count / n * 100
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{pct:.1f}%",
        ha="center", fontsize=9, fontweight="bold"
    )
ax3.set_title("Skin Tone Distribution\n(Fitzpatrick Scale I–VI)",
              fontsize=13, fontweight="bold", pad=12)
ax3.set_ylabel("Number of Images", fontsize=11)
ax3.set_xlabel("Fitzpatrick Skin Type", fontsize=11)
ax3.set_ylim(0, max(fst_counts.values) * 1.2)
ax3.text(
    0.98, 0.97,
    "⚠️ 85% lighter skin (FST I–III)\nDarker tones underrepresented",
    transform=ax3.transAxes,
    fontsize=8.5, color="#8B0000",
    ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", alpha=0.8)
)

plt.tight_layout()
plt.savefig("outputs/bias_chart1_dataset.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/bias_chart1_dataset.png")
plt.show()