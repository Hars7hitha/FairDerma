# Melanoma Classification with Bias Mitigation and Fairness Evaluation

## Overview

This project presents a **melanoma skin lesion classifier** designed to address key challenges in medical AI, including **representation bias, data leakage, and artifact bias**.

The pipeline combines **data preprocessing, artifact removal, transfer learning, fairness evaluation, and explainable AI techniques** to build a more reliable and transparent melanoma detection system.

---

## Features

### Duplicate Detection and Leakage Prevention
- Implemented **Perceptual Hashing (pHash)** to identify and remove duplicate or near-duplicate images.
- Reduced **data leakage** between training and testing datasets.
- Improved evaluation reliability and model generalization.

### Hair Artifact Removal
- Applied **DullRazor preprocessing** for automated hair artifact detection and removal.
- Reduced visual noise caused by dermoscopic hair occlusions.
- Enhanced lesion visibility and image quality.

### Deep Learning Classification
- Utilized **ResNet50** with **transfer learning** for melanoma classification.
- Leveraged pretrained weights to improve learning efficiency and model performance.
- Fine-tuned the network on dermoscopic image datasets.

### Fairness Evaluation Module
Developed a fairness analysis framework to assess model performance across **Fitzpatrick skin tone groups**.

Metrics include:
- **True Positive Rate (TPR)**
- **Area Under Curve (AUC)**
- **Equalized Odds**

This module helps identify and quantify disparities in diagnostic performance across different skin tones.

### Explainability with Grad-CAM
- Generated **Grad-CAM heatmaps** to visualize model attention.
- Highlighted image regions influencing classification decisions.
- Improved transparency and interpretability for clinical AI systems.

## Project Goals

This project aims to:

- Improve **melanoma classification performance**
- Reduce **dataset leakage**
- Mitigate **artifact-induced bias**
- Evaluate **fairness across skin tone groups**
- Increase **model interpretability and transparency**


## Explainability

Grad-CAM visualizations highlight lesion regions that contribute most strongly to model predictions, supporting transparency and helping validate model reasoning.
