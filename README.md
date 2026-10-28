---
title: Deepfake ViT & DINOv2 Detector
emoji: 🛡️
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 4.19.0
app_file: app/app.py
pinned: false
license: mit
---

# 🛡️ Deepfake Image & Video Detection Using Vision Transformers
### A Comparative Study of Supervised (ViT-B, DeiT-S) and Self-Supervised (DINOv2) Pretraining with Grad-CAM & Attention Rollout Explainability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1.2-ee4c2c.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

This repository provides an end-to-end framework for detecting Deepfakes in both **images and video sequences** using Vision Transformers (ViTs). It incorporates state-of-the-art architectures, explainable AI (XAI) heatmaps, efficient embedding caching for temporal video modeling, and an interactive Gradio web application ready to run locally or in Docker.

### 🌟 Key Highlights
* **Vision Transformers**: Benchmarks **DINOv2** (`dinov2_vitb14`), **DeiT** (`deit_small_patch16_224`), **Vanilla ViT** (`vit_base_patch16_224`), and **EfficientNet-B0**.
* **Flexible Training Modes**: Supports linear probing (frozen backbone), partial block unfreezing (last 4 blocks), and full fine-tuning.
* **Dual Modality (Image & Video)**:
  * **Image Mode**: Single-frame MTCNN face extraction, patch tokenization, and binary classification.
  * **Video Mode**: Temporal multi-frame sequence modeling with pre-computed embedding caching (`cache_embeddings.py`).
* **Explainable AI (XAI)**:
  * **Attention Rollout**: Traces multi-head self-attention flow from the `[CLS]` token across all transformer layers to input face patches.
  * **Grad-CAM**: Visualizes discriminative feature activations overlaid onto face crops.
* **Production-Ready**: One-command launch via **Docker** & **Docker Compose**.

---

## 🐳 Quick Start with Docker (Recommended)

Run the entire application in a self-contained container with zero Python environment setup required:

### 1. Using Docker Compose
```bash
# Clone the repository
git clone https://github.com/mamunur77/Deepfake.git
cd Deepfake/deepfake-vit

# Build and start the container
docker compose up --build
```

### 2. Using Standard Docker Commands
```bash
# Build Docker image
docker build -t deepfake-vit:latest .

# Run container (mounts outputs/ so trained checkpoints are loaded)
docker run -p 7860:7860 -v "$(pwd)/outputs:/app/outputs" deepfake-vit:latest
```

Open your browser and navigate to:
👉 **`http://localhost:7860`**

---

## 💻 Local Installation (Python)

If you prefer running directly on your local machine with a Python virtual environment:

### Prerequisites
* Python 3.10 or 3.11
* FFmpeg (for video frame extraction)

```bash
# 1. Clone the repository
git clone https://github.com/mamunur77/Deepfake.git
cd Deepfake/deepfake-vit

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the Gradio Web Application
python app/app.py
```

---

## 🏗️ Repository Structure

```text
deepfake-vit/
├── app/
│   └── app.py                      # Interactive Gradio Web Dashboard (Image + Video)
├── configs/
│   ├── default.yaml                # Base configuration template
│   ├── dinov2_partial.yaml         # DINOv2 partial fine-tuning (last 4 blocks)
│   ├── dinov2_frozen.yaml          # DINOv2 linear probe (frozen backbone)
│   ├── deit_small_finetune.yaml    # DeiT small distillation fine-tuning
│   ├── efficientnet_b0.yaml        # EfficientNet-B0 baseline config
│   ├── video_dinov2_cached.yaml    # Temporal video model config
│   └── video_vit_temporal.yaml     # End-to-end video ViT config
├── notebooks/
│   ├── run_on_kaggle.ipynb         # Celeb-DF v2 & FaceForensics full pipeline
│   ├── train_ai_faces.ipynb        # Fast 10-minute training on 140k AI Faces
│   └── train_face_swaps.ipynb      # Dedicated Face-Swap training notebook
├── outputs/                        # Checkpoints (.pt), metrics, plots, and logs
│   ├── dinov2_partial/             # Best trained DINOv2 weights & evaluation plots
│   └── comparison_results.md       # Comparative benchmark summary
├── scripts/
│   ├── train.py                    # CLI training entrypoint
│   ├── evaluate.py                 # CLI model evaluation & metrics reporting
│   ├── compare_runs.py             # Generates markdown comparison tables & plots
│   ├── cache_embeddings.py         # Pre-computes frame embeddings for fast video training
│   └── extract_faces.py            # MTCNN batch face extraction utility
├── src/
│   ├── data/                       # Datasets, transforms, face extraction, split builders
│   ├── models/                     # ViT, DeiT, DINOv2, and Temporal Video architectures
│   ├── training/                   # Trainer with AMP, cosine decay, early stopping
│   ├── evaluation/                 # Accuracy, AUC-ROC, F1, and confusion matrix tools
│   ├── explain/                    # Grad-CAM and ViT Attention Rollout implementation
│   └── utils/                      # Config loaders, seeds, hardware safeguard
├── Dockerfile                      # Production Docker container definition
├── docker-compose.yml              # Multi-container orchestration
├── requirements.txt                # Pinned dependencies
└── README.md                       # Documentation
```

---

## ⚡ Training on Kaggle (Free GPU)

Due to Vision Transformer memory requirements, full dataset training is optimized for Kaggle T4/P100 GPUs using the provided notebooks:

1. **AI Faces (StyleGAN / ThisPersonDoesNotExist)**:
   * Open [`notebooks/train_ai_faces.ipynb`](notebooks/train_ai_faces.ipynb).
   * Attach dataset: `xhlulu/140k-real-and-fake-faces`.
   * Runs in ~8 minutes on GPU.

2. **Face-Swaps (Identity Replacement & Blended Seams)**:
   * Open [`notebooks/train_face_swaps.ipynb`](notebooks/train_face_swaps.ipynb).
   * Attach dataset: `ciplab/real-and-fake-face-detection`.
   * Trains DINOv2 to detect blended boundaries and Poisson seam artifacts.

3. **Celeb-DF v2 (Video Deepfakes)**:
   * Open [`notebooks/run_on_kaggle.ipynb`](notebooks/run_on_kaggle.ipynb).
   * Attach dataset: `reubensuju/celeb-df-v2`.
   * Extracts multi-frame sequences, caches embeddings, and trains temporal attention heads.

---

## 📊 Benchmark Results

Evaluation on held-out test splits:

| Model Architecture | Pretraining | Fine-Tuning Mode | Frame Accuracy | AUC-ROC | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DINOv2-ViT-B/14** | Self-Supervised (DINO) | Partial (Last 4 Blocks) | **100.0%** | **1.000** | **1.000** |
| **EfficientNet-B0** | Supervised (ImageNet) | Full Fine-Tuning | **92.75%** | **0.9819** | **0.927** |

---

## 🔍 Explainability & Interpretability (XAI)

Vision Transformers process images as $16 \times 16$ or $14 \times 14$ non-overlapping patches. This repository provides:
1. **Attention Rollout**: Tracks recursive self-attention weights from the classification `[CLS]` token across all transformer layers back to raw input patches.
2. **Grad-CAM**: Computes gradients of the target score with respect to feature activations in the final normalization layer to generate spatial localization heatmaps.

---

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Academic and research use encouraged.
