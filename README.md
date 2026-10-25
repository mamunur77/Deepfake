# Deepfake Image and Video Detection Using Vision Transformers
## A Comparative Study of Supervised and Self-Supervised Pretraining

> **B.Tech Major Project** · 2024-25

---

## 📋 Overview

This project investigates how **pretraining strategy** (supervised ImageNet, DINO self-supervised, MAE self-supervised) affects the performance of Vision Transformers (ViTs) on binary deepfake detection (real vs. fake).

**Key components:**
- MTCNN face extraction from images & videos.
- Fine-tuning ViT-B/16 with three pretraining variants.
- Comprehensive metrics: Accuracy, AUC-ROC, F1, Precision, Recall.
- Grad-CAM visual explanations.
- Gradio demo for single image / video inference.

---

## 🏗️ Project Structure

```
deepfake-vit/
├── configs/
│   └── default.yaml              # Central experiment config (YAML-driven)
├── data/                          # Raw + extracted data (git-ignored)
├── src/
│   ├── data/
│   │   ├── dataset.py            # PyTorch Dataset & DataLoader factory
│   │   ├── face_extract.py       # MTCNN face detection & cropping
│   │   └── transforms.py         # Albumentations augmentation pipelines
│   ├── models/
│   │   └── factory.py            # Build ViT from timm / HuggingFace
│   ├── training/
│   │   ├── trainer.py            # Training loop (AMP, early stopping)
│   │   └── losses.py             # Loss functions
│   ├── evaluation/
│   │   └── metrics.py            # Metrics computation & confusion matrix
│   ├── explain/
│   │   └── gradcam.py            # Grad-CAM visualisations
│   └── utils/
│       ├── config.py             # YAML config loader (dot-access)
│       ├── seed.py               # Reproducibility (fixed seeds)
│       └── safeguard.py          # Kaggle-vs-local hardware guard
├── scripts/
│   ├── extract_faces.py          # CLI: face extraction pipeline
│   ├── train.py                  # CLI: training pipeline
│   └── evaluate.py               # CLI: evaluation pipeline
├── notebooks/                     # Kaggle notebooks (git-ignored .ipynb)
├── app/
│   └── gradio_app.py             # Gradio web demo
├── outputs/                       # Checkpoints, logs, plots (git-ignored)
├── requirements.txt               # Pinned dependencies
├── .gitignore
└── README.md
```

---

## 🔧 Setup

```bash
# 1. Clone the repo
git clone <repo-url> && cd deepfake-vit

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Smoke-test the scaffold (local, debug mode)

```bash
# Test that config loading, seeding, and safeguard work:
python scripts/train.py --config configs/default.yaml --debug
python scripts/extract_faces.py --debug
python scripts/evaluate.py --debug --checkpoint outputs/placeholder.pt
```

### Full pipeline (Kaggle only)

```bash
# 1. Extract faces
python scripts/extract_faces.py --config configs/default.yaml

# 2. Train
python scripts/train.py --config configs/default.yaml

# 3. Evaluate
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/best.pt

# 4. Launch Gradio demo (local is fine — single image inference)
python app/gradio_app.py --config configs/default.yaml --checkpoint outputs/best.pt
```

---

## ⚠️ Hardware Safeguard

| Environment | Allowed |
|---|---|
| **Kaggle** | Unrestricted training & extraction |
| **Local laptop** (RTX 4050 6 GB) | `--debug` mode only: 50 samples, 1 epoch, batch 4 |

If you try to train locally with > 200 samples, the safeguard in [`src/utils/safeguard.py`](src/utils/safeguard.py) will **refuse and print instructions** to use Kaggle.

---

## 📦 Checkpointing

Every checkpoint saves:
1. `model.state_dict()` → `outputs/<experiment_name>/epoch_XX.pt`
2. A copy of the config YAML → `outputs/<experiment_name>/config.yaml`

This ensures full reproducibility: you can always recreate the exact model from a checkpoint + its config.

---

## 🧪 Reproducibility

- All random seeds (Python, NumPy, PyTorch CPU/CUDA) are fixed via `seed_everything(cfg.experiment.seed)`.
- `torch.backends.cudnn.deterministic = True`.
- Config YAML is saved alongside every checkpoint.

---

## 📊 Datasets

| Dataset | Source | Notes |
|---|---|---|
| FaceForensics++ | [GitHub](https://github.com/ondyari/FaceForensics) | 4 manipulation methods |
| Celeb-DF v2 | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) | Celebrity deepfakes |
| DFDC | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) | Facebook challenge dataset |

---

## 📝 License

Academic use only – B.Tech project.
