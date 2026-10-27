import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎭 Face-Swap & Deepfake Detection — Fast 10-Minute Training\n",
                "### Training DINOv2 on Real Photos vs. Blended Face-Swaps & Synthetic Faces\n",
                "\n",
                "This notebook trains **DINOv2-ViT-B/14** and **EfficientNet-B0** to catch **face-swaps** (boundary seam blending) as well as **generative AI faces**.\n",
                "Pre-extracted crops are used, so **no slow MTCNN extraction** is required."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 1: GPU Diagnostics ──────────────────────────────────────\n",
                "import torch\n",
                "print('=' * 60)\n",
                "print('  GPU Diagnostics')\n",
                "print('=' * 60)\n",
                "print('CUDA Available :', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('GPU Device     :', torch.cuda.get_device_name(0))\n",
                "    print(f'VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n",
                "else:\n",
                "    print('WARNING: Enable GPU accelerator in Settings (top-right)')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "# ── Cell 2: Dependencies & Project Setup ─────────────────────────\n",
                "!pip install -q timm albumentations grad-cam tabulate seaborn pyyaml\n",
                "\n",
                "import os\n",
                "REPO_URL = 'https://github.com/mamunur77/Deepfake.git'\n",
                "PROJECT_DIR = '/kaggle/working/deepfake-vit'\n",
                "\n",
                "if os.path.exists(PROJECT_DIR):\n",
                "    print(f'[setup] Updating repository at {PROJECT_DIR}...')\n",
                "    !cd {PROJECT_DIR} && git pull origin main\n",
                "else:\n",
                "    print(f'[setup] Cloning repository from {REPO_URL}...')\n",
                "    !git clone {REPO_URL} {PROJECT_DIR}\n",
                "\n",
                "%cd /kaggle/working/deepfake-vit\n",
                "print('[setup] Current working directory:', os.getcwd())\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "# ── Cell 3: Build Balanced Hybrid Face-Swap Splits ───────────────\n",
                "!python src/data/build_ai_face_splits.py\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "# ── Cell 4: Train DINOv2 & EfficientNet (5 Epochs, Batch 64) ──────\n",
                "import os, yaml\n",
                "\n",
                "configs = ['configs/dinov2_partial.yaml', 'configs/efficientnet_b0.yaml']\n",
                "\n",
                "for cfg_path in configs:\n",
                "    with open(cfg_path, 'r') as f:\n",
                "        cfg = yaml.safe_load(f)\n",
                "    cfg['training']['epochs'] = 5\n",
                "    cfg['training']['batch_size'] = 64\n",
                "    cfg['training']['early_stopping_patience'] = 3\n",
                "    with open(cfg_path, 'w') as f:\n",
                "        yaml.safe_dump(cfg, f)\n",
                "\n",
                "for cfg_path in configs:\n",
                "    print('\\n' + '=' * 70)\n",
                "    print(f'  Training: {cfg_path}')\n",
                "    print('=' * 70)\n",
                "    exit_code = os.system(f'python scripts/train.py --config {cfg_path}')\n",
                "    if exit_code != 0:\n",
                "        print(f'  [WARNING] Training failed for {cfg_path}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "# ── Cell 5: Evaluation & Comparison ──────────────────────────────\n",
                "!python scripts/evaluate.py --checkpoint outputs/dinov2_partial/best_model.pt --split-csv data/splits/test.csv\n",
                "!python scripts/evaluate.py --checkpoint outputs/efficientnet_b0/best_model.pt --split-csv data/splits/test.csv\n",
                "!python scripts/compare_runs.py\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Cell 6: Zip Outputs & Download Link ──────────────────────────\n",
                "import os\n",
                "from IPython.display import FileLink\n",
                "\n",
                "!zip -r /kaggle/working/deepfake_face_swaps.zip outputs/ -x '*.tfevents*'\n",
                "\n",
                "%cd /kaggle/working\n",
                "print('\\n' + '=' * 60)\n",
                "print('  Click below to download your Face-Swap detector models:')\n",
                "print('=' * 60)\n",
                "display(FileLink('deepfake_face_swaps.zip'))\n"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

out_path = Path("notebooks/train_face_swaps.ipynb")
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Created {out_path} successfully!")
