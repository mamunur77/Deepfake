#!/usr/bin/env python3
"""
scripts/download_sota_weights.py
================================
Downloads pretrained SOTA Vision Transformer deepfake detection weights
from Hugging Face ('dima806/deepfake_vs_real_image_detection') into outputs/sota_vit_detector.
"""

from __future__ import annotations

import os
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sota_vit_detector"
MODEL_ID = "dima806/deepfake_vs_real_image_detection"


def main() -> None:
    print("=" * 70)
    print(f"  Downloading SOTA Pretrained ViT Weights: {MODEL_ID}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "model"

    print(f"[download] Fetching processor and model from Hugging Face...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

    print(f"[download] Saving weights and configuration to {save_path}...")
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)

    # Save project-compatible config
    config_data = {
        "experiment": {
            "name": "sota_vit_detector",
            "model_id": MODEL_ID,
        },
        "model": {
            "backbone": "hf_vit_base",
            "type": "hf_transformers",
            "model_path": str(save_path),
            "pretrained": True,
            "num_classes": 2,
        },
        "inference": {
            "threshold": 0.5,
        },
    }

    config_file = OUTPUT_DIR / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config_data, f)

    print(f"\n[success] Pretrained SOTA model weights successfully saved to:")
    print(f"          -> {save_path}")
    print(f"          -> {config_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
