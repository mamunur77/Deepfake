#!/usr/bin/env python3
"""
scripts/train.py  ·  CLI Runner for Deepfake ViT Training.
=========================================================
Loads config → builds data loaders → builds model → trains with AMP & Cosine LR.
Saves model.state_dict() checkpoints and logs to CSV + TensorBoard.

Usage:
    python scripts/train.py --config configs/vit_b16_finetune.yaml --debug
    python scripts/train.py --config configs/vit_b16_finetune.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import train_from_config
from src.utils.config import add_config_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake ViT Training CLI")
    parser = add_config_args(parser)
    args = parser.parse_args()

    train_from_config(args.config, debug=args.debug)


if __name__ == "__main__":
    main()
