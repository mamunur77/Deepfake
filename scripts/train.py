#!/usr/bin/env python3
"""
scripts/train.py  ·  Train a ViT model for deepfake detection.
==============================================================
Loads config → builds data loaders → builds model → trains.
Enforces the Kaggle safeguard before starting.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --debug
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, apply_debug_overrides, add_config_args
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser = add_config_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug:
        cfg.debug.enabled = True
    cfg = apply_debug_overrides(cfg)
    seed_everything(cfg.experiment.seed)

    # TODO: build dataloaders  (src.data.dataset)
    # TODO: check_environment(len(train_dataset))   ← Kaggle safeguard
    # TODO: build model        (src.models.factory)
    # TODO: train              (src.training.trainer)
    print("[train] Scaffold OK – training logic not yet implemented.")


if __name__ == "__main__":
    main()
