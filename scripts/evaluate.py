#!/usr/bin/env python3
"""
scripts/evaluate.py  ·  CLI Runner to Evaluate Model Checkpoints.
================================================================
Loads a trained checkpoint → runs inference on test set (or cross-dataset) →
reports accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and saves plots.

Usage:
    # Standard evaluation
    python scripts/evaluate.py --checkpoint outputs/vit_b16_finetune/best_model.pt

    # Cross-dataset evaluation (e.g. trained on FaceForensics, tested on Celeb-DF)
    python scripts/evaluate.py --checkpoint outputs/vit_b16_finetune/best_model.pt --test_data data/splits_celebdf/test.csv

    # Local debug run (safe for RTX 4050 laptop)
    python scripts/evaluate.py --checkpoint outputs/efficientnet_b0/best_model.pt --debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluate import run_evaluation
from src.utils.config import add_config_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Checkpoint")
    parser = add_config_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved model checkpoint (.pt).")
    parser.add_argument("--test_data", "--test-data", type=str, default=None, help="Cross-dataset test CSV.")
    parser.add_argument("--split-csv", type=str, default=None, help="Explicit split CSV to evaluate.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for metrics and plots.")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    args = parser.parse_args()

    run_evaluation(
        checkpoint=args.checkpoint,
        config=args.config if args.config != "configs/default.yaml" else None,
        test_data=args.test_data,
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        debug=args.debug,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
