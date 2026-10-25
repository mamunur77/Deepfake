#!/usr/bin/env python3
"""
scripts/evaluate.py  ·  Evaluate a trained checkpoint.
======================================================
Loads a checkpoint → runs inference on the test set → reports metrics.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/best.pt
    python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/best.pt --debug
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, apply_debug_overrides, add_config_args
from src.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation pipeline")
    parser = add_config_args(parser)
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a saved model checkpoint (.pt).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug:
        cfg.debug.enabled = True
    cfg = apply_debug_overrides(cfg)
    seed_everything(cfg.experiment.seed)

    # TODO: load checkpoint     (torch.load → model.load_state_dict)
    # TODO: build test loader   (src.data.dataset)
    # TODO: run inference       (src.evaluation.metrics)
    print("[evaluate] Scaffold OK – evaluation logic not yet implemented.")


if __name__ == "__main__":
    main()
