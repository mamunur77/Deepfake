#!/usr/bin/env python3
"""
app/gradio_app.py  ·  Gradio web UI for deepfake inference.
===========================================================
Upload a single image or short video → get real/fake prediction
with Grad-CAM overlay.

Usage:
    python app/gradio_app.py --config configs/default.yaml --checkpoint outputs/best.pt

TODO (Step 7):
    - Load model from checkpoint.
    - Build Gradio Blocks interface with:
        • Image tab   → predict + Grad-CAM.
        • Video tab   → per-frame extraction → aggregate prediction.
    - Launch on localhost (share=False by default).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, add_config_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio deepfake detection app")
    parser = add_config_args(parser)
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a saved model checkpoint (.pt).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("[gradio_app] Scaffold OK – Gradio interface not yet implemented.")
    # TODO: build and launch Gradio UI


if __name__ == "__main__":
    main()
