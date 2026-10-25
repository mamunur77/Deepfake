#!/usr/bin/env python3
"""
scripts/extract_faces.py  ·  Extract face crops from raw data.
=============================================================
Input  : data/raw/{real,fake}/  containing videos (.mp4, …) or images (.jpg, …)
Output : data/processed/{real,fake}/{item_id}/frame_XXXX.jpg

Features
--------
- N evenly-spaced frames per video (default 16).
- Largest-face detection via MTCNN + 20 % margin + 224×224 resize.
- Skips already-processed items (resume-safe).
- Logs items with no face to ``{output}/no_face_detected.log``.
- Progress bar via tqdm.
- Kaggle safeguard: refuses large runs on local hardware.

Usage
-----
    # Debug mode (auto-limits to 50 items, safe for RTX 4050)
    python scripts/extract_faces.py --debug

    # Local run with explicit small limit
    python scripts/extract_faces.py --limit 20

    # Full extraction (Kaggle only — safeguard blocks if >200 locally)
    python scripts/extract_faces.py --input data/raw --output data/processed --frames 16

    # Custom paths
    python scripts/extract_faces.py --input /kaggle/input/ff --output /kaggle/working/faces
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.face_extract import run_extraction, _collect_items
from src.utils.config import add_config_args, apply_debug_overrides, load_config
from src.utils.safeguard import check_environment
from src.utils.seed import seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face extraction pipeline  (MTCNN → crop → save)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser = add_config_args(parser)  # adds --config, --debug

    parser.add_argument(
        "--input", type=str, default="data/raw",
        help="Root input directory containing real/ and fake/ sub-dirs.",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed",
        help="Root output directory for face crops.",
    )
    parser.add_argument(
        "--frames", type=int, default=16,
        help="Number of evenly-spaced frames to sample per video.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many items (safety cap for local runs).",
    )
    parser.add_argument(
        "--margin", type=int, default=20,
        help="Bounding-box expansion (percent of box width/height).",
    )
    parser.add_argument(
        "--size", type=int, default=224,
        help="Output face crop size (square).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Load & apply config ───────────────────────────────────
    cfg = load_config(args.config)
    if args.debug:
        cfg.debug.enabled = True
    cfg = apply_debug_overrides(cfg)
    seed_everything(cfg.experiment.seed)

    # ── Resolve limit (debug mode auto-caps) ──────────────────
    limit = args.limit
    if getattr(cfg.debug, "enabled", False) and limit is None:
        limit = cfg.debug.max_samples
        print(f"[extract] Debug mode → auto-limit set to {limit}")

    # ── Count items & enforce hardware safeguard ──────────────
    input_dir = Path(args.input)
    all_items = _collect_items(input_dir)
    effective_count = min(len(all_items), limit) if limit else len(all_items)
    check_environment(effective_count)  # raises RuntimeError if local & >200

    # ── Pick device ───────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Run extraction ────────────────────────────────────────
    run_extraction(
        input_dir=args.input,
        output_dir=args.output,
        n_frames=args.frames,
        limit=limit,
        margin=args.margin,
        size=args.size,
        device=device,
    )


if __name__ == "__main__":
    main()
