#!/usr/bin/env python3
"""
scripts/cache_embeddings.py  ·  Precompute and cache frame embeddings per video.
================================================================================
Passes face crops through a frozen spatial backbone (ViT, DeiT, DINOv2, EfficientNet)
and saves per-video feature matrices as NumPy .npy files with shape (T, D).

Generated outputs in data/embeddings/{backbone}/:
    - {split}/{video_id}.npy  (features of shape T x D)
    - {split}.csv             (manifest with embedding_path, video_id, label, num_frames)

Usage:
    # Debug run (safe on local laptop, limits samples)
    python scripts/cache_embeddings.py --debug

    # With specific backbone:
    python scripts/cache_embeddings.py --backbone dinov2_vits14 --debug
    python scripts/cache_embeddings.py --backbone vit_base_patch16_224 --debug

    # Full extraction (Kaggle):
    python scripts/cache_embeddings.py --backbone dinov2_vitb14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.transforms import get_val_transforms
from src.models.video_model import build_feature_extractor
from src.utils.config import add_config_args, apply_debug_overrides, load_config
from src.utils.safeguard import check_environment
from src.utils.seed import seed_everything


def _load_and_transform_frame(path: str | Path, transform) -> torch.Tensor:
    """Read frame from disk and apply validation transforms."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Failed to read frame: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return transform(image=rgb)["image"]


def cache_split_embeddings(
    split_name: str,
    split_csv_path: Path,
    output_base_dir: Path,
    backbone: torch.nn.Module,
    embed_dim: int,
    device: torch.device,
    transform,
    batch_size: int = 16,
    limit: int | None = None,
) -> list[dict]:
    """
    Extract and save (T, D) embeddings for all videos in a split CSV.
    """
    if not split_csv_path.exists():
        print(f"[cache_embeddings] Notice: {split_csv_path} not found. Skipping {split_name}.")
        return []

    df = pd.read_csv(split_csv_path)
    if df.empty:
        return []

    # Group by video_id
    grouped = list(df.groupby("video_id"))
    if limit is not None and limit < len(grouped):
        grouped = grouped[:limit]

    split_out_dir = output_base_dir / split_name
    split_out_dir.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict] = []
    pbar = tqdm(grouped, desc=f"Caching [{split_name}]", leave=True)

    for video_id, group in pbar:
        # Sort frame paths chronologically
        sorted_rows = group.sort_values(by="frame_path")
        frame_paths = sorted_rows["frame_path"].tolist()
        label = int(sorted_rows["label"].iloc[0])

        out_npy = split_out_dir / f"{video_id}.npy"

        # Check if already cached (resume-safe)
        if out_npy.exists():
            try:
                cached = np.load(out_npy)
                manifest_records.append({
                    "embedding_path": str(out_npy.resolve()),
                    "video_id": video_id,
                    "label": label,
                    "num_frames": cached.shape[0],
                    "embed_dim": cached.shape[1],
                })
                continue
            except Exception:
                pass

        # Batch frame tensors
        frame_tensors: list[torch.Tensor] = []
        for fp in frame_paths:
            try:
                frame_tensors.append(_load_and_transform_frame(fp, transform))
            except Exception as exc:
                continue

        if len(frame_tensors) == 0:
            continue

        # Extract features in mini-batches
        embeddings_list: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(frame_tensors), batch_size):
                batch_tensors = torch.stack(frame_tensors[i : i + batch_size]).to(device)
                feats = backbone(batch_tensors)  # (B, D)
                embeddings_list.append(feats.cpu().numpy())

        # Shape: (T, D)
        video_embeddings = np.concatenate(embeddings_list, axis=0)

        # Save to .npy
        np.save(out_npy, video_embeddings)

        manifest_records.append({
            "embedding_path": str(out_npy.resolve()),
            "video_id": video_id,
            "label": label,
            "num_frames": int(video_embeddings.shape[0]),
            "embed_dim": int(video_embeddings.shape[1]),
        })

    # Save manifest CSV for this split
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = output_base_dir / f"{split_name}.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[{split_name}] Saved {len(manifest_records)} video embeddings -> {manifest_path}")
    return manifest_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache video frame embeddings (T x D)")
    parser = add_config_args(parser)
    parser.add_argument("--backbone", type=str, default=None, help="Backbone architecture.")
    parser.add_argument("--split-dir", type=str, default="data/splits", help="Path to split CSVs.")
    parser.add_argument("--output-dir", type=str, default="data/embeddings", help="Base output directory.")
    parser.add_argument("--batch-size", type=int, default=16, help="Frame feature extraction batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos per split.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.debug:
        cfg.debug.enabled = True
    cfg = apply_debug_overrides(cfg)
    seed_everything(cfg.experiment.seed)

    backbone_name = args.backbone or getattr(getattr(cfg, "model", None), "backbone", "vit_base_patch16_224")
    split_dir = Path(args.split_dir)
    output_base_dir = Path(args.output_dir) / backbone_name
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Check hardware limits
    limit = args.limit
    if getattr(cfg.debug, "enabled", False) and limit is None:
        limit = 5

    # Check total samples for local safeguard
    check_environment(limit if limit is not None else 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cache_embeddings] Loading backbone: {backbone_name} on {device} ...")

    backbone, embed_dim = build_feature_extractor(backbone_name=backbone_name, pretrained=True)
    backbone = backbone.to(device)
    backbone.eval()

    img_size = getattr(getattr(cfg, "data", None), "img_size", 224)
    transform = get_val_transforms(img_size=img_size)

    for split in ("train", "val", "test"):
        split_csv = split_dir / f"{split}.csv"
        cache_split_embeddings(
            split_name=split,
            split_csv_path=split_csv,
            output_base_dir=output_base_dir,
            backbone=backbone,
            embed_dim=embed_dim,
            device=device,
            transform=transform,
            batch_size=args.batch_size,
            limit=limit,
        )

    print(f"\n[cache_embeddings] Completed. All embeddings saved to {output_base_dir.resolve()}")


if __name__ == "__main__":
    main()
