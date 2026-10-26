"""
src.data.make_splits  ·  Video-level stratified train / val / test splitting.
=============================================================================
Builds train, val, and test splits divided strictly by VIDEO ID to eliminate
data leakage across video frames. Splits are stratified by label (real vs. fake).

Outputs generated in data/splits/:
    - train.csv
    - val.csv
    - test.csv
    - class_weights.json (class counts, balanced weights, inverse frequencies)

CSV Schema:
    frame_path, video_id, label, label_name
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

__all__ = [
    "collect_dataset_records",
    "split_by_video_id",
    "compute_class_weights",
    "create_and_save_splits",
]

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ═════════════════════════════════════════════════════════════════
#  Data collection
# ═════════════════════════════════════════════════════════════════

def collect_dataset_records(data_dir: str | Path) -> pd.DataFrame:
    """
    Scan data_dir/{real,fake} and collect all frame records.

    Expected directory structure:
        data_dir/
            real/
                video_01/
                    frame_0000.jpg ...
            fake/
                video_02/
                    frame_0000.jpg ...

    Returns
    -------
    pd.DataFrame
        Columns: ['frame_path', 'video_id', 'label', 'label_name']
        where label is 0 for 'real' and 1 for 'fake'.
    """
    data_dir = Path(data_dir)
    records: list[dict] = []

    label_map = {"real": 0, "fake": 1}

    for label_name, label_idx in label_map.items():
        label_dir = data_dir / label_name
        if not label_dir.exists():
            continue

        for item_dir in sorted(label_dir.iterdir()):
            if item_dir.is_dir():
                video_id = item_dir.name
                frames = [
                    f for f in sorted(item_dir.iterdir())
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                ]
                for f in frames:
                    records.append(
                        {
                            "frame_path": str(f.resolve()),
                            "video_id": video_id,
                            "label": label_idx,
                            "label_name": label_name,
                        }
                    )
            elif item_dir.is_file() and item_dir.suffix.lower() in IMAGE_EXTENSIONS:
                # Direct image file under real/ or fake/
                video_id = item_dir.stem
                records.append(
                    {
                        "frame_path": str(item_dir.resolve()),
                        "video_id": video_id,
                        "label": label_idx,
                        "label_name": label_name,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No frame records found in %s", data_dir)
    return df


# ═════════════════════════════════════════════════════════════════
#  Video-level Stratified Splitting
# ═════════════════════════════════════════════════════════════════

def split_by_video_id(
    df: pd.DataFrame,
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, val, and test subsets strictly by VIDEO ID.
    No video_id will appear in more than one split.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'video_id', 'label', and 'frame_path'.
    split_ratios : tuple[float, float, float]
        (train_ratio, val_ratio, test_ratio), must sum to 1.0.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    if df.empty:
        empty = pd.DataFrame(columns=["frame_path", "video_id", "label", "label_name"])
        return empty, empty, empty

    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {split_ratios}")

    # Unique videos and their label
    video_df = (
        df[["video_id", "label", "label_name"]]
        .drop_duplicates(subset=["video_id"])
        .reset_index(drop=True)
    )

    n_videos = len(video_df)
    if n_videos < 3:
        # Fallback for minimal sample test sets: partition manually
        logger.warning("Very few videos (%d). Falling back to direct assignment.", n_videos)
        train_vids = set(video_df["video_id"].iloc[:1])
        val_vids = set(video_df["video_id"].iloc[1:2]) if n_videos > 1 else set()
        test_vids = set(video_df["video_id"].iloc[2:3]) if n_videos > 2 else set()
    else:
        # Check stratification feasibility
        class_counts = video_df["label"].value_counts()
        can_stratify = (class_counts.min() >= 2)

        temp_ratio = val_ratio + test_ratio

        try:
            train_v, temp_v = train_test_split(
                video_df,
                test_size=temp_ratio,
                random_state=seed,
                stratify=video_df["label"] if can_stratify else None,
            )

            # Second split: val vs test
            val_fraction_of_temp = val_ratio / temp_ratio
            temp_can_stratify = (temp_v["label"].value_counts().min() >= 2)

            val_v, test_v = train_test_split(
                temp_v,
                train_size=val_fraction_of_temp,
                random_state=seed,
                stratify=temp_v["label"] if temp_can_stratify else None,
            )

            train_vids = set(train_v["video_id"])
            val_vids = set(val_v["video_id"])
            test_vids = set(test_v["video_id"])
        except ValueError as exc:
            logger.warning("Stratified split failed (%s). Falling back to unstratified split.", exc)
            train_v, temp_v = train_test_split(video_df, test_size=temp_ratio, random_state=seed)
            val_v, test_v = train_test_split(
                temp_v, train_size=val_ratio / temp_ratio, random_state=seed
            )
            train_vids = set(train_v["video_id"])
            val_vids = set(val_v["video_id"])
            test_vids = set(test_v["video_id"])

    # Expand video IDs to all frame records
    train_df = df[df["video_id"].isin(train_vids)].reset_index(drop=True)
    val_df = df[df["video_id"].isin(val_vids)].reset_index(drop=True)
    test_df = df[df["video_id"].isin(test_vids)].reset_index(drop=True)

    return train_df, val_df, test_df


# ═════════════════════════════════════════════════════════════════
#  Class Weight Computation
# ═════════════════════════════════════════════════════════════════

def compute_class_weights(
    labels_or_df: Sequence[int] | pd.DataFrame | np.ndarray,
    num_classes: int = 2,
    as_tensor: bool = True,
) -> tuple[torch.Tensor | np.ndarray, dict[str, float]]:
    """
    Compute balanced class weights for loss weighting.
    Formula: weight_c = N / (C * N_c)

    Parameters
    ----------
    labels_or_df : Sequence[int] | pd.DataFrame | np.ndarray
        Labels list or DataFrame with a 'label' column.
    num_classes : int
        Number of classes (default: 2 for binary classification).
    as_tensor : bool
        If True, returns weights as a PyTorch FloatTensor.

    Returns
    -------
    tuple[torch.Tensor | np.ndarray, dict[str, float]]
        (weights, summary_dict)
    """
    if isinstance(labels_or_df, pd.DataFrame):
        labels = labels_or_df["label"].to_numpy()
    else:
        labels = np.asarray(labels_or_df)

    if len(labels) == 0:
        default_weights = np.ones(num_classes, dtype=np.float32)
        tensor_w = torch.from_numpy(default_weights) if as_tensor else default_weights
        return tensor_w, {"real": 1.0, "fake": 1.0}

    total_samples = len(labels)
    classes, counts = np.unique(labels, return_counts=True)
    count_map = dict(zip(classes, counts))

    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        c_count = count_map.get(c, 0)
        if c_count > 0:
            weights[c] = float(total_samples / (num_classes * c_count))
        else:
            weights[c] = 1.0

    summary = {
        "class_0_real": float(weights[0]) if num_classes > 0 else 1.0,
        "class_1_fake": float(weights[1]) if num_classes > 1 else 1.0,
    }

    result = torch.from_numpy(weights).float() if as_tensor else weights
    return result, summary


# ═════════════════════════════════════════════════════════════════
#  End-to-end Pipeline & Saver
# ═════════════════════════════════════════════════════════════════

def create_and_save_splits(
    data_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/splits",
    split_ratios: tuple[float, float, float] | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """
    Main orchestrator: scan data, split by video_id, compute weights, and save.
    """
    if split_ratios is None:
        split_ratios = (train_ratio, val_ratio, test_ratio)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_dataset_records(data_dir)
    if df.empty:
        print(f"[splits] Warning: No records found in {data_dir}. Empty CSVs created.")
        for name in ("train", "val", "test"):
            df.to_csv(output_dir / f"{name}.csv", index=False)
        return {"total_frames": 0, "train": 0, "val": 0, "test": 0}

    train_df, val_df, test_df = split_by_video_id(df, split_ratios=split_ratios, seed=seed)

    # Save CSVs
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    # Compute and save class weights based on the training split
    weights_tensor, summary = compute_class_weights(train_df["label"].tolist(), as_tensor=False)
    
    counts_map = train_df["label_name"].value_counts().to_dict() if not train_df.empty else {}

    meta = {
        "split_ratios": {"train": split_ratios[0], "val": split_ratios[1], "test": split_ratios[2]},
        "seed": seed,
        "total_frames": len(df),
        "total_videos": int(df["video_id"].nunique()),
        "split_counts_frames": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "split_counts_videos": {
            "train": int(train_df["video_id"].nunique()),
            "val": int(val_df["video_id"].nunique()),
            "test": int(test_df["video_id"].nunique()),
        },
        "train_class_counts_frames": counts_map,
        "class_weights": {
            "balanced": weights_tensor.tolist(),
            "class_0_real": float(weights_tensor[0]),
            "class_1_fake": float(weights_tensor[1]) if len(weights_tensor) > 1 else 1.0,
        },
    }

    with open(output_dir / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("  Splits Generation Summary (Stratified by Video ID)")
    print("=" * 60)
    print(f"  Total Videos : {meta['total_videos']}  |  Total Frames: {meta['total_frames']}")
    print(f"  Train        : {meta['split_counts_videos']['train']} videos, {meta['split_counts_frames']['train']} frames")
    print(f"  Val          : {meta['split_counts_videos']['val']} videos, {meta['split_counts_frames']['val']} frames")
    print(f"  Test         : {meta['split_counts_videos']['test']} videos, {meta['split_counts_frames']['test']} frames")
    print(f"  Class Weights: Real = {weights_tensor[0]:.4f}, Fake = {weights_tensor[1]:.4f}")
    print(f"  Output Dir   : {output_dir.resolve()}")
    print("=" * 60 + "\n")

    return meta["split_counts_frames"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create video-level stratified splits.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed faces.")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Destination for split CSVs.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    create_and_save_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
