"""
src.data.dataset  ·  PyTorch Dataset classes for Deepfake Detection.
====================================================================
Provides PyTorch Dataset implementations supporting both:
1. Image Mode ("image"):
   - Each sample is an individual face crop.
   - Returns: (image_tensor, label) where image_tensor has shape (C, H, W).
2. Video Mode ("video"):
   - Each sample is a sequence of T frames belonging to the same video.
   - Evenly or randomly samples T frames from the video's frame pool.
   - Returns: (video_tensor, label) where video_tensor has shape (T, C, H, W).

DataLoader Factory:
    - get_dataloaders(cfg) returns train, val, and test DataLoaders along with
      computed class weights.
    - Honors debug mode limits (safeguard for local runs).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import get_train_transforms, get_val_transforms
from src.utils.safeguard import check_environment

__all__ = [
    "DeepfakeDataset",
    "get_dataloaders",
]

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Helper: Load RGB image
# ═════════════════════════════════════════════════════════════════

def _load_image(path: str | Path) -> np.ndarray:
    """Read image from path in RGB format as a uint8 NumPy array."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image at: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ═════════════════════════════════════════════════════════════════
#  DeepfakeDataset
# ═════════════════════════════════════════════════════════════════

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for Deepfake Detection.

    Parameters
    ----------
    csv_path_or_df : str | Path | pd.DataFrame
        Path to split CSV or an existing DataFrame containing:
        'frame_path', 'video_id', 'label'.
    mode : str, default="image"
        "image" for single frame crops, "video" for T-frame sequences.
    num_frames : int, default=16
        Number of frames (T) to return when mode="video".
    transform : Callable, optional
        Albumentations transform applied to each frame.
    is_train : bool, default=False
        If True and mode="video", allows random temporal jittering during frame sampling.
    """

    def __init__(
        self,
        csv_path_or_df: str | Path | pd.DataFrame,
        mode: str = "image",
        num_frames: int = 16,
        transform: Callable | None = None,
        is_train: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"image", "video"}:
            raise ValueError(f"Mode must be 'image' or 'video', got '{mode}'")

        self.mode = mode
        self.num_frames = num_frames
        self.transform = transform
        self.is_train = is_train

        # Load DataFrame
        if isinstance(csv_path_or_df, pd.DataFrame):
            self.df = csv_path_or_df.copy().reset_index(drop=True)
        else:
            path = Path(csv_path_or_df)
            if not path.exists():
                raise FileNotFoundError(f"Split CSV not found: {path}")
            self.df = pd.read_csv(path)

        if self.df.empty:
            logger.warning("DeepfakeDataset initialized with an empty dataframe.")
            self.video_groups: list[tuple[str, int, list[str]]] = []
            return

        # Prepare index structures based on mode
        if self.mode == "video":
            # Group by video_id to preserve label and all frame paths
            grouped = self.df.groupby("video_id")
            self.video_groups = []
            for video_id, group in grouped:
                label = int(group["label"].iloc[0])
                frame_paths = group["frame_path"].tolist()
                self.video_groups.append((video_id, label, frame_paths))
        else:
            self.video_groups = []

    def __len__(self) -> int:
        if self.mode == "video":
            return len(self.video_groups)
        return len(self.df)

    def _sample_frame_indices(self, total_frames: int) -> list[int]:
        """Sample num_frames indices uniformly or randomly across available frames."""
        t = self.num_frames
        if total_frames <= 0:
            return []
        if total_frames < t:
            # Pad with repeating indices
            indices = list(range(total_frames))
            pad_count = t - total_frames
            indices += [total_frames - 1] * pad_count
            return indices

        # Uniform linear sampling
        indices = np.linspace(0, total_frames - 1, t, dtype=int).tolist()
        return indices

    def _apply_transform(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Apply Albumentations transform and return (C, H, W) tensor."""
        if self.transform is not None:
            res = self.transform(image=img_rgb)
            tensor = res["image"]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.from_numpy(tensor).permute(2, 0, 1).float() / 255.0
            return tensor
        # Default fallback
        return torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.mode == "image":
            row = self.df.iloc[idx]
            frame_path = row["frame_path"]
            label = int(row["label"])

            img_rgb = _load_image(frame_path)
            tensor = self._apply_transform(img_rgb)
            return tensor, label

        # Video mode
        _video_id, label, frame_paths = self.video_groups[idx]
        sampled_indices = self._sample_frame_indices(len(frame_paths))

        tensors: list[torch.Tensor] = []
        for f_idx in sampled_indices:
            path = frame_paths[f_idx]
            img_rgb = _load_image(path)
            tensors.append(self._apply_transform(img_rgb))

        # Stack into (T, C, H, W)
        video_tensor = torch.stack(tensors, dim=0)
        return video_tensor, label


# ═════════════════════════════════════════════════════════════════
#  DataLoader Factory
# ═════════════════════════════════════════════════════════════════

def get_dataloaders(
    cfg: Any,
    split_dir: str | Path = "data/splits",
    mode: str = "image",
    num_frames: int = 16,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Construct train, val, and test DataLoaders from split CSVs.

    Parameters
    ----------
    cfg : Any
        Experiment config namespace.
    split_dir : str | Path
        Directory containing train.csv, val.csv, and test.csv.
    mode : str
        "image" or "video".
    num_frames : int
        Number of frames when mode="video".

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]
        (train_loader, val_loader, test_loader, class_weights)
    """
    split_dir = Path(split_dir)
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(f"Missing split CSV at {p}. Run src/data/make_splits.py first.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Debug mode safeguard
    is_debug = getattr(getattr(cfg, "debug", None), "enabled", False)
    if is_debug:
        max_s = getattr(cfg.debug, "max_samples", 50)
        train_df = train_df.iloc[:max_s]
        val_df = val_df.iloc[: min(len(val_df), max(1, max_s // 4))]
        test_df = test_df.iloc[: min(len(test_df), max(1, max_s // 4))]

    # Enforce hardware safeguard for local environment
    check_environment(len(train_df))

    # Compute balanced class weights for training
    from src.data.make_splits import compute_class_weights
    class_weights, _ = compute_class_weights(train_df["label"].tolist(), as_tensor=True)

    img_size = getattr(getattr(cfg, "data", None), "img_size", 224)
    train_tf = get_train_transforms(img_size)
    val_tf = get_val_transforms(img_size)

    train_dataset = DeepfakeDataset(train_df, mode=mode, num_frames=num_frames, transform=train_tf, is_train=True)
    val_dataset = DeepfakeDataset(val_df, mode=mode, num_frames=num_frames, transform=val_tf, is_train=False)
    test_dataset = DeepfakeDataset(test_df, mode=mode, num_frames=num_frames, transform=val_tf, is_train=False)

    batch_size = (
        getattr(cfg.debug, "batch_size", 4)
        if is_debug
        else getattr(getattr(cfg, "training", None), "batch_size", 32)
    )
    num_workers = (
        0 if is_debug else getattr(getattr(cfg, "data", None), "num_workers", 2)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, class_weights
