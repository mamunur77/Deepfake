"""
src.data.dataset  ·  PyTorch Dataset classes for Deepfake Detection.
====================================================================
Provides PyTorch Dataset implementations supporting:
1. Image Mode ("image"):
   - Each sample is an individual face crop.
   - Returns: (image_tensor, label) with shape (C, H, W).

2. Video Mode ("video"):
   - Each sample is a sequence of T frames belonging to the same video.
   - Returns: (video_tensor, label) with shape (T, C, H, W).

3. Cached Embedding Mode ("cached_embedding"):
   - Each sample is a pre-extracted feature embedding loaded from .npy.
   - Returns: (embedding_tensor, label) with shape (T, D).
   - Enables fast, lightweight training of temporal heads without spatial forward pass.
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
    "CachedEmbeddingDataset",
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
#  DeepfakeDataset (Raw Frames: Image or Video Mode)
# ═════════════════════════════════════════════════════════════════

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for raw image crops or video frame sequences.
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
            self.video_groups: list[tuple[str, int, list[str]]] = []
            return

        # Prepare index structures based on mode
        if self.mode == "video":
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
        t = self.num_frames
        if total_frames <= 0:
            return []
        if total_frames < t:
            indices = list(range(total_frames))
            pad_count = t - total_frames
            indices += [total_frames - 1] * pad_count
            return indices

        indices = np.linspace(0, total_frames - 1, t, dtype=int).tolist()
        return indices

    def _apply_transform(self, img_rgb: np.ndarray) -> torch.Tensor:
        if self.transform is not None:
            res = self.transform(image=img_rgb)
            tensor = res["image"]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.from_numpy(tensor).permute(2, 0, 1).float() / 255.0
            return tensor
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

        video_tensor = torch.stack(tensors, dim=0)
        return video_tensor, label


# ═════════════════════════════════════════════════════════════════
#  CachedEmbeddingDataset (T x D NumPy Files)
# ═════════════════════════════════════════════════════════════════

class CachedEmbeddingDataset(Dataset):
    """
    Dataset loading precomputed (T, D) video embeddings from .npy files.

    Parameters
    ----------
    manifest_csv_or_df : str | Path | pd.DataFrame
        Manifest containing 'embedding_path' and 'label'.
    num_frames : int, default=16
        Sequence length T to return. If video has fewer frames, repeats/pads.
        If more frames, samples T uniformly.
    is_train : bool, default=False
        Whether dataset is used for training.
    """

    def __init__(
        self,
        manifest_csv_or_df: str | Path | pd.DataFrame,
        num_frames: int = 16,
        is_train: bool = False,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.is_train = is_train

        if isinstance(manifest_csv_or_df, pd.DataFrame):
            self.df = manifest_csv_or_df.copy().reset_index(drop=True)
        else:
            path = Path(manifest_csv_or_df)
            if not path.exists():
                raise FileNotFoundError(f"Manifest not found: {path}")
            self.df = pd.read_csv(path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        npy_path = row["embedding_path"]
        label = int(row["label"])

        # Load features: (T_orig, D)
        embeddings = np.load(npy_path)
        t_orig, d = embeddings.shape

        t = self.num_frames
        if t_orig == 0:
            sampled = np.zeros((t, d), dtype=np.float32)
        elif t_orig < t:
            # Pad by repeating the last embedding
            pad_count = t - t_orig
            padding = np.repeat(embeddings[-1:], pad_count, axis=0)
            sampled = np.concatenate([embeddings, padding], axis=0)
        elif t_orig == t:
            sampled = embeddings
        else:
            # Uniformly sample t frames
            indices = np.linspace(0, t_orig - 1, t, dtype=int)
            sampled = embeddings[indices]

        tensor = torch.from_numpy(sampled).float()
        return tensor, label


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
    Construct DataLoaders supporting "image", "video", and "cached_embedding" modes.
    """
    mode = getattr(getattr(cfg, "data", None), "mode", mode)
    num_frames = getattr(getattr(cfg, "data", None), "num_frames", num_frames)

    # ── Branch A: Cached Embedding Mode ────────────────────────
    if mode in {"cached_embedding", "embedding", "frozen_embeddings"}:
        backbone_name = getattr(getattr(cfg, "model", None), "backbone", "vit_base_patch16_224")
        emb_dir = Path(getattr(getattr(cfg, "data", None), "embedding_dir", f"data/embeddings/{backbone_name}"))

        train_manifest = emb_dir / "train.csv"
        val_manifest = emb_dir / "val.csv"
        test_manifest = emb_dir / "test.csv"

        for p in (train_manifest, val_manifest):
            if not p.exists():
                raise FileNotFoundError(
                    f"Cached embedding manifest missing at {p}. Run scripts/cache_embeddings.py first."
                )

        train_df = pd.read_csv(train_manifest)
        val_df = pd.read_csv(val_manifest)
        test_df = pd.read_csv(test_manifest) if test_manifest.exists() else pd.DataFrame(columns=["embedding_path", "label"])

        is_debug = getattr(getattr(cfg, "debug", None), "enabled", False)
        if is_debug:
            max_s = getattr(cfg.debug, "max_samples", 50)
            train_df = train_df.iloc[:max_s]
            val_df = val_df.iloc[: min(len(val_df), max(1, max_s // 4))]
            test_df = test_df.iloc[: min(len(test_df), max(1, max_s // 4))]

        check_environment(len(train_df))

        from src.data.make_splits import compute_class_weights
        class_weights, _ = compute_class_weights(train_df["label"].tolist(), as_tensor=True)

        train_ds = CachedEmbeddingDataset(train_df, num_frames=num_frames, is_train=True)
        val_ds = CachedEmbeddingDataset(val_df, num_frames=num_frames, is_train=False)
        test_ds = CachedEmbeddingDataset(test_df, num_frames=num_frames, is_train=False)

    # ── Branch B: Raw Images or Video Frames ────────────────────
    else:
        split_dir = Path(split_dir)
        train_csv = split_dir / "train.csv"
        val_csv = split_dir / "val.csv"
        test_csv = split_dir / "test.csv"

        for p in (train_csv, val_csv, test_csv):
            if not p.exists() or p.stat().st_size < 10:
                # Check alternative candidate directories
                candidate_dirs = [
                    Path("/kaggle/working/deepfake-vit/data/splits"),
                    Path("/kaggle/working/data/splits"),
                    Path("data/splits"),
                ]
                found = False
                for c_dir in candidate_dirs:
                    cand_train = c_dir / "train.csv"
                    if cand_train.exists() and cand_train.stat().st_size > 50:
                        split_dir = c_dir
                        train_csv = split_dir / "train.csv"
                        val_csv = split_dir / "val.csv"
                        test_csv = split_dir / "test.csv"
                        found = True
                        break

                if not found:
                    try:
                        from src.data.build_ai_face_splits import auto_build_splits
                        auto_build_splits(out_dir=str(split_dir))
                        train_csv = split_dir / "train.csv"
                        val_csv = split_dir / "val.csv"
                        test_csv = split_dir / "test.csv"
                    except Exception as exc:
                        raise FileNotFoundError(
                            f"Missing or empty split CSV at {p}, and auto-build failed: {exc}"
                        )
                break

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)

        is_debug = getattr(getattr(cfg, "debug", None), "enabled", False)
        if is_debug:
            max_s = getattr(cfg.debug, "max_samples", 50)
            train_df = train_df.iloc[:max_s]
            val_df = val_df.iloc[: min(len(val_df), max(1, max_s // 4))]
            test_df = test_df.iloc[: min(len(test_df), max(1, max_s // 4))]

        check_environment(len(train_df))

        from src.data.make_splits import compute_class_weights
        class_weights, _ = compute_class_weights(train_df["label"].tolist(), as_tensor=True)

        img_size = getattr(getattr(cfg, "data", None), "img_size", 224)
        train_tf = get_train_transforms(img_size)
        val_tf = get_val_transforms(img_size)

        train_ds = DeepfakeDataset(train_df, mode=mode, num_frames=num_frames, transform=train_tf, is_train=True)
        val_ds = DeepfakeDataset(val_df, mode=mode, num_frames=num_frames, transform=val_tf, is_train=False)
        test_ds = DeepfakeDataset(test_df, mode=mode, num_frames=num_frames, transform=val_tf, is_train=False)

    # Common DataLoader construction
    batch_size = (
        getattr(cfg.debug, "batch_size", 4)
        if getattr(getattr(cfg, "debug", None), "enabled", False)
        else getattr(getattr(cfg, "training", None), "batch_size", 32)
    )
    num_workers = 0 if getattr(getattr(cfg, "debug", None), "enabled", False) else getattr(getattr(cfg, "data", None), "num_workers", 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, class_weights
