"""
src.data.transforms  ·  Albumentations-based augmentation pipelines.
====================================================================
Pipelines for Deepfake ViT Detection:

Train Pipeline:
    - Random Horizontal Flip
    - Small Rotation (+/- 15 degrees)
    - Color Jitter (brightness, contrast, saturation, hue)
    - JPEG Compression artifacts simulation
    - Gaussian Blur
    - Resize to target size (default: 224x224)
    - Normalize using standard ImageNet mean and std
    - ToTensorV2 (returns PyTorch tensor: (C, H, W) in [0, 1])

Val / Test Pipeline:
    - Resize to target size
    - Normalize using standard ImageNet mean and std
    - ToTensorV2
"""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "get_train_transforms",
    "get_val_transforms",
    "get_transforms",
]

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _resolve_img_size(size_or_cfg: int | Any) -> int:
    """Extract img_size whether passed as an integer or as a config namespace/dict."""
    if isinstance(size_or_cfg, int):
        return size_or_cfg
    if hasattr(size_or_cfg, "data") and hasattr(size_or_cfg.data, "img_size"):
        return int(size_or_cfg.data.img_size)
    if isinstance(size_or_cfg, dict) and "data" in size_or_cfg and "img_size" in size_or_cfg["data"]:
        return int(size_or_cfg["data"]["img_size"])
    return 224


def get_train_transforms(img_size: int | Any = 224, heavy_aug: bool = True) -> A.Compose:
    """
    Build training augmentation pipeline.

    Includes geometric transforms, color jitter, severe multi-quality
    JPEG compression, downscale/upscale degradation, and Gaussian blur/noise
    to prevent shortcut learning and boost cross-dataset generalization.
    """
    size = _resolve_img_size(img_size)

    if not heavy_aug:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5, border_mode=0),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5,
                ),
                A.ImageCompression(
                    quality_range=(60, 100),
                    p=0.5,
                ),
                A.GaussianBlur(
                    blur_limit=(3, 7),
                    p=0.3,
                ),
                A.Resize(height=size, width=size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    # ── Heavy SOTA Cross-Dataset Robustness Pipeline ──────────────────
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.06,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.10,
                p=0.6,
            ),
            # Simulate real-world web/social media compression & downsampling
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(30, 95), p=1.0),
                    A.Downscale(scale_range=(0.35, 0.85), p=1.0),
                ],
                p=0.7,
            ),
            # Simulate camera sensor noise and defocus/motion blur
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                ],
                p=0.5,
            ),
            A.Resize(height=size, width=size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int | Any = 224) -> A.Compose:
    """
    Build validation / evaluation transformation pipeline.

    Performs resize and ImageNet normalisation only.
    """
    size = _resolve_img_size(img_size)
    return A.Compose(
        [
            A.Resize(height=size, width=size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_transforms(split: str, img_size: int | Any = 224) -> A.Compose:
    """Convenience selector: returns train or val/test transforms by split name."""
    if split.lower() == "train":
        return get_train_transforms(img_size)
    return get_val_transforms(img_size)
