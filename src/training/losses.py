"""
src.training.losses  ·  Loss functions for Deepfake Detection.
==============================================================
Provides BCEWithLogitsLoss with optional positive class weighting
computed from balanced class distributions.
"""

from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn

__all__ = ["get_loss_fn"]


def get_loss_fn(
    class_weights: torch.Tensor | Sequence[float] | None = None,
    device: str | torch.device = "cpu",
) -> nn.BCEWithLogitsLoss:
    """
    Construct BCEWithLogitsLoss with positive class weighting.

    Parameters
    ----------
    class_weights : torch.Tensor | Sequence[float], optional
        Weights for [class_0 (real), class_1 (fake)].
        pos_weight is calculated as w_fake / w_real.
    device : str | torch.device
        Target device for pos_weight tensor.

    Returns
    -------
    nn.BCEWithLogitsLoss
    """
    pos_weight = None

    if class_weights is not None:
        if isinstance(class_weights, (list, tuple)):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        if len(class_weights) >= 2 and class_weights[0] > 0:
            ratio = float(class_weights[1] / class_weights[0])
            pos_weight = torch.tensor([ratio], dtype=torch.float32, device=device)

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
