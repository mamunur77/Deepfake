"""
src.training.losses  ·  Loss functions for Deepfake Detection.
==============================================================
Provides BCEWithLogitsLoss and BinaryFocalLossWithLogits with
positive class weighting and optional label smoothing.
"""

from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BinaryFocalLossWithLogits", "get_loss_fn"]


class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary Focal Loss with optional positive class weighting and label smoothing.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and focuses training on hard ambiguous boundary samples.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        modulating_factor = torch.clamp(1.0 - p_t, min=0.0, max=1.0) ** self.gamma

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            modulating_factor = alpha_t * modulating_factor

        return (modulating_factor * bce).mean()


def get_loss_fn(
    loss_type: str = "bce",
    class_weights: torch.Tensor | Sequence[float] | None = None,
    device: str | torch.device = "cpu",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> nn.Module:
    """
    Construct loss function: BCEWithLogitsLoss or BinaryFocalLossWithLogits.
    """
    pos_weight = None

    if class_weights is not None:
        if isinstance(class_weights, (list, tuple)):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        if len(class_weights) >= 2 and class_weights[0] > 0:
            ratio = float(class_weights[1] / class_weights[0])
            pos_weight = torch.tensor([ratio], dtype=torch.float32, device=device)

    if str(loss_type).lower() == "focal":
        return BinaryFocalLossWithLogits(
            alpha=focal_alpha,
            gamma=focal_gamma,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
