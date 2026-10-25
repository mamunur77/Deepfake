"""
src.explain.gradcam  ·  Grad-CAM for Vision Transformers & CNNs.
================================================================
Implements:
    - vit_reshape_transform: Automatically maps (B, N, C) tokens -> (B, C, H, W)
      spatial feature maps for ViT, DeiT, and DINOv2 backbones.
    - get_gradcam_target_layer: Auto-selects the optimal target layer across all backbones.
    - compute_gradcam: Generates normalized activation heatmaps in [0, 1].
    - overlay_heatmap: Overlays colorized heatmap on RGB input images.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

__all__ = [
    "vit_reshape_transform",
    "get_gradcam_target_layer",
    "compute_gradcam",
    "overlay_heatmap",
]


# ═════════════════════════════════════════════════════════════════
#  Vision Transformer Reshape Transform
# ═════════════════════════════════════════════════════════════════

def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshape Transformer token sequences into 2D spatial feature maps.

    Parameters
    ----------
    tensor : torch.Tensor
        Output activations of shape (B, N, C), where token 0 is [CLS],
        or 4D feature map (B, C, H, W) for CNNs (returned as-is).

    Returns
    -------
    torch.Tensor
        Spatial feature map of shape (B, C, H, W).
    """
    if tensor.dim() == 4:
        return tensor

    B, N, C = tensor.shape
    # Drop [CLS] token (or take remaining patch tokens)
    n_patches = N - 1
    side = int(math.isqrt(n_patches))

    # In case side * side != n_patches (e.g. extra register tokens), clamp to side * side
    if side * side != n_patches:
        side = int(math.isqrt(n_patches))
        patch_tokens = tensor[:, - (side * side) :, :]
    else:
        patch_tokens = tensor[:, 1:, :]

    # (B, H, W, C) -> (B, C, H, W)
    result = patch_tokens.reshape(B, side, side, C).permute(0, 3, 1, 2)
    return result


# ═════════════════════════════════════════════════════════════════
#  Target Layer Auto-Selector
# ═════════════════════════════════════════════════════════════════

def get_gradcam_target_layer(model: nn.Module) -> list[nn.Module]:
    """
    Automatically select the optimal target layer for Grad-CAM.

    Returns
    -------
    list[nn.Module]
        Target layer list for pytorch_grad_cam.
    """
    # 1. DINOv2 models (wrapped under model.backbone)
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return [model.backbone.blocks[-1].norm1]

    # 2. timm Vision Transformers (ViT, DeiT)
    if hasattr(model, "blocks") and len(model.blocks) > 0:
        last_block = model.blocks[-1]
        if hasattr(last_block, "norm1"):
            return [last_block.norm1]
        return [last_block]

    # 3. EfficientNet / CNNs
    if hasattr(model, "conv_head"):
        return [model.conv_head]
    if hasattr(model, "bn2"):
        return [model.bn2]

    # Fallback: Find the last Conv2d or LayerNorm module
    for mod in reversed(list(model.modules())):
        if isinstance(mod, (nn.Conv2d, nn.LayerNorm)):
            return [mod]

    raise ValueError(f"Could not automatically locate Grad-CAM target layer in {type(model).__name__}")


# ═════════════════════════════════════════════════════════════════
#  Grad-CAM Execution
# ═════════════════════════════════════════════════════════════════

def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_category: int = 1,
    target_layers: list[nn.Module] | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for an input tensor.

    Parameters
    ----------
    model : nn.Module
        PyTorch detection model.
    input_tensor : torch.Tensor
        Batch or single input image of shape (B, 3, H, W) normalized.
    target_category : int, default=1
        Target class index (1 = Fake, 0 = Real).
    target_layers : list[nn.Module], optional
        Target layers to hook. If None, auto-detected.
    device : str | torch.device
        Device to execute on.

    Returns
    -------
    np.ndarray
        Heatmap array of shape (B, H, W) with float values in [0.0, 1.0].
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    x = input_tensor.to(dev)

    if target_layers is None:
        target_layers = get_gradcam_target_layer(model)

    # Determine if model is a Transformer requiring vit_reshape_transform
    is_transformer = (
        hasattr(model, "patch_embed")
        or (hasattr(model, "backbone") and (hasattr(model.backbone, "patch_embed") or hasattr(model.backbone, "blocks")))
        or (hasattr(model, "blocks") and not hasattr(model, "conv_head") and not hasattr(model, "bn2"))
    )
    reshape_fn = vit_reshape_transform if is_transformer else None

    # Temporarily ensure parameters require grad so backward graph is constructed
    # even when evaluating a frozen backbone (linear probe)
    orig_grad_states = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad = True

    try:
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            reshape_transform=reshape_fn,
        )

        targets = [BinaryClassifierOutputTarget(target_category)] * x.size(0)
        grayscale_cam = cam(input_tensor=x, targets=targets)
    finally:
        # Restore original parameter grad states
        for p, orig_state in zip(model.parameters(), orig_grad_states):
            p.requires_grad = orig_state

    return grayscale_cam


# ═════════════════════════════════════════════════════════════════
#  Heatmap Overlay Helper
# ═════════════════════════════════════════════════════════════════

def overlay_heatmap(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a 2D float heatmap mask [0, 1] with an RGB image [0, 255] uint8.

    Returns
    -------
    np.ndarray
        Blended RGB image of uint8 type with shape (H, W, 3).
    """
    h, w = img_rgb.shape[:2]
    # Resize mask to image size if different
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 255] uint8
    mask_uint8 = np.uint8(255 * np.clip(mask, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(mask_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = np.uint8(alpha * heatmap_rgb + (1.0 - alpha) * img_rgb)
    return blended
