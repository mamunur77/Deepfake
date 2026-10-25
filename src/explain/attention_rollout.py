"""
src.explain.attention_rollout  ·  Attention Rollout for Vision Transformers.
=============================================================================
Implements the Attention Rollout algorithm (Abnar & Zuidema, 2020) for ViT,
DeiT, and DINOv2 architectures:
    1. Intercepts Multi-Head Self-Attention matrices across all transformer blocks.
    2. Fuses attention across heads (mean pooling).
    3. Adds residual connections: A_hat = 0.5 * A + 0.5 * I.
    4. Recursively multiplies across layers: R_l = A_hat_l @ R_{l-1}.
    5. Extracts [CLS]-to-patch attention flow, reshapes to 2D, and interpolates to input size.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["compute_attention_rollout"]


def _get_blocks_container(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    """Retrieve blocks module list from timm ViT or DINOv2 model."""
    if hasattr(model, "blocks"):
        return model.blocks
    if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
        return model.backbone.blocks
    raise ValueError(f"Model {type(model).__name__} does not contain transformer blocks.")


def compute_attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    Compute Attention Rollout map for an input image or batch.

    Parameters
    ----------
    model : nn.Module
        Vision Transformer or DINOv2 model.
    input_tensor : torch.Tensor
        Input image tensor with shape (B, 3, H, W).
    head_fusion : str, default="mean"
        Head aggregation function: "mean", "max", or "min".
    discard_ratio : float, default=0.0
        Optional ratio of lowest attention values to zero out per layer.
    device : str | torch.device
        Device for computation.

    Returns
    -------
    np.ndarray
        Array of shape (B, H, W) normalized to [0.0, 1.0].
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    x = input_tensor.to(dev)

    blocks = _get_blocks_container(model)
    attentions: list[torch.Tensor] = []
    hooks = []

    def make_attn_hook():
        def hook(module: nn.Module, inp: Any, outp: Any) -> None:
            inp_tensor = inp[0]  # (B, N, C)
            B, N, C = inp_tensor.shape
            num_heads = getattr(module, "num_heads", 6)
            head_dim = C // num_heads

            # Extract attention weights using qkv projection
            qkv = module.qkv(inp_tensor).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]  # (B, num_heads, N, head_dim)
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)  # (B, num_heads, N, N)
            attentions.append(attn.detach())

        return hook

    # Register forward hooks on all blocks
    for b in blocks:
        if hasattr(b, "attn"):
            hooks.append(b.attn.register_forward_hook(make_attn_hook()))

    # Run forward pass
    with torch.no_grad():
        _ = model(x)

    # Clean up hooks
    for h in hooks:
        h.remove()

    if not attentions:
        raise RuntimeError("No attention maps were captured. Ensure model has block.attn modules.")

    # Rollout computation across layers
    B, num_heads, N, _ = attentions[0].shape
    eye = torch.eye(N, device=dev).unsqueeze(0).repeat(B, 1, 1)  # (B, N, N)
    result = eye.clone()

    for attn in attentions:
        # 1. Fuse heads
        if head_fusion == "mean":
            fused = attn.mean(dim=1)
        elif head_fusion == "max":
            fused = attn.max(dim=1)[0]
        else:
            fused = attn.min(dim=1)[0]

        # 2. Optional top-k pruning
        if discard_ratio > 0.0:
            flat = fused.view(B, -1)
            k = int(flat.size(1) * (1 - discard_ratio))
            val, _ = torch.topk(flat, k=k, dim=-1)
            thresh = val[:, -1].view(B, 1, 1)
            fused = torch.where(fused >= thresh, fused, torch.zeros_like(fused))

        # 3. Add residual connection & re-normalize: A_hat = 0.5 * A + 0.5 * I
        a_hat = 0.5 * fused + 0.5 * eye
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)

        # 4. Multiply recursively
        result = torch.bmm(a_hat, result)

    # 5. Extract attention from [CLS] token (index 0) to all patch tokens
    cls_attention = result[:, 0, 1:]  # (B, N-1)
    side = int(math.isqrt(cls_attention.shape[1]))

    # Clamp in case of extra register tokens
    if side * side != cls_attention.shape[1]:
        side = int(math.isqrt(cls_attention.shape[1]))
        cls_attention = cls_attention[:, - (side * side) :]

    mask_2d = cls_attention.reshape(B, 1, side, side)

    # Interpolate to input image size
    H, W = input_tensor.shape[2], input_tensor.shape[3]
    upsampled = F.interpolate(mask_2d, size=(H, W), mode="bilinear", align_corners=False)
    masks = upsampled.squeeze(1).cpu().numpy()

    # Normalize each sample to [0, 1]
    for i in range(B):
        denom = masks[i].max() - masks[i].min() + 1e-8
        masks[i] = (masks[i] - masks[i].min()) / denom

    return masks
