"""
src.models.factory  ·  Model construction from configuration.
============================================================
Supports backbones:
    - vit_base_patch16_224   (timm, ImageNet-pretrained)
    - deit_small_patch16_224 (timm, ImageNet-pretrained)
    - dinov2_vits14          (torch.hub facebookresearch/dinov2)
    - dinov2_vitb14          (torch.hub facebookresearch/dinov2)
    - efficientnet_b0        (timm, CNN baseline)

Training Modes:
    - "frozen"   : Linear probe. All backbone layers frozen; only head is trainable.
    - "finetune" : Full fine-tuning. All layers in backbone and head are trainable.
    - "partial"  : Partial fine-tuning. Unfreezes the last K blocks + norm + head.

Output:
    - Single logit per sample (shape: [B, 1]) for real (0) vs. fake (1) classification.
"""

from __future__ import annotations

import logging
from typing import Any

import timm
import torch
import torch.nn as nn

__all__ = [
    "DINOV2_MODELS",
    "SUPPORTED_BACKBONES",
    "DINOv2Classifier",
    "build_model",
    "apply_training_mode",
    "count_parameters",
    "print_parameter_summary",
]

logger = logging.getLogger(__name__)

DINOV2_MODELS = {"dinov2_vits14", "dinov2_vitb14"}
SUPPORTED_BACKBONES = {
    "vit_base_patch16_224",
    "deit_small_patch16_224",
    "dinov2_vits14",
    "dinov2_vitb14",
    "efficientnet_b0",
}


# ═════════════════════════════════════════════════════════════════
#  DINOv2 Wrapper with Linear Classification Head
# ═════════════════════════════════════════════════════════════════

class DINOv2Classifier(nn.Module):
    """
    Wraps facebookresearch/dinov2 feature extractors with a binary
    classification head outputting a single logit per sample.

    Parameters
    ----------
    backbone : nn.Module
        Loaded DINOv2 vision transformer model.
    embed_dim : int
        Feature embedding dimension (384 for ViT-S, 768 for ViT-B).
    drop_rate : float, default=0.1
        Dropout probability before the linear projection.
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, drop_rate: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.head = nn.Sequential(
            nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity(),
            nn.Linear(embed_dim, 1),
        )

    @property
    def blocks(self) -> nn.ModuleList | nn.Sequential:
        """Expose backbone blocks for uniform partial freezing."""
        return self.backbone.blocks

    @property
    def norm(self) -> nn.Module:
        """Expose backbone final normalization layer."""
        return self.backbone.norm

    def get_classifier(self) -> nn.Module:
        """Return the classifier head."""
        return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images with shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Single logit output with shape (B, 1).
        """
        feat = self.backbone(x)  # (B, embed_dim)
        logits = self.head(feat)  # (B, 1)
        return logits


# ═════════════════════════════════════════════════════════════════
#  Parameter Counting & Reporting
# ═════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Count trainable and total parameters in the model.

    Returns
    -------
    tuple[int, int]
        (trainable_parameters, total_parameters)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def print_parameter_summary(model: nn.Module, model_name: str = "") -> None:
    """Print the trainable and total parameter statistics."""
    trainable, total = count_parameters(model)
    pct = (trainable / total * 100) if total > 0 else 0.0
    prefix = f"[{model_name}] " if model_name else ""
    print(
        f"{prefix}Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}% trainable)"
    )


# ═════════════════════════════════════════════════════════════════
#  Freezing & Fine-tuning Modes
# ═════════════════════════════════════════════════════════════════

def _get_classifier_modules(model: nn.Module) -> list[nn.Module]:
    """Find the classifier head module(s) for either timm or DINOv2 models."""
    if hasattr(model, "get_classifier"):
        cls_mod = model.get_classifier()
        if cls_mod is not None:
            return [cls_mod]

    for attr in ("head", "fc", "classifier"):
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                return [mod]

    return []


def apply_training_mode(
    model: nn.Module,
    mode: str = "finetune",
    unfreeze_blocks: int = 2,
) -> nn.Module:
    """
    Configure layer trainability according to the chosen mode.

    Parameters
    ----------
    model : nn.Module
        Model to configure.
    mode : str
        - "frozen"   : Only classification head is trainable (linear probe).
        - "finetune" : Entire model is trainable.
        - "partial"  : Head + last K blocks + final norm are trainable.
    unfreeze_blocks : int, default=2
        Number of trailing transformer/conv blocks to unfreeze in "partial" mode.

    Returns
    -------
    nn.Module
        The modified model with requires_grad set appropriately.
    """
    mode = mode.lower()
    if mode not in {"frozen", "finetune", "partial"}:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of: 'frozen', 'finetune', 'partial'.")

    # Mode: Full Fine-tuning
    if mode == "finetune":
        for param in model.parameters():
            param.requires_grad = True
        return model

    # For "frozen" and "partial": Start by freezing all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze the classifier head
    classifier_mods = _get_classifier_modules(model)
    for c_mod in classifier_mods:
        for param in c_mod.parameters():
            param.requires_grad = True

    # Mode: Partial Fine-tuning (unfreeze last K blocks + post-block norm)
    if mode == "partial":
        # Check for blocks container (ViT, DeiT, DINOv2, EfficientNet)
        blocks = getattr(model, "blocks", None)
        if blocks is None and hasattr(model, "backbone"):
            blocks = getattr(model.backbone, "blocks", None)

        if blocks is not None and len(blocks) > 0:
            k = min(unfreeze_blocks, len(blocks))
            # Unfreeze the last k blocks
            for block in blocks[-k:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Unfreeze final normalization layer before head if present
        for norm_attr in ("norm", "conv_head", "bn2"):
            norm_mod = getattr(model, norm_attr, None)
            if norm_mod is None and hasattr(model, "backbone"):
                norm_mod = getattr(model.backbone, norm_attr, None)
            if norm_mod is not None and isinstance(norm_mod, nn.Module):
                for param in norm_mod.parameters():
                    param.requires_grad = True

    return model


# ═════════════════════════════════════════════════════════════════
#  Main Factory: build_model
# ═════════════════════════════════════════════════════════════════

def build_model(cfg: Any = None, **kwargs) -> nn.Module:
    """
    Construct a deepfake detection model from config or keyword arguments.

    Parameters
    ----------
    cfg : Any, optional
        Namespace or dict containing a 'model' section with keys:
            - backbone : str
            - pretrained : bool
            - mode : str ("frozen", "finetune", "partial")
            - unfreeze_blocks : int
            - drop_rate : float
    **kwargs :
        Override or pass any configuration parameter directly.

    Returns
    -------
    nn.Module
        Configured PyTorch model producing a single logit [B, 1].
    """
    # Extract model config values
    model_cfg = getattr(cfg, "model", None) if cfg is not None else None
    if isinstance(cfg, dict) and "model" in cfg:
        model_cfg = cfg["model"]

    def _get(key: str, default: Any) -> Any:
        if key in kwargs:
            return kwargs[key]
        if model_cfg is not None:
            if hasattr(model_cfg, key):
                return getattr(model_cfg, key)
            if isinstance(model_cfg, dict) and key in model_cfg:
                return model_cfg[key]
        return default

    backbone_name: str = _get("backbone", "vit_base_patch16_224")
    pretrained: bool = _get("pretrained", True)
    mode: str = _get("mode", "finetune")
    unfreeze_blocks: int = _get("unfreeze_blocks", 2)
    drop_rate: float = _get("drop_rate", 0.1)

    print(f"\n[build_model] Building backbone: '{backbone_name}' | Pretrained: {pretrained} | Mode: '{mode}'")

    # 1. DINOv2 Models (facebookresearch/dinov2 via torch.hub)
    if backbone_name in DINOV2_MODELS:
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=pretrained,
        )
        embed_dim = getattr(backbone, "embed_dim", 384 if "vits" in backbone_name else 768)
        model = DINOv2Classifier(backbone=backbone, embed_dim=embed_dim, drop_rate=drop_rate)

    # 2. timm Backbones (ViT-B/16, DeiT-S/16, EfficientNet-B0, etc.)
    else:
        try:
            model = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=1,  # Single logit output for real/fake
                drop_rate=drop_rate,
            )
        except Exception as exc:
            logger.error("Failed to build timm model '%s': %s", backbone_name, exc)
            raise ValueError(f"Unsupported backbone: '{backbone_name}'") from exc

    # Apply training mode (frozen, finetune, partial)
    apply_training_mode(model, mode=mode, unfreeze_blocks=unfreeze_blocks)

    # Print trainable parameter statistics
    print_parameter_summary(model, model_name=f"{backbone_name}:{mode}")

    return model
