"""
src.models.video_model  ·  Temporal Transformer Models for Video Deepfake Detection.
===================================================================================
Components:
1. TemporalHead:
   - Takes input embeddings of shape (B, T, D).
   - Adds learnable positional embeddings (1, T, D).
   - Passes sequence through a 2-layer TransformerEncoder.
   - Applies temporal mean pooling across T.
   - Classifies via a linear layer returning a single logit (B, 1).

2. VideoClassifier (End-to-End):
   - Combines a 2D spatial backbone (ViT, DeiT, DINOv2, EfficientNet) with TemporalHead.
   - Takes raw video tensors of shape (B, T, C, H, W).
   - Flattens to (B*T, C, H, W) for spatial backbone feature extraction -> (B*T, D).
   - Restores sequence to (B, T, D) and forwards through TemporalHead -> (B, 1).
   - Supports freezing the spatial backbone or training end-to-end.

3. build_video_model:
   - Factory function building either standalone TemporalHead (for cached embeddings)
     or end-to-end VideoClassifier.
"""

from __future__ import annotations

import logging
from typing import Any

import timm
import torch
import torch.nn as nn

from src.models.factory import DINOV2_MODELS, count_parameters, print_parameter_summary

__all__ = [
    "TemporalHead",
    "VideoClassifier",
    "build_feature_extractor",
    "build_video_model",
]

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
#  Feature Extractor Helper
# ═════════════════════════════════════════════════════════════════

def build_feature_extractor(
    backbone_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    """
    Construct a spatial backbone that extracts a pooled feature vector (B, D).

    Returns
    -------
    tuple[nn.Module, int]
        (backbone, embed_dim)
    """
    if backbone_name in DINOV2_MODELS:
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=pretrained,
        )
        embed_dim = getattr(backbone, "embed_dim", 384 if "vits" in backbone_name else 768)
        return backbone, embed_dim
    else:
        # timm with num_classes=0 outputs pooled (B, num_features)
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        embed_dim = backbone.num_features
        return backbone, embed_dim


# ═════════════════════════════════════════════════════════════════
#  Temporal Transformer Head
# ═════════════════════════════════════════════════════════════════

class TemporalHead(nn.Module):
    """
    Temporal head with positional embeddings, 2-layer TransformerEncoder,
    mean pooling, and binary classification head.

    Parameters
    ----------
    embed_dim : int
        Feature dimension D of each frame embedding.
    num_heads : int, default=8
        Number of attention heads (must divide embed_dim).
    num_layers : int, default=2
        Number of TransformerEncoder layers.
    dim_feedforward : int, optional
        Hidden dimension in feedforward network (default: 4 * embed_dim).
    dropout : float, default=0.1
        Dropout probability.
    max_frames : int, default=64
        Maximum sequence length supported by positional embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_frames = max_frames

        # Auto-adjust num_heads if not divisible
        while num_heads > 1 and embed_dim % num_heads != 0:
            num_heads //= 2
        num_heads = max(1, num_heads)

        ff_dim = dim_feedforward or (4 * embed_dim)

        # Learnable temporal positional embeddings (1, max_frames, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames, embed_dim) * 0.02)

        # 2-layer TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head: outputs single logit (B, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Embeddings of shape (B, T, D).

        Returns
        -------
        torch.Tensor
            Single logit output with shape (B, 1).
        """
        B, T, D = x.shape
        if T > self.max_frames:
            raise ValueError(f"Sequence length T={T} exceeds max_frames={self.max_frames}")

        # Add positional embeddings
        h = x + self.pos_embed[:, :T, :]

        # Temporal TransformerEncoder
        h = self.transformer(h)
        h = self.norm(h)

        # Temporal mean pooling
        pooled = h.mean(dim=1)  # (B, D)

        # Classifier
        logits = self.classifier(pooled)  # (B, 1)
        return logits


# ═════════════════════════════════════════════════════════════════
#  End-to-End Video Classifier
# ═════════════════════════════════════════════════════════════════

class VideoClassifier(nn.Module):
    """
    End-to-end Video Classifier combining spatial 2D backbone and TemporalHead.

    Parameters
    ----------
    backbone_name : str
        Spatial backbone architecture.
    pretrained : bool
        Whether to load pretrained spatial weights.
    freeze_backbone : bool, default=False
        If True, freezes the spatial backbone parameters.
    num_heads : int, default=8
        Attention heads for TemporalHead.
    num_layers : int, default=2
        Number of transformer layers in TemporalHead.
    dropout : float, default=0.1
        Dropout probability.
    max_frames : int, default=64
        Maximum supported video frame count.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_frames: int = 64,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone

        # Build spatial backbone
        self.backbone, self.embed_dim = build_feature_extractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Build temporal transformer head
        self.temporal_head = TemporalHead(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_frames=max_frames,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input video tensor of shape (B, T, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        B, T, C, H, W = x.shape
        # Flatten B and T to extract frame features
        flat_frames = x.view(B * T, C, H, W)

        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone(flat_frames)
        else:
            feats = self.backbone(flat_frames)

        # Restore temporal dimension: (B, T, D)
        feats = feats.view(B, T, self.embed_dim)

        # Run temporal transformer head
        logits = self.temporal_head(feats)
        return logits


# ═════════════════════════════════════════════════════════════════
#  Video Model Factory
# ═════════════════════════════════════════════════════════════════

def build_video_model(cfg: Any = None, **kwargs) -> nn.Module:
    """
    Construct video model based on configuration.

    Modes supported:
        - "cached_embedding" : returns standalone TemporalHead (fast, lightweight).
        - "end_to_end"        : returns VideoClassifier (spatial backbone + TemporalHead).
    """
    model_cfg = getattr(cfg, "model", None) if cfg is not None else None
    data_cfg = getattr(cfg, "data", None) if cfg is not None else None

    def _get(key: str, default: Any) -> Any:
        if key in kwargs:
            return kwargs[key]
        if model_cfg is not None and hasattr(model_cfg, key):
            return getattr(model_cfg, key)
        if data_cfg is not None and hasattr(data_cfg, key):
            return getattr(data_cfg, key)
        return default

    backbone_name: str = _get("backbone", "vit_base_patch16_224")
    pretrained: bool = _get("pretrained", True)
    mode: str = _get("mode", "cached_embedding")
    num_heads: int = _get("num_heads", 8)
    num_layers: int = _get("num_layers", 2)
    dropout: float = _get("drop_rate", 0.1)
    max_frames: int = _get("max_frames", 64)

    # Determine embedding dimension
    if backbone_name in DINOV2_MODELS:
        embed_dim = 384 if "vits" in backbone_name else 768
    elif "efficientnet" in backbone_name:
        embed_dim = 1280
    elif "deit_small" in backbone_name:
        embed_dim = 384
    else:
        embed_dim = 768

    print(f"\n[build_video_model] Mode: '{mode}' | Backbone: '{backbone_name}' | Embed dim: {embed_dim}")

    if mode in {"cached_embedding", "embedding", "frozen_embeddings"}:
        # Standalone temporal head on cached embeddings
        model = TemporalHead(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_frames=max_frames,
        )
    else:
        # End-to-end video classifier
        freeze_backbone = (mode == "frozen_backbone" or mode == "frozen")
        model = VideoClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_frames=max_frames,
        )

    print_parameter_summary(model, model_name=f"VideoModel:{mode}")
    return model
